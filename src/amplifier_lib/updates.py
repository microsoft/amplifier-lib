"""Bundle update detection and upgrade utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from amplifier_lib.paths import get_amplifier_home
from amplifier_lib.sources.git import GitSourceHandler
from amplifier_lib.sources.protocol import SourceStatus
from amplifier_lib.sources.uri import parse_uri

if TYPE_CHECKING:
    from amplifier_lib.bundle import Bundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BundleStatus
# ---------------------------------------------------------------------------


@dataclass
class BundleStatus:
    """Status of a bundle's sources for update detection.

    Attributes:
        bundle_name: Name of the bundle.
        bundle_source: Source URI of the bundle itself (if any).
        sources: Status of each individual source URI in the bundle.
    """

    bundle_name: str
    bundle_source: str | None
    sources: list[SourceStatus] = field(default_factory=list)

    @property
    def has_updates(self) -> bool:
        """True if any source has an available update."""
        return any(s.has_update is True for s in self.sources)

    @property
    def updateable_sources(self) -> list[SourceStatus]:
        """Sources that have updates available."""
        return [s for s in self.sources if s.has_update is True]

    @property
    def up_to_date_sources(self) -> list[SourceStatus]:
        """Sources that are already up to date."""
        return [s for s in self.sources if s.has_update is False]

    @property
    def unknown_sources(self) -> list[SourceStatus]:
        """Sources whose update status could not be determined."""
        return [s for s in self.sources if s.has_update is None]

    @property
    def summary(self) -> str:
        """Human-readable summary of bundle update status."""
        total = len(self.sources)
        if total == 0:
            return f"Bundle '{self.bundle_name}': no tracked sources"

        updateable = len(self.updateable_sources)
        up_to_date = len(self.up_to_date_sources)
        unknown = len(self.unknown_sources)

        if updateable > 0:
            parts = [f"{updateable} update(s) available"]
            if up_to_date:
                parts.append(f"{up_to_date} up to date")
            if unknown:
                parts.append(f"{unknown} unknown")
            return f"Bundle '{self.bundle_name}': {', '.join(parts)}"

        if unknown > 0 and up_to_date == 0:
            return f"Bundle '{self.bundle_name}': {unknown} source(s) with unknown status"

        return f"Bundle '{self.bundle_name}': all {up_to_date} source(s) up to date"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_cache_dir() -> Path:
    """Return the default amplifier cache directory."""
    return get_amplifier_home() / "cache"


def _collect_source_uris(bundle: Bundle) -> list[str]:
    """Collect all source URIs referenced by a bundle.

    Gathers URIs from:
    - bundle.session orchestrator and context (if they have a ``source`` key)
    - bundle.providers, bundle.tools, bundle.hooks (each item's ``source`` key)
    - bundle._source_uri (the URI the bundle was loaded from, if any)

    Args:
        bundle: The bundle to inspect.

    Returns:
        Deduplicated list of source URI strings (order preserved).
    """
    seen: set[str] = set()
    uris: list[str] = []

    def _add(uri: str) -> None:
        if uri and uri not in seen:
            seen.add(uri)
            uris.append(uri)

    # Session-level modules
    session = bundle.session or {}
    for key in ("orchestrator", "context"):
        entry = session.get(key)
        if isinstance(entry, dict):
            src = entry.get("source")
            if src:
                _add(src)

    # Provider / tool / hook lists
    for section in (bundle.providers, bundle.tools, bundle.hooks):
        for item in section or []:
            if isinstance(item, dict):
                src = item.get("source")
                if src:
                    _add(src)

    # Bundle's own source URI (set by registry when loading from a remote)
    bundle_source_uri: str | None = bundle._source_uri
    if bundle_source_uri:
        _add(bundle_source_uri)

    return uris


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def check_bundle_status(
    bundle: Bundle,
    cache_dir: Path | None = None,
) -> BundleStatus:
    """Check whether a bundle's sources have available updates.

    For each source URI that uses the ``git+`` scheme, queries the remote
    to determine whether new commits are available.  Non-git sources are
    reported with ``has_update=None`` (unknown).

    Args:
        bundle: The bundle to inspect.
        cache_dir: Override the cache directory (defaults to
            ``~/.amplifier/cache``).

    Returns:
        :class:`BundleStatus` describing the update state of each source.
    """
    effective_cache_dir = cache_dir or _get_cache_dir()
    git_handler = GitSourceHandler()

    source_uris = _collect_source_uris(bundle)
    bundle_source_uri: str | None = bundle._source_uri

    statuses: list[SourceStatus] = []

    for uri in source_uris:
        parsed = parse_uri(uri)
        if git_handler.can_handle(parsed):
            try:
                status = await git_handler.get_status(parsed, effective_cache_dir)
            except Exception as exc:
                status = SourceStatus(
                    source_uri=uri,
                    is_cached=False,
                    has_update=None,
                    error=str(exc),
                    summary=f"Error checking status: {exc}",
                )
        else:
            # Non-git sources: update status is unknown
            status = SourceStatus(
                source_uri=uri,
                is_cached=False,
                has_update=None,
                summary="Update check not supported for this source type",
            )
        statuses.append(status)

    return BundleStatus(
        bundle_name=bundle.name,
        bundle_source=bundle_source_uri,
        sources=statuses,
    )


async def update_bundle(
    bundle: Bundle,
    cache_dir: Path | None = None,
    selective: list[str] | None = None,
    install_deps: bool = True,
) -> Bundle:
    """Update a bundle's sources to their latest versions.

    Fetches the latest content for all (or selected) git sources in the
    bundle.  Optionally reinstalls Python dependencies afterwards.

    Args:
        bundle: The bundle to update.
        cache_dir: Override the cache directory (defaults to
            ``~/.amplifier/cache``).
        selective: If provided, only update sources whose URI is in this list.
            URIs not in this list are left as-is even if updates are available.
        install_deps: Whether to reinstall Python dependencies after updating
            (via :class:`~amplifier_lib.modules.activator.ModuleActivator`).

    Returns:
        The same ``bundle`` object (updated in place by re-caching sources).
    """
    effective_cache_dir = cache_dir or _get_cache_dir()
    git_handler = GitSourceHandler()

    bundle_status = await check_bundle_status(bundle, cache_dir=effective_cache_dir)

    # Determine which sources to update
    sources_to_update = bundle_status.updateable_sources
    if selective is not None:
        selective_set = set(selective)
        sources_to_update = [s for s in sources_to_update if s.source_uri in selective_set]

    for source_status in sources_to_update:
        uri = source_status.source_uri
        parsed = parse_uri(uri)
        if not git_handler.can_handle(parsed):
            logger.debug("Skipping non-git source: %s", uri)
            continue

        logger.info("Updating source: %s", uri)
        try:
            await git_handler.update(parsed, effective_cache_dir)
            logger.info("Updated source: %s", uri)
        except Exception as exc:
            logger.warning("Failed to update source %s: %s", uri, exc)

    if sources_to_update:
        # Invalidate install-state fingerprints for modules whose git sources
        # were updated. Without this, the stale fingerprint would cause
        # _install_dependencies to skip reinstallation even though the
        # underlying pyproject.toml/requirements.txt may have changed.
        from amplifier_lib.modules.install_state import InstallStateManager

        install_state = InstallStateManager(effective_cache_dir)
        git_handler_for_paths = GitSourceHandler()
        for source_status in sources_to_update:
            uri = source_status.source_uri
            parsed = parse_uri(uri)
            if git_handler_for_paths.can_handle(parsed):
                cache_path = git_handler_for_paths._get_cache_path(parsed, effective_cache_dir)
                if cache_path.exists():
                    install_state.invalidate(cache_path)
                    logger.debug("Invalidated install state for updated source: %s", uri)
        install_state.save()

        if install_deps:
            from amplifier_lib.modules.activator import ModuleActivator

            activator = ModuleActivator(cache_dir=effective_cache_dir, install_deps=True)
            if bundle.base_path:
                await activator._install_dependencies(bundle.base_path)

    return bundle
