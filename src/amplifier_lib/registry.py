"""Bundle registry — central bundle management for amplifier-lib.

Handles registration, loading, caching, and update checking for bundles.
Uses ``AMPLIFIER_HOME`` env var or defaults to ``~/.amplifier``.

Filesystem layout under home::

    home/
    ├── registry.json   # Persisted state
    └── cache/          # Cached remote bundles

Quick start::

    registry = BundleRegistry()
    registry.register({"foundation": "git+https://github.com/microsoft/amplifier-foundation@main"})
    bundle = await registry.load("foundation")
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from amplifier_lib._utils import (
    parse_frontmatter,
    read_yaml,
)
from amplifier_lib.bundle import Bundle
from amplifier_lib.exceptions import BundleDependencyError, BundleLoadError, BundleNotFoundError
from amplifier_lib.paths import get_amplifier_home

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ANSI colours (terminal warning panels)
# ---------------------------------------------------------------------------


class _Colors:
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    BOX_TOP = "─"
    BOX_SIDE = "│"


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BundleState:
    """Tracked state for a registered bundle.

    Terminology:
        Root bundle:   A bundle at ``/bundle.md`` or ``/bundle.yaml`` at the
            root of a repo or directory tree.  Establishes the namespace and
            root directory for path resolution.  Tracked via ``is_root=True``.

        Nested bundle: A bundle loaded via ``#subdirectory=`` URIs or
            ``@namespace:path`` references.  Shares the namespace with its
            root bundle.  Tracked via ``is_root=False``.
    """

    uri: str
    name: str
    version: str | None = None
    loaded_at: datetime | None = None
    checked_at: datetime | None = None
    local_path: str | None = None
    includes: list[str] | None = None
    included_by: list[str] | None = None
    is_root: bool = True
    root_name: str | None = None
    explicitly_requested: bool = False
    app_bundle: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dict."""
        result: dict[str, Any] = {
            "uri": self.uri,
            "name": self.name,
            "version": self.version,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "local_path": self.local_path,
            "is_root": self.is_root,
            "explicitly_requested": self.explicitly_requested,
            "app_bundle": self.app_bundle,
        }
        if self.includes:
            result["includes"] = self.includes
        if self.included_by:
            result["included_by"] = self.included_by
        if self.root_name:
            result["root_name"] = self.root_name
        return result

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> BundleState:
        """Create from a JSON dict."""
        return cls(
            uri=data["uri"],
            name=name,
            version=data.get("version"),
            loaded_at=datetime.fromisoformat(data["loaded_at"]) if data.get("loaded_at") else None,
            checked_at=(
                datetime.fromisoformat(data["checked_at"]) if data.get("checked_at") else None
            ),
            local_path=data.get("local_path"),
            includes=data.get("includes"),
            included_by=data.get("included_by"),
            is_root=data.get("is_root", True),
            root_name=data.get("root_name"),
            explicitly_requested=data.get("explicitly_requested", False),
            app_bundle=data.get("app_bundle", False),
        )


@dataclass
class UpdateInfo:
    """Information about an available bundle update."""

    name: str
    current_version: str | None
    available_version: str
    uri: str


# ---------------------------------------------------------------------------
# BundleRegistry
# ---------------------------------------------------------------------------


class BundleRegistry:
    """Central bundle management for the Amplifier ecosystem.

    Handles registration, loading, caching, and update checking.

    Example::

        registry = BundleRegistry()
        registry.register({"foundation": "git+https://github.com/..."})
        bundle = await registry.load("foundation")
    """

    def __init__(self, home: Path | None = None, *, strict: bool = False) -> None:
        """Initialise the registry.

        Args:
            home: Base directory.  Resolved in order:
                  1. Explicit *home* parameter.
                  2. ``AMPLIFIER_HOME`` environment variable.
                  3. ``~/.amplifier`` (default).
            strict: If ``True``, include failures raise exceptions instead of
                logging warnings.  Useful for CI / validation workflows.
        """
        self._home = self._resolve_home(home)
        self._strict = strict
        self._registry: dict[str, BundleState] = {}
        self._source_resolver = self._build_source_resolver()
        # Cache: fully loaded bundles + in-progress futures for deduplication.
        self._loaded_bundles: dict[str, Bundle] = {}
        self._pending_loads: dict[str, asyncio.Future[Bundle]] = {}
        self._load_persisted_state()
        self._validate_cached_paths()

    @property
    def home(self) -> Path:
        """Base directory for all registry data."""
        return self._home

    def _resolve_home(self, home: Path | None) -> Path:
        if home is not None:
            return home.expanduser().resolve()
        return get_amplifier_home()

    def _build_source_resolver(self) -> Any:
        """Construct a SimpleSourceResolver for URI resolution."""
        from amplifier_lib.sources import SimpleSourceResolver

        return SimpleSourceResolver(
            cache_dir=self._home / "cache",
            base_path=Path.cwd(),
        )

    # =========================================================================
    # Registration
    # =========================================================================

    def register(self, bundles: dict[str, str]) -> None:
        """Register name → URI mappings.

        Always accepts a dict.  Overwrites existing registrations for the
        same names.  Does **not** persist automatically — call
        :meth:`save` to persist.

        Args:
            bundles: Dict of ``{name: uri}`` pairs,
                e.g. ``{"foundation": "git+https://..."}``.
        """
        for name, uri in bundles.items():
            existing = self._registry.get(name)
            if existing:
                existing.uri = uri
            else:
                self._registry[name] = BundleState(uri=uri, name=name)
            logger.debug("Registered bundle: %s → %s", name, uri)

    def find(self, name: str) -> str | None:
        """Return the URI registered under *name*, or None."""
        state = self._registry.get(name)
        return state.uri if state else None

    def list_registered(self) -> list[str]:
        """Return a sorted list of all registered bundle names."""
        return sorted(self._registry.keys())

    def unregister(self, name: str) -> bool:
        """Remove a bundle from the in-memory registry.

        Cleans up cross-references in related entries.
        Does **not** persist automatically — call :meth:`save` to persist.
        Does **not** delete cached files.

        Returns:
            ``True`` if the bundle was found and removed, ``False`` otherwise.
        """
        if name not in self._registry:
            return False

        state = self._registry[name]

        if state.includes:
            for child_name in state.includes:
                child = self._registry.get(child_name)
                if child and child.included_by:
                    child.included_by = [n for n in child.included_by if n != name]

        if state.included_by:
            for parent_name in state.included_by:
                parent = self._registry.get(parent_name)
                if parent and parent.includes:
                    parent.includes = [n for n in parent.includes if n != name]

        del self._registry[name]
        logger.debug("Unregistered bundle: %s", name)
        return True

    # =========================================================================
    # Loading
    # =========================================================================

    async def load(
        self,
        name_or_uri: str | None = None,
        *,
        auto_register: bool = True,
    ) -> Bundle | dict[str, Bundle]:
        """Load bundle(s).

        Args:
            name_or_uri: Registered name, direct URI, or ``None`` to load all.
            auto_register: If ``True``, direct-URI loads register using the
                bundle's extracted name.

        Returns:
            A single :class:`~amplifier_lib.bundle.Bundle` when *name_or_uri*
            is provided, or a ``{name: Bundle}`` dict when it is ``None``.
        """
        if name_or_uri is None:
            names = self.list_registered()
            if not names:
                return {}

            results = await asyncio.gather(
                *[self._load_single(n, auto_register=False) for n in names],
                return_exceptions=True,
            )

            bundles: dict[str, Bundle] = {}
            for name, result in zip(names, results, strict=True):
                if isinstance(result, Exception):
                    logger.warning("Failed to load bundle '%s': %s", name, result)
                else:
                    bundles[name] = result  # type: ignore[assignment]

            return bundles

        return await self._load_single(name_or_uri, auto_register=auto_register)

    async def _load_single(
        self,
        name_or_uri: str,
        *,
        auto_register: bool = True,
        auto_include: bool = True,
        refresh: bool = False,  # noqa: ARG002 — reserved for future cache bypass
        _loading_chain: frozenset[str] | None = None,
    ) -> Bundle:
        """Load a single bundle by name or URI.

        Args:
            name_or_uri: Registered bundle name or direct URI.
            auto_register: Register URI bundles by extracted name.
            auto_include: Load and compose included bundles.
            refresh: Bypass cache (reserved for future use).
            _loading_chain: Internal per-chain cycle-detection state.

        Raises:
            BundleNotFoundError: Bundle could not be located.
            BundleLoadError: Bundle found but failed to load.
        """
        registered_name: str | None = None
        uri: str

        if name_or_uri in self._registry:
            registered_name = name_or_uri
            uri = self._registry[name_or_uri].uri
        else:
            uri = name_or_uri

        base_uri = uri.split("#")[0] if "#" in uri else uri
        loading_chain = _loading_chain or frozenset()

        # 1. Fast path: already cached.
        if not refresh and uri in self._loaded_bundles:
            return self._loaded_bundles[uri]

        # 2. True circular dependency check (not subdirectory self-references).
        is_subdirectory = "#subdirectory=" in uri
        if not is_subdirectory and (uri in loading_chain or base_uri in loading_chain):
            raise BundleDependencyError(f"Circular dependency detected: {uri}")

        # 3. Diamond deduplication: await an in-progress load for the same URI.
        if uri in self._pending_loads:
            return await self._pending_loads[uri]

        # 4. Start a new load, protected by a Future for concurrent dedup.
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Bundle] = loop.create_future()
        self._pending_loads[uri] = future

        try:
            new_chain: frozenset[str]
            if is_subdirectory:
                new_chain = loading_chain | {uri}
            else:
                new_chain = loading_chain | {uri, base_uri}

            # Resolve URI → local path.
            resolved = await self._source_resolver.resolve(uri)
            if resolved is None:
                raise BundleNotFoundError(f"Could not resolve URI: {uri}")

            local_path: Path = resolved.active_path

            bundle = await self._load_from_path(local_path)

            # Detect nested bundles by walking up to find a root bundle.
            if local_path.is_file():
                search_start = local_path.parent.parent
            else:
                search_start = local_path.parent

            cache_root = Path.home() / ".amplifier" / "cache"
            stop_boundary = resolved.source_root if resolved.source_root else cache_root

            root_bundle_path = self._find_nearest_bundle_file(
                start=search_start, stop=stop_boundary
            )

            bundle_dir = local_path.parent if local_path.is_file() else local_path
            root_bundle_dir = root_bundle_path.parent if root_bundle_path else None

            root_bundle: Bundle | None = None
            if root_bundle_path and root_bundle_dir != bundle_dir:
                root_bundle = await self._load_from_path(root_bundle_path)
                if root_bundle.name:
                    bundle.source_base_paths[root_bundle.name] = resolved.source_root
                    logger.debug(
                        "Nested bundle '%s' registered root namespace @%s: → %s",
                        bundle.name,
                        root_bundle.name,
                        resolved.source_root,
                    )

                    if root_bundle.name not in self._registry:
                        root_uri = uri.split("#")[0] if "#" in uri else uri
                        self._registry[root_bundle.name] = BundleState(
                            uri=root_uri,
                            name=root_bundle.name,
                            version=root_bundle.version,
                            loaded_at=datetime.now(),
                            local_path=str(root_bundle_path.parent),
                            is_root=True,
                            root_name=None,
                        )
                        logger.debug("Registered root bundle: %s", root_bundle.name)

                if bundle.name and bundle.name != root_bundle.name:
                    bundle.source_base_paths[bundle.name] = resolved.source_root
                    logger.debug(
                        "Nested bundle also registered own namespace @%s: → %s",
                        bundle.name,
                        resolved.source_root,
                    )

            is_root_bundle = True
            root_bundle_name: str | None = None
            if root_bundle and root_bundle.name and root_bundle.name != bundle.name:
                is_root_bundle = False
                root_bundle_name = root_bundle.name

            # Register this bundle for namespace resolution before processing
            # its includes (self-referencing includes need the name available).
            if bundle.name and bundle.name not in self._registry:
                self._registry[bundle.name] = BundleState(
                    uri=uri,
                    name=bundle.name,
                    version=bundle.version,
                    loaded_at=datetime.now(),
                    local_path=str(local_path),
                    is_root=is_root_bundle,
                    root_name=root_bundle_name,
                )
                logger.debug(
                    "Registered bundle for namespace resolution: %s (is_root=%s, root_name=%s)",
                    bundle.name,
                    is_root_bundle,
                    root_bundle_name,
                )

            # Update state for known (pre-registered) bundles.
            update_name = registered_name or (
                bundle.name if bundle.name in self._registry else None
            )
            if update_name:
                state = self._registry[update_name]
                # Don't let a subdirectory bundle overwrite its root's entry —
                # the root URI is authoritative for update tracking.
                if state.is_root and "#subdirectory=" in uri:
                    logger.debug(
                        "Skipping registry update for '%s': root entry preserved "
                        "over subdirectory load",
                        update_name,
                    )
                else:
                    if state.uri != uri:
                        logger.debug("Updating URI for '%s': %s → %s", update_name, state.uri, uri)
                        state.uri = uri
                    state.version = bundle.version
                    state.loaded_at = datetime.now()
                    state.local_path = str(local_path)

            # Recursively load and compose includes.
            if auto_include and bundle.includes:
                bundle = await self._compose_includes(
                    bundle, parent_name=bundle.name, _loading_chain=new_chain
                )

            # Tag the bundle with its source URI for update checking.
            bundle._source_uri = uri

            self._loaded_bundles[uri] = bundle
            future.set_result(bundle)
            return bundle

        except Exception:
            future.cancel()
            raise
        finally:
            self._pending_loads.pop(uri, None)

    async def _load_from_path(self, path: Path) -> Bundle:
        """Load a bundle from a local path (file or directory).

        Raises:
            BundleLoadError: If the path cannot be parsed as a bundle.
        """
        if path.is_dir():
            bundle_md = path / "bundle.md"
            bundle_yaml = path / "bundle.yaml"
            if bundle_md.exists():
                return await self._load_markdown_bundle(bundle_md)
            if bundle_yaml.exists():
                return await self._load_yaml_bundle(bundle_yaml)
            raise BundleLoadError(f"Not a valid bundle: missing bundle.md or bundle.yaml in {path}")

        if path.suffix == ".md":
            return await self._load_markdown_bundle(path)
        if path.suffix in (".yaml", ".yml"):
            return await self._load_yaml_bundle(path)

        raise BundleLoadError(f"Unknown bundle format: {path}")

    async def _load_markdown_bundle(self, path: Path) -> Bundle:
        """Parse a markdown bundle file (YAML frontmatter + body)."""
        content = path.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)
        bundle = Bundle.from_dict(frontmatter, base_path=path.parent)
        bundle.instruction = body.strip() if body.strip() else None
        return bundle

    async def _load_yaml_bundle(self, path: Path) -> Bundle:
        """Parse a YAML bundle file."""
        data = await read_yaml(path)
        if data is None:
            data = {}
        return Bundle.from_dict(data, base_path=path.parent)

    # =========================================================================
    # Include composition
    # =========================================================================

    async def _compose_includes(
        self,
        bundle: Bundle,
        parent_name: str | None = None,
        _loading_chain: frozenset[str] | None = None,
    ) -> Bundle:
        """Load and compose all included bundles into *bundle*.

        Args:
            bundle: The bundle whose ``includes`` list should be processed.
            parent_name: Name of the parent bundle (for relationship tracking).
            _loading_chain: Internal per-chain cycle-detection state.
        """
        if not bundle.includes:
            return bundle

        # Pre-load namespace bundles so their local_path is available for
        # namespace:path resolution before we dispatch parallel loads.
        await self._preload_namespace_bundles(bundle.includes, _loading_chain)

        # Phase 1: resolve all include sources.
        include_sources: list[str] = []
        for include in bundle.includes:
            include_source = self._parse_include(include)
            if not include_source:
                continue
            try:
                resolved_source = self._resolve_include_source(include_source)
                if resolved_source is None:
                    if ":" in include_source and "://" not in include_source:
                        namespace = include_source.split(":")[0]
                        if self._registry.get(namespace):
                            raise BundleDependencyError(
                                f"Include resolution failed: '{include_source}'. "
                                f"Namespace '{namespace}' is registered but the path doesn't exist."
                            )
                    if self._strict:
                        raise BundleDependencyError(
                            f"Include resolution failed (strict mode): '{include_source}' "
                            f"could not be resolved (unregistered namespace)"
                        )
                    logger.warning("Include skipped (unregistered namespace): %s", include_source)
                    continue
                include_sources.append(resolved_source)
            except BundleNotFoundError:
                if self._strict:
                    raise BundleDependencyError(
                        f"Include not found (strict mode): '{include_source}'"
                    ) from None
                logger.warning("Include not found (skipping): %s", include_source)

        if not include_sources:
            return bundle

        # Phase 2: load all includes in parallel.
        tasks = [
            self._load_single(
                source,
                auto_register=True,
                auto_include=True,
                _loading_chain=_loading_chain,
            )
            for source in include_sources
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        included_bundles: list[Bundle] = []
        included_names: list[str] = []
        for source, result in zip(include_sources, results):
            if isinstance(result, BaseException):
                if isinstance(result, BundleDependencyError):
                    self._log_circular_dependency_warning(source, result, _loading_chain)
                else:
                    if self._strict:
                        raise BundleDependencyError(
                            f"Include failed to load (strict mode): '{source}' - {result}"
                        ) from result
                    source_name = self._extract_bundle_name(source)
                    lines = [f"Bundle: {source_name}", "", str(result)]
                    logger.warning(self._format_warning_panel("Include Failed (skipping)", lines))
            else:
                included_bundles.append(result)  # type: ignore[arg-type]
                if result.name:  # type: ignore[union-attr]
                    included_names.append(result.name)  # type: ignore[union-attr]

        if not included_bundles:
            return bundle

        if parent_name and included_names:
            self._record_include_relationships(parent_name, included_names)

        # Compose: includes first, then current bundle overrides.
        result_bundle = included_bundles[0]
        for included in included_bundles[1:]:
            result_bundle = result_bundle.compose(included)

        return result_bundle.compose(bundle)

    def _record_include_relationships(self, parent_name: str, child_names: list[str]) -> None:
        """Record which bundles include which other bundles and persist state."""
        parent_state = self._registry.get(parent_name)
        if parent_state:
            if parent_state.includes is None:
                parent_state.includes = []
            for child_name in child_names:
                if child_name not in parent_state.includes:
                    parent_state.includes.append(child_name)

        for child_name in child_names:
            child_state = self._registry.get(child_name)
            if child_state:
                if child_state.included_by is None:
                    child_state.included_by = []
                if parent_name not in child_state.included_by:
                    child_state.included_by.append(parent_name)

        self.save()
        logger.debug("Recorded include relationships: %s includes %s", parent_name, child_names)

    async def _preload_namespace_bundles(
        self,
        includes: list[Any],
        _loading_chain: frozenset[str] | None = None,
    ) -> None:
        """Pre-load namespace bundles to populate their ``local_path``.

        When processing ``namespace:path`` includes we need the namespace
        bundle's local_path to resolve the path component.  This method
        sequentially loads any namespace bundles that are registered but not
        yet cached.
        """
        namespaces_to_load: set[str] = set()

        for include in includes:
            include_source = self._parse_include(include)
            if not include_source:
                continue
            if ":" in include_source and "://" not in include_source:
                namespace = include_source.split(":")[0]
                state = self._registry.get(namespace)
                if state and not state.local_path:
                    if _loading_chain:
                        ns_uri = state.uri
                        ns_base = ns_uri.split("#")[0] if "#" in ns_uri else ns_uri
                        if ns_uri in _loading_chain or ns_base in _loading_chain:
                            logger.debug(
                                "Skipping preload of '%s' — already in loading chain", namespace
                            )
                            continue
                    namespaces_to_load.add(namespace)

        for namespace in namespaces_to_load:
            try:
                logger.debug("Pre-loading namespace bundle: %s", namespace)
                await self._load_single(
                    namespace,
                    auto_register=True,
                    auto_include=False,
                    _loading_chain=_loading_chain,
                )
            except BundleDependencyError as e:
                logger.debug("Namespace preload skipped (circular): %s — %s", namespace, e)
            except Exception as e:
                raise BundleDependencyError(
                    f"Cannot resolve includes: namespace '{namespace}' failed to load. "
                    f"Original error: {e}"
                ) from e

    def _resolve_include_source(self, source: str) -> str | None:
        """Resolve an include source string to a loadable URI.

        Resolution priority:
          1. Explicit URIs (``git+``, ``http://``, ``https://``, ``file://``)
             → returned as-is.
          2. ``namespace:path`` syntax
             → construct ``git+…#subdirectory=…`` URI or ``file://`` fallback.
          3. Plain names → returned as-is for registry lookup.

        Returns ``None`` if a ``namespace:path`` reference cannot be resolved.
        """
        if "://" in source or source.startswith("git+"):
            return source

        if ":" in source:
            namespace, rel_path = source.split(":", 1)
            state = self._registry.get(namespace)
            if not state:
                logger.debug("Namespace '%s' not found in registry", namespace)
                return None

            if state.uri and state.uri.startswith("git+"):
                base_uri = state.uri.split("#")[0]
                if state.local_path:
                    ns_path = Path(state.local_path)
                    resource_base = (
                        ns_path.parent / rel_path if ns_path.is_file() else ns_path / rel_path
                    )
                    resolved_path = self._find_resource_path(resource_base)
                    if resolved_path:
                        root_base = ns_path.parent if ns_path.is_file() else ns_path
                        rel_from_root = resolved_path.relative_to(root_base)
                        return f"{base_uri}#subdirectory={rel_from_root}"
                    logger.debug(
                        "Namespace '%s' is git-based but path '%s' not found locally",
                        namespace,
                        rel_path,
                    )
                    return None
                else:
                    logger.debug(
                        "Namespace '%s' has no local_path yet, constructing URI directly for '%s'",
                        namespace,
                        rel_path,
                    )
                    return f"{base_uri}#subdirectory={rel_path}"

            # Fallback to file:// for non-git sources.
            if state.local_path:
                ns_path = Path(state.local_path)
                resource_base = (
                    ns_path.parent / rel_path if ns_path.is_file() else ns_path / rel_path
                )
                resolved_path = self._find_resource_path(resource_base)
                if resolved_path:
                    return f"file://{resolved_path}"
                logger.debug(
                    "Namespace '%s' found but path '%s' not found within it",
                    namespace,
                    rel_path,
                )
            else:
                logger.debug("Namespace '%s' has no local_path", namespace)

            return None

        return source

    def _find_resource_path(self, base_path: Path) -> Path | None:
        """Try *base_path* with common extensions and return the first match."""
        candidates = [
            base_path,
            base_path.with_suffix(".yaml"),
            base_path.with_suffix(".yml"),
            base_path.with_suffix(".md"),
            base_path / "bundle.yaml",
            base_path / "bundle.md",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _parse_include(self, include: str | dict[str, Any]) -> str | None:
        """Parse an include directive to a source string."""
        if isinstance(include, str):
            return include
        if isinstance(include, dict):
            bundle_ref = include.get("bundle")
            if bundle_ref:
                return str(bundle_ref)
        return None

    def _find_nearest_bundle_file(self, start: Path, stop: Path) -> Path | None:
        """Walk upward from *start* to *stop* looking for ``bundle.md`` / ``.yaml``."""
        current = start.resolve()
        stop = stop.resolve()

        while current >= stop:
            for name in ("bundle.md", "bundle.yaml"):
                candidate = current / name
                if candidate.exists():
                    return candidate
            if current == stop:
                break
            current = current.parent

        return None

    # =========================================================================
    # Update checking
    # =========================================================================

    async def check_update(
        self,
        name: str | None = None,
    ) -> UpdateInfo | list[UpdateInfo] | None:
        """Check for available updates.

        Args:
            name: Bundle name, or ``None`` to check all registered bundles.

        Returns:
            :class:`UpdateInfo` if an update is available (single name), or
            a list of :class:`UpdateInfo` objects (``name=None``), or ``None``
            if up-to-date.
        """
        if name is None:
            names = self.list_registered()
            if not names:
                return []

            results = await asyncio.gather(
                *[self._check_update_single(n) for n in names],
                return_exceptions=True,
            )

            updates: list[UpdateInfo] = []
            for n, result in zip(names, results, strict=True):
                if isinstance(result, Exception):
                    logger.warning("Failed to check update for '%s': %s", n, result)
                elif result is not None:
                    updates.append(result)  # type: ignore[arg-type]

            return updates

        return await self._check_update_single(name)

    async def _check_update_single(self, name: str) -> UpdateInfo | None:
        """Check for updates on a single bundle; update ``checked_at``.

        Delegates to :func:`~amplifier_lib.updates.check_bundle_status` for
        real ``git ls-remote`` based update detection and returns an
        :class:`UpdateInfo` when any source has an available update.
        """
        state = self._registry.get(name)
        if not state:
            return None

        # Must have a loaded bundle to inspect its sources.
        bundle = self._loaded_bundles.get(state.uri)
        if bundle is None:
            state.checked_at = datetime.now()
            logger.debug("Checked for updates: %s (not loaded, skipping)", name)
            return None

        from amplifier_lib.updates import check_bundle_status

        try:
            status = await check_bundle_status(bundle, cache_dir=self._home / "cache")
        except Exception as exc:
            logger.warning("Update check failed for '%s': %s", name, exc)
            state.checked_at = datetime.now()
            return None

        state.checked_at = datetime.now()

        if status.has_updates:
            update_summaries = [s.summary for s in status.updateable_sources]
            logger.info(
                "Update available for '%s': %s",
                name,
                "; ".join(update_summaries),
            )
            return UpdateInfo(
                name=name,
                current_version=state.version,
                available_version=f"{state.version or '?'}+update",
                uri=state.uri,
            )

        logger.debug("Checked for updates: %s (up to date)", name)
        return None

    async def update(
        self,
        name: str | None = None,
    ) -> Bundle | dict[str, Bundle]:
        """Update to the latest version (bypasses cache).

        Args:
            name: Bundle name, or ``None`` to update all registered bundles.

        Raises:
            KeyError: If *name* is not registered (not raised when ``name=None``).
        """
        if name is None:
            names = self.list_registered()
            if not names:
                return {}

            results = await asyncio.gather(
                *[self._update_single(n) for n in names],
                return_exceptions=True,
            )

            bundles: dict[str, Bundle] = {}
            for n, result in zip(names, results, strict=True):
                if isinstance(result, Exception):
                    logger.warning("Failed to update bundle '%s': %s", n, result)
                else:
                    bundles[n] = result  # type: ignore[assignment]

            return bundles

        return await self._update_single(name)

    async def _update_single(self, name: str) -> Bundle:
        """Force-reload a single bundle, bypassing the bundle cache."""
        state = self._registry.get(name)
        if not state:
            raise KeyError(f"Bundle '{name}' not registered")

        # Invalidate bundle cache to force re-fetch.
        cached_uri = state.uri
        self._loaded_bundles.pop(cached_uri, None)

        bundle = await self._load_single(name, auto_register=False, refresh=True)
        state.version = bundle.version
        state.loaded_at = datetime.now()
        state.checked_at = datetime.now()
        return bundle

    # =========================================================================
    # State
    # =========================================================================

    def get_state(
        self,
        name: str | None = None,
    ) -> BundleState | dict[str, BundleState] | None:
        """Return tracked state for one or all bundles.

        Args:
            name: Bundle name, or ``None`` to return all.

        Returns:
            :class:`BundleState` (single name), ``{name: BundleState}``
            (``name=None``), or ``None`` if not registered.
        """
        if name is None:
            return dict(self._registry)
        return self._registry.get(name)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self) -> None:
        """Persist registry state to ``home/registry.json``."""
        self._home.mkdir(parents=True, exist_ok=True)
        registry_path = self._home / "registry.json"

        data = {
            "version": 1,
            "bundles": {name: state.to_dict() for name, state in self._registry.items()},
        }

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.debug("Saved registry to %s", registry_path)

    def _load_persisted_state(self) -> None:
        """Load persisted registry state from disk (best-effort)."""
        registry_path = self._home / "registry.json"
        if not registry_path.exists():
            return

        try:
            with open(registry_path, encoding="utf-8") as f:
                data = json.load(f)

            for name, bundle_data in data.get("bundles", {}).items():
                self._registry[name] = BundleState.from_dict(name, bundle_data)

            logger.debug(
                "Loaded registry from %s (%d bundles)",
                registry_path,
                len(self._registry),
            )
        except Exception as e:
            logger.warning("Failed to load registry from %s: %s", registry_path, e)

    def _validate_cached_paths(self) -> None:
        """Clear stale ``local_path`` references from registry entries.

        On startup, persisted entries may point to cached paths that no longer
        exist (e.g. user cleared the cache).  This clears those stale
        references so bundles will be re-fetched on demand.
        """
        stale: list[str] = []
        for name, state in self._registry.items():
            if state.local_path and not Path(state.local_path).exists():
                logger.info("Clearing stale cache reference for '%s'", name)
                state.local_path = None
                stale.append(name)

        if stale:
            self.save()

    # =========================================================================
    # Pre-warm  (used by amplifierd on startup)
    # =========================================================================

    async def _prewarm(self) -> None:
        """Pre-warm the bundle cache by eagerly loading all registered bundles.

        Called by amplifierd during app startup so that the first real request
        does not pay the cost of fetching and composing bundles.  Failures are
        logged but do not prevent startup.
        """
        names = self.list_registered()
        if not names:
            return

        logger.debug("Pre-warming %d registered bundle(s)…", len(names))
        results = await asyncio.gather(
            *[self._load_single(n, auto_register=False) for n in names],
            return_exceptions=True,
        )
        for name, result in zip(names, results, strict=True):
            if isinstance(result, Exception):
                logger.warning("Pre-warm failed for bundle '%s': %s", name, result)
            else:
                logger.debug("Pre-warm: '%s' ready", name)

    # =========================================================================
    # Warning / logging helpers
    # =========================================================================

    def _format_warning_panel(self, title: str, lines: list[str]) -> str:
        """Format a warning as a bordered panel for terminal visibility."""
        max_line = max((len(line) for line in lines), default=0)
        width = min(80, max(60, max_line + 4))
        border = _Colors.YELLOW + _Colors.BOX_TOP * width + _Colors.RESET

        parts = [
            "",
            border,
            f"{_Colors.YELLOW}{_Colors.BOLD}{title}{_Colors.RESET}",
            border,
            *lines,
            border,
            "",
        ]
        return "\n".join(parts)

    def _log_circular_dependency_warning(
        self,
        source: str,
        error: BundleDependencyError,
        loading_chain: frozenset[str] | None,
    ) -> None:
        """Log a helpful circular-dependency warning with resolution guidance."""
        source_name = self._extract_bundle_name(source)
        if loading_chain:
            chain_names = [self._extract_bundle_name(uri) for uri in sorted(loading_chain)]
            chain_str = " → ".join(chain_names)
        else:
            chain_str = "unknown"

        lines = [
            f"Bundle: {source_name}",
            f"Chain: {chain_str} → {source_name} (cycle)",
            "",
            "This include was skipped. The bundle will load without it.",
            "To fix: Check includes in the chain for circular references.",
        ]
        logger.warning(self._format_warning_panel("Circular Include Skipped", lines))

    def _extract_bundle_name(self, uri: str) -> str:
        """Extract a readable bundle name from a URI for log messages."""
        if "github.com" in uri:
            parts = uri.split("/")
            for i, part in enumerate(parts):
                if "github.com" in part and i + 2 < len(parts):
                    return parts[i + 2].split("@")[0].split("#")[0]
        if uri.startswith("file://"):
            return uri.split("/")[-1].split("#")[0]
        return uri.split("/")[-1].split("@")[0].split("#")[0]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


async def load_bundle(
    source: str,
    *,
    auto_include: bool = True,
    registry: BundleRegistry | None = None,
    strict: bool = False,
) -> Bundle:
    """Load a bundle from *source* (URI or registered name).

    Args:
        source: URI or bundle name.
        auto_include: Whether to load and compose ``includes``.
        registry: Optional :class:`BundleRegistry` instance.  A default
            registry is created when not provided.
        strict: If ``True``, include failures raise exceptions instead of
            logging warnings.  Ignored when *registry* is provided.

    Raises:
        ValueError: If both *registry* and *strict* are provided.
    """
    if registry is not None and strict:
        raise ValueError(
            "Cannot pass strict=True with an existing registry. "
            "Configure strict mode on the BundleRegistry directly."
        )
    if registry is None:
        registry = BundleRegistry(strict=strict)
    return await registry._load_single(source, auto_register=True, auto_include=auto_include)
