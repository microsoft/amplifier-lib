"""Bundle dataclass — the core composable unit in amplifier-lib.

A Bundle contains mount-plan configuration and resources.  Multiple bundles
can be composed together (``bundle.compose(*others)``) to build a merged
configuration that is then prepared for execution via ``bundle.prepare()``.

Public types
------------
Bundle            — Composable configuration unit.
PreparedBundle    — A fully-activated bundle ready for session creation.
BundleModuleResolver — Maps module IDs to activated local paths.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable

from amplifier_lib._utils import (
    construct_context_path,
    deep_merge,
    merge_module_lists,
    parse_frontmatter,
)
from amplifier_lib.exceptions import BundleValidationError
from amplifier_lib.spawn_utils import ProviderPreference, apply_provider_preferences_with_resolution

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bundle dataclass
# ---------------------------------------------------------------------------


@dataclass
class Bundle:
    """Composable unit containing mount plan config and resources.

    Bundles are the core composable unit in amplifier-lib.  They hold mount
    plan configuration and resources, producing mount plans for
    AmplifierSession.

    Attributes:
        name: Bundle name (namespace for @mentions).
        version: Bundle version string.
        description: Optional description.
        includes: List of bundle URIs to include.
        session: Session config (orchestrator, context).
        providers: List of provider configs.
        tools: List of tool configs.
        hooks: List of hook configs.
        spawn: Spawn config (exclude_tools, etc.).
        agents: Dict mapping agent name to definition.
        context: Dict mapping context name to file path.
        instruction: System instruction from markdown body.
        base_path: Path to bundle root directory.
        source_base_paths: Dict mapping namespace to base_path for @mention
            resolution.  Tracks original base_path for each bundle during
            composition, enabling ``@namespace:path`` references to resolve
            correctly to source files.
    """

    # Metadata
    name: str
    version: str = "1.0.0"
    description: str = ""
    includes: list[str] = field(default_factory=list)

    # Mount-plan sections
    session: dict[str, Any] = field(default_factory=dict)
    providers: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    hooks: list[dict[str, Any]] = field(default_factory=list)
    spawn: dict[str, Any] = field(default_factory=dict)

    # Resources
    agents: dict[str, dict[str, Any]] = field(default_factory=dict)
    context: dict[str, Path] = field(default_factory=dict)
    instruction: str | None = None

    # Internal
    base_path: Path | None = None
    source_base_paths: dict[str, Path] = field(default_factory=dict)
    _pending_context: dict[str, str] = field(default_factory=dict)
    _source_uri: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Guard against callers passing None for collection fields."""
        if self.context is None:
            self.context = {}
        if self.source_base_paths is None:
            self.source_base_paths = {}
        if self._pending_context is None:
            self._pending_context = {}

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self, *others: Bundle) -> Bundle:
        """Compose this bundle with others (later overrides earlier).

        Creates a new Bundle with merged configuration:
          - ``session`` / ``spawn``: deep merge (nested dicts merged, scalars
            from *later* bundle win).
          - ``providers`` / ``tools`` / ``hooks``: merge by module ID.
          - ``agents``: later overrides earlier (by agent name).
          - ``context``: accumulates with bundle-name prefix.
          - ``instruction``: later replaces earlier.

        Args:
            others: Additional bundles to compose in order.

        Returns:
            New Bundle with merged configuration.
        """
        # Seed source_base_paths from self
        initial_base_paths = dict(self.source_base_paths) if self.source_base_paths else {}
        if self.name and self.base_path and self.name not in initial_base_paths:
            initial_base_paths[self.name] = self.base_path

        # Prefix self's context keys with bundle name to avoid collisions
        initial_context: dict[str, Path] = {}
        for key, path in self.context.items():
            prefixed = f"{self.name}:{key}" if self.name and ":" not in key else key
            initial_context[prefixed] = path

        initial_pending: dict[str, str] = (
            dict(self._pending_context) if self._pending_context else {}
        )

        result = Bundle(
            name=self.name,
            version=self.version,
            description=self.description,
            includes=list(self.includes),
            session=dict(self.session),
            providers=list(self.providers),
            tools=list(self.tools),
            hooks=list(self.hooks),
            spawn=dict(self.spawn),
            agents=dict(self.agents),
            context=initial_context,
            _pending_context=initial_pending,
            instruction=self.instruction,
            base_path=self.base_path,
            source_base_paths=initial_base_paths,
        )

        for other in others:
            # Merge source_base_paths (registry-set values like source_root
            # take precedence via the first-write-wins rule)
            if other.source_base_paths:
                for ns, path in other.source_base_paths.items():
                    if ns not in result.source_base_paths:
                        result.source_base_paths[ns] = path

            if other.name and other.base_path and other.name not in result.source_base_paths:
                result.source_base_paths[other.name] = other.base_path

            # Metadata: later wins
            result.name = other.name or result.name
            result.version = other.version or result.version
            if other.description:
                result.description = other.description

            # Config sections
            result.session = deep_merge(result.session, other.session)
            result.spawn = deep_merge(result.spawn, other.spawn)

            # Module lists: merge by module ID
            result.providers = merge_module_lists(result.providers, other.providers)
            result.tools = merge_module_lists(result.tools, other.tools)
            result.hooks = merge_module_lists(result.hooks, other.hooks)

            # Agents: later overrides
            result.agents.update(other.agents)

            # Context: accumulate with bundle-name prefix
            for key, path in other.context.items():
                prefixed = f"{other.name}:{key}" if other.name and ":" not in key else key
                result.context[prefixed] = path

            # Pending context: accumulate
            if other._pending_context:
                result._pending_context.update(other._pending_context)

            # Instruction: later replaces
            if other.instruction:
                result.instruction = other.instruction

            # base_path: use the incoming bundle's path (ensures @AGENTS.md
            # resolves to user's project, not a cached copy)
            if other.base_path:
                result.base_path = other.base_path

        return result

    # ------------------------------------------------------------------
    # Mount plan
    # ------------------------------------------------------------------

    def to_mount_plan(self) -> dict[str, Any]:
        """Compile to a mount plan dict for AmplifierSession."""
        plan: dict[str, Any] = {}

        if self.session:
            plan["session"] = dict(self.session)
        if self.providers:
            plan["providers"] = list(self.providers)
        if self.tools:
            plan["tools"] = list(self.tools)
        if self.hooks:
            plan["hooks"] = list(self.hooks)
        if self.agents:
            plan["agents"] = dict(self.agents)
        if self.spawn:
            plan["spawn"] = dict(self.spawn)

        return plan

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    async def prepare(
        self,
        install_deps: bool = True,
        source_resolver: Callable[[str, str], str] | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> PreparedBundle:
        """Prepare bundle for execution by activating all modules.

        Downloads / installs all modules specified in the bundle's mount plan
        and returns a :class:`PreparedBundle` that can create sessions.

        Args:
            install_deps: Whether to install Python dependencies for modules.
            source_resolver: Optional ``(module_id, original_source) ->
                resolved_source`` callback for app-layer source overrides.
            progress_callback: Optional ``(action, detail)`` progress callback.

        Returns:
            :class:`PreparedBundle` with mount_plan and create_session() helper.
        """
        from amplifier_lib.modules.activator import ModuleActivator

        mount_plan = self.to_mount_plan()

        activator = ModuleActivator(install_deps=install_deps, base_path=self.base_path)

        if install_deps:
            if self.base_path:
                await activator.activate_bundle_package(
                    self.base_path, progress_callback=progress_callback
                )
            for _ns, bundle_path in self.source_base_paths.items():
                if bundle_path and bundle_path != self.base_path:
                    await activator.activate_bundle_package(
                        bundle_path, progress_callback=progress_callback
                    )

        # Collect all modules that require activation.
        modules_to_activate: list[dict[str, Any]] = []

        def resolve_source(mod_spec: dict[str, Any]) -> dict[str, Any]:
            if source_resolver and "module" in mod_spec and "source" in mod_spec:
                resolved = source_resolver(mod_spec["module"], mod_spec["source"])
                if resolved != mod_spec["source"]:
                    return {**mod_spec, "source": resolved}
            return mod_spec

        # Session orchestrator / context
        session_config = mount_plan.get("session", {})
        if isinstance(session_config.get("orchestrator"), dict):
            orch = session_config["orchestrator"]
            if "source" in orch:
                modules_to_activate.append(resolve_source(orch))
        if isinstance(session_config.get("context"), dict):
            ctx = session_config["context"]
            if "source" in ctx:
                modules_to_activate.append(resolve_source(ctx))

        # Providers, tools, hooks
        for section in ("providers", "tools", "hooks"):
            for mod_spec in mount_plan.get(section, []):
                if isinstance(mod_spec, dict) and "source" in mod_spec:
                    modules_to_activate.append(resolve_source(mod_spec))

        # Agent-specific modules (pre-activated for child sessions)
        for _agent_name, agent_def in mount_plan.get("agents", {}).items():
            if not isinstance(agent_def, dict):
                continue

            agent_session = agent_def.get("session", {})
            if isinstance(agent_session, dict):
                agent_orch = agent_session.get("orchestrator")
                if isinstance(agent_orch, dict) and "source" in agent_orch:
                    modules_to_activate.append(resolve_source(agent_orch))
                agent_ctx = agent_session.get("context")
                if isinstance(agent_ctx, dict) and "source" in agent_ctx:
                    modules_to_activate.append(resolve_source(agent_ctx))

            for agent_section in ("providers", "tools", "hooks"):
                agent_mods = agent_def.get(agent_section, [])
                if isinstance(agent_mods, list):
                    for mod_spec in agent_mods:
                        if isinstance(mod_spec, dict) and "source" in mod_spec:
                            modules_to_activate.append(resolve_source(mod_spec))

        module_paths = await activator.activate_all(
            modules_to_activate, progress_callback=progress_callback
        )
        activator.finalize()

        resolver = BundleModuleResolver(module_paths, activator=activator)
        bundle_package_paths = activator.bundle_package_paths

        return PreparedBundle(
            mount_plan=mount_plan,
            resolver=resolver,
            bundle=self,
            bundle_package_paths=bundle_package_paths,
        )

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def resolve_context_path(self, name: str) -> Path | None:
        """Resolve a context file by name, returning its Path or None."""
        if name in self.context:
            return self.context[name]
        if self.base_path:
            path = construct_context_path(self.base_path, name)
            if path.exists():
                return path
        return None

    def resolve_agent_path(self, name: str) -> Path | None:
        """Resolve an agent markdown file by name.

        Handles both namespaced names (``"foundation:bug-hunter"``) and
        simple names (``"bug-hunter"``).
        """
        if ":" in name:
            namespace, simple_name = name.split(":", 1)
            if namespace in self.source_base_paths:
                p = self.source_base_paths[namespace] / "agents" / f"{simple_name}.md"
                if p.exists():
                    return p
            if namespace == self.name and self.base_path:
                p = self.base_path / "agents" / f"{simple_name}.md"
                if p.exists():
                    return p
        else:
            if self.base_path:
                p = self.base_path / "agents" / f"{name}.md"
                if p.exists():
                    return p
        return None

    def get_system_instruction(self) -> str | None:
        """Return the system instruction for this bundle, or None."""
        return self.instruction

    def resolve_pending_context(self) -> None:
        """Resolve namespaced context references using ``source_base_paths``.

        Context entries with namespace prefixes (e.g. ``"foundation:ctx/f.md"``)
        are stored as pending during parsing because ``source_base_paths`` is
        not populated until after composition.  Call this before accessing
        ``self.context`` to ensure all paths are resolved.
        """
        if not self._pending_context:
            return

        for name, ref in list(self._pending_context.items()):
            if ":" not in ref:
                continue
            namespace, path_part = ref.split(":", 1)
            if namespace in self.source_base_paths:
                base = self.source_base_paths[namespace]
                self.context[name] = construct_context_path(base, path_part)
                del self._pending_context[name]
            elif self.base_path and namespace == self.name:
                self.context[name] = construct_context_path(self.base_path, path_part)
                del self._pending_context[name]

    def load_agent_metadata(self) -> None:
        """Load full metadata for all agents from their ``.md`` files in-place.

        File metadata only fills in fields not already set by inline config.
        Call after composition when ``source_base_paths`` is fully populated.
        """
        if not self.agents:
            return

        for agent_name, agent_config in self.agents.items():
            path = self.resolve_agent_path(agent_name)
            if path and path.exists():
                try:
                    file_meta = _load_agent_file_metadata(path, agent_name)
                    for key, value in file_meta.items():
                        if key not in agent_config or not agent_config.get(key):
                            agent_config[key] = value
                except Exception as e:
                    logger.warning("Failed to load metadata for agent '%s': %s", agent_name, e)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_path: Path | None = None) -> Bundle:
        """Create a Bundle from a parsed dict (YAML / frontmatter).

        Args:
            data: Dict with bundle configuration.
            base_path: Path to the bundle root directory.

        Returns:
            Bundle instance.

        Raises:
            BundleValidationError: If module lists are malformed.
        """
        bundle_meta = data.get("bundle", {})
        bundle_name = bundle_meta.get("name", "")

        providers = _validate_module_list(
            data.get("providers", []), "providers", bundle_name, base_path
        )
        tools = _validate_module_list(data.get("tools", []), "tools", bundle_name, base_path)
        hooks = _validate_module_list(data.get("hooks", []), "hooks", bundle_name, base_path)

        resolved_context, pending_context = _parse_context(data.get("context", {}), base_path)

        return cls(
            name=bundle_name,
            version=bundle_meta.get("version", "1.0.0"),
            description=bundle_meta.get("description", ""),
            includes=data.get("includes", []),
            session=data.get("session", {}),
            providers=providers,
            tools=tools,
            hooks=hooks,
            spawn=data.get("spawn", {}),
            agents=_parse_agents(data.get("agents", {}), base_path),
            context=resolved_context,
            _pending_context=pending_context,
            instruction=None,  # Set separately from markdown body
            base_path=base_path,
        )


# ---------------------------------------------------------------------------
# Bundle parsing helpers (private)
# ---------------------------------------------------------------------------


def _parse_agents(
    agents_config: dict[str, Any],
    base_path: Path | None,  # noqa: ARG001 — reserved for future path resolution
) -> dict[str, dict[str, Any]]:
    """Parse the ``agents:`` config section into a name → definition dict."""
    if not agents_config:
        return {}

    result: dict[str, dict[str, Any]] = {}

    if "include" in agents_config:
        for name in agents_config["include"]:
            result[name] = {"name": name}

    for key, value in agents_config.items():
        if key != "include" and isinstance(value, dict):
            result[key] = value

    return result


def _load_agent_file_metadata(path: Path, fallback_name: str) -> dict[str, Any]:
    """Load agent config from a ``.md`` file.

    Extracts ``meta:`` section fields plus top-level mount-plan sections
    (tools, providers, hooks, session, provider_preferences, model_role)
    and the markdown body as ``instruction``.
    """
    text = path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(text)

    meta = frontmatter.get("meta", {})
    if not meta:
        if "name" in frontmatter or "description" in frontmatter:
            meta = frontmatter
        else:
            meta = {}

    result: dict[str, Any] = {
        "name": meta.get("name", fallback_name),
        "description": meta.get("description", ""),
        **{k: v for k, v in meta.items() if k not in ("name", "description")},
    }

    for top_key in ("tools", "providers", "hooks", "session", "provider_preferences", "model_role"):
        if top_key in frontmatter:
            result[top_key] = frontmatter[top_key]

    if body and body.strip():
        result["instruction"] = body.strip()

    return result


def _parse_context(
    context_config: dict[str, Any],
    base_path: Path | None,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Parse the ``context:`` section into resolved and pending dicts.

    Context entries with a namespace prefix (``"foundation:file.md"``) are
    stored as *pending* for later resolution once ``source_base_paths`` is
    fully populated after composition.

    Returns:
        ``(resolved_context, pending_context)`` where resolved maps
        name → Path and pending maps name → original_ref.
    """
    if not context_config:
        return {}, {}

    resolved: dict[str, Path] = {}
    pending: dict[str, str] = {}

    if "include" in context_config:
        for name in context_config["include"]:
            if ":" in name:
                pending[name] = name
            elif base_path:
                resolved[name] = construct_context_path(base_path, name)

    for key, value in context_config.items():
        if key != "include" and isinstance(value, str):
            resolved[key] = (base_path / value) if base_path else Path(value)

    return resolved, pending


def _validate_module_list(
    items: Any,
    field_name: str,
    bundle_name: str,
    base_path: Path | None,
) -> list[dict[str, Any]]:
    """Validate and normalise a module list.

    Resolves relative ``source`` paths to absolute paths at parse time so
    they remain correct after composition changes ``base_path``.

    Raises:
        BundleValidationError: If *items* is not a list of dicts.
    """
    if not items:
        return []

    identifier = bundle_name or str(base_path) or "unknown"

    if not isinstance(items, list):
        raise BundleValidationError(
            f"Bundle '{identifier}' has malformed {field_name}: "
            f"expected list, got {type(items).__name__}.\n"
            f"Correct format: {field_name}: [{{module: 'module-id', source: 'git+https://...'}}]"
        )

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise BundleValidationError(
                f"Bundle '{identifier}' has malformed {field_name}[{i}]: "
                f"expected dict with 'module' and 'source' keys, "
                f"got {type(item).__name__} {item!r}.\n"
                f"Correct format: {field_name}: [{{module: 'module-id', source: 'git+https://...'}}]"
            )

    # Resolve relative source paths to absolute before composition.
    if base_path:
        resolved_items = []
        for item in items:
            source = item.get("source", "")
            if isinstance(source, str) and (source.startswith("./") or source.startswith("../")):
                resolved_source = str((base_path / source).resolve())
                item = {**item, "source": resolved_source}
            resolved_items.append(item)
        return resolved_items

    return list(items)


# ---------------------------------------------------------------------------
# Module resolver
# ---------------------------------------------------------------------------


class BundleModuleSource:
    """Simple module source that wraps a pre-resolved local path."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def resolve(self) -> Path:
        """Return the pre-resolved module path."""
        return self._path


class BundleModuleResolver:
    """Maps module IDs to their activated local paths.

    Implements the kernel's ``ModuleSourceResolver`` protocol.  Supports
    on-demand (lazy) activation for agent-specific modules that were not in
    the parent bundle's initial activation set.
    """

    def __init__(
        self,
        module_paths: dict[str, Path],
        activator: Any = None,
    ) -> None:
        """Initialise with activated module paths and an optional activator.

        Args:
            module_paths: Dict mapping module ID to local path.
            activator: Optional ``ModuleActivator`` for lazy activation of
                modules not present in *module_paths*.
        """
        self._paths = module_paths
        self._activator = activator
        self._activation_lock = asyncio.Lock()

    def resolve(
        self,
        module_id: str,
        source_hint: Any = None,
        profile_hint: Any = None,
    ) -> BundleModuleSource:
        """Synchronously resolve *module_id* to a :class:`BundleModuleSource`.

        Raises:
            ModuleNotFoundError: If *module_id* is not in the activated set.
                Use :meth:`async_resolve` for lazy-activation support.
        """
        _hint = profile_hint if profile_hint is not None else source_hint  # noqa: F841
        if module_id not in self._paths:
            raise ModuleNotFoundError(
                f"Module '{module_id}' not found in prepared bundle. "
                f"Available modules: {list(self._paths.keys())}. "
                f"Use async_resolve() for lazy activation support."
            )
        return BundleModuleSource(self._paths[module_id])

    async def async_resolve(
        self,
        module_id: str,
        source_hint: Any = None,
        profile_hint: Any = None,
    ) -> BundleModuleSource:
        """Async resolve with lazy-activation support.

        If *module_id* is already activated, returns immediately.  Otherwise
        activates it on-demand using the stored activator and *source_hint*.

        Raises:
            ModuleNotFoundError: If the module cannot be found or activated.
        """
        hint = profile_hint if profile_hint is not None else source_hint

        if module_id in self._paths:
            return BundleModuleSource(self._paths[module_id])

        if not self._activator:
            raise ModuleNotFoundError(
                f"Module '{module_id}' not found in prepared bundle and no activator available. "
                f"Available modules: {list(self._paths.keys())}"
            )

        if not hint:
            raise ModuleNotFoundError(
                f"Module '{module_id}' not found and no source hint provided for activation. "
                f"Available modules: {list(self._paths.keys())}"
            )

        async with self._activation_lock:
            # Double-check after acquiring lock.
            if module_id in self._paths:
                return BundleModuleSource(self._paths[module_id])

            logger.info("Lazy activating module '%s' from '%s'", module_id, hint)
            try:
                module_path = await self._activator.activate(module_id, hint)
                self._paths[module_id] = module_path
                logger.info("Successfully activated '%s' at %s", module_id, module_path)
                return BundleModuleSource(module_path)
            except Exception as e:
                logger.error("Failed to lazy-activate '%s': %s", module_id, e)
                raise ModuleNotFoundError(
                    f"Module '{module_id}' not found and activation failed: {e}"
                ) from e

    def get_module_source(self, module_id: str) -> str | None:
        """Return the string path for *module_id*, or None if not found.

        Compatibility shim for ``StandardModuleSourceResolver.get_module_source()``.
        """
        path = self._paths.get(module_id)
        return str(path) if path else None


# ---------------------------------------------------------------------------
# PreparedBundle
# ---------------------------------------------------------------------------


@dataclass
class PreparedBundle:
    """A bundle that has been prepared for execution.

    Contains the compiled mount plan, a module resolver for activated paths,
    and the original :class:`Bundle` (used by spawn.py to access ``agents``).

    Attributes:
        mount_plan: Configuration for mounting modules into a session.
        resolver: Resolver for finding activated module paths.
        bundle: The original Bundle that was prepared.
        bundle_package_paths: Paths to bundle ``src/`` directories added to
            ``sys.path``.  Shared with child sessions so bundle packages
            remain importable.
    """

    mount_plan: dict[str, Any]
    resolver: BundleModuleResolver
    bundle: Bundle
    bundle_package_paths: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_bundles_for_resolver(self, bundle: Bundle) -> dict[str, Bundle]:
        """Build a namespace → Bundle map for @mention resolution."""
        from dataclasses import replace as dc_replace

        bundles: dict[str, Bundle] = {}
        namespaces = list(bundle.source_base_paths.keys()) if bundle.source_base_paths else []
        if bundle.name and bundle.name not in namespaces:
            namespaces.append(bundle.name)

        for ns in namespaces:
            if not ns:
                continue
            ns_base = bundle.source_base_paths.get(ns, bundle.base_path)
            bundles[ns] = dc_replace(bundle, base_path=ns_base) if ns_base else bundle

        return bundles

    def _create_system_prompt_factory(
        self,
        bundle: Bundle,
        session: Any,
        session_cwd: Path | None = None,
    ) -> Callable[[], Awaitable[str]]:
        """Return a factory that builds a fresh system prompt on every call.

        The factory re-reads context files and re-processes @mentions each
        time, enabling dynamic content (e.g. ``AGENTS.md``) to be picked up
        immediately when modified during a session.
        """
        from amplifier_lib.mentions import (
            BaseMentionResolver,
            ContentDeduplicator,
            format_context_block,
            load_mentions,
        )

        captured_bundle = bundle
        captured_self = self
        effective_base = session_cwd or bundle.base_path or Path.cwd()

        async def factory() -> str:
            main_instruction = captured_bundle.instruction or ""

            bundles_for_resolver = captured_self._build_bundles_for_resolver(captured_bundle)
            resolver = BaseMentionResolver(
                bundles=bundles_for_resolver,  # type: ignore[arg-type]
                base_path=effective_base,
            )
            deduplicator = ContentDeduplicator()
            mention_to_path: dict[str, Path] = {}

            for ctx_name, ctx_path in captured_bundle.context.items():
                if ctx_path.exists():
                    content = ctx_path.read_text(encoding="utf-8")
                    deduplicator.add_file(ctx_path, content)
                    mention_to_path[ctx_name] = ctx_path

            mention_results = await load_mentions(
                main_instruction,
                resolver=resolver,
                deduplicator=deduplicator,
            )
            for mr in mention_results:
                if mr.resolved_path:
                    mention_to_path[mr.mention] = mr.resolved_path

            all_context = format_context_block(deduplicator, mention_to_path)

            if all_context:
                return f"{main_instruction}\n\n---\n\n{all_context}"
            return main_instruction

        return factory

    # ------------------------------------------------------------------
    # Self-healing helpers
    # ------------------------------------------------------------------

    def _has_complete_mount_failure(self, session: Any) -> bool:
        """Detect if configured modules completely failed to mount.

        The kernel intentionally swallows module load errors for resilience.
        This detects the case where *all* configured providers or *all*
        configured tools failed to load — typically caused by stale
        install-state (missing dependencies after a ``uv tool install``).

        Partial failures (some loaded, some not) are considered benign
        and do NOT trigger self-healing.
        """
        coordinator = session.coordinator

        configured_providers = self.mount_plan.get("providers", [])
        mounted_providers = coordinator.get("providers") or {}
        if configured_providers and not mounted_providers:
            logger.info(
                "Complete provider mount failure: %d configured, 0 loaded",
                len(configured_providers),
            )
            return True

        configured_tools = self.mount_plan.get("tools", [])
        mounted_tools = coordinator.get("tools") or {}
        if configured_tools and not mounted_tools:
            logger.info(
                "Complete tool mount failure: %d configured, 0 loaded",
                len(configured_tools),
            )
            return True

        return False

    def _invalidate_install_state(self) -> None:
        """Invalidate all module install state to force reinstallation.

        Clears the activator's install-state fingerprints and its
        already-activated set so all modules will be re-resolved and
        re-installed on the next ``create_session()`` call.
        """
        activator = getattr(self.resolver, "_activator", None)
        if not activator:
            logger.debug("No activator on resolver — cannot invalidate install state")
            return

        install_state = getattr(activator, "_install_state", None)
        if install_state:
            install_state.invalidate(None)
            install_state.save()
            logger.info("Invalidated all module install state for self-healing")

        activated = getattr(activator, "_activated", None)
        if activated:
            activated.clear()

    # ------------------------------------------------------------------
    # Session creation
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: str | None = None,
        parent_id: str | None = None,
        approval_system: Any = None,
        display_system: Any = None,
        session_cwd: Path | None = None,
        is_resumed: bool = False,
        self_heal: bool = True,
    ) -> Any:
        """Create and initialise an AmplifierSession from this prepared bundle.

        Steps:
          1. Creates AmplifierSession with mount plan.
          2. Mounts the module resolver.
          3. Registers capabilities (working_dir, bundle_package_paths, …).
          4. Initialises the session (loads all modules).
          5. If all providers or all tools failed to mount and *self_heal*
             is ``True``, invalidates install state and retries once.
          6. Registers a dynamic system-prompt factory if the bundle has
             an instruction or context files.

        Args:
            session_id: Optional session ID for resuming an existing session.
            parent_id: Optional parent session ID for lineage tracking.
            approval_system: Optional approval system for hooks.
            display_system: Optional display system for hooks.
            session_cwd: Working directory for resolving local @-mentions
                (e.g. ``@AGENTS.md``).  Falls back to ``bundle.base_path``.
            is_resumed: Whether this is a resumed session (controls which
                lifecycle events are emitted).
            self_heal: When ``True`` (default), automatically invalidate
                install state and retry once if all configured providers
                or tools fail to mount.

        Returns:
            Initialised AmplifierSession ready for ``execute()``.
        """
        session = await self._create_session_inner(
            session_id=session_id,
            parent_id=parent_id,
            approval_system=approval_system,
            display_system=display_system,
            session_cwd=session_cwd,
            is_resumed=is_resumed,
        )

        # Self-healing: on complete mount failure, invalidate install state
        # and retry once.  Partial failures are considered benign.
        if self_heal and self._has_complete_mount_failure(session):
            logger.warning(
                "Complete module mount failure detected — invalidating install state and retrying"
            )
            self._invalidate_install_state()
            session = await self._create_session_inner(
                session_id=session_id,
                parent_id=parent_id,
                approval_system=approval_system,
                display_system=display_system,
                session_cwd=session_cwd,
                is_resumed=is_resumed,
            )
            if self._has_complete_mount_failure(session):
                logger.warning(
                    "Self-healing retry completed but modules still failed. "
                    "Check module configuration, credentials, and dependencies."
                )

        return session

    async def _create_session_inner(
        self,
        session_id: str | None = None,
        parent_id: str | None = None,
        approval_system: Any = None,
        display_system: Any = None,
        session_cwd: Path | None = None,
        is_resumed: bool = False,
    ) -> Any:
        """Create and initialise an AmplifierSession (no self-healing)."""
        from amplifier_core import AmplifierSession

        session = AmplifierSession(
            self.mount_plan,
            session_id=session_id,
            parent_id=parent_id,
            approval_system=approval_system,
            display_system=display_system,
            is_resumed=is_resumed,  # type: ignore[call-arg]
        )

        await session.coordinator.mount("module-source-resolver", self.resolver)

        if self.bundle_package_paths:
            session.coordinator.register_capability(
                "bundle_package_paths", list(self.bundle_package_paths)
            )

        effective_cwd = session_cwd or self.bundle.base_path or Path.cwd()
        session.coordinator.register_capability("session.working_dir", str(effective_cwd.resolve()))

        await session.initialize()

        self.bundle.resolve_pending_context()

        if self.bundle.instruction or self.bundle.context or self.bundle._pending_context:
            from amplifier_lib.mentions import BaseMentionResolver, ContentDeduplicator

            bundles_for_resolver = self._build_bundles_for_resolver(self.bundle)
            resolver_base = session_cwd or self.bundle.base_path or Path.cwd()
            initial_resolver = BaseMentionResolver(
                bundles=bundles_for_resolver,  # type: ignore[arg-type]
                base_path=resolver_base,
            )
            session.coordinator.register_capability("mention_resolver", initial_resolver)
            session.coordinator.register_capability("mention_deduplicator", ContentDeduplicator())

            factory = self._create_system_prompt_factory(
                self.bundle, session, session_cwd=session_cwd
            )
            context_manager = session.coordinator.get("context")
            if context_manager and hasattr(context_manager, "set_system_prompt_factory"):
                await context_manager.set_system_prompt_factory(factory)
            elif context_manager:
                resolved_prompt = await factory()
                await context_manager.add_message({"role": "system", "content": resolved_prompt})

        return session

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    async def spawn(
        self,
        child_bundle: Bundle,
        instruction: str,
        *,
        compose: bool = True,
        parent_session: Any = None,
        session_id: str | None = None,
        orchestrator_config: dict[str, Any] | None = None,
        parent_messages: list[dict[str, Any]] | None = None,
        session_cwd: Path | None = None,
        provider_preferences: list[ProviderPreference] | None = None,
        self_delegation_depth: int = 0,
    ) -> dict[str, Any]:
        """Spawn a sub-session with *child_bundle*.

        This is the library-level spawn method.  The app layer (CLI, API
        server) typically wraps this in a spawn capability that resolves
        agent names to bundles and applies tool/hook inheritance policy.

        Args:
            child_bundle: Pre-resolved bundle for the child session.
            instruction: Task instruction for the sub-session.
            compose: Whether to compose child with parent bundle (default True).
            parent_session: Parent session for lineage tracking / UX inheritance.
            session_id: Optional session ID for resuming an existing session.
            orchestrator_config: Optional orchestrator settings to merge in.
            parent_messages: Optional parent conversation history to inject.
            session_cwd: Working directory override for the child session.
            provider_preferences: Ordered fallback chain of
                :class:`~amplifier_lib.spawn_utils.ProviderPreference` objects.
            self_delegation_depth: Current delegation depth for depth limiting.

        Returns:
            Dict with keys ``"output"``, ``"session_id"``, ``"status"``,
            ``"turn_count"``, ``"metadata"``.
        """
        effective_bundle = self.bundle.compose(child_bundle) if compose else child_bundle

        child_mount_plan = effective_bundle.to_mount_plan()

        if orchestrator_config:
            child_mount_plan.setdefault("orchestrator", {}).setdefault("config", {}).update(
                orchestrator_config
            )

        if provider_preferences:
            child_mount_plan = await apply_provider_preferences_with_resolution(
                child_mount_plan,
                provider_preferences,
                parent_session.coordinator if parent_session else None,
            )

        from amplifier_core import AmplifierSession

        child_session = AmplifierSession(
            child_mount_plan,
            session_id=session_id,
            parent_id=parent_session.session_id if parent_session else None,
            approval_system=(
                getattr(getattr(parent_session, "coordinator", None), "approval_system", None)
                if parent_session
                else None
            ),
            display_system=(
                getattr(getattr(parent_session, "coordinator", None), "display_system", None)
                if parent_session
                else None
            ),
        )

        await child_session.coordinator.mount("module-source-resolver", self.resolver)

        # Inherit working directory from parent or caller.
        if session_cwd:
            effective_child_cwd: Path = session_cwd
        elif parent_session:
            parent_wd = parent_session.coordinator.get_capability("session.working_dir")
            effective_child_cwd = (
                Path(parent_wd) if parent_wd else (self.bundle.base_path or Path.cwd())
            )
        else:
            effective_child_cwd = self.bundle.base_path or Path.cwd()

        child_session.coordinator.register_capability(
            "session.working_dir", str(effective_child_cwd.resolve())
        )

        await child_session.initialize()

        if self_delegation_depth > 0:
            child_session.coordinator.register_capability(
                "self_delegation_depth", self_delegation_depth
            )

        if parent_messages and not session_id:
            child_context = child_session.coordinator.get("context")
            if child_context and hasattr(child_context, "set_messages"):
                await child_context.set_messages(parent_messages)

        if effective_bundle.instruction or effective_bundle.context:
            factory = self._create_system_prompt_factory(
                effective_bundle, child_session, session_cwd=session_cwd
            )
            context = child_session.coordinator.get("context")
            if context and hasattr(context, "set_system_prompt_factory"):
                await context.set_system_prompt_factory(factory)
            elif context:
                resolved_prompt = await factory()
                await context.add_message({"role": "system", "content": resolved_prompt})

        from amplifier_core.models import HookResult

        completion_data: dict[str, Any] = {}

        async def _capture_complete(event: str, data: dict[str, Any]) -> HookResult:
            completion_data.update(data)
            return HookResult()

        unregister = child_session.coordinator.hooks.register(
            "orchestrator:complete",
            _capture_complete,
            priority=999,
            name="_spawn_completion_capture",
        )

        try:
            response = await child_session.execute(instruction)
        finally:
            unregister()
            await child_session.cleanup()

        return {
            "output": response,
            "session_id": child_session.session_id,
            "status": completion_data.get("status", "success"),
            "turn_count": completion_data.get("turn_count", 1),
            "metadata": completion_data.get("metadata", {}),
        }
