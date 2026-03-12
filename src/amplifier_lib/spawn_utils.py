"""Utilities for session spawning with provider/model selection.

Provides mechanisms for specifying provider/model preferences when spawning
sub-sessions:
  - Ordered list of provider/model pairs (fallback chain)
  - Model glob pattern resolution  (e.g. "claude-haiku-*")
  - Flexible provider matching     (e.g. "anthropic" matches "provider-anthropic")
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Keys that must never be overridden by ProviderPreference.config —
# credentials, endpoints, and infrastructure are always preserved from the
# bundle's own provider configuration.
PROTECTED_CONFIG_KEYS = frozenset(
    {
        # Credentials
        "api_key",
        "secret",
        "password",
        "token",
        "access_token",
        "bearer_token",
        "client_id",
        "client_secret",
        "tenant_id",
        # Endpoints / infrastructure
        "base_url",
        "host",
        "azure_endpoint",
        "api_version",
        "deployment_name",
        "organization",
        "project",
        # Azure auth control
        "managed_identity_client_id",
        "use_managed_identity",
        "use_default_credential",
        # Network control
        "proxy",
        "http_proxy",
        "https_proxy",
        "verify_ssl",
        "ssl_verify",
        "verify",
        "ca_bundle",
    }
)


@dataclass
class ProviderPreference:
    """A provider/model preference for ordered selection.

    Used with ``provider_preferences`` to specify a fallback order when
    spawning sub-sessions.  The system tries each preference in order until
    finding an available provider.

    Model supports glob patterns (e.g. ``"claude-haiku-*"``) which are
    resolved against the provider's available models at runtime.

    Attributes:
        provider: Provider identifier (e.g. "anthropic", "openai", "azure").
            Supports flexible matching — "anthropic" matches "provider-anthropic".
        model: Model name or glob pattern (e.g. "claude-haiku-*", "gpt-5-mini").
        config: Optional routing/preference config to merge into the provider's
            mount config.  Keys in ``PROTECTED_CONFIG_KEYS`` are never overridden.
            Omitted from :meth:`to_dict` when empty for backward compatibility.
    """

    provider: str
    model: str
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {"provider": self.provider, "model": self.model}
        if self.config:
            result["config"] = self.config
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderPreference:
        """Create from dictionary representation.

        Raises:
            ValueError: If required keys are missing.
        """
        if "provider" not in data:
            raise ValueError("ProviderPreference requires 'provider' key")
        if "model" not in data:
            raise ValueError("ProviderPreference requires 'model' key")
        return cls(
            provider=data["provider"],
            model=data["model"],
            config=data.get("config", {}),
        )


@dataclass
class ModelResolutionResult:
    """Result of model pattern resolution.

    Attributes:
        resolved_model: The final model name to use.
        pattern: Original pattern (None if input wasn't a pattern).
        available_models: All models available from the provider.
        matched_models: Models that matched the pattern.
    """

    resolved_model: str
    pattern: str | None = None
    available_models: list[str] | None = None
    matched_models: list[str] | None = None


def is_glob_pattern(model_hint: str) -> bool:
    """Return True if *model_hint* contains glob wildcard characters (*, ?, [)."""
    return any(c in model_hint for c in "*?[")


async def resolve_model_pattern(
    model_hint: str,
    provider_name: str | None,
    coordinator: Any,
) -> ModelResolutionResult:
    """Resolve a model glob pattern to a concrete model name.

    Resolution strategy:
      1. If not a glob pattern, return as-is.
      2. Query provider for available models via ``list_models()``.
      3. Filter with :func:`fnmatch.filter`.
      4. Sort descending (latest date/version wins).
      5. Return first match, or the original pattern if nothing matched.

    Args:
        model_hint: Exact model name or glob pattern (e.g. "claude-haiku-*").
        provider_name: Provider to query (e.g. "anthropic").
        coordinator: Amplifier coordinator for accessing mounted providers.

    Returns:
        :class:`ModelResolutionResult` with resolved model and metadata.
    """
    if not is_glob_pattern(model_hint):
        logger.debug("Model '%s' is not a pattern, using as-is", model_hint)
        return ModelResolutionResult(
            resolved_model=model_hint,
            pattern=None,
            available_models=None,
            matched_models=None,
        )

    if not provider_name:
        logger.warning(
            "Model pattern '%s' specified but no provider — cannot resolve, using as-is",
            model_hint,
        )
        return ModelResolutionResult(
            resolved_model=model_hint,
            pattern=model_hint,
            available_models=None,
            matched_models=None,
        )

    # Query the provider for its available models.
    available_models: list[str] = []
    try:
        providers = coordinator.get("providers")
        if providers:
            provider = _find_provider_instance(providers, provider_name)
            if provider and hasattr(provider, "list_models"):
                models = await provider.list_models()
                available_models = [
                    m if isinstance(m, str) else getattr(m, "id", str(m)) for m in models
                ]
                logger.debug(
                    "Provider '%s' has %d available models",
                    provider_name,
                    len(available_models),
                )
            else:
                logger.debug(
                    "Provider '%s' not found or does not support list_models()",
                    provider_name,
                )
    except Exception as e:
        logger.warning(
            "Failed to query models from provider '%s': %s",
            provider_name,
            e,
        )

    if not available_models:
        logger.warning(
            "No available models from provider '%s' for pattern '%s' — using pattern as-is",
            provider_name,
            model_hint,
        )
        return ModelResolutionResult(
            resolved_model=model_hint,
            pattern=model_hint,
            available_models=[],
            matched_models=[],
        )

    matched = fnmatch.filter(available_models, model_hint)

    if not matched:
        logger.warning(
            "Pattern '%s' matched no models from provider '%s'. "
            "Available: %s. Using pattern as-is.",
            model_hint,
            provider_name,
            ", ".join(available_models[:10]) + ("..." if len(available_models) > 10 else ""),
        )
        return ModelResolutionResult(
            resolved_model=model_hint,
            pattern=model_hint,
            available_models=available_models,
            matched_models=[],
        )

    # Sort descending — for date-suffixed names this puts the newest first.
    matched.sort(reverse=True)
    resolved = matched[0]

    logger.info(
        "Resolved model pattern '%s' -> '%s' (matched %d of %d available: %s)",
        model_hint,
        resolved,
        len(matched),
        len(available_models),
        ", ".join(matched[:5]) + ("..." if len(matched) > 5 else ""),
    )

    return ModelResolutionResult(
        resolved_model=resolved,
        pattern=model_hint,
        available_models=available_models,
        matched_models=matched,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_provider_instance(
    providers: dict[str, Any],
    provider_name: str,
) -> Any | None:
    """Find a mounted provider instance by name with flexible matching."""
    for name, provider in providers.items():
        if provider_name in (
            name,
            name.replace("provider-", ""),
            f"provider-{provider_name}",
        ):
            return provider
    return None


def _find_provider_index(
    providers: list[dict[str, Any]],
    provider_id: str,
) -> int | None:
    """Find the index of a provider spec in the mount-plan providers list.

    Supports flexible matching: ``"anthropic"``, ``"provider-anthropic"``, or
    full module ID.
    """
    for i, p in enumerate(providers):
        module_id = p.get("module", "")
        instance_id = p.get("id", "")
        if provider_id in (
            module_id,
            module_id.replace("provider-", ""),
            f"provider-{provider_id}",
            instance_id,
        ):
            return i
    return None


def _build_provider_lookup(
    providers: list[dict[str, Any]],
) -> dict[str, int]:
    """Build a name → index lookup for efficient provider matching."""
    lookup: dict[str, int] = {}
    for i, p in enumerate(providers):
        module_id = p.get("module", "")
        lookup[module_id] = i
        short_name = module_id.replace("provider-", "")
        if short_name != module_id:
            lookup[short_name] = i
        lookup[f"provider-{short_name}"] = i
        instance_id = p.get("id")
        if instance_id:
            lookup[instance_id] = i
    return lookup


def _apply_single_override(
    mount_plan: dict[str, Any],
    providers: list[dict[str, Any]],
    target_idx: int,
    model: str,
    pref_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a single provider/model override to the mount plan.

    Promotes *target_idx* to ``priority=0`` and sets ``default_model=model``.
    Non-protected keys from *pref_config* are merged in (protected keys such
    as credentials are never overridden).
    """
    new_plan = dict(mount_plan)
    new_providers = []

    for i, p in enumerate(providers):
        p_copy = dict(p)
        p_copy["config"] = dict(p.get("config", {}))

        if i == target_idx:
            if pref_config:
                for key, value in pref_config.items():
                    if key not in PROTECTED_CONFIG_KEYS:
                        p_copy["config"][key] = value
            # These invariants always win.
            p_copy["config"]["priority"] = 0
            p_copy["config"]["default_model"] = model
            logger.info(
                "Provider preference applied: %s (priority=0, model=%s)",
                p_copy.get("module"),
                model,
            )

        new_providers.append(p_copy)

    new_plan["providers"] = new_providers
    return new_plan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_provider_preferences(
    mount_plan: dict[str, Any],
    preferences: list[ProviderPreference],
) -> dict[str, Any]:
    """Apply provider preferences to a mount plan (no glob resolution).

    Finds the first preferred provider that exists in the mount plan,
    promotes it to ``priority=0`` (highest), and sets its model.

    Returns the original mount plan unchanged if no preference matches.
    """
    if not preferences:
        return mount_plan

    providers = mount_plan.get("providers", [])
    if not providers:
        logger.warning("Provider preferences specified but no providers in mount plan")
        return mount_plan

    lookup = _build_provider_lookup(providers)

    for pref in preferences:
        if pref.provider in lookup:
            target_idx = lookup[pref.provider]
            return _apply_single_override(
                mount_plan, providers, target_idx, pref.model, pref.config
            )

    logger.warning(
        "No preferred providers found in mount plan. Preferences: %s, Available: %s",
        [p.provider for p in preferences],
        list({p.get("module", "?") for p in providers}),
    )
    return mount_plan


async def apply_provider_preferences_with_resolution(
    mount_plan: dict[str, Any],
    preferences: list[ProviderPreference],
    coordinator: Any,
) -> dict[str, Any]:
    """Apply provider preferences with model glob-pattern resolution.

    Like :func:`apply_provider_preferences` but also resolves glob patterns
    in model names (e.g. ``"claude-haiku-*"`` → ``"claude-3-haiku-20240307"``).

    Args:
        mount_plan: The mount plan to modify (shallow-copied).
        preferences: Ordered list of :class:`ProviderPreference` objects.
        coordinator: Amplifier coordinator for querying provider models.

    Returns:
        New mount plan with the first matching provider promoted and its
        model pattern resolved.
    """
    if not preferences:
        return mount_plan

    providers = mount_plan.get("providers", [])
    if not providers:
        logger.warning("Provider preferences specified but no providers in mount plan")
        return mount_plan

    lookup = _build_provider_lookup(providers)

    for pref in preferences:
        if pref.provider in lookup:
            target_idx = lookup[pref.provider]

            resolved_model = pref.model
            if is_glob_pattern(pref.model):
                result = await resolve_model_pattern(pref.model, pref.provider, coordinator)
                resolved_model = result.resolved_model

            return _apply_single_override(
                mount_plan, providers, target_idx, resolved_model, pref.config
            )

    logger.warning(
        "No preferred providers found in mount plan. Preferences: %s, Available: %s",
        [p.provider for p in preferences],
        list({p.get("module", "?") for p in providers}),
    )
    return mount_plan
