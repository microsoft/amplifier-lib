"""Tests for amplifier_lib.spawn_utils — ProviderPreference, is_glob_pattern,
resolve_model_pattern, apply_provider_preferences,
apply_provider_preferences_with_resolution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_lib.spawn_utils import (
    PROTECTED_CONFIG_KEYS,
    ProviderPreference,
    apply_provider_preferences,
    apply_provider_preferences_with_resolution,
    is_glob_pattern,
    resolve_model_pattern,
)


# ---------------------------------------------------------------------------
# ProviderPreference dataclass
# ---------------------------------------------------------------------------


class TestProviderPreference:
    def test_instantiation_basic(self):
        pref = ProviderPreference(provider="anthropic", model="claude-3-opus")
        assert pref.provider == "anthropic"
        assert pref.model == "claude-3-opus"
        assert pref.config == {}

    def test_instantiation_with_config(self):
        pref = ProviderPreference(
            provider="openai", model="gpt-4", config={"temperature": 0.7}
        )
        assert pref.config == {"temperature": 0.7}

    def test_config_defaults_to_empty_dict(self):
        pref = ProviderPreference(provider="x", model="y")
        assert pref.config == {}

    def test_to_dict_without_config(self):
        pref = ProviderPreference(provider="anthropic", model="claude-3")
        d = pref.to_dict()
        assert d == {"provider": "anthropic", "model": "claude-3"}
        assert "config" not in d

    def test_to_dict_with_config(self):
        pref = ProviderPreference(provider="openai", model="gpt-4", config={"k": "v"})
        d = pref.to_dict()
        assert d["config"] == {"k": "v"}

    def test_from_dict_basic(self):
        d = {"provider": "anthropic", "model": "claude-3"}
        pref = ProviderPreference.from_dict(d)
        assert pref.provider == "anthropic"
        assert pref.model == "claude-3"
        assert pref.config == {}

    def test_from_dict_with_config(self):
        d = {"provider": "openai", "model": "gpt-4", "config": {"temp": 0.5}}
        pref = ProviderPreference.from_dict(d)
        assert pref.config == {"temp": 0.5}

    def test_from_dict_missing_provider_raises(self):
        with pytest.raises(ValueError, match="provider"):
            ProviderPreference.from_dict({"model": "gpt-4"})

    def test_from_dict_missing_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            ProviderPreference.from_dict({"provider": "openai"})

    def test_roundtrip_to_dict_from_dict(self):
        pref = ProviderPreference(provider="anthropic", model="claude-3", config={"x": 1})
        restored = ProviderPreference.from_dict(pref.to_dict())
        assert restored.provider == pref.provider
        assert restored.model == pref.model
        assert restored.config == pref.config

    def test_config_independent_instances(self):
        """Default config dict should not be shared between instances."""
        p1 = ProviderPreference(provider="a", model="b")
        p2 = ProviderPreference(provider="c", model="d")
        p1.config["key"] = "val"
        assert "key" not in p2.config


# ---------------------------------------------------------------------------
# is_glob_pattern
# ---------------------------------------------------------------------------


class TestIsGlobPattern:
    @pytest.mark.parametrize(
        "pattern",
        [
            "claude-haiku-*",
            "gpt-4-*",
            "*",
            "model-?",
            "model-[abc]",
            "prefix-*-suffix",
        ],
    )
    def test_glob_patterns_detected(self, pattern):
        assert is_glob_pattern(pattern) is True

    @pytest.mark.parametrize(
        "name",
        [
            "claude-3-opus-20240229",
            "gpt-4-turbo",
            "exact-model-name",
            "",
            "model.v2",
        ],
    )
    def test_exact_names_not_glob(self, name):
        assert is_glob_pattern(name) is False


# ---------------------------------------------------------------------------
# resolve_model_pattern
# ---------------------------------------------------------------------------


class TestResolveModelPattern:
    async def test_exact_name_returned_as_is(self):
        coordinator = MagicMock()
        result = await resolve_model_pattern("claude-3-opus", "anthropic", coordinator)
        assert result.resolved_model == "claude-3-opus"
        assert result.pattern is None

    async def test_glob_with_no_provider_returns_pattern(self):
        coordinator = MagicMock()
        result = await resolve_model_pattern("claude-haiku-*", None, coordinator)
        assert result.resolved_model == "claude-haiku-*"
        assert result.pattern == "claude-haiku-*"

    async def test_glob_resolved_to_first_sorted_match(self):
        """Pattern should resolve to the lexicographically highest match."""
        mock_provider = AsyncMock()
        mock_provider.list_models.return_value = [
            "claude-haiku-20240307",
            "claude-haiku-20241022",
            "claude-haiku-20250101",
        ]
        mock_coordinator = MagicMock()
        mock_coordinator.get.return_value = {"provider-anthropic": mock_provider}

        result = await resolve_model_pattern("claude-haiku-*", "anthropic", mock_coordinator)
        assert result.resolved_model == "claude-haiku-20250101"
        assert result.pattern == "claude-haiku-*"
        assert result.matched_models is not None
        assert len(result.matched_models) == 3

    async def test_glob_no_match_returns_pattern_as_is(self):
        mock_provider = AsyncMock()
        mock_provider.list_models.return_value = ["gpt-4", "gpt-3.5"]
        mock_coordinator = MagicMock()
        mock_coordinator.get.return_value = {"provider-openai": mock_provider}

        result = await resolve_model_pattern("claude-haiku-*", "openai", mock_coordinator)
        assert result.resolved_model == "claude-haiku-*"
        assert result.matched_models == []

    async def test_glob_provider_not_found_returns_pattern(self):
        mock_coordinator = MagicMock()
        mock_coordinator.get.return_value = {}  # no providers

        result = await resolve_model_pattern("claude-*", "anthropic", mock_coordinator)
        assert result.resolved_model == "claude-*"

    async def test_model_objects_with_id_attr(self):
        """list_models() returns objects with .id instead of strings."""
        mock_provider = AsyncMock()

        class ModelObj:
            def __init__(self, model_id: str):
                self.id = model_id

        mock_provider.list_models.return_value = [
            ModelObj("gpt-4-turbo"),
            ModelObj("gpt-4-mini"),
        ]
        mock_coordinator = MagicMock()
        mock_coordinator.get.return_value = {"provider-openai": mock_provider}

        result = await resolve_model_pattern("gpt-4-*", "openai", mock_coordinator)
        assert "gpt-4" in result.resolved_model

    async def test_list_models_exception_handled_gracefully(self):
        mock_provider = AsyncMock()
        mock_provider.list_models.side_effect = RuntimeError("API error")
        mock_coordinator = MagicMock()
        mock_coordinator.get.return_value = {"provider-anthropic": mock_provider}

        # Should not raise, falls back to pattern
        result = await resolve_model_pattern("claude-*", "anthropic", mock_coordinator)
        assert result.resolved_model == "claude-*"


# ---------------------------------------------------------------------------
# apply_provider_preferences (sync)
# ---------------------------------------------------------------------------


def _make_mount_plan(*module_ids: str) -> dict:
    """Build a minimal mount plan with the given provider module IDs."""
    return {
        "providers": [
            {"module": mid, "id": mid, "config": {}} for mid in module_ids
        ]
    }


class TestApplyProviderPreferences:
    def test_empty_preferences_returns_unchanged(self):
        plan = _make_mount_plan("provider-anthropic")
        result = apply_provider_preferences(plan, [])
        assert result is plan

    def test_no_providers_returns_unchanged(self):
        plan = {"providers": []}
        pref = ProviderPreference(provider="anthropic", model="claude-3")
        result = apply_provider_preferences(plan, [pref])
        assert result is plan

    def test_matching_provider_promoted(self):
        plan = _make_mount_plan("provider-anthropic", "provider-openai")
        pref = ProviderPreference(provider="anthropic", model="claude-3-opus")
        result = apply_provider_preferences(plan, [pref])
        # Find the anthropic provider in result
        providers = result["providers"]
        anthropic = next(p for p in providers if "anthropic" in p["module"])
        assert anthropic["config"]["priority"] == 0
        assert anthropic["config"]["default_model"] == "claude-3-opus"

    def test_first_matching_preference_wins(self):
        plan = _make_mount_plan("provider-anthropic", "provider-openai")
        prefs = [
            ProviderPreference(provider="anthropic", model="claude-3"),
            ProviderPreference(provider="openai", model="gpt-4"),
        ]
        result = apply_provider_preferences(plan, prefs)
        providers = result["providers"]
        anthropic = next(p for p in providers if "anthropic" in p["module"])
        assert anthropic["config"]["default_model"] == "claude-3"

    def test_no_matching_preference_returns_unchanged(self):
        plan = _make_mount_plan("provider-openai")
        pref = ProviderPreference(provider="anthropic", model="claude-3")
        result = apply_provider_preferences(plan, [pref])
        # No match → returned as-is
        providers = result["providers"]
        openai = next(p for p in providers if "openai" in p["module"])
        assert "priority" not in openai["config"]

    def test_short_name_matches_prefixed_module(self):
        """'anthropic' should match 'provider-anthropic'."""
        plan = _make_mount_plan("provider-anthropic")
        pref = ProviderPreference(provider="anthropic", model="claude-3")
        result = apply_provider_preferences(plan, [pref])
        providers = result["providers"]
        assert providers[0]["config"]["priority"] == 0

    def test_pref_config_non_protected_key_applied(self):
        plan = _make_mount_plan("provider-anthropic")
        pref = ProviderPreference(
            provider="anthropic",
            model="claude-3",
            config={"temperature": 0.5},
        )
        result = apply_provider_preferences(plan, [pref])
        providers = result["providers"]
        assert providers[0]["config"]["temperature"] == 0.5

    def test_pref_config_protected_keys_not_applied(self):
        plan = _make_mount_plan("provider-anthropic")
        protected = {k: "OVERRIDE" for k in PROTECTED_CONFIG_KEYS}
        pref = ProviderPreference(provider="anthropic", model="claude-3", config=protected)
        result = apply_provider_preferences(plan, [pref])
        providers = result["providers"]
        # Protected keys should NOT appear in the provider's config
        for key in PROTECTED_CONFIG_KEYS:
            assert providers[0]["config"].get(key) != "OVERRIDE"

    def test_does_not_mutate_original_plan(self):
        plan = _make_mount_plan("provider-anthropic")
        original_config = plan["providers"][0]["config"].copy()
        pref = ProviderPreference(provider="anthropic", model="claude-3")
        apply_provider_preferences(plan, [pref])
        assert plan["providers"][0]["config"] == original_config


# ---------------------------------------------------------------------------
# apply_provider_preferences_with_resolution (async)
# ---------------------------------------------------------------------------


class TestApplyProviderPreferencesWithResolution:
    async def test_empty_preferences_returns_unchanged(self):
        plan = _make_mount_plan("provider-anthropic")
        coordinator = MagicMock()
        result = await apply_provider_preferences_with_resolution(plan, [], coordinator)
        assert result is plan

    async def test_exact_model_applied_directly(self):
        plan = _make_mount_plan("provider-anthropic")
        pref = ProviderPreference(provider="anthropic", model="claude-3-opus")
        coordinator = MagicMock()
        result = await apply_provider_preferences_with_resolution(plan, [pref], coordinator)
        providers = result["providers"]
        assert providers[0]["config"]["default_model"] == "claude-3-opus"

    async def test_glob_model_resolved(self):
        """Glob patterns should be resolved via list_models."""
        mock_provider = AsyncMock()
        mock_provider.list_models.return_value = [
            "claude-haiku-20240307",
            "claude-haiku-20241022",
        ]
        coordinator = MagicMock()
        coordinator.get.return_value = {"provider-anthropic": mock_provider}

        plan = _make_mount_plan("provider-anthropic")
        pref = ProviderPreference(provider="anthropic", model="claude-haiku-*")
        result = await apply_provider_preferences_with_resolution(plan, [pref], coordinator)
        providers = result["providers"]
        # Should resolve to the highest-sorted match
        assert providers[0]["config"]["default_model"] == "claude-haiku-20241022"

    async def test_no_providers_returns_unchanged(self):
        plan = {"providers": []}
        coordinator = MagicMock()
        pref = ProviderPreference(provider="anthropic", model="claude-3")
        result = await apply_provider_preferences_with_resolution(plan, [pref], coordinator)
        assert result is plan
