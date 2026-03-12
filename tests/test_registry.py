"""Tests for amplifier_lib.registry — BundleRegistry."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_lib.bundle import Bundle
from amplifier_lib.registry import BundleRegistry, BundleState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_registry(tmp_path: Path) -> BundleRegistry:
    """Create a BundleRegistry with a mocked source resolver and tmp home."""
    mock_resolver = MagicMock()
    mock_resolver.resolve = AsyncMock(return_value=None)

    with patch.object(BundleRegistry, "_build_source_resolver", return_value=mock_resolver):
        registry = BundleRegistry(home=tmp_path)

    return registry


def make_registry_with_resolver(tmp_path: Path, mock_resolver: MagicMock) -> BundleRegistry:
    """Create a BundleRegistry with a specific source resolver."""
    with patch.object(BundleRegistry, "_build_source_resolver", return_value=mock_resolver):
        registry = BundleRegistry(home=tmp_path)
    return registry


def make_bundle_file(directory: Path, name: str = "myapp", version: str = "1.0.0") -> Path:
    """Create a minimal bundle.md file and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    bundle_file = directory / "bundle.md"
    bundle_file.write_text(
        f"""---
bundle:
  name: {name}
  version: {version}
---
You are a helpful assistant.
"""
    )
    return bundle_file


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestBundleRegistryConstructor:
    def test_constructor_with_tmp_home(self, tmp_path):
        r = make_registry(tmp_path)
        assert r.home == tmp_path

    def test_home_property(self, tmp_path):
        r = make_registry(tmp_path)
        assert isinstance(r.home, Path)

    def test_empty_registry_on_fresh_home(self, tmp_path):
        r = make_registry(tmp_path)
        assert r.list_registered() == []

    def test_loads_persisted_state_on_init(self, tmp_path):
        """If registry.json exists at home, it should be loaded."""
        registry_data = {
            "version": 1,
            "bundles": {
                "foundation": {
                    "uri": "git+https://example.com/foundation",
                    "name": "foundation",
                    "version": "1.0.0",
                    "loaded_at": None,
                    "checked_at": None,
                    "local_path": None,
                    "is_root": True,
                    "explicitly_requested": False,
                    "app_bundle": False,
                }
            },
        }
        (tmp_path / "registry.json").write_text(json.dumps(registry_data))

        r = make_registry(tmp_path)
        assert "foundation" in r.list_registered()

    def test_strict_mode_stored(self, tmp_path):
        with patch.object(BundleRegistry, "_build_source_resolver", return_value=MagicMock()):
            r = BundleRegistry(home=tmp_path, strict=True)
        assert r._strict is True


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


class TestBundleRegistryRegister:
    def test_register_single_bundle(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"myapp": "git+https://example.com/myapp"})
        assert "myapp" in r.list_registered()

    def test_register_multiple_bundles(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({
            "foundation": "git+https://example.com/foundation",
            "recipes": "git+https://example.com/recipes",
        })
        registered = r.list_registered()
        assert "foundation" in registered
        assert "recipes" in registered

    def test_register_overwrites_existing_uri(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"myapp": "git+https://example.com/v1"})
        r.register({"myapp": "git+https://example.com/v2"})
        state = r.get_state("myapp")
        assert isinstance(state, BundleState)
        assert state.uri == "git+https://example.com/v2"

    def test_register_creates_bundle_state(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"myapp": "git+https://example.com/myapp"})
        state = r.get_state("myapp")
        assert isinstance(state, BundleState)
        assert state.name == "myapp"
        assert state.uri == "git+https://example.com/myapp"

    def test_find_returns_uri_after_register(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"myapp": "git+https://example.com/myapp"})
        assert r.find("myapp") == "git+https://example.com/myapp"

    def test_find_returns_none_for_unregistered(self, tmp_path):
        r = make_registry(tmp_path)
        assert r.find("nonexistent") is None


# ---------------------------------------------------------------------------
# list_registered()
# ---------------------------------------------------------------------------


class TestBundleRegistryListRegistered:
    def test_empty_list_when_nothing_registered(self, tmp_path):
        r = make_registry(tmp_path)
        assert r.list_registered() == []

    def test_returns_sorted_names(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({
            "zebra": "uri1",
            "alpha": "uri2",
            "middle": "uri3",
        })
        assert r.list_registered() == ["alpha", "middle", "zebra"]

    def test_returns_all_registered_names(self, tmp_path):
        r = make_registry(tmp_path)
        names = ["a", "b", "c", "d"]
        r.register({n: f"uri_{n}" for n in names})
        assert set(r.list_registered()) == set(names)


# ---------------------------------------------------------------------------
# unregister()
# ---------------------------------------------------------------------------


class TestBundleRegistryUnregister:
    def test_unregister_existing_returns_true(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"myapp": "uri"})
        result = r.unregister("myapp")
        assert result is True

    def test_unregister_removes_from_registry(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"myapp": "uri"})
        r.unregister("myapp")
        assert "myapp" not in r.list_registered()

    def test_unregister_nonexistent_returns_false(self, tmp_path):
        r = make_registry(tmp_path)
        result = r.unregister("nonexistent")
        assert result is False

    def test_unregister_cleans_cross_references(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"parent": "uri_parent", "child": "uri_child"})

        # Manually set up include relationship
        parent_state = r._registry["parent"]
        child_state = r._registry["child"]
        parent_state.includes = ["child"]
        child_state.included_by = ["parent"]

        r.unregister("parent")
        # child should no longer list parent in included_by
        assert "parent" not in (child_state.included_by or [])

    def test_unregister_all_then_list_empty(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"a": "uri1", "b": "uri2"})
        r.unregister("a")
        r.unregister("b")
        assert r.list_registered() == []


# ---------------------------------------------------------------------------
# get_state()
# ---------------------------------------------------------------------------


class TestBundleRegistryGetState:
    def test_get_state_none_for_unregistered(self, tmp_path):
        r = make_registry(tmp_path)
        assert r.get_state("missing") is None

    def test_get_state_returns_bundle_state(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"myapp": "git+https://example.com/myapp"})
        state = r.get_state("myapp")
        assert isinstance(state, BundleState)

    def test_get_state_none_returns_all_states_dict(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"a": "uri_a", "b": "uri_b"})
        all_states = r.get_state(None)
        assert isinstance(all_states, dict)
        assert "a" in all_states
        assert "b" in all_states

    def test_get_state_none_empty_registry_returns_empty_dict(self, tmp_path):
        r = make_registry(tmp_path)
        assert r.get_state(None) == {}

    def test_get_state_has_correct_uri(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"myapp": "git+https://github.com/example/myapp"})
        state = r.get_state("myapp")
        assert isinstance(state, BundleState)
        assert state.uri == "git+https://github.com/example/myapp"


# ---------------------------------------------------------------------------
# save() and reload
# ---------------------------------------------------------------------------


class TestBundleRegistrySave:
    def test_save_creates_registry_json(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"myapp": "git+https://example.com/myapp"})
        r.save()
        assert (tmp_path / "registry.json").exists()

    def test_save_persists_registered_bundles(self, tmp_path):
        r = make_registry(tmp_path)
        r.register({"myapp": "git+https://example.com/myapp"})
        r.save()

        data = json.loads((tmp_path / "registry.json").read_text())
        assert "myapp" in data["bundles"]
        assert data["bundles"]["myapp"]["uri"] == "git+https://example.com/myapp"

    def test_saved_state_reloaded_by_new_instance(self, tmp_path):
        r1 = make_registry(tmp_path)
        r1.register({"foundation": "git+https://example.com/foundation"})
        r1.save()

        # Create a new registry pointing to the same home
        r2 = make_registry(tmp_path)
        assert "foundation" in r2.list_registered()

    def test_save_includes_version_field(self, tmp_path):
        r = make_registry(tmp_path)
        r.save()
        data = json.loads((tmp_path / "registry.json").read_text())
        assert data["version"] == 1

    def test_save_creates_home_directory_if_missing(self, tmp_path):
        new_home = tmp_path / "new" / "home"
        r = make_registry(new_home)
        r.register({"x": "uri"})
        r.save()
        assert (new_home / "registry.json").exists()

    def test_round_trip_preserves_multiple_bundles(self, tmp_path):
        r1 = make_registry(tmp_path)
        bundles = {"a": "uri_a", "b": "uri_b", "c": "uri_c"}
        r1.register(bundles)
        r1.save()

        r2 = make_registry(tmp_path)
        assert set(r2.list_registered()) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# load() — mocked source resolver
# ---------------------------------------------------------------------------


class TestBundleRegistryLoad:
    async def test_load_by_registered_name(self, tmp_path):
        bundle_dir = tmp_path / "mybundle"
        bundle_file = make_bundle_file(bundle_dir, name="mybundle", version="1.0.0")

        resolved = MagicMock()
        resolved.active_path = bundle_file
        resolved.source_root = bundle_dir

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=resolved)

        r = make_registry_with_resolver(tmp_path, mock_resolver)
        r.register({"mybundle": "git+https://example.com/mybundle"})

        bundle = await r.load("mybundle")

        assert isinstance(bundle, Bundle)
        assert bundle.name == "mybundle"

    async def test_load_returns_bundle_with_correct_version(self, tmp_path):
        bundle_dir = tmp_path / "vbundle"
        bundle_file = make_bundle_file(bundle_dir, name="vbundle", version="2.5.0")

        resolved = MagicMock()
        resolved.active_path = bundle_file
        resolved.source_root = bundle_dir

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=resolved)

        r = make_registry_with_resolver(tmp_path, mock_resolver)
        r.register({"vbundle": "git+https://example.com/vbundle"})

        bundle = await r.load("vbundle")
        assert bundle.version == "2.5.0"

    async def test_load_caches_bundle_on_second_call(self, tmp_path):
        bundle_dir = tmp_path / "cached"
        bundle_file = make_bundle_file(bundle_dir, name="cached")

        resolved = MagicMock()
        resolved.active_path = bundle_file
        resolved.source_root = bundle_dir

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=resolved)

        r = make_registry_with_resolver(tmp_path, mock_resolver)
        r.register({"cached": "git+https://example.com/cached"})

        b1 = await r.load("cached")
        b2 = await r.load("cached")

        # Should be the exact same object (cached)
        assert b1 is b2
        # Source resolver should only have been called once
        assert mock_resolver.resolve.call_count == 1

    async def test_load_all_when_name_is_none(self, tmp_path):
        bundle_dir_a = tmp_path / "bundleA"
        bundle_dir_b = tmp_path / "bundleB"
        file_a = make_bundle_file(bundle_dir_a, name="bundleA")
        file_b = make_bundle_file(bundle_dir_b, name="bundleB")

        def side_effect(uri):
            if "bundleA" in uri:
                resolved = MagicMock()
                resolved.active_path = file_a
                resolved.source_root = bundle_dir_a
                return resolved
            else:
                resolved = MagicMock()
                resolved.active_path = file_b
                resolved.source_root = bundle_dir_b
                return resolved

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(side_effect=side_effect)

        r = make_registry_with_resolver(tmp_path, mock_resolver)
        r.register({"bundleA": "git+https://example.com/bundleA"})
        r.register({"bundleB": "git+https://example.com/bundleB"})

        result = await r.load(None)

        assert isinstance(result, dict)
        assert "bundleA" in result
        assert "bundleB" in result

    async def test_load_empty_registry_returns_empty_dict(self, tmp_path):
        r = make_registry(tmp_path)
        result = await r.load(None)
        assert result == {}

    async def test_load_yaml_bundle_file(self, tmp_path):
        bundle_dir = tmp_path / "yamlbundle"
        bundle_dir.mkdir()
        bundle_file = bundle_dir / "bundle.yaml"
        bundle_file.write_text(
            """
bundle:
  name: yamlbundle
  version: 1.5.0
"""
        )

        resolved = MagicMock()
        resolved.active_path = bundle_file
        resolved.source_root = bundle_dir

        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=resolved)

        r = make_registry_with_resolver(tmp_path, mock_resolver)
        r.register({"yamlbundle": "git+https://example.com/yamlbundle"})

        bundle = await r.load("yamlbundle")
        assert bundle.name == "yamlbundle"
        assert bundle.version == "1.5.0"


# ---------------------------------------------------------------------------
# BundleState
# ---------------------------------------------------------------------------


class TestBundleState:
    def test_to_dict_round_trip(self):
        state = BundleState(
            uri="git+https://example.com/myapp",
            name="myapp",
            version="1.0.0",
        )
        d = state.to_dict()
        restored = BundleState.from_dict("myapp", d)
        assert restored.uri == state.uri
        assert restored.name == state.name
        assert restored.version == state.version

    def test_from_dict_sets_name_from_param(self):
        d = {
            "uri": "git+https://example.com/x",
            "version": "1.0.0",
            "loaded_at": None,
            "checked_at": None,
            "local_path": None,
        }
        state = BundleState.from_dict("explicit_name", d)
        assert state.name == "explicit_name"

    def test_default_is_root_true(self):
        state = BundleState(uri="uri", name="x")
        assert state.is_root is True

    def test_to_dict_includes_includes_when_set(self):
        state = BundleState(uri="uri", name="x", includes=["child1", "child2"])
        d = state.to_dict()
        assert d["includes"] == ["child1", "child2"]

    def test_to_dict_omits_includes_when_none(self):
        state = BundleState(uri="uri", name="x", includes=None)
        d = state.to_dict()
        assert "includes" not in d
