"""Tests for amplifier_lib.mentions.resolver — BaseMentionResolver."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from amplifier_lib.mentions.resolver import BaseMentionResolver


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_bundle(resolve_return: Path | None = None) -> MagicMock:
    """Create a mock Bundle with a controllable resolve_context_path()."""
    bundle = MagicMock()
    bundle.resolve_context_path.return_value = resolve_return
    return bundle


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestBaseMentionResolverInit:
    def test_default_bundles_empty(self):
        resolver = BaseMentionResolver()
        assert resolver.bundles == {}

    def test_default_base_path_is_cwd(self):
        resolver = BaseMentionResolver()
        assert resolver.base_path == Path.cwd()

    def test_custom_bundles(self):
        bundle = make_bundle()
        resolver = BaseMentionResolver(bundles={"myns": bundle})
        assert "myns" in resolver.bundles

    def test_custom_base_path(self, tmp_path):
        resolver = BaseMentionResolver(base_path=tmp_path)
        assert resolver.base_path == tmp_path

    def test_none_bundles_defaults_to_empty(self):
        resolver = BaseMentionResolver(bundles=None)
        assert resolver.bundles == {}


# ---------------------------------------------------------------------------
# @namespace:path resolution
# ---------------------------------------------------------------------------


class TestNamespaceResolution:
    def test_resolve_registered_bundle(self, tmp_path):
        target = tmp_path / "context" / "kernel.md"
        target.parent.mkdir(parents=True)
        target.write_text("kernel content")

        bundle = make_bundle(resolve_return=target)
        resolver = BaseMentionResolver(bundles={"foundation": bundle})

        result = resolver.resolve("@foundation:context/kernel")
        bundle.resolve_context_path.assert_called_once_with("context/kernel")
        assert result == target

    def test_resolve_returns_none_for_unregistered_namespace(self):
        resolver = BaseMentionResolver()
        result = resolver.resolve("@unknown:some/path")
        assert result is None

    def test_resolve_uses_bundle_return_value(self, tmp_path):
        target = tmp_path / "file.md"
        target.write_text("data")
        bundle = make_bundle(resolve_return=target)
        resolver = BaseMentionResolver(bundles={"ns": bundle})
        assert resolver.resolve("@ns:file") == target

    def test_resolve_returns_none_when_bundle_returns_none(self):
        bundle = make_bundle(resolve_return=None)
        resolver = BaseMentionResolver(bundles={"ns": bundle})
        assert resolver.resolve("@ns:missing") is None

    def test_multiple_bundles_correct_dispatch(self, tmp_path):
        file_a = tmp_path / "a.md"
        file_b = tmp_path / "b.md"
        file_a.write_text("a")
        file_b.write_text("b")
        bundle_a = make_bundle(resolve_return=file_a)
        bundle_b = make_bundle(resolve_return=file_b)
        resolver = BaseMentionResolver(bundles={"ns_a": bundle_a, "ns_b": bundle_b})
        assert resolver.resolve("@ns_a:x") == file_a
        assert resolver.resolve("@ns_b:x") == file_b

    def test_namespace_key_split_on_first_colon_only(self, tmp_path):
        """@ns:path:extra — namespace is 'ns', path is 'path:extra'."""
        bundle = make_bundle(resolve_return=None)
        resolver = BaseMentionResolver(bundles={"ns": bundle})
        resolver.resolve("@ns:path:extra")
        bundle.resolve_context_path.assert_called_once_with("path:extra")


# ---------------------------------------------------------------------------
# @~/path resolution (home-relative)
# ---------------------------------------------------------------------------


class TestHomeRelativeResolution:
    def test_resolve_tilde_path_that_exists(self, tmp_path, monkeypatch):
        # Create a fake home directory with a file
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        target = fake_home / "config.md"
        target.write_text("config content")

        monkeypatch.setenv("HOME", str(fake_home))

        resolver = BaseMentionResolver()
        mention = f"@~/{target.relative_to(fake_home)}"
        result = resolver.resolve(mention)
        assert result is not None
        assert result.name == "config.md"

    def test_resolve_tilde_path_not_found_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        resolver = BaseMentionResolver()
        result = resolver.resolve("@~/nonexistent/file.md")
        assert result is None

    def test_resolve_tilde_path_with_md_extension_fallback(self, tmp_path, monkeypatch):
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        target = fake_home / "notes.md"
        target.write_text("notes")

        monkeypatch.setenv("HOME", str(fake_home))

        resolver = BaseMentionResolver()
        # @~/notes (without .md extension) should fall back to notes.md
        result = resolver.resolve("@~/notes")
        assert result is not None
        assert result.name == "notes.md"


# ---------------------------------------------------------------------------
# @path resolution (base_path-relative)
# ---------------------------------------------------------------------------


class TestBasePathRelativeResolution:
    def test_resolve_file_relative_to_base_path(self, tmp_path):
        target = tmp_path / "AGENTS.md"
        target.write_text("# Agents")
        resolver = BaseMentionResolver(base_path=tmp_path)
        result = resolver.resolve("@AGENTS.md")
        assert result == target

    def test_resolve_dot_slash_relative_path(self, tmp_path):
        target = tmp_path / "docs" / "guide.md"
        target.parent.mkdir()
        target.write_text("guide content")
        resolver = BaseMentionResolver(base_path=tmp_path)
        result = resolver.resolve("@./docs/guide.md")
        assert result == target

    def test_resolve_nested_path(self, tmp_path):
        target = tmp_path / "src" / "main.py"
        target.parent.mkdir()
        target.write_text("print('hello')")
        resolver = BaseMentionResolver(base_path=tmp_path)
        result = resolver.resolve("@src/main.py")
        assert result == target

    def test_returns_none_when_file_not_found(self, tmp_path):
        resolver = BaseMentionResolver(base_path=tmp_path)
        result = resolver.resolve("@nonexistent.md")
        assert result is None

    def test_md_extension_fallback(self, tmp_path):
        """@AGENTS (no extension) should resolve to AGENTS.md if it exists."""
        target = tmp_path / "AGENTS.md"
        target.write_text("# Agents")
        resolver = BaseMentionResolver(base_path=tmp_path)
        result = resolver.resolve("@AGENTS")
        assert result == target

    def test_md_extension_fallback_not_used_when_bare_exists(self, tmp_path):
        """When @AGENTS (no extension) exists as file, return that, not @AGENTS.md."""
        bare = tmp_path / "AGENTS"
        md = tmp_path / "AGENTS.md"
        bare.write_text("no extension")
        md.write_text("with extension")
        resolver = BaseMentionResolver(base_path=tmp_path)
        result = resolver.resolve("@AGENTS")
        assert result == bare  # bare file takes priority

    def test_resolve_directory_path(self, tmp_path):
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        resolver = BaseMentionResolver(base_path=tmp_path)
        result = resolver.resolve("@mydir")
        assert result == subdir

    def test_mention_without_at_prefix_returns_none(self, tmp_path):
        target = tmp_path / "file.md"
        target.write_text("text")
        resolver = BaseMentionResolver(base_path=tmp_path)
        result = resolver.resolve("file.md")  # no @ prefix
        assert result is None


# ---------------------------------------------------------------------------
# register_bundle
# ---------------------------------------------------------------------------


class TestRegisterBundle:
    def test_register_bundle_adds_to_dict(self):
        resolver = BaseMentionResolver()
        bundle = make_bundle()
        resolver.register_bundle("myns", bundle)
        assert "myns" in resolver.bundles
        assert resolver.bundles["myns"] is bundle

    def test_register_bundle_enables_resolution(self, tmp_path):
        target = tmp_path / "ctx.md"
        target.write_text("context")
        bundle = make_bundle(resolve_return=target)
        resolver = BaseMentionResolver()
        resolver.register_bundle("myns", bundle)
        result = resolver.resolve("@myns:ctx")
        assert result == target

    def test_register_bundle_overwrites_existing(self):
        old_bundle = make_bundle()
        new_bundle = make_bundle()
        resolver = BaseMentionResolver(bundles={"ns": old_bundle})
        resolver.register_bundle("ns", new_bundle)
        assert resolver.bundles["ns"] is new_bundle
