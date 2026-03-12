"""Tests for amplifier_lib.sources.resolver — SimpleSourceResolver."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from amplifier_lib.exceptions import BundleNotFoundError
from amplifier_lib.sources.resolver import SimpleSourceResolver
from amplifier_lib.sources.uri import ParsedURI, ResolvedSource


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_resolved(path: Path) -> ResolvedSource:
    return ResolvedSource(active_path=path, source_root=path)


def make_handler(can_handle_result: bool, resolved: ResolvedSource | None = None) -> MagicMock:
    """Return a mock SourceHandlerProtocol."""
    handler = MagicMock()
    handler.can_handle.return_value = can_handle_result
    if resolved is not None:
        handler.resolve = AsyncMock(return_value=resolved)
    else:
        handler.resolve = AsyncMock(side_effect=BundleNotFoundError("not handled"))
    return handler


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


class TestSimpleSourceResolverInit:
    def test_cache_dir_defaults_to_amplifier_home_cache(self, tmp_path: Path):
        with patch("amplifier_lib.sources.resolver.get_amplifier_home", return_value=tmp_path):
            resolver = SimpleSourceResolver()
        assert resolver.cache_dir == tmp_path / "cache" / "bundles"

    def test_custom_cache_dir(self, tmp_path: Path):
        resolver = SimpleSourceResolver(cache_dir=tmp_path / "custom")
        assert resolver.cache_dir == tmp_path / "custom"

    def test_base_path_defaults_to_cwd(self):
        resolver = SimpleSourceResolver()
        assert resolver.base_path == Path.cwd()

    def test_custom_base_path(self, tmp_path: Path):
        resolver = SimpleSourceResolver(base_path=tmp_path)
        assert resolver.base_path == tmp_path

    def test_default_handlers_are_registered(self):
        resolver = SimpleSourceResolver()
        # Should have 4 default handlers
        assert len(resolver._handlers) == 4

    def test_default_handler_types(self):
        from amplifier_lib.sources.file import FileSourceHandler
        from amplifier_lib.sources.git import GitSourceHandler
        from amplifier_lib.sources.http import HttpSourceHandler
        from amplifier_lib.sources.zip import ZipSourceHandler

        resolver = SimpleSourceResolver()
        handler_types = [type(h) for h in resolver._handlers]
        assert FileSourceHandler in handler_types
        assert GitSourceHandler in handler_types
        assert HttpSourceHandler in handler_types
        assert ZipSourceHandler in handler_types


# ---------------------------------------------------------------------------
# add_handler() priority
# ---------------------------------------------------------------------------


class TestAddHandler:
    def test_custom_handler_inserted_at_front(self):
        resolver = SimpleSourceResolver()
        original_first = resolver._handlers[0]
        custom = make_handler(can_handle_result=False)
        resolver.add_handler(custom)
        assert resolver._handlers[0] is custom
        assert resolver._handlers[1] is original_first

    def test_multiple_custom_handlers_ordered_last_added_first(self):
        resolver = SimpleSourceResolver()
        h1 = make_handler(can_handle_result=False)
        h2 = make_handler(can_handle_result=False)
        resolver.add_handler(h1)
        resolver.add_handler(h2)
        assert resolver._handlers[0] is h2
        assert resolver._handlers[1] is h1


# ---------------------------------------------------------------------------
# resolve() dispatching
# ---------------------------------------------------------------------------


class TestResolve:
    async def test_resolve_dispatches_to_matching_handler(self, tmp_path: Path):
        resolver = SimpleSourceResolver(cache_dir=tmp_path)
        expected = make_resolved(tmp_path)
        custom = make_handler(can_handle_result=True, resolved=expected)
        resolver.add_handler(custom)

        result = await resolver.resolve("my-bundle")

        custom.can_handle.assert_called_once()
        custom.resolve.assert_called_once()
        assert result is expected

    async def test_resolve_skips_non_matching_handlers(self, tmp_path: Path):
        resolver = SimpleSourceResolver(cache_dir=tmp_path)
        non_match = make_handler(can_handle_result=False)
        good_result = make_resolved(tmp_path)
        match = make_handler(can_handle_result=True, resolved=good_result)

        resolver.add_handler(non_match)
        # Place the matching handler after; since add_handler inserts at 0,
        # we insert match after non_match
        resolver._handlers.insert(1, match)

        result = await resolver.resolve("my-bundle")
        non_match.resolve.assert_not_called()
        match.resolve.assert_called_once()
        assert result is good_result

    async def test_resolve_raises_bundle_not_found_when_no_handler_matches(self, tmp_path: Path):
        resolver = SimpleSourceResolver(cache_dir=tmp_path)
        # Replace all handlers with non-matching ones
        resolver._handlers = [make_handler(can_handle_result=False) for _ in range(3)]

        with pytest.raises(BundleNotFoundError, match="No handler for URI"):
            await resolver.resolve("totally-unknown://whatever")

    async def test_resolve_uses_first_matching_handler(self, tmp_path: Path):
        resolver = SimpleSourceResolver(cache_dir=tmp_path)
        result_a = make_resolved(tmp_path / "a")
        result_b = make_resolved(tmp_path / "b")
        handler_a = make_handler(can_handle_result=True, resolved=result_a)
        handler_b = make_handler(can_handle_result=True, resolved=result_b)

        # handler_a is inserted last so it ends up at index 0 (highest priority)
        resolver.add_handler(handler_b)
        resolver.add_handler(handler_a)

        result = await resolver.resolve("whatever")
        assert result is result_a
        handler_b.resolve.assert_not_called()

    async def test_resolve_passes_cache_dir_to_handler(self, tmp_path: Path):
        cache = tmp_path / "my_cache"
        resolver = SimpleSourceResolver(cache_dir=cache)
        expected = make_resolved(tmp_path)
        custom = make_handler(can_handle_result=True, resolved=expected)
        # Replace all default handlers
        resolver._handlers = [custom]

        await resolver.resolve("git+https://github.com/org/repo")
        call_args = custom.resolve.call_args
        _, kwargs = call_args if call_args[1] else (call_args[0], {})
        # resolve(parsed, cache_dir) — check positional arg
        assert custom.resolve.call_args[0][1] == cache or custom.resolve.call_args.args[1] == cache

    async def test_resolve_git_uri_hits_git_handler_by_default(self, tmp_path: Path):
        """Integration-style: git+https URI should route to GitSourceHandler."""
        from amplifier_lib.sources.git import GitSourceHandler

        resolver = SimpleSourceResolver(cache_dir=tmp_path)
        git_handler = next(h for h in resolver._handlers if isinstance(h, GitSourceHandler))

        parsed_git = ParsedURI(
            scheme="git+https", host="github.com", path="/org/repo", ref="main", subpath=""
        )
        assert git_handler.can_handle(parsed_git) is True

    async def test_resolve_file_uri_hits_file_handler_by_default(self, tmp_path: Path):
        from amplifier_lib.sources.file import FileSourceHandler

        resolver = SimpleSourceResolver(cache_dir=tmp_path)
        file_handler = next(h for h in resolver._handlers if isinstance(h, FileSourceHandler))

        parsed_file = ParsedURI(scheme="file", host="", path=str(tmp_path), ref="", subpath="")
        assert file_handler.can_handle(parsed_file) is True

    async def test_resolve_raises_for_package_name_without_custom_handler(self, tmp_path: Path):
        """Package names don't match any default handler."""
        resolver = SimpleSourceResolver(cache_dir=tmp_path)
        with pytest.raises(BundleNotFoundError):
            await resolver.resolve("my-unknown-package")
