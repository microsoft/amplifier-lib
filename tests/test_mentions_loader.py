"""Tests for amplifier_lib.mentions.loader — load_mentions() and format_context_block()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from amplifier_lib.mentions.deduplicator import ContentDeduplicator
from amplifier_lib.mentions.loader import format_context_block, load_mentions
from amplifier_lib.mentions.resolver import BaseMentionResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_resolver(base_path: Path) -> BaseMentionResolver:
    """Real BaseMentionResolver with no bundles, rooted at base_path."""
    return BaseMentionResolver(bundles={}, base_path=base_path)


# ---------------------------------------------------------------------------
# load_mentions() — basic resolution
# ---------------------------------------------------------------------------


class TestLoadMentionsBasic:
    async def test_resolves_single_file_mention(self, tmp_path):
        target = tmp_path / "AGENTS.md"
        target.write_text("# Agents\nDo things.")
        resolver = make_resolver(tmp_path)

        results = await load_mentions("Check @AGENTS.md", resolver)

        assert len(results) == 1
        r = results[0]
        assert r.mention == "@AGENTS.md"
        assert r.found is True
        assert r.content == "# Agents\nDo things."
        assert r.resolved_path == target

    async def test_no_mentions_returns_empty_list(self, tmp_path):
        resolver = make_resolver(tmp_path)
        results = await load_mentions("No mentions here", resolver)
        assert results == []

    async def test_multiple_unique_mentions(self, tmp_path):
        a = tmp_path / "a.md"
        b = tmp_path / "b.md"
        a.write_text("content a")
        b.write_text("content b")
        resolver = make_resolver(tmp_path)

        results = await load_mentions("See @a.md and @b.md", resolver)

        assert len(results) == 2
        mentions = {r.mention for r in results}
        assert mentions == {"@a.md", "@b.md"}
        assert all(r.found for r in results)

    async def test_result_order_matches_mention_order(self, tmp_path):
        for name in ("first.md", "second.md", "third.md"):
            (tmp_path / name).write_text(f"content of {name}")
        resolver = make_resolver(tmp_path)

        results = await load_mentions("@first.md then @second.md then @third.md", resolver)

        assert [r.mention for r in results] == ["@first.md", "@second.md", "@third.md"]


# ---------------------------------------------------------------------------
# load_mentions() — missing / unresolvable mentions
# ---------------------------------------------------------------------------


class TestLoadMentionsMissing:
    async def test_missing_file_returns_result_with_found_false(self, tmp_path):
        resolver = make_resolver(tmp_path)
        results = await load_mentions("Load @nonexistent.md please", resolver)

        assert len(results) == 1
        assert results[0].found is False
        assert results[0].content is None
        assert results[0].error is None  # opportunistic — no error

    async def test_unregistered_namespace_returns_not_found(self, tmp_path):
        resolver = make_resolver(tmp_path)
        results = await load_mentions("Use @unknown:ctx/thing", resolver)

        assert len(results) == 1
        assert results[0].found is False

    async def test_mix_of_found_and_missing(self, tmp_path):
        existing = tmp_path / "exists.md"
        existing.write_text("real content")
        resolver = make_resolver(tmp_path)

        results = await load_mentions("@exists.md and @missing.md", resolver)

        assert len(results) == 2
        found_map = {r.mention: r.found for r in results}
        assert found_map["@exists.md"] is True
        assert found_map["@missing.md"] is False


# ---------------------------------------------------------------------------
# load_mentions() — deduplication
# ---------------------------------------------------------------------------


class TestLoadMentionsDeduplicate:
    async def test_duplicate_mention_in_text_resolved_once(self, tmp_path):
        target = tmp_path / "file.md"
        target.write_text("shared content")
        resolver = make_resolver(tmp_path)

        # parse_mentions deduplicates, so only one result
        results = await load_mentions("@file.md and @file.md again", resolver)
        assert len(results) == 1

    async def test_shared_deduplicator_tracks_across_calls(self, tmp_path):
        target = tmp_path / "shared.md"
        target.write_text("shared")
        resolver = make_resolver(tmp_path)
        dedup = ContentDeduplicator()

        # First call
        results1 = await load_mentions("@shared.md", resolver, deduplicator=dedup)
        assert results1[0].content == "shared"

        # Second call with same deduplicator — content already seen
        results2 = await load_mentions("Also @shared.md", resolver, deduplicator=dedup)
        # Result is returned but content is None (already seen by deduplicator)
        assert results2[0].resolved_path == target
        assert results2[0].content is None

    async def test_deduplicator_populated_correctly(self, tmp_path):
        target = tmp_path / "ctx.md"
        target.write_text("the content")
        resolver = make_resolver(tmp_path)
        dedup = ContentDeduplicator()

        await load_mentions("@ctx.md", resolver, deduplicator=dedup)

        assert dedup.is_seen("the content")
        unique = dedup.get_unique_files()
        assert len(unique) == 1
        assert unique[0].content == "the content"


# ---------------------------------------------------------------------------
# load_mentions() — directory resolution
# ---------------------------------------------------------------------------


class TestLoadMentionsDirectory:
    async def test_directory_mention_returns_is_directory_true(self, tmp_path):
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "main.py").write_text("# main")
        resolver = make_resolver(tmp_path)

        results = await load_mentions("See @src for details", resolver)

        assert len(results) == 1
        r = results[0]
        assert r.is_directory is True
        assert r.found is True
        assert r.content is not None
        assert "Directory:" in r.content


# ---------------------------------------------------------------------------
# load_mentions() — recursive mention resolution
# ---------------------------------------------------------------------------


class TestLoadMentionsRecursive:
    async def test_recursive_mention_in_file_content(self, tmp_path):
        """A mention within a loaded file should also be resolved."""
        child = tmp_path / "child.md"
        child.write_text("I am the child file")

        parent = tmp_path / "parent.md"
        parent.write_text("Parent says: read @child.md")

        resolver = make_resolver(tmp_path)
        dedup = ContentDeduplicator()

        await load_mentions("@parent.md", resolver, deduplicator=dedup)

        # Both parent and child content should be in the deduplicator
        assert dedup.is_seen("Parent says: read @child.md")
        assert dedup.is_seen("I am the child file")

    async def test_max_depth_zero_prevents_recursion(self, tmp_path):
        child = tmp_path / "child.md"
        child.write_text("child content")

        parent = tmp_path / "parent.md"
        parent.write_text("Parent: @child.md")

        resolver = make_resolver(tmp_path)
        dedup = ContentDeduplicator()

        await load_mentions("@parent.md", resolver, deduplicator=dedup, max_depth=0)

        # Parent loaded, child should NOT be loaded (max_depth=0)
        assert dedup.is_seen("Parent: @child.md")
        assert not dedup.is_seen("child content")

    async def test_self_referencing_file_does_not_loop(self, tmp_path):
        """A file referencing itself should not cause infinite recursion."""
        loop_file = tmp_path / "loop.md"
        loop_file.write_text("This file mentions @loop.md recursively")

        resolver = make_resolver(tmp_path)

        # Should complete without hanging
        results = await load_mentions("@loop.md", resolver, max_depth=3)
        assert len(results) == 1
        assert results[0].found is True


# ---------------------------------------------------------------------------
# format_context_block()
# ---------------------------------------------------------------------------


class TestFormatContextBlock:
    def test_empty_deduplicator_returns_empty_string(self):
        d = ContentDeduplicator()
        result = format_context_block(d)
        assert result == ""

    def test_single_file_produces_xml_block(self, tmp_path):
        target = tmp_path / "AGENTS.md"
        target.write_text("agent instructions")
        d = ContentDeduplicator()
        d.add_file(target, "agent instructions")

        result = format_context_block(d)

        assert "<context_file" in result
        assert "agent instructions" in result
        assert "</context_file>" in result

    def test_paths_attribute_contains_resolved_path(self, tmp_path):
        target = tmp_path / "file.md"
        target.write_text("content")
        d = ContentDeduplicator()
        d.add_file(target, "content")

        result = format_context_block(d)

        assert 'paths="' in result
        assert str(target.resolve()) in result

    def test_mention_to_path_shows_arrow_format(self, tmp_path):
        target = tmp_path / "AGENTS.md"
        target.write_text("agents")
        d = ContentDeduplicator()
        d.add_file(target, "agents")

        mention_to_path = {"@AGENTS.md": target}
        result = format_context_block(d, mention_to_path=mention_to_path)

        assert "@AGENTS.md" in result
        assert "\u2192" in result  # → arrow
        assert str(target.resolve()) in result

    def test_multiple_files_separated_by_double_newline(self, tmp_path):
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("alpha content")
        f2.write_text("beta content")
        d = ContentDeduplicator()
        d.add_file(f1, "alpha content")
        d.add_file(f2, "beta content")

        result = format_context_block(d)

        # Two blocks separated by double newline
        blocks = result.split("\n\n")
        context_blocks = [b for b in blocks if "<context_file" in b]
        assert len(context_blocks) == 2

    def test_content_wrapped_inside_tags(self, tmp_path):
        target = tmp_path / "test.md"
        target.write_text("my file content")
        d = ContentDeduplicator()
        d.add_file(target, "my file content")

        result = format_context_block(d)

        assert "my file content" in result
        # Content should be between open and close tags
        open_idx = result.index("<context_file")
        close_idx = result.index("</context_file>")
        assert open_idx < close_idx

    def test_path_without_mention_shows_path_only(self, tmp_path):
        target = tmp_path / "bare.md"
        target.write_text("bare content")
        d = ContentDeduplicator()
        d.add_file(target, "bare content")

        # No mention_to_path provided — should show raw path
        result = format_context_block(d, mention_to_path=None)
        assert str(target.resolve()) in result

    def test_namespace_mention_in_attribution(self, tmp_path):
        target = tmp_path / "kernel.md"
        target.write_text("kernel content")
        d = ContentDeduplicator()
        d.add_file(target, "kernel content")

        mention_to_path = {"@foundation:context/KERNEL": target}
        result = format_context_block(d, mention_to_path=mention_to_path)

        assert "@foundation:context/KERNEL" in result

    def test_duplicate_path_tracked_once_in_output(self, tmp_path):
        """When same content found at two paths, both appear in paths attribute."""
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("shared")
        f2.write_text("shared")
        d = ContentDeduplicator()
        d.add_file(f1, "shared")
        d.add_file(f2, "shared")

        result = format_context_block(d)

        # Only one block for shared content
        assert result.count("<context_file") == 1
        # Both paths should appear
        assert "a.md" in result
        assert "b.md" in result
