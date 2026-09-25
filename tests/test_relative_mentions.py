"""Nested instructions resolve beside their source, without changing roots."""

from pathlib import Path

import pytest

from amplifier_lib.bundle import Bundle
from amplifier_lib.mentions import (
    BaseMentionResolver,
    ContentDeduplicator,
    load_mentions,
)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("mention", ["@./journal.md", "@./journal"])
async def test_nested_reference_uses_referring_file_not_workspace(tmp_path, mention):
    write(tmp_path / "rules/AGENTS.md", f"Follow {mention} and @workspace.md")
    write(tmp_path / "workspace.md", "WORKSPACE RULE")
    write(tmp_path / "rules/journal.md", "NESTED JOURNAL RULE")
    write(tmp_path / "journal.md", "WRONG WORKSPACE RULE")
    resolver = BaseMentionResolver(base_path=tmp_path)
    dedup = ContentDeduplicator()
    await load_mentions("@rules/AGENTS.md", resolver, dedup)
    contents = [entry.content for entry in dedup.get_unique_files()]
    assert "NESTED JOURNAL RULE" in contents
    assert "WORKSPACE RULE" in contents
    assert "WRONG WORKSPACE RULE" not in contents
    assert resolver.base_path == tmp_path
    assert resolver.resolve("@journal.md") == tmp_path / "journal.md"


@pytest.mark.asyncio
async def test_same_content_different_roots_preserves_both_nested_files(tmp_path):
    for name in ("global", "project"):
        write(tmp_path / name / "AGENTS.md", "Read @./journal.md")
        write(tmp_path / name / "journal.md", f"{name} journal rule @./AGENTS.md")
    dedup = ContentDeduplicator()
    await load_mentions(
        "@global/AGENTS.md @project/AGENTS.md",
        BaseMentionResolver(base_path=tmp_path),
        dedup,
    )
    entries = dedup.get_unique_files()
    assert len(entries) == 3
    assert len(entries[0].paths) == 2
    assert {entry.content for entry in entries} == {
        "Read @./journal.md",
        "global journal rule @./AGENTS.md",
        "project journal rule @./AGENTS.md",
    }


@pytest.mark.asyncio
async def test_explicit_base_and_nested_namespace_home_absolute_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    original_expanduser = Path.expanduser
    monkeypatch.setattr(
        Path,
        "expanduser",
        lambda self: (
            home / str(self)[2:] if str(self).startswith("~/") else original_expanduser(self)
        ),
    )
    write(home / "rules.md", "HOME RULE")
    absolute = write(tmp_path / "absolute.md", "ABSOLUTE RULE")
    write(tmp_path / "bundle/rules.md", "BUNDLE RULE @./child.md")
    write(tmp_path / "bundle/child.md", "BUNDLE CHILD")
    write(
        tmp_path / "workspace/AGENTS.md",
        f"@~/rules.md @{absolute} @bundle:rules.md @missing:rules.md @missing.md",
    )
    bundle = Bundle(name="bundle", base_path=tmp_path / "bundle")
    resolver = BaseMentionResolver(bundles={"bundle": bundle}, base_path=tmp_path / "wrong")
    dedup = ContentDeduplicator()
    await load_mentions("@AGENTS.md", resolver, dedup, relative_to=tmp_path / "workspace")
    contents = [entry.content for entry in dedup.get_unique_files()]
    assert all(
        value in contents
        for value in (
            "HOME RULE",
            "ABSOLUTE RULE",
            "BUNDLE RULE @./child.md",
            "BUNDLE CHILD",
        )
    )
    assert resolver.base_path == tmp_path / "wrong"


@pytest.mark.asyncio
async def test_recursion_depth_and_legacy_resolver_contract(tmp_path):
    first = write(tmp_path / "first.md", "@second.md")
    second = write(tmp_path / "second.md", "@third.md")
    third = write(tmp_path / "third.md", "THIRD")

    class LegacyResolver:
        def resolve(self, mention):
            return {"@first.md": first, "@second.md": second, "@third.md": third}.get(mention)

    dedup = ContentDeduplicator()
    await load_mentions(
        "@first.md",
        LegacyResolver(),
        dedup,
        relative_to=tmp_path / "elsewhere",
        max_depth=1,
    )
    assert [entry.content for entry in dedup.get_unique_files()] == [
        "@second.md",
        "@third.md",
    ]


def test_scoped_resolution_keeps_subclass_policy(tmp_path):
    write(tmp_path / "rules/private.md", "PRIVATE")

    class RestrictedResolver(BaseMentionResolver):
        def resolve(self, mention):
            return None if "private" in mention else super().resolve(mention)

    resolver = RestrictedResolver(base_path=tmp_path)
    assert resolver.resolve_relative("@private.md", tmp_path / "rules") is None
