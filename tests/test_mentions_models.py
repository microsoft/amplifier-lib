"""Tests for amplifier_lib.mentions.models — ContextFile and MentionResult."""

from __future__ import annotations

from pathlib import Path

import pytest

from amplifier_lib.mentions.models import ContextFile, MentionResult


# ---------------------------------------------------------------------------
# ContextFile
# ---------------------------------------------------------------------------


class TestContextFile:
    def test_construction_with_all_fields(self):
        p = Path("/some/file.md")
        cf = ContextFile(content="hello", content_hash="abc123", paths=[p])
        assert cf.content == "hello"
        assert cf.content_hash == "abc123"
        assert cf.paths == [p]

    def test_multiple_paths(self):
        paths = [Path("/a.md"), Path("/b.md"), Path("/c.md")]
        cf = ContextFile(content="shared", content_hash="hash", paths=paths)
        assert len(cf.paths) == 3
        assert Path("/b.md") in cf.paths

    def test_empty_paths_list(self):
        cf = ContextFile(content="x", content_hash="h", paths=[])
        assert cf.paths == []

    def test_is_dataclass_with_correct_field_types(self):
        cf = ContextFile(
            content="some text",
            content_hash="deadbeef",
            paths=[Path("/foo/bar.md")],
        )
        assert isinstance(cf.content, str)
        assert isinstance(cf.content_hash, str)
        assert isinstance(cf.paths, list)

    def test_equality(self):
        p = Path("/x.md")
        cf1 = ContextFile(content="a", content_hash="h", paths=[p])
        cf2 = ContextFile(content="a", content_hash="h", paths=[p])
        assert cf1 == cf2

    def test_inequality_on_different_content(self):
        p = Path("/x.md")
        cf1 = ContextFile(content="a", content_hash="h1", paths=[p])
        cf2 = ContextFile(content="b", content_hash="h2", paths=[p])
        assert cf1 != cf2


# ---------------------------------------------------------------------------
# MentionResult — found property
# ---------------------------------------------------------------------------


class TestMentionResultFound:
    def test_found_true_when_resolved_path_and_content(self):
        result = MentionResult(
            mention="@AGENTS.md",
            resolved_path=Path("/project/AGENTS.md"),
            content="# Agents",
            error=None,
        )
        assert result.found is True

    def test_found_true_when_is_directory_even_without_content(self):
        result = MentionResult(
            mention="@./src",
            resolved_path=Path("/project/src"),
            content=None,
            error=None,
            is_directory=True,
        )
        assert result.found is True

    def test_found_true_when_is_directory_with_content(self):
        result = MentionResult(
            mention="@./src",
            resolved_path=Path("/project/src"),
            content="Directory: /project/src\n\n  FILE foo.py",
            error=None,
            is_directory=True,
        )
        assert result.found is True

    def test_found_false_when_no_resolved_path(self):
        result = MentionResult(
            mention="@missing.md",
            resolved_path=None,
            content=None,
            error=None,
        )
        assert result.found is False

    def test_found_false_when_resolved_path_but_no_content_and_not_directory(self):
        result = MentionResult(
            mention="@AGENTS.md",
            resolved_path=Path("/project/AGENTS.md"),
            content=None,
            error=None,
            is_directory=False,
        )
        assert result.found is False

    def test_found_false_when_both_path_and_content_none(self):
        result = MentionResult(
            mention="@ghost",
            resolved_path=None,
            content=None,
            error="file not found",
        )
        assert result.found is False

    def test_is_directory_defaults_to_false(self):
        result = MentionResult(
            mention="@file.md",
            resolved_path=Path("/file.md"),
            content="text",
            error=None,
        )
        assert result.is_directory is False

    def test_all_fields_accessible(self):
        p = Path("/foo.md")
        result = MentionResult(
            mention="@foo.md",
            resolved_path=p,
            content="bar",
            error=None,
            is_directory=False,
        )
        assert result.mention == "@foo.md"
        assert result.resolved_path == p
        assert result.content == "bar"
        assert result.error is None
        assert result.is_directory is False

    def test_error_does_not_affect_found_when_path_missing(self):
        result = MentionResult(
            mention="@oops",
            resolved_path=None,
            content=None,
            error="permission denied",
        )
        assert result.found is False

    @pytest.mark.parametrize(
        "resolved_path,content,is_directory,expected_found",
        [
            (Path("/f.md"), "content", False, True),
            (Path("/d"), None, True, True),
            (Path("/d"), "listing", True, True),
            (None, None, False, False),
            (Path("/f.md"), None, False, False),
            (None, "orphan", False, False),  # content without path => not found
        ],
    )
    def test_found_parametrized(self, resolved_path, content, is_directory, expected_found):
        result = MentionResult(
            mention="@x",
            resolved_path=resolved_path,
            content=content,
            error=None,
            is_directory=is_directory,
        )
        assert result.found is expected_found
