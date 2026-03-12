"""Tests for amplifier_lib.mentions.deduplicator — ContentDeduplicator."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from amplifier_lib.mentions.deduplicator import ContentDeduplicator
from amplifier_lib.mentions.models import ContextFile


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# add_file
# ---------------------------------------------------------------------------


class TestAddFile:
    def test_add_new_content_returns_true(self, tmp_path):
        d = ContentDeduplicator()
        p = tmp_path / "a.md"
        p.write_text("hello")
        result = d.add_file(p, "hello")
        assert result is True

    def test_add_duplicate_content_returns_false(self, tmp_path):
        d = ContentDeduplicator()
        p1 = tmp_path / "a.md"
        p2 = tmp_path / "b.md"
        p1.write_text("shared")
        p2.write_text("shared")
        d.add_file(p1, "shared")
        result = d.add_file(p2, "shared")
        assert result is False

    def test_add_different_content_both_return_true(self, tmp_path):
        d = ContentDeduplicator()
        p1 = tmp_path / "a.md"
        p2 = tmp_path / "b.md"
        p1.write_text("alpha")
        p2.write_text("beta")
        assert d.add_file(p1, "alpha") is True
        assert d.add_file(p2, "beta") is True

    def test_same_path_same_content_returns_false_second_time(self, tmp_path):
        d = ContentDeduplicator()
        p = tmp_path / "file.md"
        p.write_text("data")
        d.add_file(p, "data")
        result = d.add_file(p, "data")
        assert result is False

    def test_duplicate_path_not_added_twice(self, tmp_path):
        d = ContentDeduplicator()
        p1 = tmp_path / "a.md"
        p2 = tmp_path / "b.md"
        p1.write_text("same")
        p2.write_text("same")
        d.add_file(p1, "same")
        d.add_file(p2, "same")

        files = d.get_unique_files()
        assert len(files) == 1
        assert len(files[0].paths) == 2

    def test_duplicate_path_not_added_twice_for_same_path(self, tmp_path):
        """Adding the same path twice with the same content only tracks path once."""
        d = ContentDeduplicator()
        p = tmp_path / "file.md"
        p.write_text("content")
        d.add_file(p, "content")
        d.add_file(p, "content")

        files = d.get_unique_files()
        assert len(files) == 1
        # The path should appear only once (dedup by resolved path)
        assert len(files[0].paths) == 1


# ---------------------------------------------------------------------------
# get_unique_files
# ---------------------------------------------------------------------------


class TestGetUniqueFiles:
    def test_empty_deduplicator_returns_empty_list(self):
        d = ContentDeduplicator()
        assert d.get_unique_files() == []

    def test_single_file_returns_one_context_file(self, tmp_path):
        d = ContentDeduplicator()
        p = tmp_path / "file.md"
        p.write_text("content")
        d.add_file(p, "content")

        files = d.get_unique_files()
        assert len(files) == 1
        assert isinstance(files[0], ContextFile)
        assert files[0].content == "content"

    def test_returns_correct_content_hash(self, tmp_path):
        d = ContentDeduplicator()
        p = tmp_path / "file.md"
        content = "test content"
        p.write_text(content)
        d.add_file(p, content)

        files = d.get_unique_files()
        assert files[0].content_hash == sha256(content)

    def test_two_unique_files_returns_two_context_files(self, tmp_path):
        d = ContentDeduplicator()
        p1 = tmp_path / "a.md"
        p2 = tmp_path / "b.md"
        p1.write_text("alpha")
        p2.write_text("beta")
        d.add_file(p1, "alpha")
        d.add_file(p2, "beta")

        files = d.get_unique_files()
        assert len(files) == 2
        contents = {f.content for f in files}
        assert contents == {"alpha", "beta"}

    def test_duplicate_content_tracked_with_multiple_paths(self, tmp_path):
        d = ContentDeduplicator()
        p1 = tmp_path / "a.md"
        p2 = tmp_path / "b.md"
        p1.write_text("shared")
        p2.write_text("shared")
        d.add_file(p1, "shared")
        d.add_file(p2, "shared")

        files = d.get_unique_files()
        assert len(files) == 1
        assert len(files[0].paths) == 2
        resolved_names = {p.name for p in files[0].paths}
        assert resolved_names == {"a.md", "b.md"}

    def test_three_files_one_duplicate(self, tmp_path):
        d = ContentDeduplicator()
        p1 = tmp_path / "a.md"
        p2 = tmp_path / "b.md"
        p3 = tmp_path / "c.md"
        p1.write_text("unique")
        p2.write_text("shared")
        p3.write_text("shared")
        d.add_file(p1, "unique")
        d.add_file(p2, "shared")
        d.add_file(p3, "shared")

        files = d.get_unique_files()
        assert len(files) == 2

        # Find which ContextFile is the shared one
        shared = next(f for f in files if f.content == "shared")
        assert len(shared.paths) == 2


# ---------------------------------------------------------------------------
# is_seen
# ---------------------------------------------------------------------------


class TestIsSeen:
    def test_is_seen_false_for_unseen_content(self):
        d = ContentDeduplicator()
        assert d.is_seen("anything") is False

    def test_is_seen_true_after_add_file(self, tmp_path):
        d = ContentDeduplicator()
        p = tmp_path / "file.md"
        p.write_text("data")
        d.add_file(p, "data")
        assert d.is_seen("data") is True

    def test_is_seen_false_for_different_content(self, tmp_path):
        d = ContentDeduplicator()
        p = tmp_path / "file.md"
        p.write_text("hello")
        d.add_file(p, "hello")
        assert d.is_seen("world") is False

    def test_is_seen_uses_hash_not_exact_equality(self, tmp_path):
        """is_seen() should work even if we pass the exact same string."""
        d = ContentDeduplicator()
        content = "some text with unicode: \u00e9\u00e0"
        p = tmp_path / "f.md"
        p.write_text(content)
        d.add_file(p, content)
        assert d.is_seen(content) is True


# ---------------------------------------------------------------------------
# get_known_hashes
# ---------------------------------------------------------------------------


class TestGetKnownHashes:
    def test_empty_returns_empty_set(self):
        d = ContentDeduplicator()
        assert d.get_known_hashes() == set()

    def test_returns_hash_of_added_content(self, tmp_path):
        d = ContentDeduplicator()
        content = "hello world"
        p = tmp_path / "f.md"
        p.write_text(content)
        d.add_file(p, content)
        hashes = d.get_known_hashes()
        assert sha256(content) in hashes

    def test_returns_all_unique_hashes(self, tmp_path):
        d = ContentDeduplicator()
        for i, text in enumerate(["a", "b", "c"]):
            p = tmp_path / f"f{i}.md"
            p.write_text(text)
            d.add_file(p, text)
        hashes = d.get_known_hashes()
        assert len(hashes) == 3

    def test_duplicate_content_counted_once(self, tmp_path):
        d = ContentDeduplicator()
        p1 = tmp_path / "a.md"
        p2 = tmp_path / "b.md"
        p1.write_text("dup")
        p2.write_text("dup")
        d.add_file(p1, "dup")
        d.add_file(p2, "dup")
        assert len(d.get_known_hashes()) == 1
