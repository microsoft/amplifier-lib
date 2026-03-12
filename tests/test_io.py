"""Tests for amplifier_lib.io — write_with_backup()."""

from pathlib import Path

import pytest

from amplifier_lib.io import write_with_backup


class TestWriteWithBackup:
    # ------------------------------------------------------------------
    # Basic write (no existing file)
    # ------------------------------------------------------------------

    def test_creates_file_when_none_exists(self, tmp_path):
        target = tmp_path / "config.json"
        write_with_backup(target, '{"key": "value"}')
        assert target.exists()
        assert target.read_text(encoding="utf-8") == '{"key": "value"}'

    def test_no_backup_created_when_no_existing_file(self, tmp_path):
        target = tmp_path / "config.json"
        write_with_backup(target, "content")
        backup = tmp_path / "config.json.backup"
        assert not backup.exists()

    def test_returns_none(self, tmp_path):
        target = tmp_path / "out.txt"
        result = write_with_backup(target, "hello")
        assert result is None

    # ------------------------------------------------------------------
    # Backup creation
    # ------------------------------------------------------------------

    def test_creates_backup_of_existing_file(self, tmp_path):
        target = tmp_path / "config.json"
        target.write_text("original content", encoding="utf-8")
        write_with_backup(target, "new content")
        backup = tmp_path / "config.json.backup"
        assert backup.exists()
        assert backup.read_text(encoding="utf-8") == "original content"

    def test_new_content_replaces_old(self, tmp_path):
        target = tmp_path / "data.txt"
        target.write_text("old", encoding="utf-8")
        write_with_backup(target, "new")
        assert target.read_text(encoding="utf-8") == "new"

    def test_backup_suffix_customizable(self, tmp_path):
        target = tmp_path / "file.yaml"
        target.write_text("original", encoding="utf-8")
        write_with_backup(target, "new", backup_suffix=".bak")
        backup = tmp_path / "file.yaml.bak"
        assert backup.exists()
        assert backup.read_text(encoding="utf-8") == "original"

    def test_backup_preserves_full_original_content(self, tmp_path):
        original = "line1\nline2\nline3\n"
        target = tmp_path / "multi.txt"
        target.write_text(original, encoding="utf-8")
        write_with_backup(target, "replaced")
        backup = tmp_path / "multi.txt.backup"
        assert backup.read_text(encoding="utf-8") == original

    def test_multiple_writes_backup_reflects_last_version(self, tmp_path):
        target = tmp_path / "config.txt"
        target.write_text("v1", encoding="utf-8")
        write_with_backup(target, "v2")
        # Now write again — backup should now contain "v2"
        write_with_backup(target, "v3")
        backup = tmp_path / "config.txt.backup"
        assert backup.read_text(encoding="utf-8") == "v2"
        assert target.read_text(encoding="utf-8") == "v3"

    # ------------------------------------------------------------------
    # Binary mode
    # ------------------------------------------------------------------

    def test_binary_write(self, tmp_path):
        target = tmp_path / "data.bin"
        content = b"\x00\x01\x02\x03binary data\xff"
        write_with_backup(target, content, mode="wb")
        assert target.read_bytes() == content

    def test_binary_write_creates_backup(self, tmp_path):
        target = tmp_path / "data.bin"
        original = b"\xde\xad\xbe\xef"
        target.write_bytes(original)
        write_with_backup(target, b"\xca\xfe", mode="wb")
        backup = tmp_path / "data.bin.backup"
        assert backup.read_bytes() == original

    # ------------------------------------------------------------------
    # Atomic write guarantees
    # ------------------------------------------------------------------

    def test_no_temp_files_left_behind(self, tmp_path):
        target = tmp_path / "clean.txt"
        write_with_backup(target, "content")
        leftovers = list(tmp_path.glob("*.tmp"))
        assert leftovers == []

    def test_write_is_complete(self, tmp_path):
        """Content must be fully present after write — no partial writes."""
        target = tmp_path / "complete.txt"
        large = "x" * 100_000
        write_with_backup(target, large)
        assert target.read_text(encoding="utf-8") == large

    # ------------------------------------------------------------------
    # Directory creation
    # ------------------------------------------------------------------

    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c" / "file.txt"
        write_with_backup(target, "nested")
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "nested"

    def test_works_with_deeply_nested_path(self, tmp_path):
        target = tmp_path / "d1" / "d2" / "d3" / "d4" / "deep.txt"
        write_with_backup(target, "deep content")
        assert target.read_text(encoding="utf-8") == "deep content"

    # ------------------------------------------------------------------
    # File with no suffix
    # ------------------------------------------------------------------

    def test_file_without_extension(self, tmp_path):
        target = tmp_path / "Makefile"
        target.write_text("original", encoding="utf-8")
        write_with_backup(target, "new")
        # backup should be Makefile.backup (suffix is "" + ".backup")
        backup = tmp_path / "Makefile.backup"
        assert backup.exists()
        assert backup.read_text(encoding="utf-8") == "original"

    # ------------------------------------------------------------------
    # Unicode content
    # ------------------------------------------------------------------

    def test_unicode_content(self, tmp_path):
        target = tmp_path / "unicode.txt"
        content = "日本語テスト 🎉 émojis and diacritics: café"
        write_with_backup(target, content)
        assert target.read_text(encoding="utf-8") == content
