"""Tests for amplifier_lib.mentions.utils — format_directory_listing()."""

from __future__ import annotations

from pathlib import Path

import pytest

from amplifier_lib.mentions.utils import format_directory_listing


# ---------------------------------------------------------------------------
# Raises on non-directory
# ---------------------------------------------------------------------------


class TestFormatDirectoryListingErrors:
    def test_raises_value_error_for_file(self, tmp_path):
        f = tmp_path / "file.md"
        f.write_text("content")
        with pytest.raises(ValueError, match="not a directory"):
            format_directory_listing(f)

    def test_raises_value_error_for_nonexistent_path(self, tmp_path):
        ghost = tmp_path / "ghost"
        with pytest.raises(ValueError, match="not a directory"):
            format_directory_listing(ghost)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


class TestFormatDirectoryListingBasic:
    def test_output_contains_directory_header(self, tmp_path):
        result = format_directory_listing(tmp_path)
        assert f"Directory: {tmp_path}" in result

    def test_empty_directory_shows_empty_message(self, tmp_path):
        result = format_directory_listing(tmp_path)
        assert "(empty directory)" in result

    def test_single_file_listed_with_file_marker(self, tmp_path):
        (tmp_path / "readme.md").write_text("readme")
        result = format_directory_listing(tmp_path)
        assert "FILE readme.md" in result

    def test_single_subdir_listed_with_dir_marker(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        result = format_directory_listing(tmp_path)
        assert "DIR  subdir" in result

    def test_mixed_files_and_dirs_all_listed(self, tmp_path):
        (tmp_path / "mydir").mkdir()
        (tmp_path / "myfile.txt").write_text("text")
        result = format_directory_listing(tmp_path)
        assert "DIR  mydir" in result
        assert "FILE myfile.txt" in result


# ---------------------------------------------------------------------------
# Sorting: dirs first, then files, alphabetically within each group
# ---------------------------------------------------------------------------


class TestFormatDirectoryListingSorting:
    def test_dirs_appear_before_files(self, tmp_path):
        (tmp_path / "alpha.txt").write_text("a")
        (tmp_path / "zeta_dir").mkdir()
        result = format_directory_listing(tmp_path)
        dir_pos = result.index("zeta_dir")
        file_pos = result.index("alpha.txt")
        assert dir_pos < file_pos  # dir listed first despite 'z' > 'a'

    def test_multiple_dirs_sorted_alphabetically(self, tmp_path):
        for name in ("charlie", "alpha", "bravo"):
            (tmp_path / name).mkdir()
        result = format_directory_listing(tmp_path)
        alpha_pos = result.index("alpha")
        bravo_pos = result.index("bravo")
        charlie_pos = result.index("charlie")
        assert alpha_pos < bravo_pos < charlie_pos

    def test_multiple_files_sorted_alphabetically(self, tmp_path):
        for name in ("z.txt", "a.txt", "m.txt"):
            (tmp_path / name).write_text(name)
        result = format_directory_listing(tmp_path)
        a_pos = result.index("a.txt")
        m_pos = result.index("m.txt")
        z_pos = result.index("z.txt")
        assert a_pos < m_pos < z_pos

    def test_dirs_before_files_with_alpha_sort_each(self, tmp_path):
        (tmp_path / "b_file.txt").write_text("b")
        (tmp_path / "a_file.txt").write_text("a")
        (tmp_path / "z_dir").mkdir()
        (tmp_path / "a_dir").mkdir()
        result = format_directory_listing(tmp_path)

        a_dir_pos = result.index("a_dir")
        z_dir_pos = result.index("z_dir")
        a_file_pos = result.index("a_file.txt")
        b_file_pos = result.index("b_file.txt")

        # All dirs before all files
        assert a_dir_pos < a_file_pos
        assert z_dir_pos < a_file_pos
        # Within dirs: alphabetical
        assert a_dir_pos < z_dir_pos
        # Within files: alphabetical
        assert a_file_pos < b_file_pos

    def test_sorting_is_case_insensitive(self, tmp_path):
        """Sorting should be case-insensitive (lowercase name key)."""
        (tmp_path / "Beta.md").write_text("b")
        (tmp_path / "alpha.md").write_text("a")
        result = format_directory_listing(tmp_path)
        alpha_pos = result.index("alpha.md")
        beta_pos = result.index("Beta.md")
        assert alpha_pos < beta_pos


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------


class TestFormatDirectoryListingFormat:
    def test_result_is_string(self, tmp_path):
        result = format_directory_listing(tmp_path)
        assert isinstance(result, str)

    def test_header_followed_by_blank_line(self, tmp_path):
        (tmp_path / "f.txt").write_text("x")
        result = format_directory_listing(tmp_path)
        # "Directory: <path>\n\n  FILE ..."
        lines = result.split("\n")
        assert lines[0].startswith("Directory: ")
        assert lines[1] == ""

    def test_entry_indented_with_two_spaces(self, tmp_path):
        (tmp_path / "thing.txt").write_text("x")
        result = format_directory_listing(tmp_path)
        entry_line = next(line for line in result.split("\n") if "thing.txt" in line)
        assert entry_line.startswith("  ")

    def test_file_marker_four_chars(self, tmp_path):
        (tmp_path / "f.txt").write_text("x")
        result = format_directory_listing(tmp_path)
        entry_line = next(line for line in result.split("\n") if "f.txt" in line)
        # "  FILE f.txt" — FILE is 4 chars
        assert "FILE" in entry_line

    def test_dir_marker_four_chars_with_space(self, tmp_path):
        (tmp_path / "mydir").mkdir()
        result = format_directory_listing(tmp_path)
        entry_line = next(line for line in result.split("\n") if "mydir" in line)
        # "  DIR  mydir" — "DIR " with extra space for alignment
        assert "DIR " in entry_line

    def test_nested_contents_not_listed(self, tmp_path):
        """Only immediate children should appear, not nested files."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("deep")
        result = format_directory_listing(tmp_path)
        assert "nested.md" not in result
        assert "subdir" in result

    def test_many_entries(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text(str(i))
        for i in range(5):
            (tmp_path / f"dir{i}").mkdir()

        result = format_directory_listing(tmp_path)
        assert result.count("FILE") == 10
        assert result.count("DIR ") == 5
