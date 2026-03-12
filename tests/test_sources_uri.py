"""Tests for amplifier_lib.sources.uri — ParsedURI, parse_uri, ResolvedSource."""

from __future__ import annotations

from pathlib import Path

import pytest

from amplifier_lib.sources.uri import ParsedURI, ResolvedSource, parse_uri


# ---------------------------------------------------------------------------
# ParsedURI property tests
# ---------------------------------------------------------------------------


class TestParsedURIProperties:
    """Tests for ParsedURI boolean property helpers."""

    # --- is_git ---

    def test_is_git_with_git_plus_https(self):
        uri = ParsedURI(scheme="git+https", host="github.com", path="/org/repo", ref="main", subpath="")
        assert uri.is_git is True

    def test_is_git_with_bare_git_scheme(self):
        uri = ParsedURI(scheme="git", host="github.com", path="/org/repo", ref="", subpath="")
        assert uri.is_git is True

    def test_is_git_false_for_file(self):
        uri = ParsedURI(scheme="file", host="", path="/some/path", ref="", subpath="")
        assert uri.is_git is False

    def test_is_git_false_for_https(self):
        uri = ParsedURI(scheme="https", host="example.com", path="/file.zip", ref="", subpath="")
        assert uri.is_git is False

    def test_is_git_false_for_empty_scheme(self):
        uri = ParsedURI(scheme="", host="", path="my-package", ref="", subpath="")
        assert uri.is_git is False

    # --- is_file ---

    def test_is_file_with_file_scheme(self):
        uri = ParsedURI(scheme="file", host="", path="/tmp/bundle", ref="", subpath="")
        assert uri.is_file is True

    def test_is_file_with_slash_in_path_and_empty_scheme(self):
        # Local absolute path stored as empty scheme with / in path
        uri = ParsedURI(scheme="", host="", path="/absolute/path", ref="", subpath="")
        assert uri.is_file is True

    def test_is_file_with_relative_path_slash(self):
        uri = ParsedURI(scheme="", host="", path="./relative/path", ref="", subpath="")
        assert uri.is_file is True

    def test_is_file_false_for_bare_package_name(self):
        uri = ParsedURI(scheme="", host="", path="my-package", ref="", subpath="")
        assert uri.is_file is False

    def test_is_file_false_for_git(self):
        uri = ParsedURI(scheme="git+https", host="github.com", path="/org/repo", ref="main", subpath="")
        assert uri.is_file is False

    # --- is_http ---

    def test_is_http_with_https_scheme(self):
        uri = ParsedURI(scheme="https", host="example.com", path="/file", ref="", subpath="")
        assert uri.is_http is True

    def test_is_http_with_plain_http_scheme(self):
        uri = ParsedURI(scheme="http", host="example.com", path="/file", ref="", subpath="")
        assert uri.is_http is True

    def test_is_http_false_for_zip_plus_https(self):
        uri = ParsedURI(scheme="zip+https", host="example.com", path="/archive.zip", ref="", subpath="")
        assert uri.is_http is False

    def test_is_http_false_for_git(self):
        uri = ParsedURI(scheme="git+https", host="github.com", path="/repo", ref="main", subpath="")
        assert uri.is_http is False

    def test_is_http_false_for_file(self):
        uri = ParsedURI(scheme="file", host="", path="/path", ref="", subpath="")
        assert uri.is_http is False

    # --- is_zip ---

    def test_is_zip_with_zip_plus_https(self):
        uri = ParsedURI(scheme="zip+https", host="example.com", path="/bundle.zip", ref="", subpath="")
        assert uri.is_zip is True

    def test_is_zip_with_zip_plus_file(self):
        uri = ParsedURI(scheme="zip+file", host="", path="/local/archive.zip", ref="", subpath="")
        assert uri.is_zip is True

    def test_is_zip_false_for_https(self):
        uri = ParsedURI(scheme="https", host="example.com", path="/bundle.zip", ref="", subpath="")
        assert uri.is_zip is False

    def test_is_zip_false_for_git(self):
        uri = ParsedURI(scheme="git+https", host="github.com", path="/repo", ref="main", subpath="")
        assert uri.is_zip is False

    # --- is_package ---

    def test_is_package_bare_name(self):
        uri = ParsedURI(scheme="", host="", path="my-bundle", ref="", subpath="")
        assert uri.is_package is True

    def test_is_package_false_for_path_with_slash(self):
        uri = ParsedURI(scheme="", host="", path="foundation", ref="", subpath="providers/anthropic")
        # scheme="" but path has no slash — is_package checks path only
        assert uri.is_package is True  # path="foundation" has no slash

    def test_is_package_false_for_slash_in_path(self):
        uri = ParsedURI(scheme="", host="", path="foundation/sub", ref="", subpath="")
        assert uri.is_package is False

    def test_is_package_false_for_file_scheme(self):
        uri = ParsedURI(scheme="file", host="", path="my-bundle", ref="", subpath="")
        assert uri.is_package is False

    def test_is_package_false_for_https(self):
        uri = ParsedURI(scheme="https", host="example.com", path="/bundle", ref="", subpath="")
        assert uri.is_package is False


# ---------------------------------------------------------------------------
# parse_uri() tests
# ---------------------------------------------------------------------------


class TestParseURI:
    """Tests for the parse_uri() factory function."""

    # --- git+https URLs ---

    def test_git_https_simple(self):
        result = parse_uri("git+https://github.com/org/repo")
        assert result.scheme == "git+https"
        assert result.host == "github.com"
        assert result.path == "/org/repo"
        assert result.ref == "main"  # defaults to main
        assert result.subpath == ""
        assert result.is_git is True

    def test_git_https_with_branch_ref(self):
        result = parse_uri("git+https://github.com/org/repo@feat/my-branch")
        assert result.ref == "feat/my-branch"
        assert result.path == "/org/repo"

    def test_git_https_with_tag_ref(self):
        result = parse_uri("git+https://github.com/org/repo@v1.2.3")
        assert result.ref == "v1.2.3"

    def test_git_https_with_commit_sha(self):
        result = parse_uri("git+https://github.com/org/repo@abc1234")
        assert result.ref == "abc1234"

    def test_git_https_with_subdirectory_fragment(self):
        result = parse_uri("git+https://github.com/org/repo@main#subdirectory=behaviors/logging")
        assert result.ref == "main"
        assert result.subpath == "behaviors/logging"

    def test_git_https_no_ref_defaults_to_main(self):
        result = parse_uri("git+https://github.com/org/repo")
        assert result.ref == "main"

    def test_git_https_subdirectory_without_ref(self):
        result = parse_uri("git+https://github.com/org/repo#subdirectory=modules/tool")
        assert result.subpath == "modules/tool"

    def test_git_https_fragment_with_multiple_params(self):
        # Only subdirectory= should be extracted
        result = parse_uri("git+https://github.com/org/repo@main#subdirectory=foo&egg=mypkg")
        assert result.subpath == "foo"

    # --- file:// URIs ---

    def test_file_scheme_absolute_path(self):
        result = parse_uri("file:///path/to/bundle")
        assert result.scheme == "file"
        assert result.path == "/path/to/bundle"
        assert result.host == ""
        assert result.ref == ""
        assert result.subpath == ""
        assert result.is_file is True

    def test_file_scheme_with_subdirectory(self):
        result = parse_uri("file:///path/to/repo#subdirectory=inner/dir")
        assert result.path == "/path/to/repo"
        assert result.subpath == "inner/dir"

    # --- Absolute local paths ---

    def test_absolute_path(self):
        result = parse_uri("/home/user/mybundle")
        assert result.scheme == "file"
        assert result.path == "/home/user/mybundle"
        assert result.is_file is True

    def test_absolute_path_with_trailing_slash(self):
        result = parse_uri("/home/user/mybundle/")
        assert result.scheme == "file"
        assert result.is_file is True

    # --- Relative paths ---

    def test_relative_path_dot_slash(self):
        result = parse_uri("./my/bundle")
        assert result.scheme == "file"
        assert result.path == "./my/bundle"
        assert result.is_file is True

    def test_relative_path_dot_dot(self):
        result = parse_uri("../parent/bundle")
        assert result.scheme == "file"
        assert result.path == "../parent/bundle"
        assert result.is_file is True

    # --- http/https URLs ---

    def test_https_url(self):
        result = parse_uri("https://example.com/bundle.yaml")
        assert result.scheme == "https"
        assert result.host == "example.com"
        assert result.path == "/bundle.yaml"
        assert result.ref == ""
        assert result.is_http is True

    def test_http_url(self):
        result = parse_uri("http://example.com/bundle.yaml")
        assert result.scheme == "http"
        assert result.is_http is True

    def test_https_url_with_subdirectory(self):
        result = parse_uri("https://example.com/bundle.yaml#subdirectory=inner")
        assert result.subpath == "inner"

    # --- zip+https URLs ---

    def test_zip_https_url(self):
        result = parse_uri("zip+https://example.com/bundle.zip")
        assert result.scheme == "zip+https"
        assert result.host == "example.com"
        assert result.path == "/bundle.zip"
        assert result.is_zip is True

    def test_zip_file_url(self):
        result = parse_uri("zip+file:///local/archive.zip")
        assert result.scheme == "zip+file"
        assert result.path == "/local/archive.zip"
        assert result.is_zip is True

    def test_zip_https_with_subdirectory(self):
        result = parse_uri("zip+https://example.com/archive.zip#subdirectory=mydir")
        assert result.subpath == "mydir"
        assert result.is_zip is True

    # --- Bare package names ---

    def test_bare_package_name(self):
        result = parse_uri("my-bundle")
        assert result.scheme == ""
        assert result.host == ""
        assert result.path == "my-bundle"
        assert result.ref == ""
        assert result.subpath == ""
        assert result.is_package is True

    def test_package_name_with_subpath(self):
        result = parse_uri("foundation/providers/anthropic")
        assert result.scheme == ""
        assert result.path == "foundation"
        assert result.subpath == "providers/anthropic"
        assert result.is_package is True

    def test_package_name_no_slash(self):
        result = parse_uri("amplifier-core")
        assert result.is_package is True
        assert result.path == "amplifier-core"


# ---------------------------------------------------------------------------
# ResolvedSource tests
# ---------------------------------------------------------------------------


class TestResolvedSource:
    """Tests for ResolvedSource dataclass and is_subdirectory property."""

    def test_is_subdirectory_false_when_equal(self, tmp_path: Path):
        rs = ResolvedSource(active_path=tmp_path, source_root=tmp_path)
        assert rs.is_subdirectory is False

    def test_is_subdirectory_true_when_different(self, tmp_path: Path):
        subdir = tmp_path / "inner"
        subdir.mkdir()
        rs = ResolvedSource(active_path=subdir, source_root=tmp_path)
        assert rs.is_subdirectory is True

    def test_active_path_attribute(self, tmp_path: Path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        rs = ResolvedSource(active_path=subdir, source_root=tmp_path)
        assert rs.active_path == subdir

    def test_source_root_attribute(self, tmp_path: Path):
        rs = ResolvedSource(active_path=tmp_path, source_root=tmp_path)
        assert rs.source_root == tmp_path

    def test_is_subdirectory_sibling_path(self, tmp_path: Path):
        """Paths that are not nested still yield is_subdirectory True (just different)."""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        rs = ResolvedSource(active_path=dir_a, source_root=dir_b)
        assert rs.is_subdirectory is True


# ---------------------------------------------------------------------------
# Parametrized round-trip property checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri, expected_scheme, expected_is_git, expected_is_file, expected_is_http, expected_is_zip, expected_is_package",
    [
        ("git+https://github.com/org/repo", "git+https", True, False, False, False, False),
        ("file:///some/path", "file", False, True, False, False, False),
        ("/absolute/local", "file", False, True, False, False, False),
        ("./relative", "file", False, True, False, False, False),
        ("https://example.com/f", "https", False, False, True, False, False),
        ("http://example.com/f", "http", False, False, True, False, False),
        ("zip+https://example.com/f.zip", "zip+https", False, False, False, True, False),
        ("zip+file:///local.zip", "zip+file", False, False, False, True, False),
        ("my-bundle", "", False, False, False, False, True),
    ],
)
def test_parse_uri_scheme_and_properties(
    uri: str,
    expected_scheme: str,
    expected_is_git: bool,
    expected_is_file: bool,
    expected_is_http: bool,
    expected_is_zip: bool,
    expected_is_package: bool,
):
    result = parse_uri(uri)
    assert result.scheme == expected_scheme
    assert result.is_git == expected_is_git
    assert result.is_file == expected_is_file
    assert result.is_http == expected_is_http
    assert result.is_zip == expected_is_zip
    assert result.is_package == expected_is_package
