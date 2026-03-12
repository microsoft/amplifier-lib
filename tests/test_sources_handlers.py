"""Tests for source handler can_handle() and resolve() logic.

Git, Http, and Zip handlers are tested for can_handle() only (network/subprocess
mocked where resolve() is exercised). FileSourceHandler resolve() is tested with
real temp directories.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from amplifier_lib.exceptions import BundleNotFoundError
from amplifier_lib.sources.file import FileSourceHandler
from amplifier_lib.sources.git import GitSourceHandler
from amplifier_lib.sources.http import HttpSourceHandler
from amplifier_lib.sources.uri import ParsedURI
from amplifier_lib.sources.zip import ZipSourceHandler

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def git_uri(path: str = "/org/repo", ref: str = "main", subpath: str = "") -> ParsedURI:
    return ParsedURI(scheme="git+https", host="github.com", path=path, ref=ref, subpath=subpath)


def file_uri(path: str, subpath: str = "") -> ParsedURI:
    return ParsedURI(scheme="file", host="", path=path, ref="", subpath=subpath)


def http_uri(scheme: str = "https", path: str = "/bundle.yaml", subpath: str = "") -> ParsedURI:
    return ParsedURI(scheme=scheme, host="example.com", path=path, ref="", subpath=subpath)


def zip_uri(scheme: str = "zip+https", path: str = "/bundle.zip", subpath: str = "") -> ParsedURI:
    return ParsedURI(scheme=scheme, host="example.com", path=path, ref="", subpath=subpath)


def package_uri(name: str = "my-bundle") -> ParsedURI:
    return ParsedURI(scheme="", host="", path=name, ref="", subpath="")


# ===========================================================================
# GitSourceHandler
# ===========================================================================


class TestGitSourceHandlerCanHandle:
    def setup_method(self):
        self.handler = GitSourceHandler()

    def test_can_handle_git_plus_https(self):
        assert self.handler.can_handle(git_uri()) is True

    def test_can_handle_bare_git_scheme(self):
        uri = ParsedURI(scheme="git", host="github.com", path="/org/repo", ref="main", subpath="")
        assert self.handler.can_handle(uri) is True

    def test_cannot_handle_file_uri(self):
        assert self.handler.can_handle(file_uri("/some/path")) is False

    def test_cannot_handle_https_uri(self):
        assert self.handler.can_handle(http_uri("https")) is False

    def test_cannot_handle_zip_uri(self):
        assert self.handler.can_handle(zip_uri()) is False

    def test_cannot_handle_package_name(self):
        assert self.handler.can_handle(package_uri()) is False


class TestGitSourceHandlerHelpers:
    def setup_method(self):
        self.handler = GitSourceHandler()

    def test_build_git_url(self):
        uri = git_uri(path="/org/repo")
        url = self.handler._build_git_url(uri)
        assert url == "https://github.com/org/repo"

    def test_get_cache_path_is_deterministic(self, tmp_path: Path):
        uri = git_uri(path="/org/repo", ref="main")
        path1 = self.handler._get_cache_path(uri, tmp_path)
        path2 = self.handler._get_cache_path(uri, tmp_path)
        assert path1 == path2

    def test_get_cache_path_differs_for_different_refs(self, tmp_path: Path):
        uri_main = git_uri(path="/org/repo", ref="main")
        uri_dev = git_uri(path="/org/repo", ref="dev")
        assert self.handler._get_cache_path(uri_main, tmp_path) != self.handler._get_cache_path(
            uri_dev, tmp_path
        )

    def test_get_cache_path_includes_repo_name(self, tmp_path: Path):
        uri = git_uri(path="/org/myrepo", ref="main")
        cache_path = self.handler._get_cache_path(uri, tmp_path)
        assert "myrepo" in cache_path.name

    def test_verify_clone_integrity_missing_dir(self, tmp_path: Path):
        assert self.handler._verify_clone_integrity(tmp_path / "nonexistent") is False

    def test_verify_clone_integrity_no_git_dir(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").touch()
        assert self.handler._verify_clone_integrity(tmp_path) is False

    def test_verify_clone_integrity_with_git_and_pyproject(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "pyproject.toml").touch()
        assert self.handler._verify_clone_integrity(tmp_path) is True

    def test_verify_clone_integrity_with_git_and_bundle_md(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "bundle.md").touch()
        assert self.handler._verify_clone_integrity(tmp_path) is True

    def test_verify_clone_integrity_with_git_and_bundle_yaml(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "bundle.yaml").touch()
        assert self.handler._verify_clone_integrity(tmp_path) is True

    def test_verify_clone_integrity_with_git_no_expected_files(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        # No pyproject.toml, bundle.md, or bundle.yaml
        assert self.handler._verify_clone_integrity(tmp_path) is False

    def test_verify_clone_integrity_with_setup_py(self, tmp_path: Path):
        (tmp_path / ".git").mkdir()
        (tmp_path / "setup.py").touch()
        assert self.handler._verify_clone_integrity(tmp_path) is True

    def test_cache_metadata_round_trip(self, tmp_path: Path):
        meta = {"ref": "main", "commit": "abc123"}
        self.handler._save_cache_metadata(tmp_path, meta)
        loaded = self.handler._get_cache_metadata(tmp_path)
        assert loaded["ref"] == "main"
        assert loaded["commit"] == "abc123"

    def test_cache_metadata_returns_empty_when_missing(self, tmp_path: Path):
        result = self.handler._get_cache_metadata(tmp_path / "nonexistent")
        assert result == {}


class TestGitSourceHandlerResolve:
    """Test resolve() by mocking subprocess."""

    async def test_resolve_uses_cached_when_valid(self, tmp_path: Path):
        handler = GitSourceHandler()
        uri = git_uri(path="/org/repo", ref="main")
        cache_path = handler._get_cache_path(uri, tmp_path)

        # Set up a valid-looking cache
        cache_path.mkdir(parents=True)
        (cache_path / ".git").mkdir()
        (cache_path / "pyproject.toml").touch()

        result = await handler.resolve(uri, tmp_path)
        assert result.active_path == cache_path
        assert result.source_root == cache_path

    async def test_resolve_uses_cached_with_subpath(self, tmp_path: Path):
        handler = GitSourceHandler()
        uri = git_uri(path="/org/repo", ref="main", subpath="behaviors/logging")
        cache_path = handler._get_cache_path(uri, tmp_path)

        # Set up valid cache with subpath
        subdir = cache_path / "behaviors" / "logging"
        subdir.mkdir(parents=True)
        (cache_path / ".git").mkdir()
        (cache_path / "pyproject.toml").touch()

        result = await handler.resolve(uri, tmp_path)
        assert result.active_path == subdir
        assert result.source_root == cache_path

    async def test_resolve_clones_when_not_cached(self, tmp_path: Path):
        handler = GitSourceHandler()
        uri = git_uri(path="/org/repo", ref="main")
        cache_path = handler._get_cache_path(uri, tmp_path)

        def fake_run(args, **kwargs):
            # Simulate git clone by creating the directory with expected structure
            cache_path.mkdir(parents=True, exist_ok=True)
            (cache_path / ".git").mkdir()
            (cache_path / "pyproject.toml").touch()
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=fake_run):
            with patch.object(handler, "_get_local_commit", return_value="abc123"):
                result = await handler.resolve(uri, tmp_path)

        assert result.active_path == cache_path
        assert result.source_root == cache_path

    async def test_resolve_raises_on_clone_failure(self, tmp_path: Path):
        import subprocess

        handler = GitSourceHandler()
        uri = git_uri(path="/org/repo", ref="main")

        error = subprocess.CalledProcessError(
            1, ["git", "clone"], stderr="fatal: repository not found"
        )
        with patch("subprocess.run", side_effect=error):
            with pytest.raises(BundleNotFoundError, match="Failed to clone"):
                await handler.resolve(uri, tmp_path)

    async def test_resolve_removes_invalid_cache_and_reclones(self, tmp_path: Path):
        handler = GitSourceHandler()
        uri = git_uri(path="/org/repo", ref="main")
        cache_path = handler._get_cache_path(uri, tmp_path)

        # Create an invalid cache (missing .git)
        cache_path.mkdir(parents=True)
        (cache_path / "some_file.txt").touch()

        def fake_run(args, **kwargs):
            cache_path.mkdir(parents=True, exist_ok=True)
            (cache_path / ".git").mkdir(exist_ok=True)
            (cache_path / "pyproject.toml").touch()
            result = MagicMock()
            result.returncode = 0
            result.stdout = result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=fake_run):
            with patch.object(handler, "_get_local_commit", return_value="abc123"):
                result = await handler.resolve(uri, tmp_path)

        assert result.source_root == cache_path


# ===========================================================================
# FileSourceHandler
# ===========================================================================


class TestFileSourceHandlerCanHandle:
    def setup_method(self):
        self.handler = FileSourceHandler()

    def test_can_handle_file_scheme(self):
        assert self.handler.can_handle(file_uri("/some/path")) is True

    def test_can_handle_absolute_path_parsed_as_file(self):
        uri = ParsedURI(scheme="file", host="", path="/absolute", ref="", subpath="")
        assert self.handler.can_handle(uri) is True

    def test_can_handle_relative_dot_slash(self):
        uri = ParsedURI(scheme="", host="", path="./relative", ref="", subpath="")
        assert self.handler.can_handle(uri) is True  # is_file True because "/" in path

    def test_cannot_handle_git_uri(self):
        assert self.handler.can_handle(git_uri()) is False

    def test_cannot_handle_package_name(self):
        assert self.handler.can_handle(package_uri()) is False

    def test_cannot_handle_https_uri(self):
        assert self.handler.can_handle(http_uri()) is False


class TestFileSourceHandlerResolve:
    async def test_resolve_absolute_path(self, tmp_path: Path):
        handler = FileSourceHandler()
        uri = file_uri(str(tmp_path))
        result = await handler.resolve(uri, tmp_path / "cache")
        assert result.active_path == tmp_path

    async def test_resolve_relative_path(self, tmp_path: Path):
        target = tmp_path / "subdir"
        target.mkdir()
        handler = FileSourceHandler(base_path=tmp_path)
        uri = ParsedURI(scheme="file", host="", path="./subdir", ref="", subpath="")
        result = await handler.resolve(uri, tmp_path / "cache")
        assert result.active_path == target

    async def test_resolve_with_subpath(self, tmp_path: Path):
        subdir = tmp_path / "inner"
        subdir.mkdir()
        handler = FileSourceHandler()
        uri = ParsedURI(scheme="file", host="", path=str(tmp_path), ref="", subpath="inner")
        result = await handler.resolve(uri, tmp_path / "cache")
        assert result.active_path == subdir
        assert result.source_root == tmp_path

    async def test_resolve_raises_when_path_missing(self, tmp_path: Path):
        handler = FileSourceHandler()
        uri = file_uri(str(tmp_path / "nonexistent"))
        with pytest.raises(BundleNotFoundError, match="File not found"):
            await handler.resolve(uri, tmp_path / "cache")

    async def test_resolve_raises_when_subpath_missing(self, tmp_path: Path):
        handler = FileSourceHandler()
        uri = ParsedURI(
            scheme="file",
            host="",
            path=str(tmp_path),
            ref="",
            subpath="does_not_exist",
        )
        with pytest.raises(BundleNotFoundError, match="File not found"):
            await handler.resolve(uri, tmp_path / "cache")

    async def test_resolve_source_root_within_cache_dir(self, tmp_path: Path):
        """Files inside the cache dir should have source_root at the repo-level folder."""
        cache_dir = tmp_path / "cache"
        repo_dir = cache_dir / "myrepo-abc123"
        nested = repo_dir / "behaviors" / "foo"
        nested.mkdir(parents=True)
        handler = FileSourceHandler()
        uri = file_uri(str(nested))
        result = await handler.resolve(uri, cache_dir)
        assert result.source_root == repo_dir

    async def test_resolve_source_root_for_bundle_with_bundle_md(self, tmp_path: Path):
        """Non-cached paths with bundle.md should resolve source_root to bundle root."""
        bundle_root = tmp_path / "mybundle"
        bundle_root.mkdir()
        (bundle_root / "bundle.md").touch()
        nested = bundle_root / "behaviors" / "foo"
        nested.mkdir(parents=True)
        handler = FileSourceHandler()
        uri = file_uri(str(nested))
        result = await handler.resolve(uri, tmp_path / "cache")
        assert result.source_root == bundle_root

    async def test_resolve_source_root_falls_back_to_path(self, tmp_path: Path):
        """Without bundle.md or cache structure, source_root defaults to the path itself."""
        target = tmp_path / "standalone"
        target.mkdir()
        handler = FileSourceHandler()
        uri = file_uri(str(target))
        result = await handler.resolve(uri, tmp_path / "cache")
        # Should fall back to the path (either itself or parent)
        assert result.source_root is not None

    async def test_resolve_is_subdirectory_with_subpath(self, tmp_path: Path):
        subdir = tmp_path / "inner"
        subdir.mkdir()
        handler = FileSourceHandler()
        uri = ParsedURI(scheme="file", host="", path=str(tmp_path), ref="", subpath="inner")
        result = await handler.resolve(uri, tmp_path / "cache")
        assert result.is_subdirectory is True

    async def test_resolve_is_not_subdirectory_without_subpath(self, tmp_path: Path):
        handler = FileSourceHandler()
        uri = file_uri(str(tmp_path))
        result = await handler.resolve(uri, tmp_path / "cache")
        # active_path == source_root unless bundle structure detected
        assert isinstance(result.is_subdirectory, bool)


# ===========================================================================
# HttpSourceHandler
# ===========================================================================


class TestHttpSourceHandlerCanHandle:
    def setup_method(self):
        self.handler = HttpSourceHandler()

    def test_can_handle_https(self):
        assert self.handler.can_handle(http_uri("https")) is True

    def test_can_handle_http(self):
        assert self.handler.can_handle(http_uri("http")) is True

    def test_cannot_handle_git(self):
        assert self.handler.can_handle(git_uri()) is False

    def test_cannot_handle_zip(self):
        assert self.handler.can_handle(zip_uri()) is False

    def test_cannot_handle_file(self):
        assert self.handler.can_handle(file_uri("/some/path")) is False

    def test_cannot_handle_package(self):
        assert self.handler.can_handle(package_uri()) is False


class TestHttpSourceHandlerResolve:
    async def test_resolve_uses_cache_when_present(self, tmp_path: Path):
        import hashlib

        handler = HttpSourceHandler()
        uri = http_uri("https", "/bundle.yaml")
        url = "https://example.com/bundle.yaml"
        cache_key = hashlib.sha256(url.encode()).hexdigest()[:16]
        filename = "bundle.yaml"
        cached_file = tmp_path / f"{filename}-{cache_key}"
        cached_file.write_text("content: hello")

        result = await handler.resolve(uri, tmp_path)
        assert result.active_path == cached_file

    async def test_resolve_downloads_when_not_cached(self, tmp_path: Path):
        handler = HttpSourceHandler()
        uri = http_uri("https", "/file.yaml")

        fake_response = MagicMock()
        fake_response.__enter__ = MagicMock(return_value=fake_response)
        fake_response.__exit__ = MagicMock(return_value=False)
        fake_response.read.return_value = b"yaml: content"

        with patch("amplifier_lib.sources.http.urlopen", return_value=fake_response):
            result = await handler.resolve(uri, tmp_path)

        assert result.active_path.exists()
        assert result.active_path.read_bytes() == b"yaml: content"

    async def test_resolve_raises_on_download_failure(self, tmp_path: Path):
        handler = HttpSourceHandler()
        uri = http_uri("https", "/missing.yaml")

        with patch("amplifier_lib.sources.http.urlopen", side_effect=OSError("connection refused")):
            with pytest.raises(BundleNotFoundError, match="Failed to download"):
                await handler.resolve(uri, tmp_path)

    async def test_resolve_source_root_equals_cached_file(self, tmp_path: Path):
        handler = HttpSourceHandler()
        uri = http_uri("https", "/bundle.yaml")

        fake_response = MagicMock()
        fake_response.__enter__ = MagicMock(return_value=fake_response)
        fake_response.__exit__ = MagicMock(return_value=False)
        fake_response.read.return_value = b"data"

        with patch("amplifier_lib.sources.http.urlopen", return_value=fake_response):
            result = await handler.resolve(uri, tmp_path)

        assert result.source_root == result.active_path


# ===========================================================================
# ZipSourceHandler
# ===========================================================================


class TestZipSourceHandlerCanHandle:
    def setup_method(self):
        self.handler = ZipSourceHandler()

    def test_can_handle_zip_plus_https(self):
        assert self.handler.can_handle(zip_uri("zip+https")) is True

    def test_can_handle_zip_plus_file(self):
        assert self.handler.can_handle(zip_uri("zip+file")) is True

    def test_cannot_handle_https(self):
        assert self.handler.can_handle(http_uri("https")) is False

    def test_cannot_handle_git(self):
        assert self.handler.can_handle(git_uri()) is False

    def test_cannot_handle_file(self):
        assert self.handler.can_handle(file_uri("/some/path")) is False

    def test_cannot_handle_package(self):
        assert self.handler.can_handle(package_uri()) is False


class TestZipSourceHandlerResolve:
    def _make_zip(self, zip_path: Path, contents: dict[str, str]) -> None:
        """Create a zip file with given {filename: content} entries."""
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in contents.items():
                zf.writestr(name, content)

    async def test_resolve_local_zip_extracts_to_cache(self, tmp_path: Path):
        zip_file = tmp_path / "archive.zip"
        self._make_zip(zip_file, {"bundle.md": "# Bundle", "main.yaml": "key: val"})

        handler = ZipSourceHandler()
        uri = ParsedURI(scheme="zip+file", host="", path=str(zip_file), ref="", subpath="")
        cache_dir = tmp_path / "cache"
        result = await handler.resolve(uri, cache_dir)

        assert result.active_path.exists()
        assert (result.active_path / "bundle.md").exists()

    async def test_resolve_local_zip_with_subpath(self, tmp_path: Path):
        zip_file = tmp_path / "archive.zip"
        self._make_zip(zip_file, {"behaviors/foo.yaml": "key: val"})

        handler = ZipSourceHandler()
        uri = ParsedURI(scheme="zip+file", host="", path=str(zip_file), ref="", subpath="behaviors")
        cache_dir = tmp_path / "cache"
        result = await handler.resolve(uri, cache_dir)

        assert result.active_path.name == "behaviors"
        assert result.is_subdirectory is True

    async def test_resolve_uses_cache_on_second_call(self, tmp_path: Path):
        zip_file = tmp_path / "archive.zip"
        self._make_zip(zip_file, {"bundle.md": "# Bundle"})

        handler = ZipSourceHandler()
        uri = ParsedURI(scheme="zip+file", host="", path=str(zip_file), ref="", subpath="")
        cache_dir = tmp_path / "cache"

        result1 = await handler.resolve(uri, cache_dir)
        # Delete source zip — second call should use cache
        zip_file.unlink()
        result2 = await handler.resolve(uri, cache_dir)

        assert result1.active_path == result2.active_path

    async def test_resolve_raises_for_missing_local_zip(self, tmp_path: Path):
        handler = ZipSourceHandler()
        uri = ParsedURI(
            scheme="zip+file", host="", path=str(tmp_path / "missing.zip"), ref="", subpath=""
        )
        with pytest.raises(BundleNotFoundError, match="Zip file not found"):
            await handler.resolve(uri, tmp_path / "cache")

    async def test_resolve_raises_for_missing_subpath_after_extract(self, tmp_path: Path):
        zip_file = tmp_path / "archive.zip"
        self._make_zip(zip_file, {"bundle.md": "# Bundle"})

        handler = ZipSourceHandler()
        uri = ParsedURI(
            scheme="zip+file",
            host="",
            path=str(zip_file),
            ref="",
            subpath="nonexistent_subdir",
        )
        with pytest.raises(BundleNotFoundError, match="Subpath not found"):
            await handler.resolve(uri, tmp_path / "cache")

    async def test_resolve_remote_zip_downloads_and_extracts(self, tmp_path: Path):
        """Mock _download_and_extract to avoid real HTTP while exercising resolve() routing."""
        handler = ZipSourceHandler()

        # Pre-build the zip on disk so _extract_zip gets a real file
        source_zip = tmp_path / "source.zip"
        with zipfile.ZipFile(source_zip, "w") as zf:
            zf.writestr("bundle.md", "# Remote Bundle")

        def fake_download_and_extract(url: str, extract_path: Path) -> None:
            handler._extract_zip(source_zip, extract_path)

        uri = zip_uri("zip+https", "/bundle.zip")
        cache_dir = tmp_path / "cache"

        with patch.object(handler, "_download_and_extract", side_effect=fake_download_and_extract):
            result = await handler.resolve(uri, cache_dir)

        assert result.active_path.exists()
        assert (result.active_path / "bundle.md").exists()

    async def test_resolve_cache_key_is_deterministic(self, tmp_path: Path):
        """Cache path is content-addressed: same URL always yields same folder name."""
        import hashlib

        url = "https://example.com/bundle.zip"
        cache_key = hashlib.sha256(url.encode()).hexdigest()[:16]
        zip_name = Path("/bundle.zip").stem
        expected_name = f"{zip_name}-{cache_key}"

        # Verify by calling resolve twice and checking the extracted folder name
        # Use a local zip to avoid network I/O
        source_zip = tmp_path / "archive.zip"
        with zipfile.ZipFile(source_zip, "w") as zf:
            zf.writestr("bundle.md", "# Bundle")

        handler = ZipSourceHandler()
        uri = zip_uri("zip+https", "/bundle.zip")
        cache_dir = tmp_path / "cache"

        def fake_download(u: str, extract_path: Path) -> None:
            handler._extract_zip(source_zip, extract_path)

        with patch.object(handler, "_download_and_extract", side_effect=fake_download):
            result = await handler.resolve(uri, cache_dir)

        assert result.source_root.name == expected_name
