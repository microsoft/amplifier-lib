"""Tests for amplifier_lib.modules.activator.ModuleActivator."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from amplifier_lib.modules.activator import ModuleActivator
from amplifier_lib.sources.uri import ResolvedSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_resolved(path: Path) -> ResolvedSource:
    return ResolvedSource(active_path=path, source_root=path)


def patch_resolver(activator: ModuleActivator, resolved: ResolvedSource) -> MagicMock:
    """Replace activator._resolver.resolve with an AsyncMock returning resolved."""
    activator._resolver.resolve = AsyncMock(return_value=resolved)
    return activator._resolver


def stub_subprocess_success() -> MagicMock:
    result = MagicMock()
    result.returncode = 0
    result.stdout = ""
    result.stderr = ""
    return result


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestModuleActivatorInit:
    def test_creates_resolver(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path)
        assert activator._resolver is not None

    def test_creates_install_state(self, tmp_path: Path):
        from amplifier_lib.modules.install_state import InstallStateManager

        activator = ModuleActivator(cache_dir=tmp_path)
        assert isinstance(activator._install_state, InstallStateManager)

    def test_default_cache_dir_uses_amplifier_home(self, tmp_path: Path):
        with patch("amplifier_lib.modules.activator.get_amplifier_home", return_value=tmp_path):
            activator = ModuleActivator()
        assert activator.cache_dir == tmp_path / "cache"

    def test_custom_cache_dir(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path / "custom")
        assert activator.cache_dir == tmp_path / "custom"

    def test_install_deps_default_true(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path)
        assert activator.install_deps is True

    def test_install_deps_can_be_disabled(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        assert activator.install_deps is False

    def test_activated_set_is_empty_initially(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path)
        assert len(activator._activated) == 0

    def test_bundle_package_paths_is_empty_initially(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path)
        assert activator.bundle_package_paths == []


# ---------------------------------------------------------------------------
# bundle_package_paths property
# ---------------------------------------------------------------------------


class TestBundlePackagePaths:
    def test_returns_copy(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path)
        activator._bundle_package_paths.append("/some/path")
        result = activator.bundle_package_paths
        result.append("/extra")
        assert len(activator._bundle_package_paths) == 1

    def test_reflects_added_paths(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path)
        activator._bundle_package_paths = ["/a", "/b"]
        assert activator.bundle_package_paths == ["/a", "/b"]


# ---------------------------------------------------------------------------
# activate()
# ---------------------------------------------------------------------------


class TestActivate:
    async def test_activate_returns_module_path(self, tmp_path: Path):
        module_dir = tmp_path / "mymod"
        module_dir.mkdir()
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        patch_resolver(activator, make_resolved(module_dir))

        result = await activator.activate("mymod", "file://" + str(module_dir))
        assert result == module_dir

    async def test_activate_adds_to_sys_path(self, tmp_path: Path):
        module_dir = tmp_path / "mymod"
        module_dir.mkdir()
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        patch_resolver(activator, make_resolved(module_dir))

        await activator.activate("mymod", "file://" + str(module_dir))
        assert str(module_dir) in sys.path

        # Cleanup
        if str(module_dir) in sys.path:
            sys.path.remove(str(module_dir))

    async def test_activate_skips_sys_path_if_already_present(self, tmp_path: Path):
        module_dir = tmp_path / "mymod2"
        module_dir.mkdir()
        sys.path.insert(0, str(module_dir))
        initial_count = sys.path.count(str(module_dir))

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        patch_resolver(activator, make_resolved(module_dir))

        await activator.activate("mymod2", "file://" + str(module_dir))
        assert sys.path.count(str(module_dir)) == initial_count

        # Cleanup
        sys.path.remove(str(module_dir))

    async def test_activate_marks_cache_key(self, tmp_path: Path):
        module_dir = tmp_path / "mymod"
        module_dir.mkdir()
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        patch_resolver(activator, make_resolved(module_dir))

        uri = "file://" + str(module_dir)
        await activator.activate("mymod", uri)
        assert f"mymod:{uri}" in activator._activated

    async def test_activate_skips_resolver_on_second_call(self, tmp_path: Path):
        module_dir = tmp_path / "mymod"
        module_dir.mkdir()
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        resolver = patch_resolver(activator, make_resolved(module_dir))

        uri = "file://" + str(module_dir)
        await activator.activate("mymod", uri)
        await activator.activate("mymod", uri)

        # resolve should be called twice: once for the initial activation,
        # once for the cache-hit path (it still calls resolve to get the path)
        assert resolver.resolve.call_count == 2

    async def test_activate_calls_progress_callback(self, tmp_path: Path):
        module_dir = tmp_path / "mymod"
        module_dir.mkdir()
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        patch_resolver(activator, make_resolved(module_dir))

        calls = []
        await activator.activate(
            "mymod",
            "file://" + str(module_dir),
            progress_callback=lambda a, d: calls.append((a, d)),
        )

        assert ("activating", "mymod") in calls

    async def test_activate_installs_deps_when_enabled(self, tmp_path: Path):
        module_dir = tmp_path / "mymod"
        module_dir.mkdir()
        (module_dir / "pyproject.toml").write_text("[project]\nname='mymod'\n")

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=True)
        patch_resolver(activator, make_resolved(module_dir))

        with patch("subprocess.run", return_value=stub_subprocess_success()) as mock_run:
            with patch("importlib.invalidate_caches"):
                with patch("site.getsitepackages", return_value=[]):
                    await activator.activate("mymod", "file://" + str(module_dir))

        mock_run.assert_called_once()
        assert "uv" in mock_run.call_args[0][0]

    async def test_activate_skips_install_when_disabled(self, tmp_path: Path):
        module_dir = tmp_path / "mymod"
        module_dir.mkdir()
        (module_dir / "pyproject.toml").write_text("[project]\nname='mymod'\n")

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        patch_resolver(activator, make_resolved(module_dir))

        with patch("subprocess.run") as mock_run:
            await activator.activate("mymod", "file://" + str(module_dir))

        mock_run.assert_not_called()

    async def test_activate_skips_install_when_already_installed(self, tmp_path: Path):
        module_dir = tmp_path / "mymod"
        module_dir.mkdir()
        (module_dir / "pyproject.toml").write_text("[project]\nname='mymod'\n")

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=True)
        patch_resolver(activator, make_resolved(module_dir))

        # Pre-mark as installed
        activator._install_state.mark_installed(module_dir)

        with patch("subprocess.run") as mock_run:
            await activator.activate("mymod", "file://" + str(module_dir))

        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# activate_all()
# ---------------------------------------------------------------------------


class TestActivateAll:
    async def test_activate_all_returns_all_modules(self, tmp_path: Path):
        mod_a = tmp_path / "mod_a"
        mod_b = tmp_path / "mod_b"
        mod_a.mkdir()
        mod_b.mkdir()

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)

        async def resolve_side_effect(uri: str) -> ResolvedSource:
            if "mod_a" in uri:
                return make_resolved(mod_a)
            return make_resolved(mod_b)

        activator._resolver.resolve = AsyncMock(side_effect=resolve_side_effect)

        modules = [
            {"module": "mod_a", "source": f"file://{mod_a}"},
            {"module": "mod_b", "source": f"file://{mod_b}"},
        ]
        result = await activator.activate_all(modules)

        assert "mod_a" in result
        assert "mod_b" in result
        assert result["mod_a"] == mod_a
        assert result["mod_b"] == mod_b

    async def test_activate_all_skips_missing_module_key(self, tmp_path: Path):
        mod = tmp_path / "mod"
        mod.mkdir()
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        activator._resolver.resolve = AsyncMock(return_value=make_resolved(mod))

        modules = [
            {"source": f"file://{mod}"},  # no 'module' key
            {"module": "mod", "source": f"file://{mod}"},
        ]
        result = await activator.activate_all(modules)
        assert "mod" in result
        assert len(result) == 1

    async def test_activate_all_skips_missing_source_key(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        activator._resolver.resolve = AsyncMock(side_effect=AssertionError("should not be called"))

        modules = [{"module": "mod"}]  # no 'source' key
        result = await activator.activate_all(modules)
        assert result == {}

    async def test_activate_all_returns_empty_for_empty_list(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        result = await activator.activate_all([])
        assert result == {}

    async def test_activate_all_logs_errors_and_continues(self, tmp_path: Path):
        mod_a = tmp_path / "mod_a"
        mod_b = tmp_path / "mod_b"
        mod_a.mkdir()
        mod_b.mkdir()

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)

        from amplifier_lib.exceptions import BundleNotFoundError

        async def resolve_side_effect(uri: str) -> ResolvedSource:
            if "mod_a" in uri:
                raise BundleNotFoundError("not found")
            return make_resolved(mod_b)

        activator._resolver.resolve = AsyncMock(side_effect=resolve_side_effect)

        modules = [
            {"module": "mod_a", "source": f"file://{mod_a}"},
            {"module": "mod_b", "source": f"file://{mod_b}"},
        ]
        result = await activator.activate_all(modules)
        assert "mod_a" not in result
        assert "mod_b" in result

    async def test_activate_all_passes_progress_callback(self, tmp_path: Path):
        mod = tmp_path / "mod"
        mod.mkdir()
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        activator._resolver.resolve = AsyncMock(return_value=make_resolved(mod))

        calls = []
        modules = [{"module": "mod", "source": f"file://{mod}"}]
        await activator.activate_all(modules, progress_callback=lambda a, d: calls.append((a, d)))
        assert any(a == "activating" for a, _ in calls)


# ---------------------------------------------------------------------------
# activate_bundle_package()
# ---------------------------------------------------------------------------


class TestActivateBundlePackage:
    async def test_skips_when_bundle_path_is_none(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        with patch("subprocess.run") as mock_run:
            await activator.activate_bundle_package(None)  # type: ignore[arg-type]
        mock_run.assert_not_called()

    async def test_skips_when_bundle_path_does_not_exist(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        with patch("subprocess.run") as mock_run:
            await activator.activate_bundle_package(tmp_path / "nonexistent")
        mock_run.assert_not_called()

    async def test_skips_when_no_pyproject_toml(self, tmp_path: Path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        with patch("subprocess.run") as mock_run:
            await activator.activate_bundle_package(bundle)
        mock_run.assert_not_called()

    async def test_skips_when_pyproject_has_no_project_section(self, tmp_path: Path):
        """Tool-config-only pyproject (no [project] or [build-system]) → skip."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=False)
        with patch("subprocess.run") as mock_run:
            await activator.activate_bundle_package(bundle)
        mock_run.assert_not_called()

    async def test_skips_when_package_already_importable(self, tmp_path: Path):
        """Package already in sys.modules / importable → skip editable install."""
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "pyproject.toml").write_text(
            "[project]\nname = 'sys'\n[build-system]\nrequires = ['hatchling']\n"
        )

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=True)

        # 'sys' is always importable
        with patch("subprocess.run") as mock_run:
            await activator.activate_bundle_package(bundle)
        mock_run.assert_not_called()

    async def test_installs_when_project_section_present_and_not_importable(self, tmp_path: Path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "pyproject.toml").write_text(
            "[project]\nname = 'totally-nonexistent-pkg-xyz123'\n"
            "[build-system]\nrequires = ['hatchling']\n"
        )

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=True)

        with patch("subprocess.run", return_value=stub_subprocess_success()) as mock_run:
            with patch("importlib.invalidate_caches"):
                with patch("site.getsitepackages", return_value=[]):
                    await activator.activate_bundle_package(bundle)

        mock_run.assert_called_once()

    async def test_adds_src_dir_to_sys_path(self, tmp_path: Path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        src = bundle / "src"
        src.mkdir()
        (bundle / "pyproject.toml").write_text(
            "[project]\nname = 'totally-nonexistent-pkg-src-test'\n"
            "[build-system]\nrequires = ['hatchling']\n"
        )

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=True)

        with patch("subprocess.run", return_value=stub_subprocess_success()):
            with patch("importlib.invalidate_caches"):
                with patch("site.getsitepackages", return_value=[]):
                    await activator.activate_bundle_package(bundle)

        assert str(src) in activator.bundle_package_paths

        # Cleanup
        if str(src) in sys.path:
            sys.path.remove(str(src))

    async def test_adds_lib_dir_to_sys_path(self, tmp_path: Path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        lib = bundle / "lib"
        lib.mkdir()
        (bundle / "pyproject.toml").write_text(
            "[project]\nname = 'totally-nonexistent-pkg-lib-test'\n"
            "[build-system]\nrequires = ['hatchling']\n"
        )

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=True)

        with patch("subprocess.run", return_value=stub_subprocess_success()):
            with patch("importlib.invalidate_caches"):
                with patch("site.getsitepackages", return_value=[]):
                    await activator.activate_bundle_package(bundle)

        assert str(lib) in activator.bundle_package_paths

        # Cleanup
        if str(lib) in sys.path:
            sys.path.remove(str(lib))

    async def test_calls_progress_callback(self, tmp_path: Path):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "pyproject.toml").write_text(
            "[project]\nname = 'totally-nonexistent-callback-pkg'\n"
            "[build-system]\nrequires = ['hatchling']\n"
        )

        activator = ModuleActivator(cache_dir=tmp_path, install_deps=True)
        calls = []

        with patch("subprocess.run", return_value=stub_subprocess_success()):
            with patch("importlib.invalidate_caches"):
                with patch("site.getsitepackages", return_value=[]):
                    await activator.activate_bundle_package(
                        bundle, progress_callback=lambda a, d: calls.append((a, d))
                    )

        assert any(a == "installing_package" for a, _ in calls)


# ---------------------------------------------------------------------------
# finalize()
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_finalize_saves_install_state(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path)
        module_dir = tmp_path / "mymod"
        module_dir.mkdir()
        activator._install_state.mark_installed(module_dir)

        activator.finalize()

        state_file = tmp_path / "install-state.json"
        assert state_file.exists()

    def test_finalize_is_idempotent(self, tmp_path: Path):
        activator = ModuleActivator(cache_dir=tmp_path)
        activator.finalize()
        activator.finalize()  # Should not raise


# ---------------------------------------------------------------------------
# _build_git_dep_overrides() (static method)
# ---------------------------------------------------------------------------


class TestBuildGitDepOverrides:
    def test_returns_empty_for_nonexistent_pyproject(self, tmp_path: Path):
        result = ModuleActivator._build_git_dep_overrides(
            tmp_path / "nonexistent" / "pyproject.toml"
        )
        assert result == []

    def test_returns_empty_when_no_git_deps(self, tmp_path: Path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname='pkg'\ndependencies = ['requests>=2.0']\n")
        result = ModuleActivator._build_git_dep_overrides(pyproject)
        assert result == []

    def test_returns_override_for_installed_git_dep(self, tmp_path: Path):
        pyproject = tmp_path / "pyproject.toml"
        # Use 'pyyaml' as a known-installed package (direct dep of amplifier-lib)
        pyproject.write_text(
            "[project]\nname='pkg'\n"
            "dependencies = ['pyyaml @ git+https://github.com/yaml/pyyaml']\n"
        )
        result = ModuleActivator._build_git_dep_overrides(pyproject)
        assert len(result) == 1
        assert result[0].startswith("PyYAML==") or result[0].startswith("pyyaml==")

    def test_skips_git_deps_not_installed(self, tmp_path: Path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            "[project]\nname='pkg'\n"
            "dependencies = ['totally-nonexistent-xyz @ git+https://github.com/x/y']\n"
        )
        result = ModuleActivator._build_git_dep_overrides(pyproject)
        assert result == []
