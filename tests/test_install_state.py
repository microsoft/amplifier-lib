"""Tests for amplifier_lib.modules.install_state.InstallStateManager."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from amplifier_lib.modules.install_state import InstallStateManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def state(tmp_path: Path) -> InstallStateManager:
    """Fresh InstallStateManager backed by a temp directory."""
    return InstallStateManager(tmp_path)


@pytest.fixture()
def module_dir(tmp_path: Path) -> Path:
    """A temp directory that simulates a module path (no dependency files initially)."""
    d = tmp_path / "my_module"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# is_installed() — basic behaviour
# ---------------------------------------------------------------------------


class TestIsInstalled:
    def test_returns_false_for_unknown_module(self, state: InstallStateManager, module_dir: Path):
        assert state.is_installed(module_dir) is False

    def test_returns_false_for_module_with_no_pyproject(
        self, state: InstallStateManager, module_dir: Path
    ):
        # Not yet marked — should be False
        assert state.is_installed(module_dir) is False

    def test_returns_true_after_mark_installed(
        self, state: InstallStateManager, module_dir: Path
    ):
        state.mark_installed(module_dir)
        assert state.is_installed(module_dir) is True

    def test_returns_false_after_pyproject_changes(
        self, state: InstallStateManager, module_dir: Path
    ):
        pyproject = module_dir / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'v1'\n")
        state.mark_installed(module_dir)

        # Change the file
        pyproject.write_text("[project]\nname = 'v2'\n")

        assert state.is_installed(module_dir) is False

    def test_fingerprint_covers_pyproject_and_requirements(
        self, state: InstallStateManager, module_dir: Path
    ):
        (module_dir / "pyproject.toml").write_text("[project]\nname = 'pkg'\n")
        (module_dir / "requirements.txt").write_text("requests>=2.0\n")
        state.mark_installed(module_dir)

        # Alter requirements.txt
        (module_dir / "requirements.txt").write_text("requests>=3.0\n")
        assert state.is_installed(module_dir) is False

    def test_returns_true_when_no_dep_files_and_marked(
        self, state: InstallStateManager, module_dir: Path
    ):
        # No pyproject or requirements — fingerprint is "none"
        state.mark_installed(module_dir)
        assert state.is_installed(module_dir) is True


# ---------------------------------------------------------------------------
# mark_installed() round-trip
# ---------------------------------------------------------------------------


class TestMarkInstalled:
    def test_marks_with_correct_fingerprint_key(
        self, state: InstallStateManager, module_dir: Path
    ):
        (module_dir / "pyproject.toml").write_text("[project]\nname = 'pkg'\n")
        state.mark_installed(module_dir)

        path_key = str(module_dir.resolve())
        entry = state._state["modules"][path_key]
        assert "pyproject_hash" in entry

    def test_mark_sets_dirty_flag(self, state: InstallStateManager, module_dir: Path):
        state._dirty = False
        state.mark_installed(module_dir)
        assert state._dirty is True

    def test_fingerprint_is_sha256_prefix(
        self, state: InstallStateManager, module_dir: Path
    ):
        (module_dir / "pyproject.toml").write_text("[project]\nname = 'pkg'\n")
        state.mark_installed(module_dir)

        path_key = str(module_dir.resolve())
        h = state._state["modules"][path_key]["pyproject_hash"]
        assert h.startswith("sha256:")

    def test_fingerprint_is_none_when_no_dep_files(
        self, state: InstallStateManager, module_dir: Path
    ):
        state.mark_installed(module_dir)
        path_key = str(module_dir.resolve())
        h = state._state["modules"][path_key]["pyproject_hash"]
        assert h == "none"


# ---------------------------------------------------------------------------
# save() + reload persistence
# ---------------------------------------------------------------------------


class TestSavePersistence:
    def test_save_writes_json_file(self, tmp_path: Path, module_dir: Path):
        state = InstallStateManager(tmp_path)
        state.mark_installed(module_dir)
        state.save()

        state_file = tmp_path / InstallStateManager.FILENAME
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["version"] == InstallStateManager.VERSION

    def test_save_is_noop_when_not_dirty(self, tmp_path: Path):
        state = InstallStateManager(tmp_path)
        state._dirty = False
        state.save()
        state_file = tmp_path / InstallStateManager.FILENAME
        assert not state_file.exists()

    def test_reload_restores_installed_modules(self, tmp_path: Path, module_dir: Path):
        state = InstallStateManager(tmp_path)
        (module_dir / "pyproject.toml").write_text("[project]\nname = 'pkg'\n")
        state.mark_installed(module_dir)
        state.save()

        # Fresh instance from same dir
        state2 = InstallStateManager(tmp_path)
        assert state2.is_installed(module_dir) is True

    def test_reload_detects_pyproject_change_after_save(self, tmp_path: Path, module_dir: Path):
        state = InstallStateManager(tmp_path)
        pyproject = module_dir / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'v1'\n")
        state.mark_installed(module_dir)
        state.save()

        pyproject.write_text("[project]\nname = 'v2'\n")

        state2 = InstallStateManager(tmp_path)
        assert state2.is_installed(module_dir) is False

    def test_corrupted_json_yields_fresh_state(self, tmp_path: Path):
        state_file = tmp_path / InstallStateManager.FILENAME
        state_file.write_text("NOT VALID JSON {{{")

        state = InstallStateManager(tmp_path)
        assert state._state["modules"] == {}

    def test_wrong_version_yields_fresh_state(self, tmp_path: Path):
        state_file = tmp_path / InstallStateManager.FILENAME
        state_file.write_text(json.dumps({"version": 999, "modules": {}}))

        state = InstallStateManager(tmp_path)
        assert state._state["modules"] == {}

    def test_save_clears_dirty_flag(self, tmp_path: Path, module_dir: Path):
        state = InstallStateManager(tmp_path)
        state.mark_installed(module_dir)
        assert state._dirty is True
        state.save()
        assert state._dirty is False

    def test_save_creates_parent_dirs(self, tmp_path: Path, module_dir: Path):
        deep_dir = tmp_path / "a" / "b" / "c"
        state = InstallStateManager(deep_dir)
        state.mark_installed(module_dir)
        state.save()
        assert (deep_dir / InstallStateManager.FILENAME).exists()


# ---------------------------------------------------------------------------
# invalidate()
# ---------------------------------------------------------------------------


class TestInvalidate:
    def test_invalidate_specific_module(self, state: InstallStateManager, module_dir: Path):
        state.mark_installed(module_dir)
        assert state.is_installed(module_dir) is True

        state.invalidate(module_dir)
        assert state.is_installed(module_dir) is False

    def test_invalidate_all_modules(self, tmp_path: Path):
        state = InstallStateManager(tmp_path)
        dirs = [tmp_path / f"mod{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()
            state.mark_installed(d)

        state.invalidate()  # None = all
        assert state._state["modules"] == {}

    def test_invalidate_specific_sets_dirty(self, state: InstallStateManager, module_dir: Path):
        state.mark_installed(module_dir)
        state._dirty = False
        state.invalidate(module_dir)
        assert state._dirty is True

    def test_invalidate_all_sets_dirty(self, tmp_path: Path, module_dir: Path):
        state = InstallStateManager(tmp_path)
        state.mark_installed(module_dir)
        state._dirty = False
        state.invalidate()
        assert state._dirty is True

    def test_invalidate_nonexistent_module_is_noop(
        self, state: InstallStateManager, module_dir: Path
    ):
        state._dirty = False
        state.invalidate(module_dir)  # Not yet marked
        assert state._dirty is False

    def test_invalidate_all_noop_when_empty(self, state: InstallStateManager):
        state._dirty = False
        state.invalidate()  # Nothing to clear
        assert state._dirty is False


# ---------------------------------------------------------------------------
# Python executable / mtime invalidation
# ---------------------------------------------------------------------------


class TestPythonExecutableInvalidation:
    def test_python_path_mismatch_triggers_fresh_state(self, tmp_path: Path, module_dir: Path):
        # Save state with the current executable
        state = InstallStateManager(tmp_path)
        state.mark_installed(module_dir)
        state.save()

        # Reload with a different Python path in the state file
        state_file = tmp_path / InstallStateManager.FILENAME
        data = json.loads(state_file.read_text())
        data["python"] = "/some/other/python"
        state_file.write_text(json.dumps(data))

        state2 = InstallStateManager(tmp_path)
        assert state2._state["modules"] == {}

    def test_python_mtime_mismatch_triggers_fresh_state(self, tmp_path: Path, module_dir: Path):
        state = InstallStateManager(tmp_path)
        state.mark_installed(module_dir)
        state.save()

        # Tamper with stored mtime
        state_file = tmp_path / InstallStateManager.FILENAME
        data = json.loads(state_file.read_text())
        data["python_mtime"] = 0  # Epoch — definitely wrong
        state_file.write_text(json.dumps(data))

        state2 = InstallStateManager(tmp_path)
        assert state2._state["modules"] == {}

    def test_fresh_state_records_current_python(self, tmp_path: Path):
        state = InstallStateManager(tmp_path)
        assert state._state["python"] == sys.executable


# ---------------------------------------------------------------------------
# _compute_fingerprint()
# ---------------------------------------------------------------------------


class TestComputeFingerprint:
    def test_fingerprint_is_none_without_dep_files(
        self, state: InstallStateManager, module_dir: Path
    ):
        fp = state._compute_fingerprint(module_dir)
        assert fp == "none"

    def test_fingerprint_changes_on_pyproject_edit(
        self, state: InstallStateManager, module_dir: Path
    ):
        pyproject = module_dir / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'v1'\n")
        fp1 = state._compute_fingerprint(module_dir)

        pyproject.write_text("[project]\nname = 'v2'\n")
        fp2 = state._compute_fingerprint(module_dir)

        assert fp1 != fp2

    def test_fingerprint_changes_on_requirements_edit(
        self, state: InstallStateManager, module_dir: Path
    ):
        req = module_dir / "requirements.txt"
        req.write_text("requests>=2\n")
        fp1 = state._compute_fingerprint(module_dir)

        req.write_text("requests>=3\n")
        fp2 = state._compute_fingerprint(module_dir)

        assert fp1 != fp2

    def test_fingerprint_is_stable(self, state: InstallStateManager, module_dir: Path):
        (module_dir / "pyproject.toml").write_text("[project]\nname = 'pkg'\n")
        fp1 = state._compute_fingerprint(module_dir)
        fp2 = state._compute_fingerprint(module_dir)
        assert fp1 == fp2
