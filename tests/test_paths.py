"""Tests for amplifier_lib.paths — get_amplifier_home()."""

from pathlib import Path

import pytest

from amplifier_lib.paths import get_amplifier_home


class TestGetAmplifierHome:
    def test_default_is_dot_amplifier_under_home(self, monkeypatch):
        """Without AMPLIFIER_HOME, resolves to ~/.amplifier."""
        monkeypatch.delenv("AMPLIFIER_HOME", raising=False)
        result = get_amplifier_home()
        expected = (Path.home() / ".amplifier").resolve()
        assert result == expected

    def test_returns_path_object(self, monkeypatch):
        monkeypatch.delenv("AMPLIFIER_HOME", raising=False)
        assert isinstance(get_amplifier_home(), Path)

    def test_default_is_absolute(self, monkeypatch):
        monkeypatch.delenv("AMPLIFIER_HOME", raising=False)
        assert get_amplifier_home().is_absolute()

    def test_env_var_overrides_default(self, monkeypatch, tmp_path):
        """When AMPLIFIER_HOME is set, it overrides ~/.amplifier."""
        monkeypatch.setenv("AMPLIFIER_HOME", str(tmp_path))
        result = get_amplifier_home()
        assert result == tmp_path.resolve()

    def test_env_var_result_is_absolute(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AMPLIFIER_HOME", str(tmp_path))
        assert get_amplifier_home().is_absolute()

    def test_env_var_is_resolved(self, monkeypatch, tmp_path):
        """The returned path should be fully resolved (no symlinks, .., etc.)."""
        monkeypatch.setenv("AMPLIFIER_HOME", str(tmp_path))
        result = get_amplifier_home()
        assert result == Path(str(tmp_path)).expanduser().resolve()

    def test_env_var_with_tilde_expansion(self, monkeypatch):
        """AMPLIFIER_HOME with ~ should be expanded."""
        monkeypatch.setenv("AMPLIFIER_HOME", "~/.my-amplifier")
        result = get_amplifier_home()
        expected = Path("~/.my-amplifier").expanduser().resolve()
        assert result == expected

    def test_env_var_with_relative_path(self, monkeypatch, tmp_path):
        """A relative AMPLIFIER_HOME should be resolved to absolute."""
        # Use a sub-directory name that exists
        sub = tmp_path / "custom_home"
        sub.mkdir()
        # We can't easily control cwd in a test, so use absolute path directly
        monkeypatch.setenv("AMPLIFIER_HOME", str(sub))
        result = get_amplifier_home()
        assert result.is_absolute()
        assert result == sub.resolve()

    def test_empty_env_var_falls_back_to_default(self, monkeypatch):
        """An empty AMPLIFIER_HOME string should fall back to ~/.amplifier."""
        monkeypatch.setenv("AMPLIFIER_HOME", "")
        result = get_amplifier_home()
        expected = (Path.home() / ".amplifier").resolve()
        assert result == expected

    def test_custom_path_does_not_need_to_exist(self, monkeypatch, tmp_path):
        """The function should not require the directory to exist."""
        non_existent = tmp_path / "does" / "not" / "exist"
        monkeypatch.setenv("AMPLIFIER_HOME", str(non_existent))
        # Should not raise
        result = get_amplifier_home()
        assert result.is_absolute()

    @pytest.mark.parametrize(
        "env_value",
        [
            "/tmp/amplifier-test",
            "/var/lib/amplifier",
        ],
    )
    def test_various_absolute_paths(self, monkeypatch, env_value):
        monkeypatch.setenv("AMPLIFIER_HOME", env_value)
        result = get_amplifier_home()
        assert result.is_absolute()
        assert str(result).endswith(Path(env_value).name)
