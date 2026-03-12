"""Tests for amplifier_lib.session.capabilities — set_working_dir, get_working_dir."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from amplifier_lib.session.capabilities import (
    WORKING_DIR_CAPABILITY,
    get_working_dir,
    set_working_dir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mock_coordinator(capability_value=None) -> MagicMock:
    """Create a mock coordinator with configurable get_capability return value."""
    coord = MagicMock()
    coord.get_capability.return_value = capability_value
    return coord


# ---------------------------------------------------------------------------
# set_working_dir
# ---------------------------------------------------------------------------


class TestSetWorkingDir:
    def test_calls_register_capability_with_correct_key(self, tmp_path):
        coord = MagicMock()
        set_working_dir(coord, tmp_path)
        coord.register_capability.assert_called_once()
        call_args = coord.register_capability.call_args
        assert call_args[0][0] == WORKING_DIR_CAPABILITY

    def test_registers_resolved_absolute_path(self, tmp_path):
        coord = MagicMock()
        set_working_dir(coord, tmp_path)
        registered_value = coord.register_capability.call_args[0][1]
        assert Path(registered_value).is_absolute()

    def test_registers_string_not_path(self, tmp_path):
        coord = MagicMock()
        set_working_dir(coord, tmp_path)
        registered_value = coord.register_capability.call_args[0][1]
        assert isinstance(registered_value, str)

    def test_accepts_string_path(self, tmp_path):
        coord = MagicMock()
        set_working_dir(coord, str(tmp_path))
        coord.register_capability.assert_called_once()

    def test_accepts_path_object(self, tmp_path):
        coord = MagicMock()
        set_working_dir(coord, tmp_path)
        coord.register_capability.assert_called_once()

    def test_resolves_relative_path(self):
        coord = MagicMock()
        set_working_dir(coord, ".")
        registered_value = coord.register_capability.call_args[0][1]
        # "." resolved should be an absolute path
        assert Path(registered_value).is_absolute()

    def test_registered_path_matches_resolved_input(self, tmp_path):
        coord = MagicMock()
        sub = tmp_path / "subdir"
        sub.mkdir()
        set_working_dir(coord, sub)
        registered_value = coord.register_capability.call_args[0][1]
        assert registered_value == str(sub.resolve())

    def test_capability_name_is_session_working_dir(self):
        assert WORKING_DIR_CAPABILITY == "session.working_dir"

    def test_overwrite_calls_register_again(self, tmp_path):
        coord = MagicMock()
        set_working_dir(coord, tmp_path)
        set_working_dir(coord, tmp_path / "other")
        assert coord.register_capability.call_count == 2


# ---------------------------------------------------------------------------
# get_working_dir
# ---------------------------------------------------------------------------


class TestGetWorkingDir:
    def test_returns_from_capability_when_set(self, tmp_path):
        coord = mock_coordinator(str(tmp_path))
        result = get_working_dir(coord)
        assert result == Path(str(tmp_path))

    def test_returns_path_object(self, tmp_path):
        coord = mock_coordinator(str(tmp_path))
        result = get_working_dir(coord)
        assert isinstance(result, Path)

    def test_fallback_to_provided_path_when_no_capability(self, tmp_path):
        coord = mock_coordinator(None)
        result = get_working_dir(coord, fallback=tmp_path)
        assert result == Path(tmp_path)

    def test_fallback_accepts_string(self, tmp_path):
        coord = mock_coordinator(None)
        result = get_working_dir(coord, fallback=str(tmp_path))
        assert result == Path(str(tmp_path))

    def test_fallback_to_cwd_when_no_capability_no_fallback(self):
        coord = mock_coordinator(None)
        result = get_working_dir(coord)
        assert result == Path.cwd()

    def test_capability_takes_precedence_over_fallback(self, tmp_path):
        cap_path = tmp_path / "from_capability"
        cap_path.mkdir()
        fallback_path = tmp_path / "fallback"
        fallback_path.mkdir()

        coord = mock_coordinator(str(cap_path))
        result = get_working_dir(coord, fallback=fallback_path)
        assert result == Path(str(cap_path))

    def test_get_capability_called_with_correct_key(self, tmp_path):
        coord = mock_coordinator(str(tmp_path))
        get_working_dir(coord)
        coord.get_capability.assert_called_once_with(WORKING_DIR_CAPABILITY)

    def test_none_fallback_uses_cwd(self):
        coord = mock_coordinator(None)
        result = get_working_dir(coord, fallback=None)
        assert result == Path.cwd()

    def test_path_from_capability_is_preserved_as_is(self, tmp_path):
        """The path from capability is used directly without resolving."""
        coord = mock_coordinator("/some/custom/path")
        result = get_working_dir(coord)
        assert str(result) == "/some/custom/path"

    def test_roundtrip_set_then_get(self, tmp_path):
        """set_working_dir then get_working_dir should return the same path."""
        store: dict = {}

        def register_capability(key: str, value: str) -> None:
            store[key] = value

        def get_capability(key: str):
            return store.get(key)

        coord = MagicMock()
        coord.register_capability.side_effect = register_capability
        coord.get_capability.side_effect = get_capability

        target = tmp_path / "work"
        target.mkdir()

        set_working_dir(coord, target)
        result = get_working_dir(coord)
        assert result == target.resolve()
