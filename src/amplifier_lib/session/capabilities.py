"""Session capability helpers.

Provides a consistent interface for modules to read and write session-scoped
capabilities registered by the app layer during session creation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Capability name constant — matches the key used by the app layer.
WORKING_DIR_CAPABILITY = "session.working_dir"


def get_working_dir(coordinator: Any, fallback: Path | str | None = None) -> Path:
    """Return the session's working directory from the coordinator capability.

    This is the canonical way for modules to determine where file operations
    should be rooted.  The app layer registers the working directory at
    session creation; modules should call this rather than ``Path.cwd()``
    so that the path follows dynamic ``cd`` operations during a session.

    Args:
        coordinator: The session coordinator (must have a ``get_capability``
            method).
        fallback: Path to use when the capability is not set.  Defaults to
            ``Path.cwd()`` when *fallback* is ``None``.

    Returns:
        Working directory as an absolute ``Path``.

    Example:
        working_dir = get_working_dir(coordinator)
        file_path = working_dir / "relative/path/file.txt"

        # With an explicit fallback from config:
        working_dir = get_working_dir(coordinator, fallback=config.get("working_dir", "."))
    """
    value = coordinator.get_capability(WORKING_DIR_CAPABILITY)
    if value is not None:
        return Path(value)
    if fallback is not None:
        return Path(fallback)
    return Path.cwd()


def set_working_dir(coordinator: Any, path: Path | str) -> None:
    """Update the session's working directory capability.

    Re-registers the capability with a resolved absolute path, overwriting
    any previously stored value.  Call this whenever the assistant changes
    the active directory (e.g. after a ``cd`` tool call).

    Args:
        coordinator: The session coordinator (must have a
            ``register_capability`` method).
        path: New working directory.  Resolved to an absolute path before
            storing.
    """
    coordinator.register_capability(WORKING_DIR_CAPABILITY, str(Path(path).resolve()))
