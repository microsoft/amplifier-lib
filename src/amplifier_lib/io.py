"""File I/O utilities for amplifier-lib.

Provides atomic writes and backup-before-write for robust file operations.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _write_atomic(
    path: Path,
    content: str | bytes,
    *,
    mode: str = "w",
    encoding: str | None = "utf-8",
) -> None:
    """Write file atomically using a temp file + rename pattern.

    Ensures the file is never partially written — either the old content
    exists or the new content exists, never a mix.

    Args:
        path: Target file path.
        content: Content to write.
        mode: Write mode ("w" for text, "wb" for binary).
        encoding: Encoding for text mode (ignored for binary).

    Raises:
        OSError: If the write or rename fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode,
            encoding=encoding if "b" not in mode else None,
            dir=path.parent,
            prefix=f".{path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)
            tmp_file.write(content)
            tmp_file.flush()

        # File is closed — safe to rename (works cross-platform including Windows)
        temp_path.replace(path)

    except Exception as e:
        if temp_path:
            with contextlib.suppress(Exception):
                temp_path.unlink()
        raise OSError(f"Failed to write atomically to {path}: {e}") from e


def write_with_backup(
    path: Path,
    content: str | bytes,
    *,
    backup_suffix: str = ".backup",
    mode: str = "w",
    encoding: str | None = "utf-8",
) -> None:
    """Write file with a backup of the previous version.

    Creates a backup before writing, enabling recovery if the new write
    is corrupted or interrupted.

    Args:
        path: Target file path.
        content: Content to write.
        backup_suffix: Suffix appended to path for the backup file (default: ".backup").
        mode: Write mode ("w" for text, "wb" for binary).
        encoding: Encoding for text mode.

    Example:
        # Creates config.json.backup before writing config.json
        write_with_backup(Path("config.json"), json.dumps(config))
    """
    backup_path = path.with_suffix(path.suffix + backup_suffix)

    # Best-effort backup — don't let a backup failure block the write
    if path.exists():
        with contextlib.suppress(Exception):
            shutil.copy2(path, backup_path)

    _write_atomic(path, content, mode=mode, encoding=encoding)
