"""Installation state tracking for fast module startup.

Tracks fingerprints of installed modules to skip redundant `uv pip install`
calls. When a module's pyproject.toml/requirements.txt hasn't changed, the
install step is skipped entirely, significantly speeding up startup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class InstallStateManager:
    """Tracks module installation state for fast startup.

    Stores fingerprints (dependency file hashes) for installed modules.
    If a fingerprint matches the stored value, `uv pip install` is skipped.

    Self-healing: corrupted JSON or schema mismatch creates fresh state.
    Invalidates all entries if the Python executable or its mtime changes.
    """

    VERSION = 1
    FILENAME = "install-state.json"

    def __init__(self, cache_dir: Path) -> None:
        """Initialize the install state manager.

        Args:
            cache_dir: Directory for storing the state file (e.g., ~/.amplifier).
        """
        self._state_file = cache_dir / self.FILENAME
        self._dirty = False
        self._state = self._load()

    def _get_python_mtime(self) -> int | None:
        """Get Python executable mtime as integer seconds.

        Uses os.lstat() (no symlink following) so that `uv tool install --force`
        — which recreates the venv symlink — is detected even when the underlying
        CPython binary is unchanged.

        Returns:
            mtime as integer seconds, or None if unavailable.
        """
        try:
            return int(os.lstat(sys.executable).st_mtime)
        except OSError:
            return None

    def _load(self) -> dict:
        """Load state from disk, returning fresh state if loading fails."""
        if not self._state_file.exists():
            return self._fresh_state()

        try:
            with open(self._state_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Creating fresh install state (load failed: {e})")
            return self._fresh_state()

        if data.get("version") != self.VERSION:
            logger.debug(
                f"Creating fresh install state (version {data.get('version')} != {self.VERSION})"
            )
            return self._fresh_state()

        if data.get("python") != sys.executable:
            logger.debug(
                f"Clearing install state (Python changed: {data.get('python')} -> {sys.executable})"
            )
            return self._fresh_state()

        current_mtime = self._get_python_mtime()
        if data.get("python_mtime") != current_mtime:
            stored_mtime = data.get("python_mtime")
            logger.debug(
                f"Clearing install state (Python mtime changed: {stored_mtime} -> {current_mtime})"
            )
            return self._fresh_state()

        return data

    def _fresh_state(self) -> dict:
        """Create a fresh empty state dict."""
        self._dirty = True
        return {
            "version": self.VERSION,
            "python": sys.executable,
            "python_mtime": self._get_python_mtime(),
            "modules": {},
        }

    def _compute_fingerprint(self, module_path: Path) -> str:
        """Compute a fingerprint for a module's dependency files.

        Hashes pyproject.toml and requirements.txt if present.
        Returns "none" when no dependency files exist.
        """
        hasher = hashlib.sha256()
        files_hashed = 0

        for filename in ("pyproject.toml", "requirements.txt"):
            filepath = module_path / filename
            if filepath.exists():
                try:
                    content = filepath.read_bytes()
                    hasher.update(filename.encode())
                    hasher.update(content)
                    files_hashed += 1
                except OSError:
                    pass

        if files_hashed == 0:
            return "none"

        return f"sha256:{hasher.hexdigest()}"

    def is_installed(self, module_path: Path) -> bool:
        """Check if a module is already installed with a matching fingerprint.

        Args:
            module_path: Path to the module directory.

        Returns:
            True if the module is installed and its fingerprint matches.
        """
        path_key = str(module_path.resolve())
        entry = self._state["modules"].get(path_key)

        if not entry:
            return False

        current = self._compute_fingerprint(module_path)
        stored = entry.get("pyproject_hash")

        if current != stored:
            logger.debug(f"Fingerprint mismatch for {module_path.name}: {stored} -> {current}")
            return False

        return True

    def mark_installed(self, module_path: Path) -> None:
        """Record that a module was successfully installed.

        Args:
            module_path: Path to the module directory.
        """
        path_key = str(module_path.resolve())
        self._state["modules"][path_key] = {
            "pyproject_hash": self._compute_fingerprint(module_path)
        }
        self._dirty = True

    def save(self) -> None:
        """Persist state to disk if changed.

        Uses atomic write (temp file + rename) to avoid corruption.
        """
        if not self._dirty:
            return

        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            fd, temp_path = tempfile.mkstemp(
                dir=self._state_file.parent,
                prefix=".install-state-",
                suffix=".tmp",
            )
            try:
                with open(fd, "w") as f:
                    json.dump(self._state, f, indent=2)
                Path(temp_path).rename(self._state_file)
                self._dirty = False
            except Exception:
                Path(temp_path).unlink(missing_ok=True)
                raise
        except OSError as e:
            logger.warning(f"Failed to save install state: {e}")

    def invalidate(self, module_path: Path | None = None) -> None:
        """Clear install state for one module or all modules.

        Args:
            module_path: Path to a specific module to invalidate,
                         or None to invalidate all modules.
        """
        if module_path is None:
            if self._state["modules"]:
                self._state["modules"] = {}
                self._dirty = True
                logger.debug("Invalidated all module install states")
        else:
            path_key = str(module_path.resolve())
            if path_key in self._state["modules"]:
                del self._state["modules"][path_key]
                self._dirty = True
                logger.debug(f"Invalidated install state for {module_path.name}")
