"""Path utilities for amplifier-lib."""

from __future__ import annotations

import os
from pathlib import Path


def get_amplifier_home() -> Path:
    """Get the Amplifier home directory.

    Resolves in order:
    1. AMPLIFIER_HOME environment variable
    2. ~/.amplifier (default)

    This is the single source of truth for all Amplifier path resolution.
    All components should use this for determining cache and data directories.

    Returns:
        Resolved path to Amplifier home directory.
    """
    env_home = os.environ.get("AMPLIFIER_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".amplifier").resolve()
