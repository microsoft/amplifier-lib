"""Protocol for @mention resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class MentionResolverProtocol(Protocol):
    """Protocol for resolving @mentions to file paths.

    BaseMentionResolver provides a minimal implementation.
    Apps extend with additional shortcuts like @user:, @project:.
    """

    def resolve(self, mention: str) -> Path | None:
        """Resolve an @mention to a file path.

        Args:
            mention: The mention string (including @ prefix).

        Returns:
            Path to the resolved file, or None if not found.
        """
        ...
