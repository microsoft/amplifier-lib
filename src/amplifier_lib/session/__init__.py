"""Session utilities for amplifier-lib.

Public API re-exported from submodules:

- **slice**        — turn-boundary slicing and orphaned-tool helpers
- **fork**         — in-memory session forking
- **capabilities** — session-scoped capability accessors
"""

from amplifier_lib.session.capabilities import (
    WORKING_DIR_CAPABILITY,
    get_working_dir,
    set_working_dir,
)
from amplifier_lib.session.fork import (
    ForkResult,
    fork_session,
    fork_session_in_memory,
    get_fork_preview,
    get_session_lineage,
    list_session_forks,
)
from amplifier_lib.session.slice import (
    add_synthetic_tool_results,
    count_turns,
    find_orphaned_tool_calls,
    get_turn_boundaries,
    get_turn_summary,
)

__all__ = [
    # slice
    "get_turn_boundaries",
    "count_turns",
    "get_turn_summary",
    "find_orphaned_tool_calls",
    "add_synthetic_tool_results",
    # fork
    "ForkResult",
    "fork_session_in_memory",
    "fork_session",
    "get_fork_preview",
    "get_session_lineage",
    "list_session_forks",
    # capabilities
    "WORKING_DIR_CAPABILITY",
    "get_working_dir",
    "set_working_dir",
]
