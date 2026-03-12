"""amplifier-lib: core primitives for building Amplifier applications."""

__version__ = "0.1.0"

# Exceptions
# Bundle system
from amplifier_lib.bundle import Bundle, BundleModuleResolver, PreparedBundle
from amplifier_lib.exceptions import (
    BundleDependencyError,
    BundleError,
    BundleLoadError,
    BundleNotFoundError,
    BundleValidationError,
)

# IO / Serialization
from amplifier_lib.io import write_with_backup

# Paths
from amplifier_lib.paths import get_amplifier_home
from amplifier_lib.registry import BundleRegistry, load_bundle
from amplifier_lib.serialization import sanitize_message

# Session utilities
from amplifier_lib.session import (
    ForkResult,
    add_synthetic_tool_results,
    count_turns,
    find_orphaned_tool_calls,
    fork_session,
    fork_session_in_memory,
    get_fork_preview,
    get_session_lineage,
    get_turn_boundaries,
    get_turn_summary,
    list_session_forks,
    set_working_dir,
)

# Spawn utilities
from amplifier_lib.spawn_utils import apply_provider_preferences_with_resolution

# Tracing
from amplifier_lib.tracing import generate_sub_session_id

# Bundle updates
from amplifier_lib.updates import BundleStatus, check_bundle_status, update_bundle

__all__ = [
    # Exceptions
    "BundleDependencyError",
    "BundleError",
    "BundleLoadError",
    "BundleNotFoundError",
    "BundleValidationError",
    # Paths
    "get_amplifier_home",
    # IO / Serialization
    "sanitize_message",
    "write_with_backup",
    # Bundle system
    "Bundle",
    "BundleModuleResolver",
    "BundleRegistry",
    "PreparedBundle",
    "apply_provider_preferences_with_resolution",
    "load_bundle",
    # Bundle updates
    "BundleStatus",
    "check_bundle_status",
    "update_bundle",
    # Session
    "ForkResult",
    "add_synthetic_tool_results",
    "count_turns",
    "find_orphaned_tool_calls",
    "fork_session",
    "fork_session_in_memory",
    "get_fork_preview",
    "get_session_lineage",
    "get_turn_boundaries",
    "get_turn_summary",
    "list_session_forks",
    "set_working_dir",
    # Tracing
    "generate_sub_session_id",
]
