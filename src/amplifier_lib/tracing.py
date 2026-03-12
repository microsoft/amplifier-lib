"""Session ID generation with W3C Trace Context lineage."""

from __future__ import annotations

import re
import uuid

_SPAN_HEX_LEN = 16
_DEFAULT_PARENT_SPAN = "0" * _SPAN_HEX_LEN
_SPAN_PATTERN = re.compile(r"^([0-9a-f]{16})-([0-9a-f]{16})_")
_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def generate_sub_session_id(
    agent_name: str | None = None,
    parent_session_id: str | None = None,
    parent_trace_id: str | None = None,
) -> str:
    """Generate a sub-session ID with W3C Trace Context lineage.

    Format: {parent-span}-{child-span}_{agent-name}

    Args:
        agent_name: Name of the sub-agent (for human readability).
        parent_session_id: Parent session's ID (for span extraction).
        parent_trace_id: Parent trace ID if using distributed tracing.

    Returns:
        Sub-session ID with embedded trace context.
    """
    raw_name = (agent_name or "").lower()
    sanitized = re.sub(r"[^a-z0-9]+", "-", raw_name)
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    sanitized = sanitized.strip("-").lstrip(".")
    if not sanitized:
        sanitized = "agent"

    parent_span = _DEFAULT_PARENT_SPAN

    if parent_session_id:
        match = _SPAN_PATTERN.match(parent_session_id)
        if match:
            parent_span = match.group(2)

    if (
        parent_span == _DEFAULT_PARENT_SPAN
        and parent_trace_id
        and _TRACE_ID_PATTERN.fullmatch(parent_trace_id)
    ):
        parent_span = parent_trace_id[8:24]

    child_span = uuid.uuid4().hex[:_SPAN_HEX_LEN]

    return f"{parent_span}-{child_span}_{sanitized}"
