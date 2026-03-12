"""Message slicing utilities for session fork operations.

Provides pure functions for slicing conversation messages at turn boundaries.
A "turn" is a user message plus all subsequent non-user messages until the
next user message. Turns are 1-indexed for user-facing operations.
"""

from __future__ import annotations

import json
from typing import Any


def get_turn_boundaries(messages: list[dict[str, Any]]) -> list[int]:
    """Return 0-indexed positions of all user messages in the conversation.

    Each user message marks the start of a new turn.

    Args:
        messages: List of conversation messages with a 'role' field.

    Returns:
        List of indices where user messages appear.

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Q1"},
        ...     {"role": "assistant", "content": "A1"},
        ...     {"role": "user", "content": "Q2"},
        ... ]
        >>> get_turn_boundaries(messages)
        [0, 2]
    """
    return [i for i, msg in enumerate(messages) if msg.get("role") == "user"]


def count_turns(messages: list[dict[str, Any]]) -> int:
    """Return the number of turns (user messages) in a conversation."""
    return len(get_turn_boundaries(messages))


def get_turn_summary(
    messages: list[dict[str, Any]],
    turn: int,
    *,
    max_length: int = 100,
) -> dict[str, Any]:
    """Get a summary of a specific turn for display purposes.

    Args:
        messages: Full conversation message list.
        turn: 1-indexed turn number to summarise.
        max_length: Maximum characters to include from user/assistant content.

    Returns:
        Dict with keys: ``turn``, ``user_content``, ``assistant_content``,
        ``tool_count``, ``message_count``.

    Raises:
        ValueError: If *turn* is out of range (1-based, must be <= max turns).
    """
    boundaries = get_turn_boundaries(messages)
    max_turns = len(boundaries)
    if turn < 1 or turn > max_turns:
        raise ValueError(f"Turn {turn} out of range (1-{max_turns})")

    start_idx = boundaries[turn - 1]
    end_idx = boundaries[turn] if turn < max_turns else len(messages)
    turn_messages = messages[start_idx:end_idx]

    user_content = ""
    assistant_content = ""
    tool_count = 0

    for msg in turn_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, str):
                user_content = content[:max_length]
                if len(content) > max_length:
                    user_content += "..."
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        user_content = block.get("text", "")[:max_length]
                        break

        elif role == "assistant":
            if isinstance(content, str):
                if not assistant_content:
                    assistant_content = content[:max_length]
                    if len(content) > max_length:
                        assistant_content += "..."
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text" and not assistant_content:
                            assistant_content = block.get("text", "")[:max_length]
                        elif block.get("type") == "tool_use":
                            tool_count += 1

            if "tool_calls" in msg:
                tool_count += len(msg["tool_calls"])

    return {
        "turn": turn,
        "user_content": user_content,
        "assistant_content": assistant_content,
        "tool_count": tool_count,
        "message_count": len(turn_messages),
    }


def find_orphaned_tool_calls(messages: list[dict[str, Any]]) -> list[str]:
    """Find tool-call IDs that have no corresponding tool result.

    Scans assistant messages for tool calls in both OpenAI format
    (``tool_calls`` array) and Anthropic format (``content`` blocks with
    ``type == "tool_use"``), then returns IDs with no matching ``tool`` result.

    Args:
        messages: List of conversation messages.

    Returns:
        List of orphaned tool-call IDs.
    """
    called_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            # OpenAI format: tool_calls array
            for tc in msg.get("tool_calls", []):
                if "id" in tc:
                    called_ids.add(tc["id"])
            # Anthropic format: content blocks
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if "id" in block:
                            called_ids.add(block["id"])

    result_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            result_ids.add(msg["tool_call_id"])

    return list(called_ids - result_ids)


def add_synthetic_tool_results(
    messages: list[dict[str, Any]],
    orphaned_ids: list[str],
) -> list[dict[str, Any]]:
    """Insert synthetic error results for orphaned tool calls.

    When a session is forked mid-turn some tool calls may not yet have
    results.  This inserts a synthetic ``role: tool`` error message directly
    after the assistant message that issued each orphaned call, keeping the
    conversation valid for replay.

    Args:
        messages: List of conversation messages.
        orphaned_ids: Tool-call IDs that need synthetic results.

    Returns:
        New message list with synthetic results inserted at the correct
        positions.
    """
    if not orphaned_ids:
        return messages

    orphaned_set = set(orphaned_ids)

    # Build tool_call_id -> tool_name mapping from assistant messages.
    tool_names: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                tc_id = tc.get("id", "")
                tc_name = tc.get("function", {}).get("name", "") or tc.get("name", "")
                if tc_id and tc_name:
                    tool_names[tc_id] = tc_name
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tc_id = block.get("id", "")
                        tc_name = block.get("name", "")
                        if tc_id and tc_name:
                            tool_names[tc_id] = tc_name

    result: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        result.append(msg)

        if msg.get("role") != "assistant":
            continue

        # Collect orphaned IDs belonging to this assistant message.
        msg_orphans: list[str] = []
        for tc in msg.get("tool_calls", []):
            tc_id = tc.get("id", "")
            if tc_id in orphaned_set:
                msg_orphans.append(tc_id)
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tc_id = block.get("id", "")
                    if tc_id in orphaned_set:
                        msg_orphans.append(tc_id)

        if not msg_orphans:
            continue

        # Insert one synthetic result per orphaned call.
        for tc_id in msg_orphans:
            synthetic: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps(
                    {
                        "error": "Tool execution interrupted by session fork",
                        "forked": True,
                        "message": (
                            "This tool call was in progress when the session was "
                            "forked. The result is not available in this forked session."
                        ),
                    }
                ),
            }
            tool_name = tool_names.get(tc_id)
            if tool_name:
                synthetic["name"] = tool_name
            result.append(synthetic)

        # If the next original message is a user message, close the interrupted
        # turn with a synthetic assistant response first.
        next_idx = i + 1
        if next_idx < len(messages) and messages[next_idx].get("role") == "user":
            result.append(
                {
                    "role": "assistant",
                    "content": (
                        "The previous tool calls were interrupted by a session fork. "
                        "Results are not available in this forked session."
                    ),
                }
            )

    return result


def slice_to_turn(
    messages: list[dict[str, Any]],
    turn: int,
    *,
    handle_orphaned_tools: str = "complete",
) -> list[dict[str, Any]]:
    """Slice messages to include only up to turn N (1-indexed).

    Turn N covers the Nth user message and all responses up to (but not
    including) the next user message.

    Args:
        messages: Full conversation message list.
        turn: 1-indexed turn number to slice at.
        handle_orphaned_tools: What to do when a tool_use has no result:
            - ``"complete"`` — insert a synthetic error result (default)
            - ``"remove"``   — strip the orphaned tool_use block
            - ``"error"``    — raise ``ValueError``

    Returns:
        Sliced message list with orphaned tools handled.

    Raises:
        ValueError: If *turn* is out of range, or if *handle_orphaned_tools*
            is ``"error"`` and orphaned calls are found.
    """
    if turn < 1:
        raise ValueError(f"Turn must be >= 1, got {turn}")

    boundaries = get_turn_boundaries(messages)
    max_turns = len(boundaries)

    if max_turns == 0:
        raise ValueError("No user messages found in conversation")

    if turn > max_turns:
        raise ValueError(f"Turn {turn} exceeds max turns ({max_turns}). Valid range: 1-{max_turns}")

    end_idx = boundaries[turn] if turn < max_turns else len(messages)
    sliced = messages[:end_idx]

    orphaned = find_orphaned_tool_calls(sliced)
    if orphaned:
        if handle_orphaned_tools == "error":
            raise ValueError(
                f"Orphaned tool calls at fork boundary: {orphaned}. "
                "These tool_use blocks have no matching tool_result."
            )
        elif handle_orphaned_tools == "remove":
            sliced = _remove_orphaned_tool_calls(sliced, orphaned)
        else:  # "complete" is the default
            sliced = add_synthetic_tool_results(sliced, orphaned)

    return sliced


# --- Private helpers ---


def _remove_orphaned_tool_calls(
    messages: list[dict[str, Any]],
    orphaned_ids: list[str],
) -> list[dict[str, Any]]:
    """Remove orphaned tool-call entries from assistant messages."""
    orphaned_set = set(orphaned_ids)
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") != "assistant":
            result.append(msg)
            continue

        new_msg = dict(msg)

        # Filter OpenAI-format tool_calls array.
        if "tool_calls" in new_msg:
            new_msg["tool_calls"] = [
                tc for tc in new_msg["tool_calls"] if tc.get("id") not in orphaned_set
            ]
            if not new_msg["tool_calls"]:
                del new_msg["tool_calls"]

        # Filter Anthropic-format content blocks.
        content = new_msg.get("content")
        if isinstance(content, list):
            new_msg["content"] = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("id") in orphaned_set
                )
            ]

        result.append(new_msg)

    return result
