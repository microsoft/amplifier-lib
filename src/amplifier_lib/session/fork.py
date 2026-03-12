"""In-memory session fork operations.

Provides ``fork_session_in_memory()`` for creating a forked view of a
conversation at a specific turn boundary without any filesystem I/O.

Key concepts:
- Fork: A new conversation derived from an existing one at a specific turn.
- Turn: A user message plus all subsequent non-user messages until the next
  user message (1-indexed).
- Lineage: Parent–child relationship tracked via *parent_id*.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .slice import count_turns, find_orphaned_tool_calls, get_turn_boundaries, slice_to_turn


@dataclass
class ForkResult:
    """Result of a fork operation.

    Attributes:
        session_id: Unique identifier for the new forked session.
        session_dir: Path to the session directory, or ``None`` for in-memory
            forks.
        parent_id: The parent session's ID used for lineage tracking.
        forked_from_turn: Turn number at which the fork was taken.
        message_count: Number of messages in the forked conversation.
        messages: The forked message list (populated for in-memory forks only).
        events_count: Number of events copied (file-based forks only).
    """

    session_id: str
    session_dir: Path | None
    parent_id: str
    forked_from_turn: int
    message_count: int
    messages: list[dict[str, Any]] | None = None
    events_count: int = field(default=0)


def fork_session_in_memory(
    messages: list[dict[str, Any]],
    *,
    turn: int | None = None,
    parent_id: str | None = None,
    handle_orphaned_tools: str = "complete",
) -> ForkResult:
    """Fork a conversation in memory without any file I/O.

    Slices *messages* to the requested *turn* and returns a ``ForkResult``
    with the forked message list attached.  Useful for testing fork logic,
    generating previews, or in-process forking via a ContextManager.

    Args:
        messages: Source conversation messages to fork from.
        turn: 1-indexed turn to fork at.  ``None`` forks at the last turn
            (i.e. a full copy).
        parent_id: Parent session ID recorded in the result for lineage
            tracking.  Defaults to ``"unknown"`` when not provided.
        handle_orphaned_tools: How to handle tool_use blocks without a
            matching tool_result at the fork boundary:
            - ``"complete"`` — insert a synthetic error result (default)
            - ``"remove"``   — strip the orphaned tool_use block
            - ``"error"``    — raise ``ValueError``

    Returns:
        ``ForkResult`` with ``session_dir=None`` and ``messages`` populated.

    Raises:
        ValueError: If *turn* is out of range, or if *handle_orphaned_tools*
            is ``"error"`` and orphaned calls are present.

    Example:
        >>> messages = await context.get_messages()
        >>> result = fork_session_in_memory(messages, turn=2)
        >>> await new_context.set_messages(result.messages)
    """
    max_turns = count_turns(messages)

    if turn is None:
        turn = max_turns if max_turns > 0 else 0

    if max_turns == 0:
        return ForkResult(
            session_id=str(uuid.uuid4()),
            session_dir=None,
            parent_id=parent_id or "unknown",
            forked_from_turn=0,
            message_count=0,
            messages=[],
        )

    sliced = slice_to_turn(messages, turn, handle_orphaned_tools=handle_orphaned_tools)

    return ForkResult(
        session_id=str(uuid.uuid4()),
        session_dir=None,
        parent_id=parent_id or "unknown",
        forked_from_turn=turn,
        message_count=len(sliced),
        messages=sliced,
    )


# ---------------------------------------------------------------------------
# Private filesystem helpers
# ---------------------------------------------------------------------------


def _load_transcript(path: Path) -> list[dict[str, Any]]:
    """Load transcript.jsonl into a list of message dicts."""
    messages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            messages.append(json.loads(line))
    return messages


def _write_transcript(path: Path, messages: list[dict[str, Any]]) -> None:
    """Write messages to transcript.jsonl."""
    with path.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")


def _load_metadata(path: Path) -> dict[str, Any]:
    """Load metadata.json."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write metadata.json."""
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def _extract_text_content(content: Any) -> str:
    """Extract text from message content (string or list-of-blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
    return ""


# ---------------------------------------------------------------------------
# Public filesystem-level fork functions
# ---------------------------------------------------------------------------


def list_session_forks(session_dir: Path, sessions_root: Path) -> list[dict[str, Any]]:
    """Find all direct forks (children) of a session.

    Args:
        session_dir: Path to the parent session directory.
        sessions_root: Root directory containing all session directories.

    Returns:
        List of dicts describing each child fork, with keys:
        ``session_id``, ``forked_from_turn``, ``forked_at``.
    """
    session_dir = session_dir.resolve()
    metadata_path = session_dir / "metadata.json"
    session_id = (
        _load_metadata(metadata_path).get("session_id", session_dir.name)
        if metadata_path.exists()
        else session_dir.name
    )

    children = []
    for child_dir in sessions_root.iterdir():
        if not child_dir.is_dir() or child_dir.resolve() == session_dir:
            continue
        child_meta_path = child_dir / "metadata.json"
        if child_meta_path.exists():
            child_meta = _load_metadata(child_meta_path)
            if child_meta.get("parent_id") == session_id:
                children.append(
                    {
                        "session_id": child_meta.get("session_id", child_dir.name),
                        "forked_from_turn": child_meta.get("forked_from_turn"),
                        "forked_at": child_meta.get("forked_at"),
                    }
                )
    return children


def fork_session(
    parent_session_dir: Path,
    *,
    turn: int | None = None,
    new_session_id: str | None = None,
    target_dir: Path | None = None,
    include_events: bool = True,
    handle_orphaned_tools: str = "complete",
) -> ForkResult:
    """Fork a session at a specific turn, writing a new session directory.

    Loads ``transcript.jsonl`` from *parent_session_dir*, slices it to
    *turn*, writes the result to a new directory alongside the parent, and
    records lineage in ``metadata.json``.

    Args:
        parent_session_dir: Path to the existing session directory.
        turn: 1-indexed turn to fork at.  ``None`` copies the full history.
        new_session_id: Override the generated session ID for the fork.
        target_dir: Override where the new session directory is created.
            Defaults to a sibling of *parent_session_dir* named after the
            new session ID.
        include_events: If ``True``, attempt to copy ``events.jsonl``.
            Failures are silently swallowed and an empty file is written.
        handle_orphaned_tools: Forwarded to :func:`slice_to_turn`.

    Returns:
        :class:`ForkResult` with ``session_dir`` pointing to the new
        directory and ``messages=None`` (filesystem fork).

    Raises:
        FileNotFoundError: If ``transcript.jsonl`` is missing.
        ValueError: If *turn* is out of range.
    """
    parent_session_dir = parent_session_dir.resolve()
    transcript_path = parent_session_dir / "transcript.jsonl"

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    messages = _load_transcript(transcript_path)
    max_turns = count_turns(messages)

    if turn is None:
        effective_turn = max_turns if max_turns > 0 else 0
    else:
        effective_turn = turn

    if max_turns == 0:
        sliced: list[dict[str, Any]] = []
    else:
        if effective_turn < 1 or effective_turn > max_turns:
            raise ValueError(f"Turn {effective_turn} out of range (1-{max_turns})")
        sliced = slice_to_turn(
            messages, effective_turn, handle_orphaned_tools=handle_orphaned_tools
        )

    # Load parent metadata for lineage tracking.
    parent_meta_path = parent_session_dir / "metadata.json"
    if parent_meta_path.exists():
        parent_meta = _load_metadata(parent_meta_path)
        parent_id = parent_meta.get("session_id", parent_session_dir.name)
    else:
        parent_id = parent_session_dir.name

    session_id = new_session_id or str(uuid.uuid4())

    # Determine (and create) the target directory.
    resolved_target = Path(target_dir) if target_dir else parent_session_dir.parent / session_id
    resolved_target.mkdir(parents=True, exist_ok=True)

    # Write sliced transcript.
    _write_transcript(resolved_target / "transcript.jsonl", sliced)

    # Write metadata with parent lineage.
    now = datetime.now(tz=UTC).isoformat()
    metadata: dict[str, Any] = {
        "session_id": session_id,
        "parent_id": parent_id,
        "forked_from_turn": effective_turn,
        "forked_at": now,
        "message_count": len(sliced),
    }
    _write_metadata(resolved_target / "metadata.json", metadata)

    # Optionally copy events (complex slicing skipped — copy as-is or skip).
    events_count = 0
    if include_events:
        try:
            events_src = parent_session_dir / "events.jsonl"
            events_dst = resolved_target / "events.jsonl"
            if events_src.exists():
                events_data = events_src.read_text(encoding="utf-8")
                events_dst.write_text(events_data, encoding="utf-8")
                events_count = sum(1 for line in events_data.splitlines() if line.strip())
            else:
                events_dst.write_text("", encoding="utf-8")
        except Exception:
            (resolved_target / "events.jsonl").write_text("", encoding="utf-8")

    return ForkResult(
        session_id=session_id,
        session_dir=resolved_target,
        parent_id=parent_id,
        forked_from_turn=effective_turn,
        message_count=len(sliced),
        events_count=events_count,
    )


def get_fork_preview(
    parent_session_dir: Path,
    turn: int,
) -> dict[str, Any]:
    """Return a preview of what a fork at *turn* would look like.

    Does not write any files.

    Args:
        parent_session_dir: Path to the parent session directory.
        turn: 1-indexed turn to preview the fork at.

    Returns:
        Dict with keys: ``parent_id``, ``turn``, ``max_turns``,
        ``message_count``, ``has_orphaned_tools``, ``orphaned_tool_count``,
        ``last_user_message``, ``last_assistant_message``.

    Raises:
        FileNotFoundError: If ``transcript.jsonl`` is missing.
        ValueError: If *turn* is out of range.
    """
    parent_session_dir = parent_session_dir.resolve()
    transcript_path = parent_session_dir / "transcript.jsonl"

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    messages = _load_transcript(transcript_path)
    boundaries = get_turn_boundaries(messages)
    max_turns = len(boundaries)

    if turn < 1 or turn > max_turns:
        raise ValueError(f"Turn {turn} out of range (1-{max_turns})")

    # Slice without handling orphans so we can report them accurately.
    start_idx = boundaries[turn - 1]
    end_idx = boundaries[turn] if turn < max_turns else len(messages)
    turn_messages = messages[start_idx:end_idx]
    preview_messages = messages[:end_idx]

    orphaned = find_orphaned_tool_calls(preview_messages)

    # Extract last user / assistant content from the turn slice.
    last_user = ""
    last_assistant = ""
    for msg in turn_messages:
        role = msg.get("role")
        if role == "user":
            last_user = _extract_text_content(msg.get("content", ""))
        elif role == "assistant":
            last_assistant = _extract_text_content(msg.get("content", ""))

    # Resolve parent_id from metadata if available.
    parent_meta_path = parent_session_dir / "metadata.json"
    if parent_meta_path.exists():
        parent_id = _load_metadata(parent_meta_path).get("session_id", parent_session_dir.name)
    else:
        parent_id = parent_session_dir.name

    return {
        "parent_id": parent_id,
        "turn": turn,
        "max_turns": max_turns,
        "message_count": end_idx,
        "has_orphaned_tools": bool(orphaned),
        "orphaned_tool_count": len(orphaned),
        "last_user_message": last_user,
        "last_assistant_message": last_assistant,
    }


def get_session_lineage(
    session_dir: Path,
    sessions_root: Path | None = None,
) -> dict[str, Any]:
    """Trace the full lineage of a session: ancestors and direct children.

    Args:
        session_dir: Path to the session directory to inspect.
        sessions_root: Root directory used to discover sibling sessions when
            searching for children.  Defaults to the parent of *session_dir*.

    Returns:
        Dict with keys: ``session_id``, ``parent_id``, ``forked_from_turn``,
        ``ancestors`` (ordered oldest-first), ``children``, ``depth``
        (0 = root, no parent).
    """
    session_dir = session_dir.resolve()
    if sessions_root is None:
        sessions_root = session_dir.parent

    meta_path = session_dir / "metadata.json"
    if meta_path.exists():
        meta = _load_metadata(meta_path)
    else:
        meta = {}

    session_id = meta.get("session_id", session_dir.name)
    parent_id: str | None = meta.get("parent_id")
    forked_from_turn: int | None = meta.get("forked_from_turn")

    # Walk ancestor chain.
    ancestors: list[dict[str, Any]] = []
    visited: set[str] = {session_id}
    current_parent_id = parent_id

    while current_parent_id:
        # Find the ancestor directory by matching session_id in metadata.
        ancestor_dir: Path | None = None
        for candidate in sessions_root.iterdir():
            if not candidate.is_dir():
                continue
            cand_meta_path = candidate / "metadata.json"
            if cand_meta_path.exists():
                cand_meta = _load_metadata(cand_meta_path)
                if cand_meta.get("session_id") == current_parent_id:
                    ancestor_dir = candidate
                    break

        if ancestor_dir is None:
            # Ancestor not found in sessions_root — record as stub and stop.
            ancestors.insert(
                0,
                {
                    "session_id": current_parent_id,
                    "parent_id": None,
                    "forked_from_turn": None,
                },
            )
            break

        anc_meta_path = ancestor_dir / "metadata.json"
        anc_meta = _load_metadata(anc_meta_path) if anc_meta_path.exists() else {}
        anc_id = anc_meta.get("session_id", ancestor_dir.name)

        if anc_id in visited:
            break  # Guard against circular references.
        visited.add(anc_id)

        ancestors.insert(
            0,
            {
                "session_id": anc_id,
                "parent_id": anc_meta.get("parent_id"),
                "forked_from_turn": anc_meta.get("forked_from_turn"),
            },
        )
        current_parent_id = anc_meta.get("parent_id")

    children = list_session_forks(session_dir, sessions_root)

    return {
        "session_id": session_id,
        "parent_id": parent_id,
        "forked_from_turn": forked_from_turn,
        "ancestors": ancestors,
        "children": children,
        "depth": len(ancestors),
    }
