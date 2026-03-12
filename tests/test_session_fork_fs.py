"""Tests for filesystem-level session fork functions:
fork_session, get_fork_preview, get_session_lineage, list_session_forks.
"""

import json
from pathlib import Path

import pytest

from amplifier_lib.session.fork import (
    ForkResult,
    fork_session,
    get_fork_preview,
    get_session_lineage,
    list_session_forks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_transcript(session_dir: Path, messages: list[dict]) -> None:
    """Write a list of message dicts to transcript.jsonl."""
    with (session_dir / "transcript.jsonl").open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


def write_metadata(session_dir: Path, metadata: dict) -> None:
    """Write metadata.json."""
    (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def make_session(
    root: Path,
    session_id: str,
    messages: list[dict],
    parent_id: str | None = None,
    forked_from_turn: int | None = None,
) -> Path:
    """Create a minimal session directory with transcript and metadata."""
    session_dir = root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    write_transcript(session_dir, messages)
    meta: dict = {"session_id": session_id}
    if parent_id is not None:
        meta["parent_id"] = parent_id
    if forked_from_turn is not None:
        meta["forked_from_turn"] = forked_from_turn
    write_metadata(session_dir, meta)
    return session_dir


def make_conversation(*qa_pairs: tuple[str, str]) -> list[dict]:
    """Build a conversation from (question, answer) pairs."""
    msgs: list[dict] = []
    for q, a in qa_pairs:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def load_transcript(session_dir: Path) -> list[dict]:
    """Read back a transcript.jsonl as a list of dicts."""
    lines = (session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_metadata(session_dir: Path) -> dict:
    """Read back metadata.json as a dict."""
    return json.loads((session_dir / "metadata.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# fork_session — basic creation
# ---------------------------------------------------------------------------


class TestForkSession:
    def test_creates_new_directory(self, tmp_path):
        session_dir = make_session(
            tmp_path, "parent-1", make_conversation(("Q1", "A1"), ("Q2", "A2"))
        )
        result = fork_session(session_dir, turn=1)
        assert result.session_dir is not None
        assert result.session_dir.exists()
        assert result.session_dir.is_dir()

    def test_returns_fork_result_instance(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q1", "A1")))
        result = fork_session(session_dir, turn=1)
        assert isinstance(result, ForkResult)

    def test_transcript_sliced_to_requested_turn(self, tmp_path):
        session_dir = make_session(
            tmp_path,
            "parent-1",
            make_conversation(("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")),
        )
        result = fork_session(session_dir, turn=2)
        assert result.session_dir is not None
        transcript = load_transcript(result.session_dir)
        contents = [m["content"] for m in transcript]
        assert "Q1" in contents
        assert "A1" in contents
        assert "Q2" in contents
        assert "A2" in contents
        assert "Q3" not in contents

    def test_transcript_full_copy_when_turn_none(self, tmp_path):
        msgs = make_conversation(("Q1", "A1"), ("Q2", "A2"))
        session_dir = make_session(tmp_path, "parent-1", msgs)
        result = fork_session(session_dir, turn=None)
        assert result.session_dir is not None
        transcript = load_transcript(result.session_dir)
        assert len(transcript) == len(msgs)

    def test_metadata_has_parent_id(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-abc", make_conversation(("Q1", "A1")))
        result = fork_session(session_dir, turn=1)
        assert result.session_dir is not None
        meta = load_metadata(result.session_dir)
        assert meta["parent_id"] == "parent-abc"

    def test_metadata_has_forked_from_turn(self, tmp_path):
        session_dir = make_session(
            tmp_path, "parent-1", make_conversation(("Q1", "A1"), ("Q2", "A2"))
        )
        result = fork_session(session_dir, turn=1)
        assert result.session_dir is not None
        meta = load_metadata(result.session_dir)
        assert meta["forked_from_turn"] == 1

    def test_metadata_session_id_matches_result(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q1", "A1")))
        result = fork_session(session_dir, turn=1)
        assert result.session_dir is not None
        meta = load_metadata(result.session_dir)
        assert meta["session_id"] == result.session_id

    def test_result_parent_id_matches_parent_session(self, tmp_path):
        session_dir = make_session(tmp_path, "my-parent", make_conversation(("Q1", "A1")))
        result = fork_session(session_dir, turn=1)
        assert result.parent_id == "my-parent"

    def test_custom_session_id_used(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q1", "A1")))
        result = fork_session(session_dir, turn=1, new_session_id="my-fork-id")
        assert result.session_id == "my-fork-id"

    def test_custom_target_dir_used(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q1", "A1")))
        target = tmp_path / "custom-target"
        result = fork_session(session_dir, turn=1, target_dir=target)
        assert result.session_dir == target.resolve()
        assert target.exists()

    def test_message_count_in_result(self, tmp_path):
        session_dir = make_session(
            tmp_path, "parent-1", make_conversation(("Q1", "A1"), ("Q2", "A2"))
        )
        result = fork_session(session_dir, turn=1)
        assert result.message_count == 2

    def test_forked_from_turn_in_result(self, tmp_path):
        session_dir = make_session(
            tmp_path, "parent-1", make_conversation(("Q1", "A1"), ("Q2", "A2"))
        )
        result = fork_session(session_dir, turn=2)
        assert result.forked_from_turn == 2

    def test_invalid_turn_raises_value_error(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q1", "A1")))
        with pytest.raises(ValueError, match="out of range"):
            fork_session(session_dir, turn=99)

    def test_missing_transcript_raises_file_not_found(self, tmp_path):
        session_dir = tmp_path / "empty-session"
        session_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            fork_session(session_dir, turn=1)

    def test_events_file_created_when_source_exists(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q1", "A1")))
        (session_dir / "events.jsonl").write_text('{"event": "start"}\n', encoding="utf-8")
        result = fork_session(session_dir, turn=1, include_events=True)
        assert result.session_dir is not None
        assert (result.session_dir / "events.jsonl").exists()

    def test_events_count_reflects_copied_events(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q1", "A1")))
        (session_dir / "events.jsonl").write_text(
            '{"event": "e1"}\n{"event": "e2"}\n', encoding="utf-8"
        )
        result = fork_session(session_dir, turn=1, include_events=True)
        assert result.events_count == 2

    def test_session_dir_is_none_when_using_messages_field(self, tmp_path):
        """Filesystem fork should have session_dir set, not None."""
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q1", "A1")))
        result = fork_session(session_dir, turn=1)
        assert result.session_dir is not None


# ---------------------------------------------------------------------------
# get_fork_preview
# ---------------------------------------------------------------------------


class TestGetForkPreview:
    def test_returns_dict(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q1", "A1")))
        preview = get_fork_preview(session_dir, 1)
        assert isinstance(preview, dict)

    def test_expected_keys_present(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q1", "A1")))
        preview = get_fork_preview(session_dir, 1)
        expected_keys = {
            "parent_id",
            "turn",
            "max_turns",
            "message_count",
            "has_orphaned_tools",
            "orphaned_tool_count",
            "last_user_message",
            "last_assistant_message",
        }
        assert set(preview.keys()) == expected_keys

    def test_turn_field_matches_requested_turn(self, tmp_path):
        session_dir = make_session(
            tmp_path, "parent-1", make_conversation(("Q1", "A1"), ("Q2", "A2"))
        )
        preview = get_fork_preview(session_dir, 2)
        assert preview["turn"] == 2

    def test_max_turns_correct(self, tmp_path):
        session_dir = make_session(
            tmp_path,
            "parent-1",
            make_conversation(("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")),
        )
        preview = get_fork_preview(session_dir, 1)
        assert preview["max_turns"] == 3

    def test_last_user_message_extracted(self, tmp_path):
        session_dir = make_session(
            tmp_path, "parent-1", make_conversation(("Hello user", "Hello assistant"))
        )
        preview = get_fork_preview(session_dir, 1)
        assert preview["last_user_message"] == "Hello user"

    def test_last_assistant_message_extracted(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q", "My reply")))
        preview = get_fork_preview(session_dir, 1)
        assert preview["last_assistant_message"] == "My reply"

    def test_no_orphaned_tools_when_clean(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q", "A")))
        preview = get_fork_preview(session_dir, 1)
        assert preview["has_orphaned_tools"] is False
        assert preview["orphaned_tool_count"] == 0

    def test_detects_orphaned_tools(self, tmp_path):
        msgs = [
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "id1",
                        "type": "function",
                        "function": {"name": "fn", "arguments": "{}"},
                    }
                ],
            },
            # No tool result — orphaned
        ]
        session_dir = make_session(tmp_path, "parent-1", msgs)
        preview = get_fork_preview(session_dir, 1)
        assert preview["has_orphaned_tools"] is True
        assert preview["orphaned_tool_count"] == 1

    def test_parent_id_from_metadata(self, tmp_path):
        session_dir = make_session(tmp_path, "root-session", make_conversation(("Q", "A")))
        preview = get_fork_preview(session_dir, 1)
        assert preview["parent_id"] == "root-session"

    def test_missing_transcript_raises_file_not_found(self, tmp_path):
        session_dir = tmp_path / "no-transcript"
        session_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            get_fork_preview(session_dir, 1)

    def test_out_of_range_turn_raises_value_error(self, tmp_path):
        session_dir = make_session(tmp_path, "parent-1", make_conversation(("Q", "A")))
        with pytest.raises(ValueError, match="out of range"):
            get_fork_preview(session_dir, 99)

    def test_message_count_includes_all_messages_up_to_turn(self, tmp_path):
        session_dir = make_session(
            tmp_path,
            "parent-1",
            make_conversation(("Q1", "A1"), ("Q2", "A2")),
        )
        preview = get_fork_preview(session_dir, 2)
        # All 4 messages are included when previewing turn 2
        assert preview["message_count"] == 4


# ---------------------------------------------------------------------------
# get_session_lineage
# ---------------------------------------------------------------------------


class TestGetSessionLineage:
    def test_returns_dict(self, tmp_path):
        session_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        lineage = get_session_lineage(session_dir)
        assert isinstance(lineage, dict)

    def test_expected_keys_present(self, tmp_path):
        session_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        lineage = get_session_lineage(session_dir)
        assert set(lineage.keys()) == {
            "session_id",
            "parent_id",
            "forked_from_turn",
            "ancestors",
            "children",
            "depth",
        }

    def test_root_session_has_no_parent(self, tmp_path):
        session_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        lineage = get_session_lineage(session_dir)
        assert lineage["parent_id"] is None

    def test_root_session_depth_zero(self, tmp_path):
        session_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        lineage = get_session_lineage(session_dir)
        assert lineage["depth"] == 0

    def test_root_session_no_ancestors(self, tmp_path):
        session_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        lineage = get_session_lineage(session_dir)
        assert lineage["ancestors"] == []

    def test_session_id_from_metadata(self, tmp_path):
        session_dir = make_session(tmp_path, "my-session", make_conversation(("Q", "A")))
        lineage = get_session_lineage(session_dir)
        assert lineage["session_id"] == "my-session"

    def test_child_has_parent_in_ancestors(self, tmp_path):
        # root → child
        make_session(tmp_path, "root", make_conversation(("Q", "A")))
        child_dir = make_session(
            tmp_path,
            "child",
            make_conversation(("Q", "A")),
            parent_id="root",
            forked_from_turn=1,
        )
        lineage = get_session_lineage(child_dir, sessions_root=tmp_path)
        ancestor_ids = [a["session_id"] for a in lineage["ancestors"]]
        assert "root" in ancestor_ids

    def test_child_depth_is_one(self, tmp_path):
        make_session(tmp_path, "root", make_conversation(("Q", "A")))
        child_dir = make_session(
            tmp_path,
            "child",
            make_conversation(("Q", "A")),
            parent_id="root",
            forked_from_turn=1,
        )
        lineage = get_session_lineage(child_dir, sessions_root=tmp_path)
        assert lineage["depth"] == 1

    def test_grandchild_depth_is_two(self, tmp_path):
        make_session(tmp_path, "root", make_conversation(("Q", "A")))
        make_session(
            tmp_path,
            "child",
            make_conversation(("Q", "A")),
            parent_id="root",
            forked_from_turn=1,
        )
        grandchild_dir = make_session(
            tmp_path,
            "grandchild",
            make_conversation(("Q", "A")),
            parent_id="child",
            forked_from_turn=1,
        )
        lineage = get_session_lineage(grandchild_dir, sessions_root=tmp_path)
        assert lineage["depth"] == 2

    def test_ancestors_ordered_oldest_first(self, tmp_path):
        make_session(tmp_path, "root", make_conversation(("Q", "A")))
        make_session(
            tmp_path,
            "child",
            make_conversation(("Q", "A")),
            parent_id="root",
            forked_from_turn=1,
        )
        grandchild_dir = make_session(
            tmp_path,
            "grandchild",
            make_conversation(("Q", "A")),
            parent_id="child",
            forked_from_turn=1,
        )
        lineage = get_session_lineage(grandchild_dir, sessions_root=tmp_path)
        ancestor_ids = [a["session_id"] for a in lineage["ancestors"]]
        assert ancestor_ids == ["root", "child"]

    def test_finds_direct_children(self, tmp_path):
        root_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        make_session(
            tmp_path,
            "fork-1",
            make_conversation(("Q", "A")),
            parent_id="root",
            forked_from_turn=1,
        )
        make_session(
            tmp_path,
            "fork-2",
            make_conversation(("Q", "A")),
            parent_id="root",
            forked_from_turn=1,
        )
        lineage = get_session_lineage(root_dir, sessions_root=tmp_path)
        child_ids = {c["session_id"] for c in lineage["children"]}
        assert child_ids == {"fork-1", "fork-2"}

    def test_no_children_for_leaf_session(self, tmp_path):
        session_dir = make_session(tmp_path, "leaf", make_conversation(("Q", "A")))
        lineage = get_session_lineage(session_dir, sessions_root=tmp_path)
        assert lineage["children"] == []

    def test_parent_id_in_lineage_matches_metadata(self, tmp_path):
        make_session(tmp_path, "root", make_conversation(("Q", "A")))
        child_dir = make_session(
            tmp_path,
            "child",
            make_conversation(("Q", "A")),
            parent_id="root",
            forked_from_turn=1,
        )
        lineage = get_session_lineage(child_dir, sessions_root=tmp_path)
        assert lineage["parent_id"] == "root"

    def test_forked_from_turn_in_lineage(self, tmp_path):
        make_session(tmp_path, "root", make_conversation(("Q", "A")))
        child_dir = make_session(
            tmp_path,
            "child",
            make_conversation(("Q", "A")),
            parent_id="root",
            forked_from_turn=3,
        )
        lineage = get_session_lineage(child_dir, sessions_root=tmp_path)
        assert lineage["forked_from_turn"] == 3


# ---------------------------------------------------------------------------
# list_session_forks
# ---------------------------------------------------------------------------


class TestListSessionForks:
    def test_returns_empty_when_no_forks(self, tmp_path):
        session_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        forks = list_session_forks(session_dir, tmp_path)
        assert forks == []

    def test_finds_one_fork(self, tmp_path):
        root_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        make_session(
            tmp_path,
            "fork-1",
            make_conversation(("Q", "A")),
            parent_id="root",
            forked_from_turn=1,
        )
        forks = list_session_forks(root_dir, tmp_path)
        assert len(forks) == 1
        assert forks[0]["session_id"] == "fork-1"

    def test_finds_multiple_forks(self, tmp_path):
        root_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        for i in range(3):
            make_session(
                tmp_path,
                f"fork-{i}",
                make_conversation(("Q", "A")),
                parent_id="root",
                forked_from_turn=1,
            )
        forks = list_session_forks(root_dir, tmp_path)
        assert len(forks) == 3

    def test_does_not_include_unrelated_sessions(self, tmp_path):
        root_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        make_session(tmp_path, "unrelated", make_conversation(("Q", "A")))
        forks = list_session_forks(root_dir, tmp_path)
        fork_ids = [f["session_id"] for f in forks]
        assert "unrelated" not in fork_ids

    def test_fork_entry_has_expected_keys(self, tmp_path):
        root_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        make_session(
            tmp_path,
            "fork-1",
            make_conversation(("Q", "A")),
            parent_id="root",
            forked_from_turn=2,
        )
        forks = list_session_forks(root_dir, tmp_path)
        assert set(forks[0].keys()) == {"session_id", "forked_from_turn", "forked_at"}

    def test_forked_from_turn_recorded(self, tmp_path):
        root_dir = make_session(tmp_path, "root", make_conversation(("Q", "A")))
        make_session(
            tmp_path,
            "fork-1",
            make_conversation(("Q", "A")),
            parent_id="root",
            forked_from_turn=5,
        )
        forks = list_session_forks(root_dir, tmp_path)
        assert forks[0]["forked_from_turn"] == 5
