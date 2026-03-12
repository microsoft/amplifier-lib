"""Tests for amplifier_lib.session.fork — fork_session_in_memory()."""

import uuid

import pytest

from amplifier_lib.session.fork import ForkResult, fork_session_in_memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant(content: str = "OK") -> dict:
    return {"role": "assistant", "content": content}


def make_conversation(*qa_pairs: tuple[str, str]) -> list[dict]:
    """Build a conversation from (question, answer) pairs."""
    msgs = []
    for q, a in qa_pairs:
        msgs.append(user(q))
        msgs.append(assistant(a))
    return msgs


# ---------------------------------------------------------------------------
# ForkResult dataclass
# ---------------------------------------------------------------------------


class TestForkResultFields:
    def test_all_fields_present(self):
        result = ForkResult(
            session_id="abc-123",
            session_dir=None,
            parent_id="parent-456",
            forked_from_turn=2,
            message_count=4,
            messages=[],
        )
        assert result.session_id == "abc-123"
        assert result.session_dir is None
        assert result.parent_id == "parent-456"
        assert result.forked_from_turn == 2
        assert result.message_count == 4
        assert result.messages == []
        assert result.events_count == 0

    def test_events_count_defaults_to_zero(self):
        result = ForkResult(
            session_id="x", session_dir=None, parent_id="p",
            forked_from_turn=1, message_count=1,
        )
        assert result.events_count == 0


# ---------------------------------------------------------------------------
# fork_session_in_memory — empty conversation
# ---------------------------------------------------------------------------


class TestForkEmptyConversation:
    def test_empty_messages_returns_fork_result(self):
        result = fork_session_in_memory([])
        assert isinstance(result, ForkResult)

    def test_empty_messages_session_dir_none(self):
        result = fork_session_in_memory([])
        assert result.session_dir is None

    def test_empty_messages_message_count_zero(self):
        result = fork_session_in_memory([])
        assert result.message_count == 0

    def test_empty_messages_messages_empty_list(self):
        result = fork_session_in_memory([])
        assert result.messages == []

    def test_empty_messages_forked_from_turn_zero(self):
        result = fork_session_in_memory([])
        assert result.forked_from_turn == 0

    def test_empty_messages_session_id_is_uuid(self):
        result = fork_session_in_memory([])
        assert uuid.UUID(result.session_id)  # should not raise

    def test_empty_messages_unknown_parent_id(self):
        result = fork_session_in_memory([])
        assert result.parent_id == "unknown"

    def test_empty_messages_custom_parent_id(self):
        result = fork_session_in_memory([], parent_id="my-parent")
        assert result.parent_id == "my-parent"


# ---------------------------------------------------------------------------
# fork_session_in_memory — turn selection
# ---------------------------------------------------------------------------


class TestForkAtTurn:
    def test_fork_at_turn_1_includes_only_first_qa(self):
        msgs = make_conversation(("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3"))
        result = fork_session_in_memory(msgs, turn=1)
        assert result.forked_from_turn == 1
        assert result.messages is not None
        # Should include user(Q1) and assistant(A1) only
        assert len(result.messages) == 2
        assert result.messages[0]["content"] == "Q1"
        assert result.messages[1]["content"] == "A1"

    def test_fork_at_turn_2_includes_first_two_turns(self):
        msgs = make_conversation(("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3"))
        result = fork_session_in_memory(msgs, turn=2)
        assert result.forked_from_turn == 2
        assert result.messages is not None
        assert len(result.messages) == 4
        contents = [m["content"] for m in result.messages]
        assert "Q1" in contents
        assert "A1" in contents
        assert "Q2" in contents
        assert "A2" in contents
        assert "Q3" not in contents

    def test_fork_at_last_turn_returns_all_messages(self):
        msgs = make_conversation(("Q1", "A1"), ("Q2", "A2"))
        result = fork_session_in_memory(msgs, turn=2)
        assert result.messages == msgs

    def test_fork_with_turn_none_returns_all(self):
        msgs = make_conversation(("Q1", "A1"), ("Q2", "A2"))
        result = fork_session_in_memory(msgs, turn=None)
        assert result.messages == msgs

    def test_fork_message_count_matches_messages_len(self):
        msgs = make_conversation(("Q1", "A1"), ("Q2", "A2"))
        result = fork_session_in_memory(msgs, turn=1)
        assert result.message_count == len(result.messages)

    def test_fork_session_id_is_unique(self):
        msgs = make_conversation(("Q", "A"))
        r1 = fork_session_in_memory(msgs, turn=1)
        r2 = fork_session_in_memory(msgs, turn=1)
        assert r1.session_id != r2.session_id

    def test_fork_session_id_is_valid_uuid(self):
        msgs = make_conversation(("Q", "A"))
        result = fork_session_in_memory(msgs, turn=1)
        uuid.UUID(result.session_id)  # should not raise

    def test_fork_session_dir_is_none_for_in_memory(self):
        msgs = make_conversation(("Q", "A"))
        result = fork_session_in_memory(msgs, turn=1)
        assert result.session_dir is None

    def test_parent_id_default_unknown(self):
        msgs = make_conversation(("Q", "A"))
        result = fork_session_in_memory(msgs, turn=1)
        assert result.parent_id == "unknown"

    def test_parent_id_custom(self):
        msgs = make_conversation(("Q", "A"))
        result = fork_session_in_memory(msgs, turn=1, parent_id="session-abc")
        assert result.parent_id == "session-abc"

    def test_fork_does_not_mutate_original(self):
        msgs = make_conversation(("Q1", "A1"), ("Q2", "A2"))
        original_len = len(msgs)
        fork_session_in_memory(msgs, turn=1)
        assert len(msgs) == original_len

    def test_turn_out_of_range_raises_value_error(self):
        msgs = make_conversation(("Q1", "A1"))  # only 1 turn
        with pytest.raises(ValueError):
            fork_session_in_memory(msgs, turn=5)

    def test_single_turn_conversation_fork_at_1(self):
        msgs = [user("Only question"), assistant("Only answer")]
        result = fork_session_in_memory(msgs, turn=1)
        assert result.message_count == 2
        assert result.forked_from_turn == 1

    def test_conversation_with_only_user_messages(self):
        msgs = [user("Q1"), user("Q2")]
        # Both are user messages — 2 turns; fork at turn 1 → just Q1
        result = fork_session_in_memory(msgs, turn=1)
        assert result.message_count == 1
        assert result.messages[0]["content"] == "Q1"

    def test_assistant_only_messages_treated_as_empty(self):
        msgs = [assistant("A1"), assistant("A2")]
        result = fork_session_in_memory(msgs)
        assert result.message_count == 0
        assert result.messages == []


# ---------------------------------------------------------------------------
# fork_session_in_memory — orphaned tool handling
# ---------------------------------------------------------------------------


class TestForkOrphanedTools:
    def _make_openai_tool_call(self, call_id: str, fn_name: str) -> dict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": fn_name, "arguments": "{}"},
                }
            ],
        }

    def test_complete_mode_adds_synthetic_result(self):
        msgs = [
            user("Q"),
            self._make_openai_tool_call("id1", "fn"),
            # No tool result
        ]
        result = fork_session_in_memory(msgs, turn=1, handle_orphaned_tools="complete")
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1

    def test_error_mode_raises_on_orphaned(self):
        msgs = [
            user("Q"),
            self._make_openai_tool_call("id1", "fn"),
        ]
        with pytest.raises(ValueError, match="[Oo]rphaned"):
            fork_session_in_memory(msgs, turn=1, handle_orphaned_tools="error")

    def test_remove_mode_strips_orphaned_tool_calls(self):
        msgs = [
            user("Q"),
            self._make_openai_tool_call("id1", "fn"),
        ]
        result = fork_session_in_memory(msgs, turn=1, handle_orphaned_tools="remove")
        # After removal, no tool_calls should remain in assistant message
        for msg in result.messages:
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                assert tc.get("id") != "id1"
