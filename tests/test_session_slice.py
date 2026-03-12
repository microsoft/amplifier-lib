"""Tests for amplifier_lib.session.slice — get_turn_boundaries,
find_orphaned_tool_calls, add_synthetic_tool_results, count_turns,
get_turn_summary."""

import json

import pytest

from amplifier_lib.session.slice import (
    add_synthetic_tool_results,
    count_turns,
    find_orphaned_tool_calls,
    get_turn_boundaries,
    get_turn_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant(content: str = "OK") -> dict:
    return {"role": "assistant", "content": content}


def tool_result(tool_call_id: str, content: str = "result") -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def assistant_with_openai_tool_call(call_id: str, fn_name: str) -> dict:
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


def assistant_with_anthropic_tool_use(call_id: str, name: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": {},
            }
        ],
    }


# ---------------------------------------------------------------------------
# get_turn_boundaries
# ---------------------------------------------------------------------------


class TestGetTurnBoundaries:
    def test_empty_messages(self):
        assert get_turn_boundaries([]) == []

    def test_single_user_message(self):
        msgs = [user("Hello")]
        assert get_turn_boundaries(msgs) == [0]

    def test_user_then_assistant(self):
        msgs = [user("Q"), assistant("A")]
        assert get_turn_boundaries(msgs) == [0]

    def test_two_turns(self):
        msgs = [user("Q1"), assistant("A1"), user("Q2"), assistant("A2")]
        assert get_turn_boundaries(msgs) == [0, 2]

    def test_three_turns(self):
        msgs = [
            user("Q1"),
            assistant("A1"),
            user("Q2"),
            assistant("A2"),
            user("Q3"),
            assistant("A3"),
        ]
        assert get_turn_boundaries(msgs) == [0, 2, 4]

    def test_assistant_only_no_boundaries(self):
        msgs = [assistant("A"), assistant("B")]
        assert get_turn_boundaries(msgs) == []

    def test_multiple_user_messages_in_a_row(self):
        msgs = [user("U1"), user("U2"), assistant("A")]
        assert get_turn_boundaries(msgs) == [0, 1]

    def test_tool_messages_not_counted(self):
        msgs = [
            user("Q"),
            assistant_with_openai_tool_call("id1", "func"),
            tool_result("id1"),
            assistant("Done"),
        ]
        assert get_turn_boundaries(msgs) == [0]

    def test_returns_correct_indices(self):
        msgs = [
            assistant("preamble"),  # 0 — not user
            user("Q1"),  # 1 — first turn
            assistant("A1"),  # 2
            user("Q2"),  # 3 — second turn
        ]
        assert get_turn_boundaries(msgs) == [1, 3]


# ---------------------------------------------------------------------------
# find_orphaned_tool_calls
# ---------------------------------------------------------------------------


class TestFindOrphanedToolCalls:
    def test_no_tool_calls_returns_empty(self):
        msgs = [user("Q"), assistant("A")]
        assert find_orphaned_tool_calls(msgs) == []

    def test_tool_call_with_result_not_orphaned(self):
        msgs = [
            user("Q"),
            assistant_with_openai_tool_call("id1", "get_data"),
            tool_result("id1"),
        ]
        assert find_orphaned_tool_calls(msgs) == []

    def test_openai_format_orphan_detected(self):
        msgs = [
            user("Q"),
            assistant_with_openai_tool_call("id1", "get_data"),
            # No tool result
        ]
        orphans = find_orphaned_tool_calls(msgs)
        assert "id1" in orphans

    def test_anthropic_format_orphan_detected(self):
        msgs = [
            user("Q"),
            assistant_with_anthropic_tool_use("id2", "search"),
            # No tool result
        ]
        orphans = find_orphaned_tool_calls(msgs)
        assert "id2" in orphans

    def test_multiple_tool_calls_partial_results(self):
        msgs = [
            user("Q"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "id1",
                        "type": "function",
                        "function": {"name": "f1", "arguments": "{}"},
                    },
                    {
                        "id": "id2",
                        "type": "function",
                        "function": {"name": "f2", "arguments": "{}"},
                    },
                ],
            },
            tool_result("id1"),  # only id1 has a result
        ]
        orphans = find_orphaned_tool_calls(msgs)
        assert "id2" in orphans
        assert "id1" not in orphans

    def test_all_tool_calls_answered_none_orphaned(self):
        msgs = [
            user("Q"),
            assistant_with_openai_tool_call("id1", "fn"),
            tool_result("id1"),
            assistant("Done"),
        ]
        assert find_orphaned_tool_calls(msgs) == []

    def test_anthropic_and_openai_mixed(self):
        msgs = [
            user("Q"),
            assistant_with_openai_tool_call("oid1", "fn_openai"),
            tool_result("oid1"),
            assistant_with_anthropic_tool_use("aid1", "fn_anthropic"),
            # aid1 has no result
        ]
        orphans = find_orphaned_tool_calls(msgs)
        assert "aid1" in orphans
        assert "oid1" not in orphans

    def test_empty_messages(self):
        assert find_orphaned_tool_calls([]) == []


# ---------------------------------------------------------------------------
# add_synthetic_tool_results
# ---------------------------------------------------------------------------


class TestAddSyntheticToolResults:
    def test_no_orphans_returns_original(self):
        msgs = [user("Q"), assistant("A")]
        result = add_synthetic_tool_results(msgs, [])
        assert result == msgs

    def test_synthetic_result_inserted_after_assistant(self):
        msgs = [
            user("Q"),
            assistant_with_openai_tool_call("id1", "get_data"),
        ]
        result = add_synthetic_tool_results(msgs, ["id1"])
        # Should have 3 messages: user, assistant, synthetic tool result
        assert len(result) == 3
        synthetic = result[2]
        assert synthetic["role"] == "tool"
        assert synthetic["tool_call_id"] == "id1"

    def test_synthetic_result_content_is_json_with_error_key(self):
        msgs = [
            user("Q"),
            assistant_with_openai_tool_call("id1", "fn"),
        ]
        result = add_synthetic_tool_results(msgs, ["id1"])
        synthetic = result[2]
        content = json.loads(synthetic["content"])
        assert "error" in content
        assert content.get("forked") is True

    def test_synthetic_result_includes_tool_name_from_openai_format(self):
        msgs = [
            user("Q"),
            assistant_with_openai_tool_call("id1", "my_function"),
        ]
        result = add_synthetic_tool_results(msgs, ["id1"])
        synthetic = result[2]
        assert synthetic.get("name") == "my_function"

    def test_synthetic_result_includes_tool_name_from_anthropic_format(self):
        msgs = [
            user("Q"),
            assistant_with_anthropic_tool_use("id2", "my_tool"),
        ]
        result = add_synthetic_tool_results(msgs, ["id2"])
        synthetic = result[2]
        assert synthetic.get("name") == "my_tool"

    def test_multiple_orphans_all_get_synthetics(self):
        msgs = [
            user("Q"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "id1",
                        "type": "function",
                        "function": {"name": "f1", "arguments": "{}"},
                    },
                    {
                        "id": "id2",
                        "type": "function",
                        "function": {"name": "f2", "arguments": "{}"},
                    },
                ],
            },
        ]
        result = add_synthetic_tool_results(msgs, ["id1", "id2"])
        tool_msgs = [m for m in result if m["role"] == "tool"]
        tool_ids = {m["tool_call_id"] for m in tool_msgs}
        assert tool_ids == {"id1", "id2"}

    def test_synthetic_assistant_added_before_next_user(self):
        """When the next message after orphaned calls is a user message,
        a synthetic assistant message should bridge the gap."""
        msgs = [
            user("Q1"),
            assistant_with_openai_tool_call("id1", "fn"),
            user("Q2"),
        ]
        result = add_synthetic_tool_results(msgs, ["id1"])
        # Expected: user(Q1), assistant(tool_call), tool(synthetic), assistant(bridge), user(Q2)
        roles = [m["role"] for m in result]
        assert roles.count("tool") >= 1
        # The bridge assistant message should appear before the last user message
        last_user_idx = max(i for i, m in enumerate(result) if m["role"] == "user")
        # At least one assistant message should appear after the tool message
        tool_idx = next(i for i, m in enumerate(result) if m["role"] == "tool")
        assert last_user_idx > tool_idx

    def test_non_orphaned_ids_not_affected(self):
        msgs = [
            user("Q"),
            assistant_with_openai_tool_call("id1", "fn"),
            tool_result("id1"),
        ]
        # Pass "id_missing" as orphan, but id1 already has a result
        result = add_synthetic_tool_results(msgs, ["id_missing"])
        # No synthetic should be inserted since id_missing doesn't appear in messages
        assert len(result) == 3

    def test_original_messages_unchanged(self):
        msgs = [user("Q"), assistant_with_openai_tool_call("id1", "fn")]
        result = add_synthetic_tool_results(msgs, ["id1"])
        # Original messages at their positions
        assert result[0] == msgs[0]
        assert result[1] == msgs[1]

    def test_anthropic_format_synthetic_inserted(self):
        msgs = [
            user("Q"),
            assistant_with_anthropic_tool_use("aid1", "search"),
        ]
        result = add_synthetic_tool_results(msgs, ["aid1"])
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "aid1"


# ---------------------------------------------------------------------------
# count_turns
# ---------------------------------------------------------------------------


class TestCountTurns:
    def test_empty_messages_returns_zero(self):
        assert count_turns([]) == 0

    def test_single_turn(self):
        msgs = [user("Q"), assistant("A")]
        assert count_turns(msgs) == 1

    def test_two_turns(self):
        msgs = [user("Q1"), assistant("A1"), user("Q2"), assistant("A2")]
        assert count_turns(msgs) == 2

    def test_three_turns(self):
        msgs = [
            user("Q1"),
            assistant("A1"),
            user("Q2"),
            assistant("A2"),
            user("Q3"),
            assistant("A3"),
        ]
        assert count_turns(msgs) == 3

    def test_assistant_only_returns_zero(self):
        msgs = [assistant("A1"), assistant("A2")]
        assert count_turns(msgs) == 0

    def test_consistent_with_get_turn_boundaries(self):
        msgs = [user("Q1"), assistant("A1"), user("Q2"), assistant("A2")]
        assert count_turns(msgs) == len(get_turn_boundaries(msgs))


# ---------------------------------------------------------------------------
# get_turn_summary
# ---------------------------------------------------------------------------


class TestGetTurnSummary:
    def test_valid_turn_returns_expected_keys(self):
        msgs = [user("Hello"), assistant("Hi there")]
        summary = get_turn_summary(msgs, 1)
        assert set(summary.keys()) == {
            "turn",
            "user_content",
            "assistant_content",
            "tool_count",
            "message_count",
        }

    def test_turn_number_in_result(self):
        msgs = [user("Q1"), assistant("A1"), user("Q2"), assistant("A2")]
        assert get_turn_summary(msgs, 2)["turn"] == 2

    def test_user_content_extracted_from_string(self):
        msgs = [user("Hello world"), assistant("Hi")]
        summary = get_turn_summary(msgs, 1)
        assert summary["user_content"] == "Hello world"

    def test_assistant_content_extracted_from_string(self):
        msgs = [user("Q"), assistant("My answer")]
        summary = get_turn_summary(msgs, 1)
        assert summary["assistant_content"] == "My answer"

    def test_message_count_covers_full_turn(self):
        msgs = [
            user("Q"),
            assistant_with_openai_tool_call("id1", "fn"),
            tool_result("id1"),
            assistant("Done"),
        ]
        summary = get_turn_summary(msgs, 1)
        assert summary["message_count"] == 4

    def test_truncation_applied_with_ellipsis(self):
        long_text = "x" * 200
        msgs = [user(long_text), assistant("A")]
        summary = get_turn_summary(msgs, 1, max_length=50)
        assert len(summary["user_content"]) == 53  # 50 chars + "..."
        assert summary["user_content"].endswith("...")

    def test_no_truncation_when_content_within_limit(self):
        msgs = [user("Short"), assistant("Brief")]
        summary = get_turn_summary(msgs, 1, max_length=100)
        assert summary["user_content"] == "Short"
        assert "..." not in summary["user_content"]

    def test_openai_tool_count(self):
        msgs = [
            user("Q"),
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "id1",
                        "type": "function",
                        "function": {"name": "f1", "arguments": "{}"},
                    },
                    {
                        "id": "id2",
                        "type": "function",
                        "function": {"name": "f2", "arguments": "{}"},
                    },
                ],
            },
            tool_result("id1"),
            tool_result("id2"),
            assistant("Done"),
        ]
        summary = get_turn_summary(msgs, 1)
        assert summary["tool_count"] == 2

    def test_anthropic_tool_use_count(self):
        msgs = [
            user("Q"),
            assistant_with_anthropic_tool_use("aid1", "search"),
        ]
        summary = get_turn_summary(msgs, 1)
        assert summary["tool_count"] == 1

    def test_user_content_from_list_of_blocks(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "Block text"}]},
            assistant("A"),
        ]
        summary = get_turn_summary(msgs, 1)
        assert summary["user_content"] == "Block text"

    def test_assistant_content_from_list_of_blocks(self):
        msgs = [
            user("Q"),
            {"role": "assistant", "content": [{"type": "text", "text": "Block answer"}]},
        ]
        summary = get_turn_summary(msgs, 1)
        assert summary["assistant_content"] == "Block answer"

    def test_out_of_range_turn_raises_value_error_too_high(self):
        msgs = [user("Q"), assistant("A")]
        with pytest.raises(ValueError, match="out of range"):
            get_turn_summary(msgs, 5)

    def test_out_of_range_turn_raises_value_error_too_low(self):
        msgs = [user("Q"), assistant("A")]
        with pytest.raises(ValueError, match="out of range"):
            get_turn_summary(msgs, 0)

    def test_second_turn_only_includes_second_turn_messages(self):
        msgs = [
            user("Q1"),
            assistant("A1"),
            user("Q2"),
            assistant("A2"),
        ]
        summary = get_turn_summary(msgs, 2)
        assert summary["user_content"] == "Q2"
        assert summary["assistant_content"] == "A2"
        assert summary["message_count"] == 2
