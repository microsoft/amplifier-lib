"""Tests for amplifier_lib.serialization — sanitize_message() and sanitize_for_json()."""

import json
from typing import Any

import pytest

from amplifier_lib.serialization import sanitize_for_json, sanitize_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleObj:
    """Object that exposes state via __dict__."""

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ModelDumpObj:
    """Mimics a Pydantic v2 model with __slots__ (no __dict__) but has model_dump.

    Using __slots__ prevents __dict__ from being created, which means
    sanitize_for_json will skip the __dict__ path and reach model_dump.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return self._data


class TextObj:
    """Object with a .text attribute (used for thinking_block)."""

    def __init__(self, text: str):
        self.text = text


# ---------------------------------------------------------------------------
# sanitize_for_json
# ---------------------------------------------------------------------------


class TestSanitizeForJson:
    def test_none_passes_through(self):
        assert sanitize_for_json(None) is None

    @pytest.mark.parametrize("value", [True, False, 0, 1, 3.14, -42, "hello", ""])
    def test_primitives_pass_through(self, value):
        assert sanitize_for_json(value) == value

    def test_plain_dict_passes_through(self):
        d = {"a": 1, "b": "two", "c": True}
        assert sanitize_for_json(d) == d

    def test_dict_with_none_values_filtered(self):
        # Non-serializable values get converted to None and then filtered
        result = sanitize_for_json({"keep": "yes", "drop": object()})
        assert "keep" in result
        assert "drop" not in result

    def test_nested_dict(self):
        d = {"outer": {"inner": 42}}
        assert sanitize_for_json(d) == {"outer": {"inner": 42}}

    def test_plain_list_passes_through(self):
        lst = [1, "two", 3.0, True, None]
        # None items are filtered from lists
        result = sanitize_for_json(lst)
        assert 1 in result
        assert "two" in result

    def test_list_none_items_filtered(self):
        result = sanitize_for_json([1, object(), 3])
        # object() is not serializable → None → filtered
        assert result == [1, 3]

    def test_tuple_converted_to_list(self):
        result = sanitize_for_json((1, 2, 3))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_object_with_dict(self):
        obj = SimpleObj(name="alice", age=30)
        result = sanitize_for_json(obj)
        assert result == {"name": "alice", "age": 30}

    def test_object_with_model_dump(self):
        obj = ModelDumpObj({"x": 1, "y": "hello"})
        result = sanitize_for_json(obj)
        assert result == {"x": 1, "y": "hello"}

    def test_non_serializable_returns_none(self):
        result = sanitize_for_json(object())
        assert result is None

    def test_depth_limit_returns_none(self):
        result = sanitize_for_json({"key": "value"}, max_depth=0)
        assert result is None

    def test_depth_limit_one_processes_primitives(self):
        # max_depth=1 means we process the top value but not nested structures
        # For a primitive, max_depth > 0 works fine
        assert sanitize_for_json(42, max_depth=1) == 42

    def test_result_is_json_serializable(self):
        data = {
            "str": "hello",
            "num": 42,
            "float": 3.14,
            "bool": True,
            "none_val": None,
            "list": [1, 2, 3],
            "nested": {"a": "b"},
        }
        result = sanitize_for_json(data)
        # Should not raise
        json.dumps(result)

    def test_deeply_nested_structure(self):
        d: dict = {}
        current = d
        for i in range(10):
            current["level"] = {}
            current = current["level"]
        current["value"] = "deep"
        result = sanitize_for_json(d)
        assert result is not None  # within default depth limit

    def test_list_of_dicts(self):
        data = [{"a": 1}, {"b": 2}]
        result = sanitize_for_json(data)
        assert result == [{"a": 1}, {"b": 2}]


# ---------------------------------------------------------------------------
# sanitize_message
# ---------------------------------------------------------------------------


class TestSanitizeMessage:
    def test_plain_dict_passes_through(self):
        msg = {"role": "user", "content": "Hello!"}
        result = sanitize_message(msg)
        assert result == {"role": "user", "content": "Hello!"}

    def test_empty_dict(self):
        assert sanitize_message({}) == {}

    # --- thinking_block ---

    def test_thinking_block_dict_extracted(self):
        msg = {
            "role": "assistant",
            "thinking_block": {"type": "thinking", "text": "Let me think..."},
        }
        result = sanitize_message(msg)
        assert "thinking_block" not in result
        assert result["thinking_text"] == "Let me think..."

    def test_thinking_block_object_with_text_attr(self):
        obj = TextObj("my thoughts")
        msg = {"role": "assistant", "thinking_block": obj}
        result = sanitize_message(msg)
        assert "thinking_block" not in result
        assert result["thinking_text"] == "my thoughts"

    def test_thinking_block_dict_without_text_key_skipped(self):
        """If thinking_block dict has no 'text', no thinking_text is produced."""
        msg = {"role": "assistant", "thinking_block": {"type": "thinking"}}
        result = sanitize_message(msg)
        assert "thinking_block" not in result
        assert "thinking_text" not in result

    def test_thinking_block_preserves_other_fields(self):
        msg = {
            "role": "assistant",
            "content": "Response text",
            "thinking_block": {"text": "reasoning"},
        }
        result = sanitize_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Response text"
        assert result["thinking_text"] == "reasoning"

    # --- content_blocks ---

    def test_content_blocks_skipped(self):
        msg = {
            "role": "assistant",
            "content": "text",
            "content_blocks": [{"type": "text", "text": "raw api object"}],
        }
        result = sanitize_message(msg)
        assert "content_blocks" not in result
        assert result["content"] == "text"

    def test_content_blocks_only_skipped(self):
        """A message with only content_blocks results in an empty dict."""
        msg = {"content_blocks": [{"raw": "data"}]}
        result = sanitize_message(msg)
        assert result == {}

    # --- nested sanitization ---

    def test_nested_non_serializable_cleaned(self):
        msg = {
            "role": "user",
            "metadata": {"obj": object(), "value": 42},
        }
        result = sanitize_message(msg)
        assert result["metadata"] == {"value": 42}

    def test_nested_list_with_bad_values(self):
        msg = {
            "role": "assistant",
            "tool_calls": [{"id": "call_1", "obj": object()}, {"id": "call_2"}],
        }
        result = sanitize_message(msg)
        # Non-serializable nested values are filtered
        assert "tool_calls" in result

    # --- non-dict input ---

    def test_non_dict_input_returns_empty_dict_for_non_convertible(self):
        result = sanitize_message("not a dict")  # type: ignore[arg-type]
        # str → sanitize_for_json returns str → not a dict → {}
        assert result == {}

    def test_none_input_returns_empty_dict(self):
        result = sanitize_message(None)  # type: ignore[arg-type]
        assert result == {}

    def test_values_with_none_result_omitted(self):
        """Keys whose values sanitize to None are omitted from output."""
        msg = {"role": "user", "bad_field": object()}
        result = sanitize_message(msg)
        assert "bad_field" not in result
        assert result["role"] == "user"

    # --- full round-trip ---

    def test_result_is_json_serializable(self):
        msg = {
            "role": "assistant",
            "content": "Sure, let me help!",
            "thinking_block": {"text": "Planning..."},
            "content_blocks": [object()],
            "metadata": {"tokens": 42, "model": "claude-3"},
        }
        result = sanitize_message(msg)
        # Should not raise
        json.dumps(result)
        assert result["thinking_text"] == "Planning..."
        assert "content_blocks" not in result

    def test_list_content_sanitized(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
        }
        result = sanitize_message(msg)
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2

    def test_thinking_block_none_value_noop(self):
        """thinking_block: None should produce no thinking_text."""
        msg = {"role": "assistant", "thinking_block": None, "content": "hi"}
        result = sanitize_message(msg)
        assert "thinking_text" not in result
        assert result["content"] == "hi"
