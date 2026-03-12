"""Tests for amplifier_lib.tracing — generate_sub_session_id."""

from __future__ import annotations

import re

import pytest

from amplifier_lib.tracing import (
    _DEFAULT_PARENT_SPAN,
    _SPAN_HEX_LEN,
    generate_sub_session_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FORMAT_RE = re.compile(r"^([0-9a-f]{16})-([0-9a-f]{16})_(.+)$")


def _parse(session_id: str) -> tuple[str, str, str]:
    """Return (parent_span, child_span, name) from a session ID."""
    m = _FORMAT_RE.fullmatch(session_id)
    assert m is not None, f"Session ID does not match expected format: {session_id!r}"
    return m.group(1), m.group(2), m.group(3)


# ---------------------------------------------------------------------------
# Format / default behaviour
# ---------------------------------------------------------------------------


class TestDefaultBehaviour:
    def test_returns_string(self):
        result = generate_sub_session_id()
        assert isinstance(result, str)

    def test_format_no_args(self):
        result = generate_sub_session_id()
        parent_span, child_span, name = _parse(result)
        assert len(parent_span) == _SPAN_HEX_LEN
        assert len(child_span) == _SPAN_HEX_LEN
        assert name == "agent"

    def test_default_parent_span_is_all_zeros(self):
        result = generate_sub_session_id()
        parent_span, _, _ = _parse(result)
        assert parent_span == _DEFAULT_PARENT_SPAN

    def test_child_span_is_hex(self):
        result = generate_sub_session_id()
        _, child_span, _ = _parse(result)
        assert all(c in "0123456789abcdef" for c in child_span)

    def test_unique_child_spans(self):
        spans = {_parse(generate_sub_session_id())[1] for _ in range(20)}
        assert len(spans) == 20, "Child spans should be unique across calls"


# ---------------------------------------------------------------------------
# Agent name sanitisation
# ---------------------------------------------------------------------------


class TestAgentNameSanitisation:
    def test_custom_name_preserved_lowercase(self):
        _, _, name = _parse(generate_sub_session_id(agent_name="myagent"))
        assert name == "myagent"

    def test_uppercase_lowercased(self):
        _, _, name = _parse(generate_sub_session_id(agent_name="MyAgent"))
        assert name == "myagent"

    def test_spaces_replaced_by_hyphens(self):
        _, _, name = _parse(generate_sub_session_id(agent_name="my agent"))
        assert name == "my-agent"

    def test_special_chars_replaced_by_hyphens(self):
        _, _, name = _parse(generate_sub_session_id(agent_name="my@agent!"))
        assert name == "my-agent"

    def test_multiple_consecutive_special_chars_collapsed(self):
        _, _, name = _parse(generate_sub_session_id(agent_name="my--agent"))
        assert name == "my-agent"

    def test_mixed_special_chars_collapsed(self):
        _, _, name = _parse(generate_sub_session_id(agent_name="my  !!  agent"))
        assert name == "my-agent"

    def test_leading_trailing_hyphens_stripped(self):
        _, _, name = _parse(generate_sub_session_id(agent_name="--my-agent--"))
        assert name == "my-agent"

    def test_none_agent_name_defaults_to_agent(self):
        _, _, name = _parse(generate_sub_session_id(agent_name=None))
        assert name == "agent"

    def test_empty_string_agent_name_defaults_to_agent(self):
        _, _, name = _parse(generate_sub_session_id(agent_name=""))
        assert name == "agent"

    def test_whitespace_only_agent_name_defaults_to_agent(self):
        _, _, name = _parse(generate_sub_session_id(agent_name="   "))
        assert name == "agent"

    def test_special_chars_only_defaults_to_agent(self):
        _, _, name = _parse(generate_sub_session_id(agent_name="!!!"))
        assert name == "agent"

    def test_numbers_preserved(self):
        _, _, name = _parse(generate_sub_session_id(agent_name="agent42"))
        assert name == "agent42"

    def test_hyphens_in_name_preserved(self):
        _, _, name = _parse(generate_sub_session_id(agent_name="code-reviewer"))
        assert name == "code-reviewer"


# ---------------------------------------------------------------------------
# Parent session ID extraction
# ---------------------------------------------------------------------------


class TestParentSessionId:
    def _make_parent(self, parent_span: str, child_span: str, name: str = "agent") -> str:
        return f"{parent_span}-{child_span}_{name}"

    def test_extracts_child_span_as_parent(self):
        child_span = "abcdef1234567890"
        parent_id = self._make_parent(_DEFAULT_PARENT_SPAN, child_span)
        result = generate_sub_session_id(parent_session_id=parent_id)
        extracted_parent, _, _ = _parse(result)
        assert extracted_parent == child_span

    def test_no_match_falls_back_to_zeros(self):
        result = generate_sub_session_id(parent_session_id="not-a-valid-session-id")
        parent_span, _, _ = _parse(result)
        assert parent_span == _DEFAULT_PARENT_SPAN

    def test_empty_parent_session_id_falls_back_to_zeros(self):
        result = generate_sub_session_id(parent_session_id="")
        parent_span, _, _ = _parse(result)
        assert parent_span == _DEFAULT_PARENT_SPAN

    def test_none_parent_session_id_falls_back_to_zeros(self):
        result = generate_sub_session_id(parent_session_id=None)
        parent_span, _, _ = _parse(result)
        assert parent_span == _DEFAULT_PARENT_SPAN

    def test_chained_lineage(self):
        """A grandchild's parent_span should equal the child's child_span."""
        child = generate_sub_session_id(agent_name="child")
        _, child_child_span, _ = _parse(child)

        grandchild = generate_sub_session_id(agent_name="grandchild", parent_session_id=child)
        grandchild_parent_span, _, _ = _parse(grandchild)

        assert grandchild_parent_span == child_child_span


# ---------------------------------------------------------------------------
# Parent trace ID extraction
# ---------------------------------------------------------------------------


class TestParentTraceId:
    _VALID_TRACE_ID = "aabbccdd" + "1122334455667788" + "eeff0011"

    def test_extracts_middle_16_chars_when_no_parent_session(self):
        result = generate_sub_session_id(parent_trace_id=self._VALID_TRACE_ID)
        parent_span, _, _ = _parse(result)
        # Characters [8:24] of trace ID
        expected = self._VALID_TRACE_ID[8:24]
        assert parent_span == expected

    def test_invalid_trace_id_ignored(self):
        result = generate_sub_session_id(parent_trace_id="not-a-valid-trace-id")
        parent_span, _, _ = _parse(result)
        assert parent_span == _DEFAULT_PARENT_SPAN

    def test_trace_id_wrong_length_ignored(self):
        result = generate_sub_session_id(parent_trace_id="abc123")
        parent_span, _, _ = _parse(result)
        assert parent_span == _DEFAULT_PARENT_SPAN

    def test_trace_id_with_uppercase_ignored(self):
        uppercase_trace = "AABBCCDD1122334455667788EEFF0011"
        result = generate_sub_session_id(parent_trace_id=uppercase_trace)
        parent_span, _, _ = _parse(result)
        assert parent_span == _DEFAULT_PARENT_SPAN

    def test_none_trace_id_falls_back_to_zeros(self):
        result = generate_sub_session_id(parent_trace_id=None)
        parent_span, _, _ = _parse(result)
        assert parent_span == _DEFAULT_PARENT_SPAN


# ---------------------------------------------------------------------------
# Precedence: parent_session_id over parent_trace_id
# ---------------------------------------------------------------------------


class TestPrecedence:
    _VALID_TRACE_ID = "aabbccdd1122334455667788eeff0011"

    def _make_parent(self, child_span: str) -> str:
        return f"{_DEFAULT_PARENT_SPAN}-{child_span}_agent"

    def test_parent_session_id_takes_precedence_over_trace_id(self):
        child_span = "deadbeef12345678"
        parent_id = self._make_parent(child_span)

        result = generate_sub_session_id(
            parent_session_id=parent_id,
            parent_trace_id=self._VALID_TRACE_ID,
        )
        extracted_parent, _, _ = _parse(result)
        # Should use child_span from parent_session_id, NOT the trace_id middle bytes
        assert extracted_parent == child_span

    def test_trace_id_used_when_parent_session_id_not_matching(self):
        """Falls through to trace_id when parent_session_id doesn't match pattern."""
        result = generate_sub_session_id(
            parent_session_id="invalid-session-id",
            parent_trace_id=self._VALID_TRACE_ID,
        )
        extracted_parent, _, _ = _parse(result)
        expected = self._VALID_TRACE_ID[8:24]
        assert extracted_parent == expected
