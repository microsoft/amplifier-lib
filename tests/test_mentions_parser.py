"""Tests for amplifier_lib.mentions.parser — parse_mentions()."""

from __future__ import annotations

import pytest

from amplifier_lib.mentions.parser import parse_mentions


# ---------------------------------------------------------------------------
# Basic extraction
# ---------------------------------------------------------------------------


def test_single_mention_bare_path():
    result = parse_mentions("Check @AGENTS.md for details")
    assert result == ["@AGENTS.md"]


def test_single_mention_namespace_colon():
    result = parse_mentions("See @foundation:context/kernel for info")
    assert result == ["@foundation:context/kernel"]


def test_multiple_distinct_mentions():
    result = parse_mentions("Use @foundation:tools and @recipes:cook for this task")
    assert result == ["@foundation:tools", "@recipes:cook"]


def test_mentions_extracted_in_order():
    text = "First @alpha then @beta and finally @gamma"
    result = parse_mentions(text)
    assert result == ["@alpha", "@beta", "@gamma"]


# ---------------------------------------------------------------------------
# Unique / deduplication
# ---------------------------------------------------------------------------


def test_duplicate_mentions_returned_once():
    text = "@AGENTS.md is great. Also see @AGENTS.md for more details."
    result = parse_mentions(text)
    assert result == ["@AGENTS.md"]
    assert len(result) == 1


def test_duplicate_preserves_first_occurrence_order():
    text = "@alpha @beta @alpha @gamma @beta"
    result = parse_mentions(text)
    assert result == ["@alpha", "@beta", "@gamma"]


# ---------------------------------------------------------------------------
# Path variants
# ---------------------------------------------------------------------------


def test_dot_slash_relative_path():
    result = parse_mentions("Load @./relative/path.md here")
    assert result == ["@./relative/path.md"]


def test_tilde_home_path():
    result = parse_mentions("See @~/my/config.md please")
    assert result == ["@~/my/config.md"]


def test_namespace_colon_path_with_slashes():
    result = parse_mentions("Context: @bundle-name:ctx/KERNEL.md")
    assert result == ["@bundle-name:ctx/KERNEL.md"]


def test_mention_with_hyphens_in_namespace():
    result = parse_mentions("Load @my-bundle:thing")
    assert result == ["@my-bundle:thing"]


def test_mention_with_underscores():
    result = parse_mentions("Use @my_bundle:my_context")
    assert result == ["@my_bundle:my_context"]


# ---------------------------------------------------------------------------
# Email address handling (actual behavior)
# ---------------------------------------------------------------------------
#
# Note: The regex uses a negative lookahead to attempt email exclusion, but
# it only prevents double-@ patterns (e.g. abc@def@ghi.com -> @ghi.com skipped
# for the first @, but @ghi.com is still extracted from the last @).
# For plain user@domain.com addresses, the DOMAIN PART (@domain.com) IS
# captured because the lookahead checks what follows the @, and "domain.com"
# doesn't itself contain another @, so the lookahead doesn't trigger.
# Tests here document the actual behaviour.


def test_email_domain_part_extracted_as_mention():
    """The domain portion after @ in an email is captured as a mention."""
    result = parse_mentions("Send to user@example.com for help")
    assert result == ["@example.com"]


def test_email_mixed_with_bare_mention():
    """Both domain-part and an explicit mention are extracted."""
    result = parse_mentions("Email user@example.com or use @AGENTS.md")
    assert "@AGENTS.md" in result
    # Domain part is also captured
    assert "@example.com" in result


def test_multiple_email_domain_parts_extracted():
    text = "Contact a@b.com and c@d.org for support"
    result = parse_mentions(text)
    # Domain parts from both emails are extracted
    assert "@b.com" in result
    assert "@d.org" in result


# ---------------------------------------------------------------------------
# Inline code block exclusion
# ---------------------------------------------------------------------------


def test_mention_in_inline_code_excluded():
    result = parse_mentions("The `@AGENTS.md` file is referenced here")
    assert result == []


def test_mention_outside_inline_code_extracted():
    result = parse_mentions("Use `code` and then @AGENTS.md")
    assert result == ["@AGENTS.md"]


def test_mention_inline_code_and_bare_mixed():
    result = parse_mentions("Try `@foundation:skip` but do use @foundation:keep")
    assert result == ["@foundation:keep"]


# ---------------------------------------------------------------------------
# Fenced code block exclusion
# ---------------------------------------------------------------------------


def test_mention_in_fenced_block_excluded():
    text = "Before\n```\n@AGENTS.md\n```\nAfter"
    result = parse_mentions(text)
    assert result == []


def test_mention_in_fenced_block_with_language_excluded():
    text = "Here:\n```python\nimport @something\n```\nDone"
    result = parse_mentions(text)
    assert result == []


def test_mention_outside_fenced_block_extracted():
    text = "Use @outside here\n```\n@inside\n```\nEnd"
    result = parse_mentions(text)
    assert result == ["@outside"]


def test_mention_after_fenced_block_extracted():
    text = "```\n@ignore\n```\nNow use @AGENTS.md"
    result = parse_mentions(text)
    assert result == ["@AGENTS.md"]


def test_multiple_fenced_blocks():
    text = "```\n@skip1\n```\nUse @keep\n```\n@skip2\n```"
    result = parse_mentions(text)
    assert result == ["@keep"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_string():
    result = parse_mentions("")
    assert result == []


def test_text_with_no_mentions():
    result = parse_mentions("This is plain text without any mentions.")
    assert result == []


def test_at_sign_alone_not_extracted():
    # Bare @ with no alphanumeric following
    result = parse_mentions("Email @ domain is invalid")
    assert result == []


def test_mention_at_start_of_string():
    result = parse_mentions("@AGENTS.md is the entrypoint")
    assert result == ["@AGENTS.md"]


def test_mention_at_end_of_string():
    result = parse_mentions("The entrypoint is @AGENTS.md")
    assert result == ["@AGENTS.md"]


def test_deeply_nested_namespace_path():
    result = parse_mentions("Use @ns:a/b/c/d.md for this")
    assert result == ["@ns:a/b/c/d.md"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("@foundation:kernel", ["@foundation:kernel"]),
        ("@./local.md", ["@./local.md"]),
        ("@~/home/file.md", ["@~/home/file.md"]),
        # Email: domain part is extracted (see email-handling tests above)
        ("user@example.com", ["@example.com"]),
        ("`@hidden`", []),
        ("", []),
    ],
)
def test_parametrized_mention_variants(text, expected):
    assert parse_mentions(text) == expected
