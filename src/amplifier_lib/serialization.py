"""Serialization utilities for amplifier-lib.

Provides sanitization for safely persisting data that may contain
non-serializable objects (common with LLM API responses).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def sanitize_for_json(value: Any, *, max_depth: int = 50) -> Any:
    """Recursively sanitize a value to ensure it's JSON-serializable.

    Handles common cases from LLM responses:
    - Non-serializable objects (returns None or extracts useful text)
    - Nested dicts and lists
    - Objects with __dict__

    Args:
        value: Any value that may or may not be serializable.
        max_depth: Maximum recursion depth (prevents infinite loops).

    Returns:
        Sanitized value that's JSON-serializable.
    """
    if max_depth <= 0:
        return None

    # None and primitives are always serializable
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, dict):
        return {
            k: sanitize_for_json(v, max_depth=max_depth - 1)
            for k, v in value.items()
            if sanitize_for_json(v, max_depth=max_depth - 1) is not None
        }

    if isinstance(value, list):
        sanitized = []
        for item in value:
            clean = sanitize_for_json(item, max_depth=max_depth - 1)
            if clean is not None:
                sanitized.append(clean)
        return sanitized

    if isinstance(value, tuple):
        return sanitize_for_json(list(value), max_depth=max_depth - 1)

    # Try objects with __dict__ (e.g. Pydantic models)
    if hasattr(value, "__dict__"):
        try:
            return sanitize_for_json(vars(value), max_depth=max_depth - 1)
        except Exception:
            pass

    # Try model_dump for Pydantic v2
    if hasattr(value, "model_dump"):
        try:
            return sanitize_for_json(value.model_dump(), max_depth=max_depth - 1)
        except Exception:
            pass

    # Last resort: attempt direct serialization
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        logger.debug(f"Skipping non-serializable value of type {type(value).__name__}")
        return None


def sanitize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a chat message for persistence.

    Special handling for known non-serializable fields from LLM APIs:
    - thinking_block: Extracts text content
    - content_blocks: Skipped (often contain raw API objects)

    Args:
        message: Chat message dict (may contain non-serializable fields).

    Returns:
        Sanitized message safe for JSON serialization.
    """
    if not isinstance(message, dict):
        result = sanitize_for_json(message)
        return result if isinstance(result, dict) else {}

    sanitized: dict[str, Any] = {}

    for key, value in message.items():
        if key == "thinking_block":
            if isinstance(value, dict) and "text" in value:
                sanitized["thinking_text"] = value["text"]
            elif hasattr(value, "text"):
                sanitized["thinking_text"] = value.text  # pyright: ignore[reportAttributeAccessIssue]
            continue

        if key == "content_blocks":
            continue

        clean_value = sanitize_for_json(value)
        if clean_value is not None:
            sanitized[key] = clean_value

    return sanitized
