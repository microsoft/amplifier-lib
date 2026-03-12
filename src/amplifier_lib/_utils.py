"""Internal utilities for amplifier-lib.

Shared helpers used by bundle.py and registry.py:
  - deep_merge / merge_module_lists  (dict merging)
  - construct_context_path           (bundle-relative paths)
  - parse_frontmatter                (YAML frontmatter from markdown)
  - read_yaml                        (async YAML with retry)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dict merge utilities
# ---------------------------------------------------------------------------


def deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Child values override parent values.  For nested dicts, merge recursively.
    For lists, concatenate with deduplication (parent items first).
    For scalars, child replaces parent.
    """
    result = parent.copy()

    for key, child_value in child.items():
        if key in result:
            parent_value = result[key]
            if isinstance(parent_value, dict) and isinstance(child_value, dict):
                result[key] = deep_merge(parent_value, child_value)
            elif isinstance(parent_value, list) and isinstance(child_value, list):
                seen: set[Any] = set()
                merged: list[Any] = []
                for item in parent_value + child_value:
                    if isinstance(item, (str, int, float, bool, type(None))):
                        dedup_key: Any = (type(item), item)
                    else:
                        try:
                            dedup_key = (
                                type(item),
                                json.dumps(item, sort_keys=True, default=str),
                            )
                        except (TypeError, ValueError):
                            merged.append(item)
                            continue
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        merged.append(item)
                result[key] = merged
            else:
                result[key] = child_value
        else:
            result[key] = child_value

    return result


def merge_module_lists(
    parent: list[dict[str, Any]],
    child: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge two lists of module configs by module ID.

    If both lists have config for the same module ID, deep merge them
    (child overrides parent).
    """
    by_id: dict[str, dict[str, Any]] = {}
    for i, config in enumerate(parent):
        if not isinstance(config, dict):
            raise TypeError(
                f"Malformed module config at index {i}: expected dict with 'module' key, "
                f"got {type(config).__name__} {config!r}"
            )
        module_id = config.get("id") or config.get("module")
        if module_id:
            by_id[module_id] = config.copy()

    for i, config in enumerate(child):
        if not isinstance(config, dict):
            raise TypeError(
                f"Malformed module config at index {i}: expected dict with 'module' key, "
                f"got {type(config).__name__} {config!r}"
            )
        module_id = config.get("id") or config.get("module")
        if not module_id:
            continue
        if module_id in by_id:
            by_id[module_id] = deep_merge(by_id[module_id], config)
        else:
            by_id[module_id] = config.copy()

    return list(by_id.values())


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def construct_context_path(base: Path, name: str) -> Path:
    """Construct an absolute path to a bundle resource file.

    ``name`` is a path relative to the bundle root directory.  Leading ``/``
    is stripped to prevent accidentally making the result absolute.
    """
    name = name.lstrip("/")
    if not name:
        return base
    return base / name


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown text.

    Returns:
        Tuple of (frontmatter_dict, body_text).
        If no frontmatter, returns ({}, original_text).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    frontmatter = yaml.safe_load(match.group(1)) or {}
    body = text[match.end() :]
    return frontmatter, body


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


async def _read_with_retry(
    path: Path,
    max_retries: int = 3,
    initial_delay: float = 0.1,
) -> str:
    """Read file with retry logic for cloud-sync delays (errno 5)."""
    delay = initial_delay
    last_error: OSError | None = None

    for attempt in range(max_retries):
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            last_error = e
            if e.errno == 5 and attempt < max_retries - 1:
                if attempt == 0:
                    logger.warning("File I/O error reading %s – retrying (cloud-sync?)", path)
                await asyncio.sleep(delay)
                delay *= 2
            else:
                raise

    raise last_error  # type: ignore[misc]


async def read_yaml(path: Path) -> dict[str, Any] | None:
    """Read a YAML file and return parsed content, or None if missing."""
    if not path.exists():
        return None
    content = await _read_with_retry(path)
    return yaml.safe_load(content) or {}
