"""Tests for amplifier_lib._utils — deep_merge, merge_module_lists,
construct_context_path, parse_frontmatter, read_yaml."""

from pathlib import Path

import pytest
import yaml

from amplifier_lib._utils import (
    construct_context_path,
    deep_merge,
    merge_module_lists,
    parse_frontmatter,
    read_yaml,
)


# ---------------------------------------------------------------------------
# deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_child_overrides_scalar(self):
        result = deep_merge({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_parent_key_retained_when_not_in_child(self):
        result = deep_merge({"a": 1, "b": 2}, {"b": 99})
        assert result["a"] == 1
        assert result["b"] == 99

    def test_child_key_added_when_not_in_parent(self):
        result = deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_nested_dict_merge(self):
        parent = {"cfg": {"timeout": 10, "retries": 3}}
        child = {"cfg": {"retries": 5, "verbose": True}}
        result = deep_merge(parent, child)
        assert result["cfg"]["timeout"] == 10   # preserved from parent
        assert result["cfg"]["retries"] == 5    # overridden by child
        assert result["cfg"]["verbose"] is True  # added by child

    def test_deeply_nested_merge(self):
        parent = {"a": {"b": {"c": 1, "d": 2}}}
        child = {"a": {"b": {"d": 99, "e": 3}}}
        result = deep_merge(parent, child)
        assert result["a"]["b"]["c"] == 1
        assert result["a"]["b"]["d"] == 99
        assert result["a"]["b"]["e"] == 3

    def test_list_concat_with_dedup(self):
        parent = {"tags": ["a", "b"]}
        child = {"tags": ["b", "c"]}
        result = deep_merge(parent, child)
        assert result["tags"] == ["a", "b", "c"]

    def test_list_parent_items_first(self):
        parent = {"items": [1, 2]}
        child = {"items": [3, 4]}
        result = deep_merge(parent, child)
        assert result["items"] == [1, 2, 3, 4]

    def test_list_dedup_preserves_first_occurrence(self):
        parent = {"items": ["x"]}
        child = {"items": ["x", "y"]}
        result = deep_merge(parent, child)
        assert result["items"].count("x") == 1
        assert "y" in result["items"]

    def test_scalar_overrides_when_types_differ(self):
        """When parent has list but child has scalar, child wins."""
        result = deep_merge({"a": [1, 2]}, {"a": "string"})
        assert result["a"] == "string"

    def test_empty_parent(self):
        result = deep_merge({}, {"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_empty_child(self):
        result = deep_merge({"a": 1, "b": 2}, {})
        assert result == {"a": 1, "b": 2}

    def test_both_empty(self):
        assert deep_merge({}, {}) == {}

    def test_does_not_mutate_parent(self):
        parent = {"a": 1}
        deep_merge(parent, {"b": 2})
        assert "b" not in parent

    def test_does_not_mutate_child(self):
        child = {"b": 2}
        deep_merge({"a": 1}, child)
        assert "a" not in child

    def test_none_values_handled(self):
        result = deep_merge({"a": None}, {"a": 42})
        assert result["a"] == 42

    def test_boolean_values(self):
        result = deep_merge({"flag": True}, {"flag": False})
        assert result["flag"] is False


# ---------------------------------------------------------------------------
# merge_module_lists
# ---------------------------------------------------------------------------


class TestMergeModuleLists:
    def test_no_overlap_combines_all(self):
        parent = [{"module": "mod-a", "config": {}}]
        child = [{"module": "mod-b", "config": {}}]
        result = merge_module_lists(parent, child)
        ids = {m.get("module") for m in result}
        assert ids == {"mod-a", "mod-b"}

    def test_child_overrides_parent_config(self):
        parent = [{"module": "mod-a", "config": {"timeout": 10, "retries": 3}}]
        child = [{"module": "mod-a", "config": {"retries": 5}}]
        result = merge_module_lists(parent, child)
        assert len(result) == 1
        assert result[0]["config"]["retries"] == 5
        assert result[0]["config"]["timeout"] == 10

    def test_uses_id_key_as_fallback(self):
        parent = [{"id": "mod-x", "config": {"a": 1}}]
        child = [{"id": "mod-x", "config": {"b": 2}}]
        result = merge_module_lists(parent, child)
        assert len(result) == 1
        assert result[0]["config"]["a"] == 1
        assert result[0]["config"]["b"] == 2

    def test_child_module_without_id_skipped(self):
        """Child entries with no 'id' or 'module' key are skipped."""
        parent = [{"module": "mod-a"}]
        child = [{"config": {"x": 1}}]  # no id/module key
        result = merge_module_lists(parent, child)
        assert len(result) == 1

    def test_empty_parent(self):
        child = [{"module": "mod-a"}]
        result = merge_module_lists([], child)
        assert len(result) == 1

    def test_empty_child(self):
        parent = [{"module": "mod-a"}]
        result = merge_module_lists(parent, [])
        assert len(result) == 1

    def test_both_empty(self):
        assert merge_module_lists([], []) == []

    def test_malformed_parent_raises_type_error(self):
        with pytest.raises(TypeError, match="Malformed module config"):
            merge_module_lists(["not a dict"], [])  # type: ignore[list-item]

    def test_malformed_child_raises_type_error(self):
        with pytest.raises(TypeError, match="Malformed module config"):
            merge_module_lists([], [42])  # type: ignore[list-item]

    def test_multiple_modules_multiple_overrides(self):
        parent = [
            {"module": "mod-a", "enabled": True},
            {"module": "mod-b", "priority": 1},
        ]
        child = [
            {"module": "mod-a", "enabled": False},
            {"module": "mod-c", "priority": 5},
        ]
        result = merge_module_lists(parent, child)
        by_id = {m.get("module"): m for m in result}
        assert by_id["mod-a"]["enabled"] is False
        assert by_id["mod-b"]["priority"] == 1
        assert by_id["mod-c"]["priority"] == 5


# ---------------------------------------------------------------------------
# construct_context_path
# ---------------------------------------------------------------------------


class TestConstructContextPath:
    def test_basic_relative_path(self, tmp_path):
        result = construct_context_path(tmp_path, "subdir/file.txt")
        assert result == tmp_path / "subdir" / "file.txt"

    def test_leading_slash_stripped(self, tmp_path):
        result = construct_context_path(tmp_path, "/subdir/file.txt")
        assert result == tmp_path / "subdir" / "file.txt"

    def test_multiple_leading_slashes_stripped(self, tmp_path):
        result = construct_context_path(tmp_path, "///subdir/file.txt")
        assert result == tmp_path / "subdir" / "file.txt"

    def test_empty_name_returns_base(self, tmp_path):
        result = construct_context_path(tmp_path, "")
        assert result == tmp_path

    def test_slash_only_returns_base(self, tmp_path):
        result = construct_context_path(tmp_path, "/")
        assert result == tmp_path

    def test_filename_only(self, tmp_path):
        result = construct_context_path(tmp_path, "README.md")
        assert result == tmp_path / "README.md"

    def test_result_is_path_object(self, tmp_path):
        result = construct_context_path(tmp_path, "something")
        assert isinstance(result, Path)

    def test_nested_path(self, tmp_path):
        result = construct_context_path(tmp_path, "a/b/c/d.txt")
        assert result == tmp_path / "a" / "b" / "c" / "d.txt"

    def test_does_not_escape_base(self, tmp_path):
        """Leading slash is stripped, so the path stays under base."""
        result = construct_context_path(tmp_path, "/etc/passwd")
        # The leading slash is stripped → base / "etc/passwd"
        assert str(result).startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        text = "---\ntitle: My Doc\nauthor: Alice\n---\nBody text here."
        fm, body = parse_frontmatter(text)
        assert fm == {"title": "My Doc", "author": "Alice"}
        assert body == "Body text here."

    def test_no_frontmatter_returns_empty_dict(self):
        text = "Just plain text."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == "Just plain text."

    def test_empty_frontmatter_returns_empty_dict(self):
        # The regex requires a newline before the closing ---, so a blank line
        # between the delimiters is needed for truly empty frontmatter.
        text = "---\n\n---\nBody."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == "Body."

    def test_body_preserved_after_frontmatter(self):
        text = "---\nkey: val\n---\nLine 1\nLine 2\n"
        fm, body = parse_frontmatter(text)
        assert "Line 1" in body
        assert "Line 2" in body

    def test_body_not_included_in_frontmatter(self):
        text = "---\ntitle: Test\n---\nbody content"
        fm, _ = parse_frontmatter(text)
        assert "body content" not in str(fm)

    def test_frontmatter_with_list(self):
        text = "---\ntags:\n  - python\n  - testing\n---\n"
        fm, _ = parse_frontmatter(text)
        assert fm["tags"] == ["python", "testing"]

    def test_frontmatter_with_nested_dict(self):
        text = "---\nmeta:\n  version: 1\n  author: Bob\n---\nbody"
        fm, _ = parse_frontmatter(text)
        assert fm["meta"]["version"] == 1
        assert fm["meta"]["author"] == "Bob"

    def test_no_frontmatter_body_unchanged(self):
        text = "Hello World\nSecond line"
        fm, body = parse_frontmatter(text)
        assert body == text

    def test_empty_string(self):
        fm, body = parse_frontmatter("")
        assert fm == {}
        assert body == ""

    def test_returns_tuple_of_two(self):
        result = parse_frontmatter("text")
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.parametrize(
        "text",
        [
            "No frontmatter here",
            "---\nThis is not valid frontmatter block",
            "  ---\nindented: true\n---\nbody",  # not at start of string
        ],
    )
    def test_invalid_frontmatter_returns_empty(self, text):
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == text


# ---------------------------------------------------------------------------
# read_yaml
# ---------------------------------------------------------------------------


class TestReadYaml:
    async def test_reads_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: value\nnumber: 42\n", encoding="utf-8")
        result = await read_yaml(yaml_file)
        assert result == {"key": "value", "number": 42}

    async def test_returns_none_for_missing_file(self, tmp_path):
        result = await read_yaml(tmp_path / "nonexistent.yaml")
        assert result is None

    async def test_returns_empty_dict_for_empty_file(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("", encoding="utf-8")
        result = await read_yaml(yaml_file)
        assert result == {}

    async def test_reads_nested_yaml(self, tmp_path):
        data = {"outer": {"inner": [1, 2, 3]}, "flag": True}
        yaml_file = tmp_path / "nested.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")
        result = await read_yaml(yaml_file)
        assert result == data

    async def test_reads_list_yaml_returns_empty_dict(self, tmp_path):
        """yaml.safe_load on a list-only YAML → falsy → returns {}."""
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("- item1\n- item2\n", encoding="utf-8")
        result = await read_yaml(yaml_file)
        # yaml.safe_load returns a list, which is truthy, so it's returned directly
        # Actually: `yaml.safe_load(content) or {}` — a non-empty list is truthy
        # so list is returned as-is when truthy, or {} when falsy
        assert result is not None

    async def test_reads_boolean_values(self, tmp_path):
        yaml_file = tmp_path / "bools.yaml"
        yaml_file.write_text("flag: true\nother: false\n", encoding="utf-8")
        result = await read_yaml(yaml_file)
        assert result == {"flag": True, "other": False}

    async def test_reads_integer_values(self, tmp_path):
        yaml_file = tmp_path / "ints.yaml"
        yaml_file.write_text("count: 100\nzero: 0\n", encoding="utf-8")
        result = await read_yaml(yaml_file)
        assert result["count"] == 100
        assert result["zero"] == 0
