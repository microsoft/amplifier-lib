"""Tests for amplifier_lib.bundle — Bundle dataclass, from_dict(), compose(), to_mount_plan()."""

from __future__ import annotations

from pathlib import Path

import pytest

from amplifier_lib.bundle import Bundle
from amplifier_lib.exceptions import BundleValidationError


# ---------------------------------------------------------------------------
# Constructor / defaults
# ---------------------------------------------------------------------------


class TestBundleConstructor:
    def test_minimal_construction(self):
        b = Bundle(name="myapp")
        assert b.name == "myapp"
        assert b.version == "1.0.0"
        assert b.description == ""
        assert b.includes == []
        assert b.session == {}
        assert b.providers == []
        assert b.tools == []
        assert b.hooks == []
        assert b.spawn == {}
        assert b.agents == {}
        assert b.context == {}
        assert b.instruction is None
        assert b.base_path is None

    def test_all_fields_set(self, tmp_path):
        b = Bundle(
            name="full",
            version="2.0.0",
            description="A full bundle",
            includes=["other"],
            session={"orchestrator": {"module": "m"}},
            providers=[{"module": "prov", "source": "git+https://example.com/prov"}],
            tools=[{"module": "tool", "source": "git+https://example.com/tool"}],
            hooks=[{"module": "hook", "source": "git+https://example.com/hook"}],
            spawn={"exclude_tools": ["dangerous"]},
            agents={"agent1": {"name": "Agent One"}},
            context={"ctx": tmp_path / "ctx.md"},
            instruction="You are a helpful assistant.",
            base_path=tmp_path,
        )
        assert b.name == "full"
        assert b.version == "2.0.0"
        assert b.description == "A full bundle"
        assert b.includes == ["other"]
        assert b.instruction == "You are a helpful assistant."
        assert b.base_path == tmp_path

    def test_post_init_converts_none_context_to_dict(self):
        b = Bundle(name="x", context=None)  # type: ignore[arg-type]
        assert b.context == {}

    def test_post_init_converts_none_source_base_paths_to_dict(self):
        b = Bundle(name="x", source_base_paths=None)  # type: ignore[arg-type]
        assert b.source_base_paths == {}

    def test_kwargs_matching_amplifierd_usage(self):
        """Ensure Bundle can be constructed with the kwargs amplifierd uses."""
        b = Bundle(
            name="app",
            version="1.0.0",
            session={"orchestrator": {"module": "default-orchestrator"}},
            providers=[{"module": "anthropic", "source": "git+https://example.com"}],
            tools=[],
            hooks=[],
            instruction="You help users.",
        )
        assert b.name == "app"
        assert len(b.providers) == 1
        assert b.instruction == "You help users."


# ---------------------------------------------------------------------------
# from_dict()
# ---------------------------------------------------------------------------


class TestBundleFromDict:
    def test_from_empty_dict(self):
        b = Bundle.from_dict({})
        assert b.name == ""
        assert b.version == "1.0.0"
        assert b.providers == []
        assert b.tools == []
        assert b.hooks == []

    def test_from_dict_with_bundle_metadata(self):
        data = {
            "bundle": {"name": "myapp", "version": "3.1.0", "description": "Test app"},
        }
        b = Bundle.from_dict(data)
        assert b.name == "myapp"
        assert b.version == "3.1.0"
        assert b.description == "Test app"

    def test_from_dict_with_session(self):
        data = {
            "bundle": {"name": "app"},
            "session": {"orchestrator": {"module": "m", "source": "git+..."}},
        }
        b = Bundle.from_dict(data)
        assert b.session["orchestrator"]["module"] == "m"

    def test_from_dict_with_providers_list(self):
        data = {
            "bundle": {"name": "app"},
            "providers": [{"module": "anthropic", "source": "git+https://example.com"}],
        }
        b = Bundle.from_dict(data)
        assert len(b.providers) == 1
        assert b.providers[0]["module"] == "anthropic"

    def test_from_dict_with_tools_list(self):
        data = {
            "bundle": {"name": "app"},
            "tools": [
                {"module": "tool-a", "source": "git+https://example.com/a"},
                {"module": "tool-b", "source": "git+https://example.com/b"},
            ],
        }
        b = Bundle.from_dict(data)
        assert len(b.tools) == 2

    def test_from_dict_with_base_path(self, tmp_path):
        data = {"bundle": {"name": "app"}}
        b = Bundle.from_dict(data, base_path=tmp_path)
        assert b.base_path == tmp_path

    def test_from_dict_raises_on_malformed_providers(self):
        data = {"bundle": {"name": "app"}, "providers": "not-a-list"}
        with pytest.raises(BundleValidationError):
            Bundle.from_dict(data)

    def test_from_dict_raises_on_non_dict_provider_item(self):
        data = {"bundle": {"name": "app"}, "providers": ["string-item"]}
        with pytest.raises(BundleValidationError):
            Bundle.from_dict(data)

    def test_from_dict_resolves_relative_source_paths(self, tmp_path):
        data = {
            "bundle": {"name": "app"},
            "tools": [{"module": "local-tool", "source": "./tools/mytool"}],
        }
        b = Bundle.from_dict(data, base_path=tmp_path)
        assert b.tools[0]["source"] == str((tmp_path / "./tools/mytool").resolve())

    def test_from_dict_with_agents_include(self):
        data = {
            "bundle": {"name": "app"},
            "agents": {"include": ["agent1", "agent2"]},
        }
        b = Bundle.from_dict(data)
        assert "agent1" in b.agents
        assert "agent2" in b.agents

    def test_from_dict_instruction_is_none(self):
        """from_dict() always sets instruction=None (set separately from body)."""
        data = {"bundle": {"name": "app"}}
        b = Bundle.from_dict(data)
        assert b.instruction is None


# ---------------------------------------------------------------------------
# compose()
# ---------------------------------------------------------------------------


class TestBundleCompose:
    def test_compose_no_others_returns_copy(self):
        b = Bundle(
            name="base",
            version="1.0.0",
            providers=[{"module": "p", "source": "src"}],
            instruction="base instruction",
        )
        result = b.compose()
        assert result.name == "base"
        assert result.instruction == "base instruction"
        assert len(result.providers) == 1

    def test_compose_later_name_wins(self):
        base = Bundle(name="base")
        child = Bundle(name="child")
        result = base.compose(child)
        assert result.name == "child"

    def test_compose_later_instruction_overrides(self):
        base = Bundle(name="base", instruction="base instruction")
        child = Bundle(name="child", instruction="child instruction")
        result = base.compose(child)
        assert result.instruction == "child instruction"

    def test_compose_base_instruction_preserved_when_child_has_none(self):
        base = Bundle(name="base", instruction="base instruction")
        child = Bundle(name="child", instruction=None)
        result = base.compose(child)
        assert result.instruction == "base instruction"

    def test_compose_providers_merged(self):
        base = Bundle(name="base", providers=[{"module": "pA", "source": "srcA"}])
        child = Bundle(name="child", providers=[{"module": "pB", "source": "srcB"}])
        result = base.compose(child)
        module_ids = [p["module"] for p in result.providers]
        assert "pA" in module_ids
        assert "pB" in module_ids

    def test_compose_tools_merged(self):
        base = Bundle(name="base", tools=[{"module": "tA", "source": "srcA"}])
        child = Bundle(name="child", tools=[{"module": "tB", "source": "srcB"}])
        result = base.compose(child)
        module_ids = [t["module"] for t in result.tools]
        assert "tA" in module_ids
        assert "tB" in module_ids

    def test_compose_hooks_merged(self):
        base = Bundle(name="base", hooks=[{"module": "hA", "source": "srcA"}])
        child = Bundle(name="child", hooks=[{"module": "hB", "source": "srcB"}])
        result = base.compose(child)
        module_ids = [h["module"] for h in result.hooks]
        assert "hA" in module_ids
        assert "hB" in module_ids

    def test_compose_same_module_deep_merges(self):
        """Two bundles with the same module ID should merge their configs."""
        base = Bundle(name="base", tools=[{"module": "shared", "source": "src", "config": {"a": 1}}])
        child = Bundle(name="child", tools=[{"module": "shared", "source": "src", "config": {"b": 2}}])
        result = base.compose(child)
        # Only one entry for the shared module
        shared = next(t for t in result.tools if t["module"] == "shared")
        assert shared["config"]["a"] == 1
        assert shared["config"]["b"] == 2

    def test_compose_session_deep_merged(self):
        base = Bundle(name="base", session={"orchestrator": {"module": "m1"}})
        child = Bundle(name="child", session={"context": {"module": "ctx"}})
        result = base.compose(child)
        assert result.session["orchestrator"]["module"] == "m1"
        assert result.session["context"]["module"] == "ctx"

    def test_compose_agents_later_overrides(self):
        base = Bundle(name="base", agents={"agent1": {"name": "Old Agent"}})
        child = Bundle(name="child", agents={"agent1": {"name": "New Agent"}})
        result = base.compose(child)
        assert result.agents["agent1"]["name"] == "New Agent"

    def test_compose_agents_accumulate(self):
        base = Bundle(name="base", agents={"a1": {"name": "A1"}})
        child = Bundle(name="child", agents={"a2": {"name": "A2"}})
        result = base.compose(child)
        assert "a1" in result.agents
        assert "a2" in result.agents

    def test_compose_base_path_from_child(self, tmp_path):
        child_path = tmp_path / "child"
        child_path.mkdir()
        base = Bundle(name="base", base_path=tmp_path)
        child = Bundle(name="child", base_path=child_path)
        result = base.compose(child)
        assert result.base_path == child_path

    def test_compose_source_base_paths_accumulated(self, tmp_path):
        base = Bundle(name="base", base_path=tmp_path)
        child_path = tmp_path / "child"
        child_path.mkdir()
        child = Bundle(name="child", base_path=child_path)
        result = base.compose(child)
        assert "base" in result.source_base_paths
        assert "child" in result.source_base_paths

    def test_compose_multiple_others(self):
        base = Bundle(name="base", instruction="base")
        mid = Bundle(name="mid", instruction="mid")
        top = Bundle(name="top", instruction="top")
        result = base.compose(mid, top)
        assert result.instruction == "top"

    def test_compose_returns_new_bundle_not_mutating_originals(self):
        base = Bundle(name="base", providers=[{"module": "p", "source": "src"}])
        child = Bundle(name="child", providers=[{"module": "q", "source": "src"}])
        result = base.compose(child)
        assert len(base.providers) == 1  # not mutated
        assert len(child.providers) == 1  # not mutated
        assert len(result.providers) == 2


# ---------------------------------------------------------------------------
# to_mount_plan()
# ---------------------------------------------------------------------------


class TestBundleToMountPlan:
    def test_empty_bundle_returns_empty_plan(self):
        b = Bundle(name="empty")
        plan = b.to_mount_plan()
        assert plan == {}

    def test_session_in_plan_when_set(self):
        b = Bundle(name="app", session={"orchestrator": {"module": "m"}})
        plan = b.to_mount_plan()
        assert "session" in plan
        assert plan["session"]["orchestrator"]["module"] == "m"

    def test_providers_in_plan_when_set(self):
        b = Bundle(name="app", providers=[{"module": "p", "source": "src"}])
        plan = b.to_mount_plan()
        assert "providers" in plan
        assert plan["providers"][0]["module"] == "p"

    def test_tools_in_plan_when_set(self):
        b = Bundle(name="app", tools=[{"module": "t", "source": "src"}])
        plan = b.to_mount_plan()
        assert "tools" in plan
        assert plan["tools"][0]["module"] == "t"

    def test_hooks_in_plan_when_set(self):
        b = Bundle(name="app", hooks=[{"module": "h", "source": "src"}])
        plan = b.to_mount_plan()
        assert "hooks" in plan

    def test_agents_in_plan_when_set(self):
        b = Bundle(name="app", agents={"a1": {"name": "Agent1"}})
        plan = b.to_mount_plan()
        assert "agents" in plan
        assert "a1" in plan["agents"]

    def test_spawn_in_plan_when_set(self):
        b = Bundle(name="app", spawn={"exclude_tools": ["dangerous"]})
        plan = b.to_mount_plan()
        assert "spawn" in plan
        assert plan["spawn"]["exclude_tools"] == ["dangerous"]

    def test_empty_lists_not_in_plan(self):
        b = Bundle(name="app", providers=[], tools=[], hooks=[])
        plan = b.to_mount_plan()
        assert "providers" not in plan
        assert "tools" not in plan
        assert "hooks" not in plan

    def test_plan_is_new_dict_not_reference(self):
        """Modifying the plan should not affect the bundle."""
        b = Bundle(name="app", session={"key": "val"})
        plan = b.to_mount_plan()
        plan["session"]["key"] = "modified"
        assert b.session["key"] == "val"

    def test_instruction_not_in_plan(self):
        """instruction is not part of the mount plan."""
        b = Bundle(name="app", instruction="do things")
        plan = b.to_mount_plan()
        assert "instruction" not in plan

    def test_full_bundle_produces_correct_plan_structure(self):
        b = Bundle(
            name="app",
            session={"orchestrator": {"module": "orch"}},
            providers=[{"module": "prov", "source": "src"}],
            tools=[{"module": "tool", "source": "src"}],
            hooks=[{"module": "hook", "source": "src"}],
            agents={"a1": {"name": "A1"}},
            spawn={"exclude_tools": []},
        )
        plan = b.to_mount_plan()
        assert set(plan.keys()) == {"session", "providers", "tools", "hooks", "agents", "spawn"}


# ---------------------------------------------------------------------------
# resolve_context_path()
# ---------------------------------------------------------------------------


class TestResolveContextPath:
    def test_resolves_from_context_dict(self, tmp_path):
        target = tmp_path / "ctx.md"
        target.write_text("context")
        b = Bundle(name="app", context={"myctx": target})
        result = b.resolve_context_path("myctx")
        assert result == target

    def test_returns_none_when_not_in_context_and_no_base_path(self):
        b = Bundle(name="app")
        assert b.resolve_context_path("missing") is None

    def test_returns_none_when_file_does_not_exist(self, tmp_path):
        b = Bundle(name="app", base_path=tmp_path)
        assert b.resolve_context_path("nonexistent") is None
