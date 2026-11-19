"""Offline ReflAct tool-boundary contract tests; no provider or simulator."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyrualean.arms import ArmResult, ArmToolkit
from pyrualean.reflact_harness import ReflActHarness, validate_options

SUMMARIES = {
    "observed_state": "The item is visible.",
    "goal_gap": "Retrieval is incomplete.",
}
# Exact helper-shadowing definition from the failed layout-0 real-model cell_001.py.
HISTORICAL_SHADOW = """def reflact(observed_state='', goal_gap=''):
    print('REFLACT:', observed_state, '||', goal_gap)
"""


def kit_for(robo, tmp_path, **options):
    kit = ArmToolkit(
        robo, arm="cells", workspace=tmp_path / "workspace", images="none", feedback="pure"
    )
    kit.general_harness = ReflActHarness(kit, {"kind": "reflact", **options}, "Retrieve the item")
    return kit


def events(kit, event=None):
    path = kit.general_harness.directory / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [row for row in rows if event is None or row["event"] == event]


def arguments(code="pass", **updates):
    return {**SUMMARIES, "code": code, **updates}


@pytest.mark.parametrize("limit", [80, 320, 640])
def test_valid_options_and_checked_in_config(limit):
    validate_options({"kind": "reflact", "max_summary_chars": limit})
    path = Path(__file__).resolve().parents[1] / (
        "integrations/migrations/general_harness/reflact.json"
    )
    validate_options(json.loads(path.read_text()))


@pytest.mark.parametrize(
    "config",
    [None, [], {}, {"kind": "pos"}, {"kind": "reflact", "completion_guard": True},
     *[{"kind": "reflact", "max_summary_chars": value}
       for value in (True, 79, 641, 320.0, "320", None)]],
)
def test_invalid_options(config):
    with pytest.raises(ValueError):
        validate_options(config)


def test_only_cells_and_no_python_namespace_dependency(robo, tmp_path):
    kit = ArmToolkit(robo, arm="program", workspace=tmp_path / "workspace")
    with pytest.raises(ValueError, match="arm=cells"):
        ReflActHarness(kit, {"kind": "reflact"}, "goal")
    kit.arm = "cells"
    existing = object()
    kit._globals["reflact"] = existing
    harness = ReflActHarness(kit, {"kind": "reflact"}, "goal")
    assert kit._globals["reflact"] is existing
    assert harness.provenance["implementation"] == "pyrualean-reflact-v3-interface-v11"
    assert not getattr(harness, "controls_loop", False)
    assert not hasattr(harness, "before_request")


def test_schema_is_explicit_repeatable_and_baseline_is_unchanged(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace")
    baseline = kit.get_tools_spec()
    original = copy.deepcopy(baseline)
    kit.general_harness = ReflActHarness(kit, {"kind": "reflact"}, "goal")
    specs = kit.get_tools_spec()
    assert specs == kit.get_tools_spec()
    assert baseline == original
    schema = next(s for s in specs if s["name"] == "python")["input_schema"]
    assert schema["required"] == ["observed_state", "goal_gap", "code"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["goal_gap"]["maxLength"] == 320
    assert next(s for s in specs if s["name"] == "finish") == baseline[1]
    kit.general_harness = None
    assert kit.get_tools_spec() == baseline


def test_multiaction_cell_records_before_any_python_primitive(robo, backend, tmp_path):
    kit = kit_for(robo, tmp_path)
    before_calls = len(robo._ledger)
    out = kit.execute_tool("python", arguments(
        "robo.move_to([0.05, -0.1, 0.6]); robo.move_to([0.05, 0.1, 0.6])"))
    assert "exception" not in out.result and "error" not in out.result
    assert len(robo._ledger) - before_calls == 3
    assert len([name for name, _ in backend.calls if name == "move_to"]) == 2
    row = events(kit, "after_tool")[-1]
    assert row["adoption"] == "recorded" and row["reflection_count"] == 1
    assert row["stateful_count"] == 2 and row["primitive_count"] == 3
    assert row["cell"] == 1 and row["turn"] == 0
    assert row["reflection_usage"] == [{
        "id": "reflection-1", "before_first_primitive": True, "before_first_stateful": True,
        "subsequent_primitive_count": 3, "subsequent_stateful_count": 2,
    }]
    assert kit.general_harness.pending == []
    assert "reflact" not in kit._globals


@pytest.mark.parametrize("field", ["observed_state", "goal_gap", "code"])
@pytest.mark.parametrize("bad", [None, "", "  ", True, 3])
def test_bad_arguments_never_consume_cell_or_execute_actions(robo, backend, tmp_path, field, bad):
    kit = kit_for(robo, tmp_path)
    payload = arguments("marker = 1; robo.move_to([0.05, -0.1, 0.6])")
    payload[field] = bad
    calls_before = list(backend.calls)
    result = kit.execute_tool("python", payload)
    assert field in result.result["error"] and "No code was executed" in result.result["error"]
    assert kit._cell_index == 0 and "marker" not in kit._globals
    assert backend.calls == calls_before and len(robo._ledger) == 0
    assert not list(kit.workspace.glob("cell_*.py"))
    assert not events(kit, "reflection_recorded")
    assert events(kit, "after_tool")[-1]["adoption"] == "rejected"


def test_missing_long_and_unknown_fields_are_reported_together(robo, tmp_path):
    kit = kit_for(robo, tmp_path)
    result = kit.execute_tool("python", {"code": "marker=1", "goal_gap": "x" * 321,
                                         "success": True})
    issues = result.result["reflact_protocol"]["issues"]
    assert len(issues) == 3
    assert any("observed_state" in e for e in issues)
    assert any("goal_gap" in e for e in issues)
    assert any("success" in e for e in issues)
    assert "marker" not in kit._globals
    assert kit._cell_index == 0


def test_historical_shadowing_cannot_replace_tool_boundary_record(robo, tmp_path):
    kit = kit_for(robo, tmp_path)
    # The original cell omitted the two tool fields and cannot run in v2.
    rejected = kit.execute_tool("python", {"code": HISTORICAL_SHADOW + "marker = 1"})
    assert "error" in rejected.result and "reflact" not in kit._globals
    # Adding explicit fields records them before the same definition executes.
    out = kit.execute_tool("python", arguments(HISTORICAL_SHADOW + "marker = 2"))
    assert "exception" not in out.result and kit._globals["marker"] == 2
    assert callable(kit._globals["reflact"])
    kit.execute_tool("python", arguments("reflact = None; marker = 3"))
    rows = events(kit, "reflection_recorded")
    assert len(rows) == 2 and all(r["interface"] == "tool_arguments" for r in rows)
    assert [r["cell"] for r in rows] == [1, 2]
    assert [r["turn"] for r in rows] == [1, 2]
    assert [r["adoption"] for r in events(kit, "after_tool")] == [
        "rejected", "recorded", "recorded"]


def test_no_stale_summary_reuse_after_a_valid_cell(robo, tmp_path):
    kit = kit_for(robo, tmp_path)
    kit.execute_tool("python", arguments("marker=1"))
    out = kit.execute_tool("python", {"code": "marker=2"})
    assert "error" in out.result
    assert kit._globals["marker"] == 1 and kit._cell_index == 1
    assert len(events(kit, "reflection_recorded")) == 1
    assert events(kit, "after_tool")[-1]["reflection_ids"] == []


@pytest.mark.parametrize("code", ["raise ValueError('failure')", "def broken(:", "import os"])
def test_failed_or_blocked_python_keeps_real_decision_record(robo, tmp_path, code):
    kit = kit_for(robo, tmp_path)
    result = kit.execute_tool("python", arguments(code))
    assert "exception" in result.result
    row = events(kit, "after_tool")[-1]
    assert row["adoption"] == "recorded" and row["tool_failed"]
    assert row["stateful_count"] == 0 and row["reflection_count"] == 1
    assert row["reflection_usage"][0]["before_first_stateful"] is None


def test_exhausted_cell_budget_does_not_record_an_unexecuted_decision(robo, tmp_path):
    kit = kit_for(robo, tmp_path)
    kit.max_turns = 1
    kit.execute_tool("python", arguments())
    result = kit.execute_tool("python", arguments("marker=1"))
    assert "budget exhausted" in result.result["error"]
    assert len(events(kit, "reflection_recorded")) == 1
    assert events(kit, "after_tool")[-1]["adoption"] == "not_executed"
    assert "marker" not in kit._globals


def test_recording_does_not_certify_success_or_gate_finish(robo, backend, tmp_path):
    kit = kit_for(robo, tmp_path)
    kit.execute_tool("python", arguments(observed_state="All done.", goal_gap="No remaining gap."))
    assert not backend.solved()
    result = kit.execute_tool("finish", {"status": "success", "summary": "agent claim"})
    assert result.result["ok"] and kit.finish["status"] == "success"
    assert not backend.solved()
    row = events(kit, "after_tool")[-1]
    assert row["adoption"] == "not_applicable" and row["reflection_count"] == 0
    assert events(kit, "reflection_recorded")[0]["source"] == "agent_report"


def test_audit_uses_only_public_call_metadata_and_preserves_feedback(tmp_path):
    class CounterOnly:
        def __len__(self):
            return 2

        def __getattr__(self, key):
            raise AssertionError(f"not permitted: {key}")

    class MetadataOnly:
        _ledger = CounterOnly()

        def __getattr__(self, key):
            raise AssertionError(f"hidden state access: {key}")

    class NeverRead:
        def __iter__(self):
            raise AssertionError("images must stay in normal context")

        def __str__(self):
            raise AssertionError("task must stay in normal context")

    kit = SimpleNamespace(arm="cells", workspace=tmp_path / "workspace", _globals={},
                          turns=[], _cell_index=1, robo=MetadataOnly())
    harness = ReflActHarness(kit, {"kind": "reflact"}, NeverRead(), NeverRead())
    kit.general_harness = harness
    harness.reflect("uncertain state", "unverified outcome")
    result = ArmResult({"cell": 1, "stdout": "x" * 50000, "exception": "public failure"},
                       [b"unchanged-image"])
    result.rendered_text = "unchanged rendering"
    original = copy.deepcopy(result)
    record = SimpleNamespace(index=2, stateful=True, error="public primitive failure")
    assert harness.after_tool("python", result, [record]) is result
    assert result.result == original.result and result.images == original.images
    assert result.rendered_text == original.rendered_text
    row = events(kit, "after_tool")[-1]
    assert row["primitive_error_count"] == 1 and row["tool_failed"]
    assert row["reflection_usage"][0]["subsequent_stateful_count"] == 1
