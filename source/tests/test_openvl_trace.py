"""Temporal evidence links must not become robot execution policy."""

import json

import pytest

from tests.test_openvl_native import ROOT, setup


def local(robo, tmp_path):
    return setup(robo, tmp_path, str(ROOT / "capabilities_local.yml"))[0]


def test_multiaction_states_and_images_keep_their_own_steps(robo, backend, tmp_path):
    kit = local(robo, tmp_path)
    out = kit.execute_tool(
        "python", {"code": "robo.move_to([0,0,2]); robo.move_to([0,0,0.6]); marker=17"}
    )
    assert "exception" not in out.result
    trace = kit.evidence_adapter.trace
    actions = trace.find(kind="action", tool="move_to", cell=0)
    assert len(actions) == 2
    summaries = trace.related(actions[0]["reference"])
    visual_summary = next(row for row in summaries if row["kind"] == "visual")
    assert visual_summary["image_ref"] == trace.read(visual_summary["reference"])["image_ref"]
    first, second = [trace.read(x["reference"]) for x in reversed(actions)]
    assert first["result"]["reached"] is False and second["result"]["reached"] is True
    assert first["step"] == second["step_before"] < second["step"]
    assert trace.read(first["state_before"])["relation"] == "cached_before_action_not_refreshed"
    assert trace.read(second["state_before"])["value"] == trace.read(first["state_after"])["value"]
    assert "cached" in trace.read(second["state_before"])["relation"]
    for action in actions:
        visual = [
            trace.read(x["reference"])
            for x in trace.related(action["reference"])
            if x["kind"] == "visual"
        ]
        assert len(visual) == len(robo.CAMERAS)
        assert all(v["step"] == action["step"] for v in visual)
        assert all(v["relation"] == "after_action_same_step" for v in visual)
    assert len([n for n, _ in backend.calls if n == "move_to"]) == 2
    assert not any(n == "segment" for n, _ in backend.calls)
    assert robo._action_feedback is None and kit.stage_verifier is None
    end = trace.read(out.result["robot_trace"]["end"])
    assert all(x["reference"] in end["links"] for x in actions)
    first["result"]["reached"] = "mutated"
    assert trace.read(actions[-1]["reference"])["result"]["reached"] is False
    assert trace.find(kind="action", step_from=second["step"], step_to=second["step"])
    assert not trace.find(time_from=10**20)
    manifest = json.loads((tmp_path / "openvl_evidence/native/manifest.json").read_text())
    assert actions[0]["reference"] in manifest["bindings"]
    later = kit.execute_tool(
        "python",
        {
            "code": "print(marker); "
            "print(trace_read(trace_find(kind='action')[0]['reference'])['tool'])"
        },
    )
    assert "17" in later.result["stdout"] and "move_to" in later.result["stdout"]


def test_caught_primitive_and_uncaught_cell_errors_both_link_actions(robo, backend, tmp_path):
    kit = local(robo, tmp_path)
    backend.error_next = {"error": "fixture failure", "code": "bad_target"}
    out = kit.execute_tool(
        "python",
        {
            "code": "try:\n    robo.move_to([0,0,0.6])\nexcept Exception:\n    pass\n"
            "robo.move_to([0,0,0.55])\nraise ValueError('cell failure')"
        },
    )
    assert "cell failure" in out.result["exception"]
    trace = kit.evidence_adapter.trace
    errors = [trace.read(x["reference"]) for x in trace.find(kind="exception")]
    assert {e["scope"] for e in errors} == {"primitive", "cell"}
    primitive = next(e for e in errors if e["scope"] == "primitive")
    assert "fixture failure" in primitive["error"]
    actions = trace.find(kind="action")
    assert len(actions) == 2 and actions[-1]["reference"] in primitive["links"]
    assert trace.read(actions[-1]["reference"])["result"]["code"] == "bad_target"
    assert len([n for n, _ in backend.calls if n == "move_to"]) == 2
    assert "recovery_required" not in out.result


def test_perception_links_are_historical_context_not_targets(robo, tmp_path):
    kit = local(robo, tmp_path)
    kit.execute_tool("python", {"code": "robo.segment('bowl', camera='agentview')"})
    adapter = kit.evidence_adapter
    ref = adapter.bridge.refs[-1]
    kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6])"})
    action = adapter.trace.read(adapter.trace.find(kind="action")[0]["reference"])
    assert action["historical_context"] == [ref]
    assert action["association"] == "temporal_only_target_unknown"
    assert adapter.trace.read(ref)["historical"]
    assert adapter.trace.related(ref)
    with pytest.raises(ValueError, match="Historical"):
        adapter.read_data(ref)
    with pytest.raises(ValueError):
        adapter.trace.read("invented-reference")


def test_missing_images_and_audit_failure_preserve_execution(robo, tmp_path, monkeypatch):
    kit = local(robo, tmp_path)
    monkeypatch.setattr(robo, "image", lambda *a, **k: (_ for _ in ()).throw(OSError("no camera")))
    out = kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6]); saved=3"})
    assert "exception" not in out.result and kit._globals["saved"] == 3
    trace = kit.evidence_adapter.trace
    assert all(
        trace.read(x["reference"])["availability"] == "unavailable"
        for x in trace.find(kind="visual")
    )
    monkeypatch.setattr(
        trace, "action", lambda *a: (_ for _ in ()).throw(OSError("archive failed"))
    )
    out = kit.execute_tool("python", {"code": "robo.move_to([0,0,0.55]); saved=4"})
    assert "exception" not in out.result and kit._globals["saved"] == 4
    assert any("archive failed" in e for e in out.result["robot_trace"]["audit_errors"])


def test_native_artifact_links_and_images_none(robo, tmp_path):
    kit = local(robo, tmp_path)
    out = kit.execute_tool("Zoom_In_Tool", {"image_id": "head.png", "coords": [0, 0, 400, 400]})
    ref = out.result["data"]["image"]
    assert ref in kit.evidence_adapter.trace.read(out.result["robot_trace"]["end"])["links"]
    kit.images = "none"
    before = len(kit.evidence_adapter.trace.find(kind="visual"))
    kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6])"})
    assert len(kit.evidence_adapter.trace.find(kind="visual")) == before
