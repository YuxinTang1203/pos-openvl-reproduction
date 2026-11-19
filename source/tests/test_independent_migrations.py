import io
import json
from pathlib import Path

import pytest
from PIL import Image

from pyrualean.arms import ArmToolkit
from pyrualean.migrations import configure_migration, migration_prompt, validate_migration
from pyrualean.openai_runtime import run_openai_agent
from tests.test_openai_runtime import ScriptedTransport, _reply

PROJECTS = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parents[1]


def configure(robo, tmp_path, kind):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace",
                     images="on-demand", feedback="pure", max_turns=100)
    config = {"kind": kind}
    if kind == "harnesspro":
        policy = json.loads((BASE / "integrations/migrations/harnesspro_policy.json").read_text())
        # Exercise the real hooks without spending a network request on classification.
        policy.pop("classifier", None)
        policy["profiles"]["standard"].update(economy_after_tools=1, strict_after_tools=2,
                                              finalize_after_tools=3)
        p = tmp_path / "policy.json"
        p.write_text(json.dumps(policy))
        config.update(root=str(PROJECTS / "HarnessPro"), policy=str(p))
    elif kind in {"vista", "openvl_vista"}:
        config["root"] = str(PROJECTS / "VISTA")
    stream = io.BytesIO()
    Image.new("RGB", (64, 48), "red").save(stream, format="PNG")
    record = configure_migration(kit, config, task="Move an object",
                                 initial_images={"head.png": stream.getvalue()})
    return kit, record


@pytest.mark.parametrize("kind", ["baseline", "harnesspro", "vista", "openvl_evidence", "openvl_vista"])
def test_independent_tools_and_persistent_execution(robo, tmp_path, kind):
    kit, record = configure(robo, tmp_path, kind)
    tools = {t["name"] for t in kit.get_tools_spec()}
    assert ("inspect" in tools) == (kind in {"vista", "openvl_vista"})
    assert ("Robot_Perception_Tool" in tools) == (kind in {"openvl_evidence", "openvl_vista"})
    assert bool(kit.harnesspro_hook) == (kind == "harnesspro")
    assert kit.robo._action_feedback is None and kit.stage_verifier is None
    assert "checked" not in kit._globals and "action_scope" not in kit._globals
    assert record["execution_recovery"] is False
    if kind == "baseline":
        assert tools == {"python", "finish"}
        assert migration_prompt("unchanged", kit) == "unchanged"
    first = kit.execute_tool("python", {"code": "value = 41"})
    second = kit.execute_tool("python", {"code": "print(value + 1)"})
    assert "error" not in first.result and "42" in second.result["stdout"]
    assert kit.robo._action_feedback is None and kit.stage_verifier is None


def test_vista_native_inspection_and_notes_without_harness(robo, tmp_path):
    kit, _ = configure(robo, tmp_path, "vista")
    written = kit.execute_tool("write_working", {"content": "Observed red object"})
    read = kit.execute_tool("read_working", {})
    assert written.result["ok"] and "Observed red object" in read.result["text"]
    inspected = kit.execute_tool("inspect", {"question": "What color?", "views": [
        {"label": "initial", "event": 0, "view": "head.png",
         "region": {"x": 0, "y": 0, "width": 16, "height": 12}}]})
    assert inspected.result["ok"] and inspected.images
    assert kit.harnesspro_hook is None and kit.evidence_adapter is None


def test_native_harness_budget_and_output_compaction(robo, tmp_path):
    kit, _ = configure(robo, tmp_path, "harnesspro")
    large = kit.execute_tool("python", {"code": "print('x' * 18000)"})
    assert large.rendered_text and "Tool output compacted" in large.rendered_text
    for _ in range(2):
        kit.execute_tool("python", {"code": "pass"})
    denied = kit.execute_tool("python", {"code": "should_not_exist = True"})
    assert "budget exhausted" in denied.result["error"]
    assert "should_not_exist" not in kit._globals
    assert kit.execute_tool("finish", {"status": "stuck", "summary": "budget"}).result["ok"]


def test_fixed_profile_counts_and_protected_failure(robo, tmp_path):
    kit, record = configure(robo, tmp_path, "harnesspro")
    assert record["native"]["classification"] is None
    assert kit.harnesspro_hook.selection.profile.name == "standard"
    out = kit.execute_tool("python", {"code":
        "robo.move_to([0,0,2]); robo.move_to([0,0,0.6]); print('x'*18000)"})
    assert out.result["robot_execution"]["counts"]["python_cells"] == 1
    assert out.result["robot_execution"]["counts"]["robot_primitives"] == 2
    assert out.result["robot_execution"]["counts"]["environment_steps"] > 0
    assert '"reached": false' in out.rendered_text
    assert 'Protected robot feedback' in out.rendered_text
    assert kit.robo._action_feedback is None


def test_harness_compression_preserves_perception_and_pose(robo, tmp_path):
    kit, _ = configure(robo, tmp_path, "harnesspro")
    out = kit.execute_tool("python", {"code":
        "robo.segment('missing object', camera='agentview'); "
        "robo.move_to([0,0,2]); print('x'*18000)"})
    protected = json.loads(out.rendered_text.split(
        "Protected robot feedback (not a task verdict):\n", 1)[1])
    feedback = protected["robot_execution"]["primitive_feedback"]
    assert any(item["tool"] == "segment" and item["feedback"]["found"] is False
               for item in feedback)
    move = next(item["feedback"] for item in feedback if item["tool"] == "move_to")
    assert "final_eef_pos" in move and "target_xyz" in move
    assert "terminated" in move
    assert protected["robot_execution"]["counts"]["robot_primitives"] == 1


def test_harness_hook_interface_is_independent_of_parent_env(robo, tmp_path, monkeypatch):
    monkeypatch.setenv("RHO_RUNTIME_INTERFACE", "claude-code")
    kit, _ = configure(robo, tmp_path, "harnesspro")
    out = kit.execute_tool("python", {"code": "print('x'*18000)"})
    assert out.rendered_text and "Protected robot feedback" in out.rendered_text


def test_request_cap_counts_model_calls_separately(robo, tmp_path):
    kit, _ = configure(robo, tmp_path, "harnesspro")
    transport = ScriptedTransport([_reply([("a", "python", {"code": "pass"})])])
    result = run_openai_agent("test", [], kit, model="test", base_url="http://unused/v1",
        api_key=None, max_turns=100, max_requests=1, budget_s=10,
        out_dir=tmp_path / "episode", transport=transport)
    assert result["requests"] == 1
    assert kit.harnesspro_hook.counters["model_requests"] == 1
    assert kit.harnesspro_hook.counters["python_cells"] == 1
    assert kit.harnesspro_hook.counters["robot_primitives"] == 0


@pytest.mark.parametrize("kind", ["baseline", "harnesspro", "vista", "openvl_evidence", "openvl_vista"])
def test_same_model_loop_executes_each_migration(robo, tmp_path, kind):
    kit, _ = configure(robo, tmp_path, kind)
    transport = ScriptedTransport([
        _reply([("a", "python", {"code": "x = 7"})]),
        _reply([("b", "python", {"code": "print(x + 1)"})]),
        _reply([("c", "finish", {"status": "stuck", "summary": "smoke"})]),
    ])
    generation = run_openai_agent(
        "Test", [], kit, model="qwen3.8-max", base_url="http://unused/v1", api_key=None,
        max_turns=100, budget_s=60, out_dir=tmp_path / "episode", transport=transport)
    assert generation["status"] == "completed" and generation["requests"] == 3
    assert all(r["body"]["model"] == "qwen3.8-max" for r in transport.requests)
    assert kit._globals["x"] == 7 and kit.robo._action_feedback is None


@pytest.mark.parametrize("config", [
    {"kind": "vista", "root": "/tmp", "execution_recovery": True},
    {"kind": "openvl_evidence", "stage_verification": True},
    {"kind": "harnesspro_vista"},
    {"kind": "openvl_vista", "root": "/tmp", "execution_recovery": True},
    {"kind": "openvl_vista"},
])
def test_combined_mechanisms_rejected(config):
    with pytest.raises(ValueError):
        validate_migration(config)


def test_combined_prompt_and_evidence_history(robo, tmp_path):
    from pyrualean.openvl_adapter import adapt_prompt
    kit, record = configure(robo, tmp_path, "openvl_vista")
    names = [t["name"] for t in kit.get_tools_spec()]
    assert len(names) == len(set(names))
    prompt = migration_prompt(adapt_prompt("Act through ONE tool,\n`python(code)`.", kit), kit)
    assert "VISTA visual memory" in prompt and "Robot_Evidence_Tool" in prompt
    assert prompt.count("# VISTA visual memory") == 1
    assert set(record["native"]) == {"vista", "openvl_evidence"}
    out = kit.execute_tool("Robot_Perception_Tool", {
        "operation": "segment", "query": "bowl", "camera": "agentview"})
    assert "error" not in out.result, out.result
    ref = kit.evidence_adapter.bridge.refs[-1]
    link = out.result["openvl_vista_link"]
    assert ref in link["new_bindings"]
    assert link["vista_event"] == out.result["vista_evidence"]["event"]
    out = kit.execute_tool("python", {"code": "evidence_inspect(evidence_refs[-1])"})
    assert out.images and out.result["vista_evidence"]["views"]
    event = out.result["vista_evidence"]["event"]
    view = out.result["vista_evidence"]["views"][0]
    inspected = kit.execute_tool("inspect", {"question": "Compare stored view", "views": [
        {"event": event, "view": view, "label": "stored"}]})
    assert inspected.result["ok"] and inspected.images
    # Reading history does not make old geometry current after an action.
    kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6])"})
    with pytest.raises(ValueError, match="Historical"):
        kit.evidence_adapter.bridge.read(ref)
    assert kit.execute_tool("write_working", {"content": "Need fresh observation"}).result["ok"]
    assert kit.robo._action_feedback is None and kit.stage_verifier is None


def test_combined_failure_automatically_links_stale_context(robo, tmp_path):
    kit, _ = configure(robo, tmp_path, "openvl_vista")
    previous = kit.execute_tool("Robot_Perception_Tool", {
        "operation": "segment", "query": "bowl", "camera": "agentview"})
    ref = kit.evidence_adapter.bridge.refs[-1]
    out = kit.execute_tool("python", {"code": "robo.move_to([0,0,2])"})
    review = out.result["evidence_update"]["recovery_evidence"]
    assert review["candidates"] == []  # Recent context is not a declared target.
    recent = review["recent_evidence_context"][0]
    assert recent["reference"] == ref and not recent["validity"]["usable"]
    assert recent["history"] == previous.result["openvl_vista_link"]["binding_history"][ref]
    assert review["target_review"]["current_target_geometry"] == "unknown"
    assert review["target_review"]["commanded_target"] == [0, 0, 2]
    assert review["current_vista_event"]["event"] == out.result["vista_evidence"]["event"]
    # Same pixels have one image payload but both temporal names remain visible.
    assert review["image_aliases"]
    assert len(out.images) == len(out.result["images"])


def test_combined_history_keeps_capture_pixels_before_same_cell_motion(robo, tmp_path, monkeypatch):
    kit, _ = configure(robo, tmp_path, "openvl_vista")
    backend = robo._backend
    source_step = backend.env_steps()

    def artifact(name, step=-1):
        stream = io.BytesIO()
        color = "red" if backend.env_steps() == source_step else "blue"
        Image.new("RGB", (64, 48), color).save(stream, format="PNG")
        return stream.getvalue()

    monkeypatch.setattr(backend, "artifact", artifact)
    out = kit.execute_tool("python", {"code":
        "robo.segment('bowl', camera='agentview'); "
        "robo.move_to([0,0,0.6]); robo.show('agentview')"})
    ref = kit.evidence_adapter.bridge.refs[-1]
    history = kit.evidence_adapter.binding_history[ref]
    assert history["source_step"] == source_step
    assert history["event"] != out.result["vista_evidence"]["event"]
    assert history["views"]
    inspected = kit.execute_tool("inspect", {"question": "Read original measurement image", "views": [
        {"event": history["event"], "view": history["views"][0], "label": "source"}]})
    assert inspected.result["ok"], inspected.result
    # Native inspect may add a border; its center must contain the old camera pixels.
    with Image.open(io.BytesIO(inspected.images[0])) as picture:
        assert picture.convert("RGB").getpixel((picture.width // 2, picture.height // 2)) == (255, 0, 0)
    assert not kit.evidence_adapter.bridge.validity(ref)["usable"]


def test_vista_archive_error_preserves_executed_cell_feedback(robo, tmp_path, monkeypatch):
    kit, _ = configure(robo, tmp_path, "openvl_vista")

    def broken_archive(*args):
        raise OSError("archive unavailable")

    monkeypatch.setattr(kit.native_vista, "record", broken_archive)
    out = kit.execute_tool("python", {"code":
        "answer = 42; print(robo.move_to([0,0,0.6]).reached)"})
    assert "True" in out.result["stdout"]
    assert kit._globals["answer"] == 42
    assert out.result["vista_error"] == "OSError: archive unavailable"
    assert kit.turns[-1]["moved"]
