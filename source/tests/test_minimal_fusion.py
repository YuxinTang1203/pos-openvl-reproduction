"""Behavioral integration through the real toolkit; no model or simulator service."""
import copy
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.failure_dependencies import FailureDependencies
from pyrualean.minimal_fusion import MinimalFusionHarness, validate_options
from pyrualean.migrations import cli_migration, configure_migration
from pyrualean.openai_runtime import run_openai_agent
from tests.test_composite_v10_feedback import directive
from tests.test_openai_runtime import ScriptedTransport, _reply
from tests.test_openvl_routed import Candidates, CONFIG
from tests.test_pos_harness import Script, audit
from tests.test_robodojo import FakeRobodojoBackend
from pyrualean.robodojo import RobodojoRobot


def setup(tmp_path, mode="advisory", visual=False):
    backend = FakeRobodojoBackend()
    robo = RobodojoRobot(backend)
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace",
                     images="on-demand", feedback="pure")
    if visual:
        from pyrualean.openvl_routed import OpenVLRoutedAdapter
        kit.evidence_adapter = OpenVLRoutedAdapter(kit, config=CONFIG,
            initial_images={}, segmenter=Candidates())
    kit.general_harness = MinimalFusionHarness(kit, {
        "kind": "minimal_fusion", "openvl": {"enabled": visual, "config": str(CONFIG)},
        "dependency_mode": mode}, "Insert all three tubes, one by one")
    return backend, robo, kit


def events(kit, kind):
    return [x for line in (kit.general_harness.directory / "events.jsonl").read_text().splitlines()
            if (x := json.loads(line))["event"] == kind]


def record(i, tool, params, feedback, error=None):
    return SimpleNamespace(index=i, tool=tool, kwargs=params, summary=feedback,
                           error=error, stateful=True)


def observe(tracker, r):
    return tracker.observe(r, close=1, open_=-1, cell=1)


def test_empty_grasp_invalidates_only_declared_dependents_and_preserves_other_arm():
    t = FailureDependencies()
    belief = {"world": {"states": [{"id": "s-held"}], "relations": []}}
    t.begin(1, {"goal": "transport", "dependencies": [
        {"fact": "holding:right", "belief_ids": ["s-held"]}]}, belief)
    observe(t, record(0, "grip", {"arm": "left", "gripper": 1}, {"holding": True}))
    e = observe(t, record(1, "grip", {"arm": "right", "gripper": 1}, {"holding": False}))
    assert e["reason"] == "empty_grasp"
    assert e["affected_declared_dependencies"][0]["belief_ids"] == ["s-held"]
    assert t.facts["holding:left"]["status"] == "reported_holding"
    assert t.facts["holding:right"]["status"] == "contradicted"
    assert not e["identity_inferred"] and not e["task_success_inferred"]


def test_open_is_release_not_empty_grasp_and_true_never_means_identity():
    t = FailureDependencies()
    e = observe(t, record(0, "grip", {"arm": "right", "gripper": -1}, {"holding": False}))
    assert e["reason"] == "release" and not e["review_required"]
    e = observe(t, record(1, "grip", {"arm": "right", "gripper": 1}, {"holding": True}))
    assert e["reason"] == "holding_report" and not e["identity_inferred"]


@pytest.mark.parametrize("summary,reason", [
    ({"reached": False, "steps_used": 0}, "partial_or_unknown_motion"),
    ({"reached": False, "ik_error": "out of reach", "steps_used": 0}, "ik_refused_no_execution"),
    ({"reached": False, "ik_error": "IK stopped", "steps_used": 3}, "partial_or_unknown_motion"),
    ({"reached": False, "ik_error": "IK stopped"}, "partial_or_unknown_motion"),
    ({"reached": False, "blocked": "something in the way", "steps_used": 4}, "partial_or_unknown_motion"),
])
def test_failure_types_require_explicit_execution_evidence(summary, reason):
    t = FailureDependencies()
    e = observe(t, record(0, "move_pose", {"arm": "right"}, summary))
    assert e["reason"] == reason
    assert e["task_success_inferred"] is False
    assert e["updates"]["arrival:right"] == ("contradicted" if reason == "ik_refused_no_execution" else "unknown")


def test_refused_open_does_not_claim_release_or_invalidate_prior_holding_report():
    t = FailureDependencies()
    observe(t, record(0, "grip", {"arm": "right", "gripper": 1}, {"holding": True}))
    observe(t, record(1, "move_pose", {"arm": "right", "gripper": -1},
                      {"reached": False, "ik_error": "refused", "steps_used": 0}))
    assert t.facts["holding:right"]["status"] == "reported_holding"


def test_partial_motion_does_not_preserve_certified_retention():
    t = FailureDependencies()
    observe(t, record(0, "grip", {"arm": "right", "gripper": 1}, {"holding": True}))
    e = observe(t, record(1, "move_pose", {"arm": "right", "gripper": 1},
                          {"reached": False, "blocked": True, "steps_used": 2}))
    assert t.facts["holding:right"]["status"] == "unknown"
    assert "collision" not in e["reason"]


def test_reached_with_blocked_preserves_both_reports_without_inventing_task_failure():
    t = FailureDependencies()
    e = observe(t, record(0, "move_pose", {"arm": "right"},
                          {"reached": True, "blocked": "stopped short", "steps_used": 9}))
    assert e["review_required"] and not e["task_success_inferred"]
    assert e["updates"]["arrival:right"] == "reported_reached"
    assert e["updates"]["geometry:right"] == "unknown"


def test_tool_error_cannot_promote_success_fields_and_both_arm_actions_invalidate_both():
    t = FailureDependencies()
    e = observe(t, record(0, "grip", {"arm": "right", "gripper": 1},
                          {"holding": True}, error="transport interrupted"))
    assert e["reason"] == "tool_error_unknown_effect"
    assert t.facts["holding:right"]["status"] == "unknown"
    observe(t, record(1, "home", {"arm": "both"}, {"reached": True, "steps_used": 3}))
    assert t.facts["holding:left"]["status"] == t.facts["holding:right"]["status"] == "unknown"


def test_declaring_dependency_on_prior_failure_never_certifies_readiness():
    t = FailureDependencies()
    observe(t, record(0, "grip", {"arm": "right", "gripper": 1}, {"holding": False}))
    t.begin(2, {"dependencies": [{"fact": "holding:right", "belief_ids": []}]},
            {"world": {"states": [], "relations": []}})
    prerequisite = t.context()["declared_prerequisites"][0]
    assert prerequisite["report"]["status"] == "contradicted"
    assert prerequisite["satisfied_by_host"] is None


def test_advisory_preserves_native_returns_and_does_not_suppress_followup(tmp_path):
    backend, robo, kit = setup(tmp_path)
    out = kit.execute_tool("python", {"code":
        "g=robo.set_gripper('right',robo.CLOSE)\n"
        "m=robo.move_to([.3,-.2,.95],arm='right',gripper=robo.CLOSE)"})
    assert kit._globals["g"].holding is False
    assert kit._globals["m"].reached is True
    assert not out.result.get("general_harness_error") and not out.result.get("exception")
    assert events(kit, "dependency_followup")[0]["same_cell_after_failure"]
    public = out.result["failure_dependencies"]
    assert public["events"][0]["reason"] == "empty_grasp"
    assert len([x for x in backend.calls if x[0] == "grip"]) == 1
    assert len([x for x in backend.calls if x[0] == "move_pose"]) == 1
    stored = json.loads(kit.general_harness.evidence["obs-1"]["text"])
    assert stored["failure_dependencies"] == public


def test_pause_returns_failed_result_then_stops_next_action_without_replay(tmp_path):
    backend, robo, kit = setup(tmp_path, "pause_cell")
    out = kit.execute_tool("python", {"code":
        "visits=1\ng=robo.set_gripper('right',robo.CLOSE)\n"
        "robo.show('cam_head')\nrobo.move_to([.3,-.2,.95])\nvisits+=1"})
    assert "DependencyReviewRequired" in out.result["exception"]
    assert kit._globals["g"].holding is False and kit._globals["visits"] == 1
    assert not any(x[0] == "move_pose" for x in backend.calls)
    assert out.images  # Observation is allowed before the cell boundary intervention.
    out = kit.execute_tool("python", {"code": "robo.move_to([.3,-.2,.95])"})
    assert not out.result.get("exception")
    assert sum(x[0] == "grip" for x in backend.calls) == 1
    assert sum(x[0] == "move_pose" for x in backend.calls) == 1


def test_exact_retry_is_logged_across_cells_and_different_path_is_distinguished(tmp_path):
    _, _, kit = setup(tmp_path)
    for path in ("line", "line", "auto"):
        kit.execute_tool("python", {"code": f"robo.move_to([.3,-.2,.7],path={path!r})"})
    requests = events(kit, "dependency_request")
    assert [e["exact_rejected_request_repeated"] for e in requests] == [False, True, False]


def test_bad_metadata_never_replays_code_or_removes_original_obligations(tmp_path):
    _, _, kit = setup(tmp_path)
    before = copy.deepcopy(kit.general_harness.belief["achievement"])
    out = kit.execute_tool("python", {"code": "marker=123", "step": {
        "dependencies": [{"fact": "holding:right", "belief_ids": ["invented"]}]}})
    assert kit._globals["marker"] == 123 and kit._cell_index == 1
    assert kit.general_harness.dependencies.warnings
    assert kit.general_harness.belief["achievement"] == before
    assert not out.result.get("general_harness_error")


def test_bound_physical_record_is_invalidated_inline_before_cell_returns(tmp_path):
    _, _, kit = setup(tmp_path)
    h = kit.general_harness
    h.belief["world"]["states"] = [{"id": "s-held", "status": "current", "evidence": []}]
    original_obligations = copy.deepcopy(h.belief["achievement"])
    # The probe observes bookkeeping in the host only; it is not an agent capability.
    statuses = []
    kit._globals["test_probe"] = lambda: statuses.append(h.belief["world"]["states"][0]["status"])
    out = kit.execute_tool("python", {"code": "robo.set_gripper('right',robo.CLOSE)\ntest_probe()",
        "step": {"dependencies": [{"fact": "holding:right", "belief_ids": ["s-held"]}]}})
    assert statuses == ["uncertain"]
    assert h.belief["achievement"] == original_obligations
    assert out.result["failure_dependencies"]["events"][0]["affected_declared_dependencies"][0]["belief_ids"] == ["s-held"]


def test_new_observation_never_clears_failed_prerequisite(tmp_path):
    _, _, kit = setup(tmp_path)
    kit.execute_tool("python", {"code": "robo.set_gripper('right',robo.CLOSE)"})
    kit.execute_tool("python", {"code": "robo.show('cam_head')"})
    assert kit.general_harness.dependencies.facts["holding:right"]["status"] == "contradicted"


def test_all_auxiliary_roles_receive_same_dependency_sources(tmp_path):
    _, _, kit = setup(tmp_path)
    kit.execute_tool("python", {"code": "robo.set_gripper('right',robo.CLOSE)"})
    h = kit.general_harness
    bad = copy.deepcopy(h.belief)
    bad["extra"] = 1
    model = Script(bad, copy.deepcopy(h.belief), {**audit(None), "directive": directive()})
    h.before_request(model)
    assert [x["role"] for x in model.calls] == ["pos_belief", "pos_repair", "pos_sentinel"]
    contexts = [x["payload"]["failure_dependencies"] for x in model.calls]
    assert contexts[0] == contexts[1] == contexts[2]
    assert contexts[0]["facts"]["holding:right"]["status"] == "contradicted"
    assert h.last_status == "committed"


def test_openvl_selection_sources_survive_minimal_fusion(tmp_path):
    _, _, kit = setup(tmp_path, visual=True)
    out = kit.execute_tool("python", {"code": "robo.segment('blocks')"})
    assert "PerceptionChoiceRequired" in out.result["exception"]
    packet = kit.general_harness.visual_handoff()["recent_packets"][-1]
    assert "candidate_or_routing_feedback" in packet["evidence_kinds"]
    assert not packet["current_image_evidence_ids"]
    assert not kit.general_harness.dependencies.facts


def test_registered_migration_and_relative_config_path(tmp_path, monkeypatch):
    import pyrualean.openvl_routed as routed
    original = routed.OpenVLRoutedAdapter
    monkeypatch.setattr(routed, "OpenVLRoutedAdapter", lambda *a, **kw:
                        original(*a, **kw, segmenter=Candidates()))
    backend = FakeRobodojoBackend()
    robo = RobodojoRobot(backend)
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace")
    cfg = tmp_path / "minimal.json"
    cfg.write_text(json.dumps({"kind": "minimal_fusion", "openvl": {"config": str(CONFIG)}}))
    config = cli_migration(Namespace(general_harness_config=cfg, runtime="openai"))
    r = configure_migration(kit, config, task="Insert tubes", initial_images={})
    assert isinstance(kit.general_harness, MinimalFusionHarness) and kit.evidence_adapter
    assert r["adaptation"]["dependency_mode"] == "advisory"


def test_runtime_has_no_added_model_calls_and_finish_still_requires_audit(tmp_path):
    _, _, kit = setup(tmp_path)
    pending = kit.execute_tool("finish", {"status": "success", "summary": "claim"})
    assert pending.result["completion_review_pending"] and kit.finish is None
    transport = ScriptedTransport([
        _reply([("grip", "python", {"code": "robo.set_gripper('right',robo.CLOSE)"})]),
        _reply([("stop", "finish", {"status": "stuck", "summary": "offline test"})])])
    r = run_openai_agent(kit.general_harness.prompt(), [], kit, model="offline",
        base_url="http://unused/v1", api_key=None, transport=transport, budget_s=60,
        max_turns=3, max_requests=3, auxiliary_budget_s=0, out_dir=tmp_path / "runtime")
    assert r["requests"] == 2 and r["auxiliary_requests"] == 0 and r["python_cells"] == 1
    assert not r["errors"]


def test_off_ablation_does_not_instrument_or_change_prompt(tmp_path):
    _, _, kit = setup(tmp_path, "off")
    kit.execute_tool("python", {"code": "robo.set_gripper('right',robo.CLOSE)"})
    assert not events(kit, "dependency_update")
    assert "# Failure dependencies" not in kit.general_harness.prompt()
    schema = next(t for t in kit.get_tools_spec() if t["name"] == "python")["input_schema"]
    assert schema["required"] == ["code"]
    assert "dependencies" not in schema["properties"]["step"]["properties"]


@pytest.mark.parametrize("patch", [{"reflact": {"enabled": True}}, {"action_focus": {"enabled": "invalid"}},
                                   {"dependency_mode": "invalid"}, {"pos": {"unknown": 1}}])
def test_minimal_config_rejects_extra_layers(patch):
    with pytest.raises(ValueError):
        validate_options({"kind": "minimal_fusion", "openvl": {"enabled": False}, **patch})
