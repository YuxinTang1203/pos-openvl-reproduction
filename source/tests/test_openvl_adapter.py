"""Real upstream evidence modules, original PyRUA loop, scripted public robot API."""

import json

import numpy as np
import pytest

pytest.importorskip("openvlharness")

from pyrualean.arms import ArmToolkit
from pyrualean.openai_runtime import run_openai_agent
from pyrualean.openvl_adapter import adapt_prompt
from tests.test_openai_runtime import ScriptedTransport, _reply


def kit_for(robo, tmp_path, **options):
    kit = ArmToolkit(
        robo,
        arm=options.pop("arm", "cells"),
        workspace=tmp_path / "workspace",
        images=options.pop("images", "on-demand"),
        feedback="pure",
    )
    kit.enable_openvl_evidence(**options)
    return kit


def perceive(kit, query="bowl"):
    result = kit.execute_tool(
        "Robot_Perception_Tool", {"operation": "segment", "query": query, "camera": "agentview"}
    )
    assert "error" not in result.result, result.result
    return kit.evidence_adapter.bridge.refs[-1], result


@pytest.mark.parametrize("helper", ["evidence", "evidence_check", "evidence_inspect", "failure_evidence"])
def test_summary_record_is_rejected_with_id_extraction_hint(robo, backend, tmp_path, helper):
    kit = kit_for(robo, tmp_path, temporal_feedback=True)
    ref, _ = perceive(kit)
    row = robo.evidence_find('bowl')[0]
    before = len(backend.calls)
    call = kit._globals[helper]
    with pytest.raises(TypeError, match="dict") as error:
        call(reference=row) if helper == "failure_evidence" else call(row)
    assert "matches[0]['reference']" in str(error.value)
    assert len(backend.calls) == before
    assert kit.evidence_adapter.bridge.read(ref)["evidence_validity"]["usable"]


@pytest.mark.parametrize("bad_ref", [None, [], 1, True, "", "   ", "unknown-id"])
def test_target_requires_explicit_valid_id_before_any_motion(robo, backend, tmp_path, bad_ref):
    kit = kit_for(robo, tmp_path, temporal_feedback=True)
    perceive(kit)
    before = len(backend.calls)
    with pytest.raises((TypeError, ValueError)) as error:
        with robo.evidence_target(bad_ref):
            robo.move_to([0, 0, .6])
    assert "matches[0]['reference']" in str(error.value)
    assert len(backend.calls) == before
    assert kit.evidence_adapter.active_target is None
    assert kit.evidence_adapter.action_links == []


def test_model_can_fix_summary_argument_without_bypassing_binding(robo, backend, tmp_path):
    kit = kit_for(robo, tmp_path, temporal_feedback=True)
    ref, _ = perceive(kit)
    before = len(backend.calls)
    bad = kit.execute_tool('python', {'code':
        "matches = evidence_find('bowl')\n"
        "with evidence_target(matches[0]):\n"
        "    robo.move_to([0,0,0.6])"})
    assert "matches[0]['reference']" in str(bad.result['exception'])
    assert not any(n == 'move_to' for n, _ in backend.calls[before:])
    fixed = kit.execute_tool('python', {'code':
        "ref = matches[0]['reference']\n"
        "with evidence_target(ref):\n"
        "    robo.move_to([0,0,0.6])"})
    assert 'exception' not in fixed.result
    link = fixed.result['evidence_update']['action_targets'][0]
    assert link['target']['reference'] == ref
    assert link['association'] == 'model_declared' and link['validity_before']['usable']
    # Correcting the type must not bypass freshness after the motion.
    with pytest.raises(ValueError, match='Historical'):
        with robo.evidence_target(ref):
            pass


def test_image_id_is_not_a_robot_binding_and_read_default_is_preserved(robo, tmp_path):
    kit = kit_for(robo, tmp_path)
    ref, _ = perceive(kit)
    value = robo.evidence()
    assert value == robo.evidence(None) == robo.evidence(ref)
    with pytest.raises(ValueError, match='image_ref and VISTA event/view'):
        robo.evidence(value['source']['image_ref'])


@pytest.mark.parametrize('operation', ['check', 'inspect'])
@pytest.mark.parametrize('reference_args', [{}, {'reference': ''}, {'reference': {}}, {'reference': 'unknown-id'}])
def test_native_evidence_tool_reference_errors_are_actionable(robo, backend, tmp_path, operation, reference_args):
    kit = kit_for(robo, tmp_path)
    ref, _ = perceive(kit)
    before = len(backend.calls)
    failed = kit.execute_tool('Robot_Evidence_Tool', {'operation': operation, **reference_args})
    assert 'error' in failed.result
    assert "matches[0]['reference']" in failed.result['reference_hint']
    assert len(backend.calls) == before
    fixed = kit.execute_tool('Robot_Evidence_Tool', {'operation': operation, 'reference': ref})
    assert 'error' not in fixed.result


def geometry_backend(backend, monkeypatch):
    original = backend._readonly
    state = {"depth": 0.42, "camera_shift": 0}

    def read(name, args):
        if name == "view_camera_meta":
            return {
                "camera_meta": {
                    "shape": [1024, 1024, 3],
                    "intrinsic_matrix": np.eye(3).tolist(),
                    "extrinsic_matrix": (np.eye(4) + state["camera_shift"]).tolist(),
                    "has_depth": True,
                }
            }
        value = original(name, args)
        if name == "segment":
            value["box"] = [10, 10, 50, 50]
            value["mask_rle"] = {"counts": [100, 200, 300], "size": [1024, 1024]}
        elif name == "back_project":
            value["world_xyz"][2] = state["depth"]
        return value

    monkeypatch.setattr(backend, "_readonly", read)
    return state


def test_native_bindings_masks_and_delta_rendering(robo, backend, tmp_path, monkeypatch):
    geometry_backend(backend, monkeypatch)
    kit = kit_for(robo, tmp_path)
    adapter = kit.evidence_adapter
    ref, result = perceive(kit)
    value = adapter.bridge.read(ref)
    assert value["mask_rle"]["counts"] == [100, 200, 300]
    assert value["source"]["timestamp_utc"] and value["source"]["image_ref"]
    assert "mask_rle" not in json.dumps(result.result)
    value["world_xyz"][0] = 999
    assert adapter.bridge.read(ref)["world_xyz"][0] != 999
    count = sum(n == "segment" for n, _ in backend.calls)
    later = kit.execute_tool(
        "python",
        {
            "code": (
                "print(evidence_find(query='bowl')[0]['reference'])\n"
                "x = evidence(evidence_refs[-1], fields=['world_xyz'])"
            )
        },
    )
    assert "evidence_update" not in later.result
    assert sum(n == "segment" for n, _ in backend.calls) == count
    assert adapter.event_counts["native_capability"] == 1
    assert adapter.provenance["mode"] == "evidence_only"
    assert kit.stage_verifier is None and robo._action_feedback is None


def test_motion_stales_evidence_revalidate_without_sam(robo, backend, tmp_path, monkeypatch):
    geometry_backend(backend, monkeypatch)
    kit = kit_for(robo, tmp_path)
    ref, _ = perceive(kit)
    bridge = kit.evidence_adapter.bridge
    assert ref in bridge.proofs, bridge.entries[ref]
    kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6])"})
    with pytest.raises(ValueError, match="Historical"):
        bridge.read(ref)
    before = sum(n == "segment" for n, _ in backend.calls)
    assert bridge.check(ref)["status"] == "revalidated_visible_region"
    assert bridge.read(ref)["evidence_validity"]["usable"]
    assert sum(n == "segment" for n, _ in backend.calls) == before
    assert len(bridge.refs) == 1  # depth checks do not recursively create bindings


@pytest.mark.parametrize("change", ["depth", "camera", "rgb", "missing"])
def test_changed_evidence_fails_closed(robo, backend, tmp_path, monkeypatch, change):
    state = geometry_backend(backend, monkeypatch)
    kit = kit_for(robo, tmp_path)
    ref, _ = perceive(kit)
    if change == "depth":
        state["depth"] += 0.01
    elif change == "camera":
        state["camera_shift"] = 1
    elif change == "rgb":
        monkeypatch.setattr(robo, "image", lambda *a, **k: np.zeros((1024, 1024, 3), dtype="uint8"))
    else:
        kit.evidence_adapter.bridge.proofs.clear()
    kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6])"})
    assert not kit.evidence_adapter.bridge.check(ref)["usable"]
    with pytest.raises(ValueError, match="Historical"):
        kit.evidence_adapter.bridge.read(ref)
    assert kit.evidence_adapter.bridge.read(ref, allow_historical=True)["world_xyz"]


def test_revision_invalidates_even_without_reported_step_change(
    robo, backend, tmp_path, monkeypatch
):
    kit = kit_for(robo, tmp_path, validate_geometry=False)
    ref, _ = perceive(kit)
    monkeypatch.setattr(backend, "env_steps", lambda: 0)
    kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6])"})
    assert not kit.evidence_adapter.bridge.validity(ref)["usable"]


def test_explicit_target_and_failure_time_survive_later_actions(robo, backend, tmp_path, monkeypatch):
    kit = kit_for(robo, tmp_path, temporal_feedback=True)
    target, _ = perceive(kit, "bowl")
    other, _ = perceive(kit, "plate")
    # A recent unrelated read must not replace the declared target.
    kit.evidence_adapter.bridge.read(other)
    monkeypatch.setattr(robo, "image", lambda *a, **k: np.full(
        (20, 20, 3), len(backend.steps) % 256, dtype=np.uint8))
    out = kit.execute_tool("python", {"code": (
        f"with evidence_target({target!r}, intent='move'):\n"
        "    robo.move_to([0,0,2])\n"
        "    robo.move_to([0,0,0.6])\n"
        "still_executing = True\n")})
    assert "error" not in out.result, out.result
    context = out.result["evidence_update"]["recovery_evidence"]
    failure = context["failure"]
    assert failure["target"]["reference"] == target
    assert context["candidates"][0]["reference"] == target
    assert failure["snapshot"]["step"] < context["current_step"]
    assert out.images[0] != out.images[1]
    assert kit._globals["still_executing"] is True
    assert kit.evidence_adapter.active_target is None
    assert kit.robo._action_feedback is None
    with pytest.raises(ValueError, match="Historical"):
        with kit.evidence_adapter.target_scope(target):
            pass


def test_unassigned_failure_does_not_claim_latest_target(robo, tmp_path):
    kit = kit_for(robo, tmp_path, temporal_feedback=True)
    perceive(kit)
    out = kit.execute_tool("python", {"code": "robo.move_to([0,0,2])"})
    context = out.result["evidence_update"]["recovery_evidence"]
    assert context["failure"]["target"] is None
    assert context["candidates"] == []


def test_snapshot_failure_does_not_hide_action_result(robo, tmp_path, monkeypatch):
    kit = kit_for(robo, tmp_path, temporal_feedback=True)
    def broken(*args, **kwargs):
        raise RuntimeError("camera unavailable")
    monkeypatch.setattr(robo, "image", broken)
    out = kit.execute_tool("python", {"code": "m = robo.move_to([0,0,2]); print(m.reached)"})
    assert "error" not in out.result
    assert "False" in out.result["stdout"]
    assert "camera unavailable" in out.result["evidence_update"]["recovery_evidence"]["failure"]["snapshot_error"]


def test_failure_returns_evidence_and_current_image_preserves_python(robo, tmp_path):
    kit = kit_for(robo, tmp_path, execution_recovery=True)
    ref, _ = perceive(kit)
    result = kit.execute_tool(
        "python",
        {"code": "counter=11\nx=evidence(evidence_refs[-1])\nrobo.move_to([0,0,2])\ncounter=99"},
    )
    assert result.result["recovery_required"]
    context = result.result["evidence_update"]["recovery_evidence"]
    assert context["candidates"][0]["reference"] == ref
    assert context["failure"]["feedback"]["reached"] is False
    assert any(n.startswith("evidence_") for n in result.result["images"])
    assert any(n.startswith("current_") for n in result.result["images"])
    second = kit.execute_tool("python", {"code": "print(counter)\nrobo.move_to([0,0,0.6])"})
    assert second.result["stdout"] == "11\n" and "exception" not in second.result
    # A normal release must never be treated as an empty grasp.
    adapter = kit.evidence_adapter
    before = len(adapter.failures)
    adapter.observe("grip", {"gripper": robo.OPEN}, {"holding": False}, 100)
    assert len(adapter.failures) == before


def test_failure_without_prior_perception_and_images_none(robo, tmp_path):
    kit = kit_for(robo, tmp_path, images="none")
    result = kit.execute_tool("python", {"code": "robo.move_to([0,0,2])"})
    assert result.result["evidence_update"]["recovery_evidence"]["candidates"] == []
    assert not result.images


def test_native_schema_rejects_without_motion(robo, backend, tmp_path):
    kit = kit_for(robo, tmp_path)
    before = len(backend.calls)
    result = kit.execute_tool("Robot_Evidence_Tool", {"operation": "move"})
    assert "error" in result.result
    assert len(backend.calls) == before


def test_original_model_loop_multiple_calls_and_persistent_multiaction_cells(
    robo, backend, tmp_path, monkeypatch
):
    from openvlharness.agent.agent import MainAgent

    def forbidden(*a, **k):
        raise AssertionError("B must not call MainAgent")

    monkeypatch.setattr(MainAgent, "generate", forbidden)
    kit = kit_for(robo, tmp_path)
    provider = ScriptedTransport(
        [
            _reply(
                [
                    (
                        "p1",
                        "Robot_Perception_Tool",
                        {"operation": "segment", "query": "bowl", "camera": "agentview"},
                    ),
                    (
                        "p2",
                        "python",
                        {"code": "counter=3\nrobo.move_to([0,0,0.60])\nrobo.move_to([0,0,0.55])"},
                    ),
                ]
            ),
            _reply(
                [
                    (
                        "p3",
                        "python",
                        {
                            "code": (
                                "print(counter+4)\n"
                                'print(evidence_find(query="bowl")[0]["validity"])'
                            )
                        },
                    )
                ]
            ),
            _reply([("p4", "finish", {"status": "stuck", "summary": "probe"})]),
        ]
    )
    result = run_openai_agent(
        adapt_prompt("Use the robot through ONE tool, python.", kit),
        [],
        kit,
        model="test",
        base_url="http://provider.test/v1",
        api_key="test",
        max_turns=10,
        budget_s=60,
        out_dir=tmp_path / "out",
        transport=provider,
    )
    assert result["status"] == "completed", result
    assert len(provider.requests) == 3
    assert len(kit.turns) == 4 and kit.turns[0]["tool"] == "Robot_Perception_Tool"
    assert sum(n == "move_to" for n, _ in backend.calls) == 2
    assert kit._globals["counter"] == 3
    output = next(
        m for m in provider.requests[-1]["body"]["messages"] if m.get("tool_call_id") == "p3"
    )
    assert json.loads(output["content"])["stdout"].startswith("7\n")
    assert "through ONE tool" not in provider.requests[0]["body"]["messages"][0]["content"]


def test_program_helpers_and_explicit_inspection(robo, tmp_path):
    kit = kit_for(robo, tmp_path, arm="program")
    result = kit.execute_tool(
        "run_program",
        {
            "code": """def run(robo):
    robo.segment('bowl', camera='agentview')
    ref = robo.evidence_find(query='bowl')[0]['reference']
    print(robo.evidence(ref)['world_xyz'])
    robo.evidence_inspect(ref)
"""
        },
    )
    assert "exception" not in result.result, result.result
    assert len(result.images) == 1
    assert kit.evidence_adapter.event_counts["evidence_image_read"] == 1


def test_rgb_only_validation_does_not_call_sam_backprojection(robo, backend, tmp_path, monkeypatch):
    geometry_backend(backend, monkeypatch)
    original = backend._readonly

    def rgb_only(name, args):
        if name == "back_project":
            raise AssertionError("RGB back_project can call SAM: must not be used by cheap check")
        value = original(name, args)
        if name == "view_camera_meta":
            value["camera_meta"].pop("has_depth")
        return value

    monkeypatch.setattr(backend, "_readonly", rgb_only)
    kit = kit_for(robo, tmp_path)
    ref, _ = perceive(kit)
    kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6])"})
    state = kit.evidence_adapter.bridge.check(ref)
    assert state["visual_unchanged"] and not state["usable"]
    assert sum(n == "segment" for n, _ in backend.calls) == 1
    assert not any(n == "back_project" for n, _ in backend.calls)


def test_not_found_preserved_as_negative_evidence(robo, tmp_path):
    kit = kit_for(robo, tmp_path)
    ref, result = perceive(kit, "missing")
    assert kit.evidence_adapter.bridge.raw(ref)["found"] is False
    assert (
        result.result["evidence_update"]["recovery_evidence"]["failure"]["reason"]
        == "object_not_found"
    )
    assert (kit.evidence_adapter.root / f"{ref}.json").is_file()
    assert (kit.evidence_adapter.root / f"{ref}.png").is_file()


def test_actual_robodojo_host_retains_existing_mask_without_resegmenting():
    from pyrualean.hosts import robodojo_native, robodojo_tools
    from tests.test_robodojo_motion import _overhead, _Primitives, _Sam3, _SimArms

    cam, mask = _overhead((0.1, -0.1), 0.1, 0.04, 0.05, 30)
    env = _SimArms()
    env.vision = {"cam_head": cam}
    sam = _Sam3(mask)
    primitives = _Primitives(env, sam)
    value = robodojo_tools.locate(primitives, "red block", include_evidence=True)
    assert len(sam.calls) == 1 and value["found"]
    rle = value["mask_rle"]
    decoded = np.repeat(np.arange(len(rle["counts"])) % 2, rle["counts"]).reshape(rle["size"])
    assert np.array_equal(mask, decoded)

    class Toolkit:
        def execute_tool(self, name, args):
            return robodojo_tools.locate(primitives, **args)

    backend = robodojo_native.NativeBackend(Toolkit())
    plain = backend.execute_tool("locate", {"text_prompt": "red block"})
    assert "mask_rle" not in plain and "box_px" not in plain
    backend.enable_perception_evidence()
    archived = backend.execute_tool("locate", {"text_prompt": "red block"})
    assert archived["mask_rle"] == rle and len(sam.calls) == 3
    # Segment uses the same already-computed mask; provide its normal service bbox.
    original = sam.segment

    def segment(*a, **k):
        res = original(*a, **k)
        res.box = value["box_px"]
        return res

    sam.segment = segment
    segmented = robodojo_native.segment(primitives, "red block", include_evidence=True)
    assert segmented["mask_rle"] == rle and len(sam.calls) == 4


def test_plugin_helpers_do_not_change_baseline_api(robo, tmp_path):
    from pyrualean.prompt import api_reference

    before = api_reference(type(robo))
    kit_for(robo, tmp_path)
    assert api_reference(type(robo)) == before
    assert "robo.evidence" not in before


@pytest.mark.parametrize("enabled", [False, True])
def test_play_episode_evidence_option_uses_original_runtime(
    backend, tmp_path, monkeypatch, enabled
):
    from types import SimpleNamespace

    from PIL import Image

    from pyrualean import play
    from pyrualean.robots import get_adapter

    png = tmp_path / "initial.png"
    Image.new("RGB", (64, 64), "red").save(png)
    card = SimpleNamespace(
        robot="libero",
        max_env_steps=600,
        task_language="move bowl",
        images={"agentview": str(png)},
        render=lambda: "move bowl",
    )
    host = SimpleNamespace(
        boot=lambda *a, **k: (backend, []), dump_card=lambda *a: card, make_backend=lambda x: x
    )
    adapter = SimpleNamespace(**vars(get_adapter("libero")), host_module=lambda: host)
    monkeypatch.setattr(play, "get_adapter", lambda name: adapter)
    monkeypatch.setattr(play, "resolve_rpent_root", lambda explicit=None: tmp_path)
    monkeypatch.chdir(tmp_path)
    cell = SimpleNamespace(
        suite="test", task=0, seed=0, max_episode_steps=600, cuda_device=0, tag="test"
    )
    provider = ScriptedTransport(
        [_reply([("end", "finish", {"status": "stuck", "summary": "probe"})])]
    )
    result = play.play_episode(
        cell,
        tmp_path / "episode",
        arm="cells",
        model="test",
        runtime="openai",
        base_url="http://provider.test/v1",
        budget_s=30,
        openai_options={"transport": provider},
        openvl_evidence={} if enabled else None,
    )
    assert result["status"] == "completed", result
    assert result["generation"]["runtime"] == "openai"
    assert ("openvl_evidence" in result) == enabled
    tools = {t["function"]["name"] for t in provider.requests[0]["body"]["tools"]}
    assert ("Robot_Evidence_Tool" in tools) == enabled
    assert not (tmp_path / "episode/openvlharness_source.json").exists()
    if enabled:
        assert result["openvl_evidence"]["mode"] == "evidence_only"
