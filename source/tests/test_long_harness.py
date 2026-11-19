"""Evidence lifecycle tests with scripted maintenance calls, no model/simulator."""

import copy
import json
from types import SimpleNamespace

import pytest

from pyrualean.arms import ArmResult
from pyrualean.long_harness import LongHarness, validate_options


def make_harness(tmp_path, **options):
    kit = SimpleNamespace(arm="cells", workspace=tmp_path / "workspace")
    return LongHarness(kit, {"kind": "longharness", **options}, "Put block in bowl")


def record(index=0, *, stateful=False, summary=None):
    return SimpleNamespace(index=index, stateful=stateful, tool="move_to" if stateful else "locate",
                           kwargs={}, summary=summary or {"found": True}, error=None)


def observe(h, *, stateful=False):
    h.after_tool("python", ArmResult({"cell": h.cell_count + 1}), [record(stateful=stateful)])
    return next(reversed(h.sources))


def evidence(eid, source, *, slot="block.location", scope="world", statement="Block beside bowl"):
    return {"id": eid, "sources": [source], "slot": slot,
            "statement": statement, "scope": scope}


def add(h, entry):
    return h.apply_operations([{"op": "upsert", **entry}])[0]


def claim(eid="e1", text="Block beside bowl", scope="world"):
    return {"id": "c1", "text": text, "evidence": [eid], "scope": scope}


def test_unregistered_sources_rejected_without_mutating_bank(tmp_path):
    h = make_harness(tmp_path)
    receipt = add(h, evidence("e1", "fabricated-camera"))
    assert receipt["accepted"] is False and "unregistered" in receipt["reason"]
    assert not h.evidence and h.revision == 0


def test_conflicting_slot_requires_explicit_revision_and_flags_old_claim(tmp_path):
    h = make_harness(tmp_path)
    source = observe(h)
    assert add(h, evidence("e1", source))["accepted"]
    h.update_conclusions({"established": [claim()]})
    assert not add(h, evidence("e2", source, statement="Block in bowl"))["accepted"]
    receipt = h.apply_operations([{"op": "revise", "id": "e1", "replacement":
                                  evidence("e2", source, statement="Block in bowl")}])[0]
    assert receipt["accepted"] and h.evidence["e1"]["status"] == "superseded"
    # Keep prior inference for audit, explicitly expose invalid support.
    assert h.conclusions["established"][0]["text"] == "Block beside bowl"
    assert h.context()["unsupported_claims"][0]["references"] == ["e1"]
    h.update_conclusions({"established": [claim("e2", "Block in bowl")]})
    assert not h.context()["unsupported_claims"]


def test_motion_expires_world_support_and_cannot_be_refreshed_by_recitation(tmp_path):
    h = make_harness(tmp_path)
    source = observe(h)
    add(h, evidence("e1", source))
    h.update_conclusions({"established": [claim()]})
    observe(h, stateful=True)
    assert h.evidence["e1"]["status"] == "stale"
    context = h.context()
    assert context["conclusions"]["established"] == []
    assert context["unsupported_claims"][0]["support_valid"] is False
    assert context["unsupported_claims"][0]["text"] == "Block beside bowl"
    # Keep the original claim in persistent audit state, not in active conclusions.
    assert h.conclusions["established"][0]["text"] == "Block beside bowl"
    assert not add(h, evidence("e2", source))["accepted"]
    # Historical accounts remain possible, but cannot certify present conditions.
    assert add(h, evidence("past", source, scope="history"))["accepted"]
    with pytest.raises(ValueError, match="current world"):
        h.update_conclusions({"established": [claim("past")]})


def test_source_chronology_within_one_multi_action_cell(tmp_path):
    h = make_harness(tmp_path)
    h.after_tool("python", ArmResult({}), [record(0), record(1, stateful=True), record(2)])
    assert not add(h, evidence("before", "s0002"))["accepted"]
    assert add(h, evidence("after", "s0004"))["accepted"]


def test_retraction_cannot_be_cited_and_ids_cannot_be_reused(tmp_path):
    h = make_harness(tmp_path)
    source = observe(h)
    add(h, evidence("e1", source))
    h.apply_operations([{"op": "revise", "id": "e1", "replacement": None}])
    with pytest.raises(ValueError, match="inactive"):
        h.update_conclusions({"hypotheses": [claim()]})
    assert not add(h, evidence("e1", source))["accepted"]


def test_conclusion_partial_merge_is_atomic_and_gaps_drive_observations(tmp_path):
    h = make_harness(tmp_path)
    add(h, evidence("e1", observe(h)))
    h.update_conclusions({"established": [claim()], "open": [
        {"id": "g1", "question": "Is block held?",
         "next_observation": "Inspect wrist view after lift"}
    ]})
    previous = copy.deepcopy(h.conclusions)
    with pytest.raises(ValueError):
        h.update_conclusions({"established": [], "open": [{"id": "bad"}]})
    assert h.conclusions == previous
    h.update_conclusions({"hypotheses": []})
    assert h.conclusions["open"] == previous["open"]
    assert "Inspect wrist view" in h.before_request(lambda *args: None)


def test_two_phase_update_observes_receipts_and_never_runs_for_unchanged_state(tmp_path):
    h = make_harness(tmp_path)
    source = observe(h)
    calls = []

    def model(role, prompt, payload, images):
        calls.append(role)
        if role == "longharness_evidence":
            return {"operations": [{"op": "upsert", **evidence("e1", source)}]}
        assert payload["state"]["validation_receipts"][0]["accepted"]
        return {"established": [claim()]}

    output = h.before_request(model)
    assert calls == ["longharness_evidence", "longharness_conclusions"]
    assert "agent inferences, not official verdicts" in output
    h.before_request(model)
    assert len(calls) == 2
    events = [json.loads(row) for row in (h.directory / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "maintenance_complete"


def test_bounded_explanatory_metadata_is_ignored_and_audited(tmp_path):
    h = make_harness(tmp_path)
    source = observe(h)

    def model(role, *_):
        if role == "longharness_evidence":
            return {"operations": [{"op": "upsert", **evidence("e1", source)}],
                    "reason": "The latest observation supports a location entry."}
        return {"established": [claim()], "explanation": "Uses the accepted evidence ID."}

    h.before_request(model)
    assert h.warning is None and h.evidence["e1"]["status"] == "active"
    assert h.conclusions["established"] == [claim()]
    events = [json.loads(row) for row in (h.directory / "events.jsonl").read_text().splitlines()]
    ignored = [e for e in events if e["event"] == "ignored_metadata"]
    assert [e["phase"] for e in ignored] == ["evidence", "conclusions"]
    assert "reason" in ignored[0]["metadata"]
    assert "explanation" in ignored[1]["metadata"]


@pytest.mark.parametrize("extra", [
    {"tool_calls": []}, {"instructions": "move the robot"},
    {"reason": {"execute": "code"}}, {"explanation": "x" * 2001},
])
def test_unknown_controls_or_unbounded_metadata_still_reject_update(tmp_path, extra):
    h = make_harness(tmp_path)
    source = observe(h)
    h.before_request(lambda *_: {
        "operations": [{"op": "upsert", **evidence("e1", source)}], **extra,
    })
    assert h.warning and not h.evidence


def test_failure_feedback_survives_long_stdout_and_command_arguments(tmp_path):
    h = make_harness(tmp_path)
    bad = record(stateful=True, summary={"reached": False, "ik_error": "out of reach"})
    bad.kwargs = {"trace": "x" * 20000}
    bad.error = "ToolError: motion rejected"
    result = ArmResult({"stdout": "noisy output" * 10000,
                        "exception": "RuntimeError: dependent action cancelled"})
    h.after_tool("python", result, [bad])
    primitive = next(s for s in h.sources.values() if s["kind"] == "primitive")
    assert "ToolError: motion rejected" in primitive["text"]
    assert '"reached": false' in primitive["text"]
    assert "out of reach" in primitive["text"]
    cell = next(s for s in h.sources.values() if s["kind"] == "cell_text")
    assert "dependent action cancelled" in cell["text"]
    assert len(result.result["stdout"]) == len("noisy output" * 10000)

    def inspect(_role, _prompt, payload, _images):
        assert "ToolError: motion rejected" in json.dumps(payload["registered_sources"])
        return None

    h.before_request(inspect)


def test_budget_exhaustion_between_phases_preserves_evidence_and_warns(tmp_path):
    h = make_harness(tmp_path)
    source = observe(h)

    def model(role, *_):
        return {"operations": [{"op": "upsert", **evidence("e1", source)}]} if (
            role == "longharness_evidence") else None

    output = h.before_request(model)
    assert "shared request budget exhausted" in output
    assert h.evidence["e1"]["status"] == "active"
    assert not h.conclusions["established"]


def recorded_qwen_scope_failure():
    """Relevant unmodified entries from the real 2026-11-04 Qwen protocol response.

    Origin: protocol/DEVELOPMENT_CALLS.jsonl, longharness_conclusions request 4.
    Keep the fixture independent of machine-local run archives.
    """
    return {
        "established": [{
            "id": "i1",
            "text": "Task goal (instruction-scope): move the test cube to its target location; "
                    "completion must be judged by the cube occupying that target, not by any "
                    "motion command succeeding.",
            "evidence": ["e1"], "scope": "instruction",
        }],
        "hypotheses": [{
            "id": "h1",
            "text": "The scene contains exactly one test cube and one designated target location "
                    "(marker/zone) that can be detected by an available sensor or scene query.",
            "evidence": ["e1", "e2"], "scope": "world",
        }],
        "open": [{
            "id": "g1",
            "question": "Where is the test cube currently (position/orientation, "
                        "and is it occluded or out of view)?",
            "next_observation": "Run an inspection/observation step (camera snapshot "
                                "or scene-state query) and record the cube's detected pose.",
        }],
    }


def test_real_qwen_scope_error_repaired_once_without_relaxing_source_checks(tmp_path):
    h = make_harness(tmp_path)
    source = next(iter(h.sources))
    bad = recorded_qwen_scope_failure()
    calls = []

    def model(role, prompt, payload, _images):
        calls.append(role)
        if role == "longharness_evidence":
            return {"operations": [
                {"op": "upsert", **evidence("e1", source, scope="instruction", slot="task.goal")},
                {"op": "upsert", **evidence("e2", source, scope="instruction", slot="inspect")},
            ]}
        assert payload["allowed_evidence_by_scope"]["world"] == []
        assert payload["allowed_evidence_by_scope"]["instruction"] == ["e1", "e2"]
        assert "BOTH established AND hypotheses" in prompt
        if role == "longharness_conclusions":
            return bad
        assert role == "longharness_conclusion_repair"
        assert payload["rejected_proposal"] == bad
        assert "hypotheses h1" in payload["validation_error"]
        assert "e1 has scope=instruction" in payload["validation_error"]
        assert not h.conclusions["established"]  # invalid first patch was atomic
        return {"established": bad["established"], "hypotheses": [], "open": [{
            "id": "g1", "question": "Where are the object and target?",
            "next_observation": "Inspect a current public camera image.",
        }]}

    h.before_request(model)
    assert calls == ["longharness_evidence", "longharness_conclusions",
                     "longharness_conclusion_repair"]
    assert h.warning is None and not h.conclusions["hypotheses"]
    assert h.conclusions["established"][0]["scope"] == "instruction"
    events = [json.loads(row) for row in (h.directory / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "conclusion_rejected" and e["proposal"] == bad for e in events)
    assert events[-1]["event"] == "maintenance_complete"


@pytest.mark.parametrize("repair_answer", [None, "same_invalid"])
def test_conclusion_repair_failure_is_bounded_and_does_not_accept_original(tmp_path, repair_answer):
    h = make_harness(tmp_path)
    source = next(iter(h.sources))
    bad = recorded_qwen_scope_failure()
    calls = []

    def model(role, *_):
        calls.append(role)
        if role == "longharness_evidence":
            return {"operations": [
                {"op": "upsert", **evidence("e1", source, scope="instruction", slot="a")},
                {"op": "upsert", **evidence("e2", source, scope="instruction", slot="b")},
            ]}
        if role == "longharness_conclusions" or repair_answer == "same_invalid":
            return bad
        return None

    h.before_request(model)
    assert len(calls) == 3 and h.warning
    assert not h.conclusions["established"] and not h.conclusions["hypotheses"]
    assert h.evidence["e1"]["scope"] == "instruction"


def test_unavailable_conclusion_request_does_not_trigger_repair(tmp_path):
    h = make_harness(tmp_path)
    calls = []

    def model(role, *_):
        calls.append(role)
        return {"operations": []} if role == "longharness_evidence" else None

    h.before_request(model)
    assert calls == ["longharness_evidence", "longharness_conclusions"]
    assert h.warning and h.provenance["max_maintenance_requests"] == 3


@pytest.mark.parametrize("bad", [None, [], {"evidence": []}, {"operations": "bad"}])
def test_bad_model_response_is_logged_without_losing_observations(tmp_path, bad):
    h = make_harness(tmp_path)
    source = observe(h)
    h.before_request(lambda *_: bad)
    assert h.warning and source in h.sources and not h.evidence


def test_active_state_limits_retire_sources_and_old_evidence(tmp_path):
    h = make_harness(tmp_path, max_sources=4, max_evidence=2, max_claims=1)
    for i in range(5):
        source = observe(h)
        assert add(h, evidence(f"e{i}", source, slot=f"slot{i}"))["accepted"]
    assert len(h.sources) == 4 and len(h.evidence) == 2
    assert "s0001" in h.sources  # task source is pinned
    with pytest.raises(ValueError, match="budget"):
        h.update_conclusions({"established": [claim("e3"), claim("e4")]})
    # The append-only source log retains observations removed from active memory.
    assert len((h.directory / "sources.jsonl").read_text().splitlines()) == 6


def test_display_images_and_printed_claims_cannot_assert_current_world(tmp_path):
    h = make_harness(tmp_path)
    h.after_tool("python", ArmResult({"stdout": "It is grasped", "images": ["display_0"]},
                                    [b"png-bytes"]), [])
    display_source = next(reversed(h.sources))

    def model(role, *_):
        if role == "longharness_evidence":
            return {"operations": [{"op": "upsert", **evidence("e1", display_source)}]}
        return {}

    h.before_request(model)
    assert not h.evidence and h.receipts[0]["accepted"] is False
    assert not add(h, evidence("e2", "s0002"))["accepted"]


def test_camera_image_admitted_only_after_actual_maintenance_inspection(tmp_path):
    h = make_harness(tmp_path)
    h.after_tool("python", ArmResult({"images": ["wrist.png"]}, [b"image"]), [])
    source = next(reversed(h.sources))
    assert not add(h, evidence("e1", source))["accepted"]

    def model(role, _prompt, _payload, images):
        if role == "longharness_evidence":
            assert images[0][0].startswith(source)
            return {"operations": [{"op": "upsert", **evidence("e1", source)}]}
        return {"hypotheses": [claim()]}

    h.before_request(model)
    assert h.evidence["e1"]["status"] == "active"


def test_attached_image_labels_resolve_exact_registry_sources(tmp_path):
    h = make_harness(tmp_path, max_images=2)
    h.after_tool("python", ArmResult({"images": ["front.png", "wrist.png", "display_0"]},
                                    [b"front", b"wrist", b"display"]), [])
    calls = []

    def model(role, _prompt, payload, images):
        calls.append(role)
        if role == "longharness_evidence":
            sources = {s["id"]: s for s in payload["registered_sources"]}
            assert len(images) == 2
            assert [data for _, data in images] == [b"front", b"wrist"]
            for label, _ in images:
                sid, name = label.split(":", 1)
                assert sources[sid]["text"] == name
                assert sources[sid]["kind"] == "image"
            assert sources["s0002"]["inspected"] is False  # before the first response
            assert sources["s0002"]["attached_this_request"] is True
            assert sources["s0004"]["attached_this_request"] is False
            return {"operations": []}
        return {}

    h.before_request(model)
    assert h.sources["s0002"]["inspected"] is True
    assert h.sources["s0003"]["inspected"] is True
    assert h.sources["s0004"]["inspected"] is False
    assert h.sources["s0004"]["current_eligible"] is False


def test_initial_three_camera_regression_keeps_head_and_rejects_unshown_source(tmp_path):
    """The stack trial previously registered head but sent only the two wrists."""
    kit = SimpleNamespace(arm="cells", workspace=tmp_path / "workspace")
    h = LongHarness(kit, {"kind": "longharness", "max_images": 2}, "Stack three blocks", initial_images={
        "cam_head.png": b"head", "cam_left_wrist.png": b"left", "cam_right_wrist.png": b"right",
    })

    def model(role, _prompt, payload, images):
        if role == "longharness_evidence":
            assert len(images) == 2 and images[0] == ("s0002:cam_head.png", b"head")
            sent = payload["image_delivery"]["selected_image_ids"]
            assert sent == [label.split(":", 1)[0] for label, _ in images]
            unshown = next(s["id"] for s in payload["registered_sources"]
                          if s["kind"] == "image" and not s["attached_this_request"])
            return {"operations": [{"op": "upsert", **evidence("invented_view", unshown)}]}
        return {}

    h.before_request(model)
    assert h.receipts[0]["accepted"] is False and not h.evidence
    assert h.sources["s0002"]["inspected"] is True
    assert sum(bool(s.get("inspected")) for s in h.sources.values()) == 2


def test_image_selection_follows_question_then_action_arm_without_taking_new_views(tmp_path):
    h = make_harness(tmp_path, max_images=1)
    h.conclusions["open"] = [{"id": "g1", "question": "Is the left gripper holding the block?",
                               "next_observation": "Inspect cam_left_wrist"}]
    names = ["cam_head.png", "cam_left_wrist.png", "cam_right_wrist.png"]
    h.after_tool("python", ArmResult({"images": names}, [b"head", b"left", b"right"]), [])
    chosen = h._select_image_ids()
    assert [h.images[s][1] for s in chosen] == [b"left"]
    h.conclusions["open"] = []
    motion = record(1, stateful=True)
    motion.kwargs = {"arm": "right"}
    h.after_tool("python", ArmResult({"images": names}, [b"new head", b"new left", b"new right"]), [motion])
    chosen = h._select_image_ids()
    assert [h.images[s][1] for s in chosen] == [b"new right"]
    assert len(h.images) == 3  # latest artifact per view, not unbounded frame retention
    h.before_request(lambda *_: None)
    assert not any(s.get("inspected") for s in h.sources.values())
    logs = [json.loads(line) for line in (h.directory / "events.jsonl").read_text().splitlines()]
    delivered = next(e for e in logs if e["event"] == "image_delivery")
    assert delivered["actually_sent_image_ids"] is None  # skipped/failed callback is not proof of delivery


def test_unshown_images_not_admitted_and_official_verdict_fields_filtered(tmp_path):
    h = make_harness(tmp_path, max_images=0)
    h.after_tool("python", ArmResult({"images": ["wrist.png"], "episode_done": True}, [b"x"]),
                 [record(summary={"found": True, "reward": 1, "eval_success": True})])
    source = next(reversed(h.sources))
    h.before_request(lambda role, *_: {"operations": [{"op": "upsert", **evidence("e1", source)}]}
                     if role == "longharness_evidence" else {})
    assert not h.evidence
    assert "eval_success" not in json.dumps(list(h.sources.values()))
    assert "reward" not in json.dumps(list(h.sources.values()))


@pytest.mark.parametrize("config", [
    {"kind": "longharness", "update_every": True},
    {"kind": "longharness", "max_sources": 0},
    {"kind": "longharness", "extra": 1},
])
def test_invalid_options(config):
    with pytest.raises(ValueError):
        validate_options(config)
