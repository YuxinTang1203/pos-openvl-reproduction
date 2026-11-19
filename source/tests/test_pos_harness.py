"""PoS state/audit semantics with a scripted maintainer, never a live provider."""

import copy
import json
from types import SimpleNamespace

import pytest

from pyrualean.arms import ArmResult
from pyrualean.pos_harness import (
    DEFAULTS, REFERENCE_TEXT_BUDGET, PoSHarness, trapping_health, validate_options,
)


def harness(tmp_path, **options):
    kit = SimpleNamespace(arm="cells", workspace=tmp_path / "workspace")
    return PoSHarness(
        kit,
        {"kind": "pos", **options},
        "Place the block",
        initial_images={"front.png": b"image-bytes"},
    )


def belief(ref="obs-0-image-0", *, value="on table", confidence=0.9):
    return {
        "world": {
            "entities": [{"id": "block", "name": "block", "evidence": [ref]}],
            "states": [
                {
                    "id": "block-location",
                    "entity": "block",
                    "property": "location",
                    "value": value,
                    "confidence": confidence,
                    "status": "current",
                    "evidence": [ref],
                }
            ],
            "relations": [],
        },
        "epistemic": [{"id": "e1", "target": "Confirm block grasp", "entities": ["block"]}],
        "achievement": [{"id": "a1", "target": "Place the block", "entities": ["block"]}],
        "active_gap": {"type": "epistemic", "id": "e1"},
    }


def audit(progress=False, accepted=True):
    return {
        "accepted": accepted,
        "issues": [] if accepted else ["No grasp evidence"],
        "progress": progress,
        "progress_reason": "public observation",
    }


class Script:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return copy.deepcopy(self.outputs.pop(0))


def observe(h, *, moved=True, result=None):
    records = [
        SimpleNamespace(
            index=h.sequence,
            tool="move_to",
            kwargs={"xyz": [0, 0, 1]},
            summary={"reached": False},
            error=None,
            stateful=moved,
        )
    ]
    return h.after_tool("python", result or ArmResult({"cell": h.sequence + 1}), records)


def test_maintainer_and_independent_sentinel_commit_once(tmp_path):
    h = harness(tmp_path)
    model = Script(belief(), audit())
    text = h.before_request(model)
    assert [c["role"] for c in model.calls] == ["pos_belief", "pos_sentinel"]
    assert h.revision == 1 and h.last_status == "committed"
    assert "POS_CONTEXT" in text and h.belief["world"]["states"][0]["value"] == "on table"
    assert h.belief["active_gap"] == {"type": "epistemic", "id": "e1"}
    assert h.before_request(model) == "" and len(model.calls) == 2
    assert not h.transitions  # initial construction is not an execution transition


def test_old_locate_height_text_reaches_next_sentinel_without_becoming_fresh(tmp_path):
    """Stack PoS req12 lost obs-2 height_m and confused it with width_m."""
    h = harness(tmp_path, repair_attempts=0)
    h.before_request(Script(belief(), audit()))
    row = SimpleNamespace(index=1, tool="locate", stateful=False, kwargs={"text_prompt": "purple block"},
                          summary={"found": True, "height_m": 0.0321, "width_m": 0.0335}, error=None)
    h.after_tool("python", ArmResult({"stdout": "Locate(height_m=0.0321, width_m=0.0335)"}), [row])
    h.before_request(Script(belief("obs-1"), audit()))
    old_source = copy.deepcopy(h.evidence["obs-1"])
    observe(h, moved=True)
    candidate = belief("obs-1")
    candidate["world"]["states"][0]["status"] = "uncertain"
    model = Script(candidate, audit(None))
    h.before_request(model)
    for call in model.calls:  # both maintainer and independently invoked sentinel
        old = next(e for e in call["payload"]["referenced_evidence"] if e["id"] == "obs-1")
        assert old["text_complete"] and "height_m" in old["text"] and "0.0321" in old["text"]
        assert old["motion_epoch"] == 0 and call["payload"]["motion_epoch"] == 1
        assert "obs-1" not in call["payload"]["eligible_current_evidence"]
        assert "0 is KEEP, not OPEN" in call["payload"]["public_api_semantics"]["gripper_commands"]
    assert h.evidence["obs-1"] == old_source
    assert h.belief["world"]["states"][0]["status"] == "uncertain"


def test_historical_text_budget_and_unattached_images_are_explicit(tmp_path):
    h = harness(tmp_path)
    h.evidence["obs-0-image-0"]["inspected"] = True
    for index in range(3):
        h.evidence[f"old-{index}"] = {
            "id": f"old-{index}", "type": "tool", "sequence": 0, "motion_epoch": 0,
            "current_eligible": True, "text": f"old observation {index} " + "x" * 9000,
        }
    old = copy.deepcopy(h.evidence)
    candidate = belief()
    candidate["world"]["entities"][0]["evidence"] = [
        "old-0", "old-1", "old-2", "obs-0-image-0", "retired-source",
    ]
    packet = h._reference_payload({"candidate": candidate}, [], set())
    delivery = packet["evidence_delivery"]
    assert delivery["historical_text_chars"] == REFERENCE_TEXT_BUDGET
    assert delivery["partial_text_reference_ids"] == ["old-0", "old-1"]
    assert set(delivery["not_provided_reference_ids"]) == {"old-2", "obs-0-image-0", "retired-source"}
    assert delivery["attached_image_ids"] == [] and delivery["provided_text_ids"] == []
    assert h.evidence == old  # delivery cannot upgrade age, freshness or inspection


def test_unknown_evidence_cannot_be_approved_by_sentinel(tmp_path):
    h = harness(tmp_path, repair_attempts=0)
    model = Script(belief("imaginary-view"))
    h.before_request(model)
    assert h.revision == 0 and h.last_status == "candidate_rejected"
    assert len(model.calls) == 1


def test_motion_invalidates_old_current_claim_and_rejects_stale_candidate(tmp_path):
    h = harness(tmp_path, repair_attempts=0)
    h.before_request(Script(belief(), audit()))
    observe(h)
    assert h.belief["world"]["states"][0]["status"] == "uncertain"
    model = Script(belief("obs-0", value="held"))
    h.before_request(model)
    assert h.revision == 1 and h.last_status == "candidate_rejected"
    assert h.belief["world"]["states"][0]["value"] == "on table"
    observe(h)
    h.before_request(Script(belief("obs-2", value="held"), audit(True)))
    assert h.revision == 2 and h.belief["world"]["states"][0]["status"] == "current"


def test_conflicting_properties_rejected_atomically(tmp_path):
    h = harness(tmp_path, repair_attempts=0)
    candidate = belief()
    other = copy.deepcopy(candidate["world"]["states"][0])
    other.update(id="other-location", value="held")
    candidate["world"]["states"].append(other)
    h.before_request(Script(candidate))
    assert h.revision == 0 and not h.belief["world"]["states"]


def test_sentinel_repair_is_bounded_and_old_belief_stays_until_accepted(tmp_path):
    h = harness(tmp_path)
    model = Script(belief(value="held"), audit(accepted=False), belief(), audit())
    h.before_request(model)
    assert [c["role"] for c in model.calls] == [
        "pos_belief",
        "pos_sentinel",
        "pos_repair",
        "pos_sentinel",
    ]
    assert model.calls[2]["payload"]["repair_issues"] == ["No grasp evidence"]
    assert not model.calls[2]["payload"]["previous_belief"]["world"]["states"]
    assert h.belief["world"]["states"][0]["value"] == "on table"


@pytest.mark.parametrize("where", ["updater", "sentinel", "exception"])
def test_budget_or_provider_failure_is_auditable_and_not_zero_progress(tmp_path, where):
    h = harness(tmp_path)
    model = Script(None) if where == "updater" else Script(belief(), None)
    if where == "exception":

        def model(**kwargs):
            raise TimeoutError("not enough budget")

    h.before_request(model)
    assert h.revision == 0 and not h.transitions
    events = [json.loads(line) for line in (h.directory / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "maintenance_degraded"
    assert h.before_request(model) == ""  # do not burn all remaining requests retrying maintenance


def test_unverified_progress_does_not_trigger_stagnation(tmp_path):
    h = harness(tmp_path)
    h.before_request(Script(belief(), audit()))
    for i in range(6):
        observe(h)
        h.before_request(Script(belief(f"obs-{i + 1}"), audit(None)))
    assert h.revision == 7 and not h.transitions and not h.health["window_ready"]


@pytest.mark.parametrize("pattern", ["static", "cycle", "drift"])
def test_paper_health_distinguishes_trapping_patterns(pattern):
    histories = []
    for index in range(4):
        item = belief(value=str(index % 2) if pattern == "cycle" else "same")
        if pattern == "drift":
            item["world"]["entities"].append({"id": "other", "name": "unrelated", "evidence": []})
            item["world"]["states"].append(
                {
                    "id": "counter",
                    "entity": "other",
                    "property": "count",
                    "value": str(index),
                    "confidence": 1,
                    "status": "current",
                    "evidence": [],
                }
            )
        histories.append({"progress": False, "belief": item})
    result = trapping_health(histories, histories[-1]["belief"], DEFAULTS)
    assert result["health"] == 0 and result["trapped"]
    assert result["pattern"] == pattern and result["gap_type"] == "epistemic"


def test_static_recovery_has_gap_and_action_evidence(tmp_path):
    h = harness(tmp_path)
    h.before_request(Script(belief(), audit()))
    for i in range(4):
        observe(h)
        h.before_request(Script(belief(f"obs-{i + 1}"), audit(False)))
    assert h.health["pattern"] == "static"
    assert h.recovery["target"]["id"] == "e1"
    assert h.recovery["recent_transitions"][-1]["actions"][0]["summary"]["reached"] is False


def test_active_gap_cannot_be_silently_switched_while_unresolved(tmp_path):
    h = harness(tmp_path)
    h.before_request(Script(belief(), audit()))
    observe(h)
    candidate = belief("obs-1")
    candidate["epistemic"][0]["id"] = "new-name-for-same-gap"
    candidate["active_gap"] = {"type": "achievement", "id": "a1"}
    h.before_request(Script(candidate, audit(False)))
    assert h.belief["epistemic"][0]["id"] == "e1"
    assert h.belief["active_gap"] == {"type": "epistemic", "id": "e1"}


def test_verbose_stdout_never_erases_primitive_failure_evidence(tmp_path):
    h = harness(tmp_path)
    failure = SimpleNamespace(
        index=0,
        tool="move_to",
        kwargs={},
        stateful=True,
        summary={
            "reached": False,
            "ik_error": "target unreachable",
            "reward": 1,
            "environment_success": True,
        },
        error=None,
    )
    h.after_tool("python", ArmResult({"stdout": "x" * 9000}), [failure])
    model = Script(None)
    h.before_request(model)
    evidence = next(e for e in model.calls[0]["payload"]["new_evidence"] if e["id"] == "obs-1")
    trace = json.loads(evidence["text"])
    assert trace["actions"][0]["summary"] == {"reached": False, "ik_error": "target unreachable"}
    assert len(trace["result"]["stdout"]) < 2100


@pytest.mark.parametrize("image_name", ["display_0", None])
def test_display_or_unbound_image_cannot_refresh_current_physical_state(tmp_path, image_name):
    h = harness(tmp_path, repair_attempts=0)
    h.before_request(Script(belief(), audit()))
    result = ArmResult({"images": [image_name]} if image_name else {}, [b"old-array-image"])
    observe(h, result=result)
    model = Script(belief("obs-1-image-0", value="held"))
    h.before_request(model)
    assert h.last_status == "candidate_rejected" and h.revision == 1
    assert not h.evidence["obs-1-image-0"]["current_eligible"]


def test_plain_printed_claim_is_not_current_world_evidence(tmp_path):
    h = harness(tmp_path, repair_attempts=0)
    h.before_request(Script(belief(), audit()))
    h.after_tool("python", ArmResult({"stdout": "The block is grasped"}), [])
    h.before_request(Script(belief("obs-1", value="held")))
    assert h.last_status == "candidate_rejected" and h.revision == 1


def test_unshown_camera_evidence_cannot_support_current_state(tmp_path):
    h = harness(tmp_path, repair_attempts=0, max_images=1)
    h.after_tool("python", ArmResult({"images": ["wrist.png"]}, [b"new-image"]), [])
    # Initial image exists, but only the newest image fits the maintenance call.
    model = Script(belief("obs-0-image-0"))
    h.before_request(model)
    assert [ref for ref, _ in model.calls[0]["images"]] == ["obs-1-image-0"]
    assert h.last_status == "candidate_rejected" and h.revision == 0


def test_confidence_and_freshness_noise_cannot_hide_physical_static_trap():
    histories = []
    for index in range(4):
        item = belief(confidence=0.5 + index * 0.1)
        item["world"]["states"][0]["status"] = "uncertain" if index % 2 else "current"
        histories.append({"progress": False, "belief": item})
    health = trapping_health(histories, histories[-1]["belief"], DEFAULTS)
    assert health["trapped"] and health["pattern"] == "static"


def test_real_protocol_goal_and_observation_meta_states_have_record_specific_repair(tmp_path):
    """Regression from the first real Qwen synthetic protocol smoke, not robot tasks."""
    kit = SimpleNamespace(arm="cells", workspace=tmp_path / "workspace")
    h = PoSHarness(kit, {"kind": "pos"}, "Move an object to its target", initial_images={})
    bad = {
        "world": {
            "entities": [
                {"id": "cube", "name": "object", "evidence": ["obs-0"]},
                {"id": "target", "name": "target", "evidence": ["obs-0"]},
            ],
            "states": [
                {
                    "id": "s-observation-availability",
                    "entity": "cube",
                    "property": "current observation available",
                    "value": "none",
                    "status": "current",
                    "confidence": 0.8,
                    "evidence": ["obs-0"],
                }
            ],
            "relations": [
                {
                    "id": "cube-target-task",
                    "source": "cube",
                    "relation": "must be moved to",
                    "target": "target",
                    "status": "current",
                    "confidence": 1.0,
                    "evidence": ["obs-0"],
                }
            ],
        },
        "epistemic": [
            {
                "id": "inspect",
                "target": "Obtain a current scene observation",
                "entities": ["cube", "target"],
            }
        ],
        "achievement": [
            {
                "id": "a-root",
                "target": "Move an object to its target",
                "entities": ["cube", "target"],
            }
        ],
        "active_gap": {"type": "epistemic", "id": "inspect"},
    }
    repaired = copy.deepcopy(bad)
    repaired["world"]["states"] = []
    repaired["world"]["relations"] = []
    model = Script(bad, repaired, audit(None))
    h.before_request(model)
    assert [c["role"] for c in model.calls] == ["pos_belief", "pos_repair", "pos_sentinel"]
    assert model.calls[0]["payload"]["eligible_current_evidence"] == []
    assert model.calls[1]["payload"]["rejected_candidate"] == bad
    issues = " ".join(model.calls[1]["payload"]["repair_issues"])
    assert "s-observation-availability" in issues and "cube-target-task" in issues
    assert "status=current lacks temporally eligible physical evidence" in issues
    assert "PHYSICAL entities" in model.calls[0]["prompt"]
    assert h.revision == 1 and h.last_status == "committed"
    assert not h.belief["world"]["states"] and not h.belief["world"]["relations"]
    assert h.belief["epistemic"] and h.belief["achievement"]


def test_no_hidden_verdicts_in_maintainer_and_no_calls_after_finish(tmp_path):
    h = harness(tmp_path)
    observe(
        h,
        result=ArmResult(
            {
                "stdout": "hello",
                "environment_success": True,
                "state": {"privileged_object_position": [1, 2, 3]},
            }
        ),
    )
    model = Script(None)
    h.before_request(model)
    serialized = json.dumps(model.calls[0]["payload"])
    assert "hello" in serialized
    assert (
        "environment_success" not in serialized and "privileged_object_position" not in serialized
    )
    h.after_tool("finish", ArmResult({}), [])
    assert h.before_request(Script()) == ""


def test_images_are_bound_to_evidence_and_retirement_invalidates(tmp_path):
    kit = SimpleNamespace(arm="cells", workspace=tmp_path / "workspace")
    h = PoSHarness(
        kit,
        {"kind": "pos", "max_evidence": 2},
        "Place the block",
        initial_images={"front": b"fake-png"},
    )
    model = Script(belief("obs-0-image-0"), audit())
    h.before_request(model)
    assert model.calls[0]["images"] == [("obs-0-image-0", b"fake-png")]
    assert (h.directory / "evidence/obs-0-image-0.png").read_bytes() == b"fake-png"
    observe(h, moved=False)
    observe(h, moved=False)
    state = h.belief["world"]["states"][0]
    assert state["evidence"] == [] and state["status"] == "uncertain"
    assert len(h.evidence) == 2


@pytest.mark.parametrize(
    "options",
    [
        {"health_window": True},
        {"max_cycle_length": 4},
        {"repair_attempts": 2},
        {"health_threshold": float("nan")},
        {"max_images": 0},
        {"unknown": 3},
    ],
)
def test_bad_options_rejected(options):
    with pytest.raises(ValueError):
        validate_options({"kind": "pos", **options})
