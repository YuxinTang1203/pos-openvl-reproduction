"""Closed PoS wire contracts and archived-response replay; no network or simulator."""
import copy
import json
from collections import OrderedDict
from pathlib import Path

import pytest

from pyrualean.openai_runtime import parse_harness_payload
from pyrualean.pos_harness import ProtocolIssues
from tests.test_pos_harness import Script, audit, belief, harness


def test_role_schemas_close_nested_records_and_declare_both_required_stages(tmp_path):
    h = harness(tmp_path)
    schema = h.response_schema("pos_belief")
    assert schema == h.response_schema("pos_repair")
    world = schema["properties"]["world"]
    assert set(world["properties"]) == {"entities", "states", "relations"}
    assert world["additionalProperties"] is False
    entity = world["properties"]["entities"]["items"]
    assert "evidence" in entity["required"]
    assert entity["properties"]["evidence"]["maxItems"] == 8
    assert "obs-0-image-0" in entity["properties"]["evidence"]["items"]["enum"]
    assert world["properties"]["entities"]["maxItems"] == h.options["max_records"]
    assert "COMBINED" in world["description"]
    assert h.maintenance_requirements() == {
        "min_calls": 2,
        "role_followup_calls": {"pos_belief": 1, "pos_repair": 1, "pos_sentinel": 0},
    }
    sentinel = h.response_schema("pos_sentinel")
    assert set(sentinel["required"]) == {"accepted", "issues", "progress", "progress_reason"}
    assert sentinel["properties"]["progress"]["type"] == ["boolean", "null"]
    with pytest.raises(ValueError, match="unknown PoS response role"):
        h.response_schema("planner")


def test_published_schema_is_valid_json_schema_and_rejects_wrong_sibling_nesting(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    h = harness(tmp_path)
    for role, value in (("pos_belief", belief()), ("pos_repair", belief()),
                        ("pos_sentinel", audit())):
        schema = h.response_schema(role)
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(value)
    bad = belief()
    bad["world"]["epistemic"] = []
    del bad["world"]["entities"][0]["evidence"]
    errors = list(jsonschema.Draft202012Validator(h.response_schema("pos_belief")).iter_errors(bad))
    assert len(errors) == 2
    assert any(e.validator == "additionalProperties" for e in errors)
    assert any(e.validator == "required" for e in errors)


def test_all_structural_issues_reach_repair_without_mutating_candidate(tmp_path):
    h = harness(tmp_path)
    bad = belief()
    bad["world"].update(epistemic=[], achievement=[], active_gap=None)
    del bad["world"]["entities"][0]["evidence"]
    # Each group fits individually while their combined total exceeds max_records.
    bad["world"]["entities"] += [
        {"id": f"object-{i}", "name": f"object {i}"} for i in range(13)
    ]
    bad["world"]["states"] *= 14
    bad["world"]["relations"] = [
        {"id": f"r{i}", "source": "block", "relation": "beside", "target": "block",
         "status": "uncertain", "confidence": .5, "evidence": []} for i in range(5)
    ]
    original = copy.deepcopy(bad)
    script = Script(bad, belief(), audit())
    h.before_request(script)
    issues = script.calls[1]["payload"]["repair_issues"]
    assert len([e for e in issues if "missing required fields ['evidence']" in e]) == 14
    assert any("unexpected fields ['achievement', 'active_gap', 'epistemic']" in e for e in issues)
    assert any("total 33 exceeds max_records=24" in e for e in issues)
    assert bad == original and script.calls[1]["payload"]["rejected_candidate"] == original
    assert h.revision == 1  # only the separately supplied, independently audited repair commits


@pytest.mark.parametrize("changed", [
    {"accepted": True},
    {"accepted": True, "issues": [], "progress": "false", "progress_reason": "unknown"},
    {"accepted": True, "issues": "none", "progress": None, "progress_reason": "unknown"},
    {"accepted": 1, "issues": [], "progress": None, "progress_reason": "unknown"},
])
def test_malformed_sentinel_cannot_commit_even_with_truthy_acceptance(tmp_path, changed):
    h = harness(tmp_path, repair_attempts=0)
    h.before_request(Script(belief(), changed))
    assert h.revision == 0 and h.last_status == "audit_rejected"
    assert not h.transitions and not h.belief["world"]["states"]


def test_sentinel_protocol_repair_receives_all_missing_and_extra_fields(tmp_path):
    h = harness(tmp_path)
    model = Script(belief(), {"accepted": True, "unexpected": 1}, belief(), audit(None))
    h.before_request(model)
    issues = model.calls[2]["payload"]["repair_issues"]
    assert any("missing required fields" in e and "progress_reason" in e for e in issues)
    assert any("unexpected fields ['unexpected']" in e for e in issues)
    assert h.revision == 1


@pytest.mark.parametrize("accepted", [True, False])
def test_long_audit_annotations_preserve_decision_and_full_text(tmp_path, accepted):
    h = harness(tmp_path, repair_attempts=0)
    report = audit(accepted=accepted)
    report["issues"] = ["Evidence caveat. " * 80]
    report["progress_reason"] = "Public evidence supports this assessment. " * 80
    original = copy.deepcopy(report)
    assert h._protocol_issues(report, "pos_sentinel") == []
    h.before_request(Script(belief(), report))
    assert h.revision == int(accepted)
    assert h.last_status == ("committed" if accepted else "audit_rejected")
    events = [json.loads(line) for line in (h.directory / "events.jsonl").read_text().splitlines()]
    decision = next(e for e in events if e.get("event") == h.last_status)
    assert decision["audit"] == original
    assert report == original


def test_long_annotations_do_not_bypass_candidate_evidence_validation(tmp_path):
    h = harness(tmp_path, repair_attempts=0)
    invalid = belief("unknown-evidence")
    script = Script(invalid, audit())
    h.before_request(script)
    assert h.revision == 0 and h.last_status == "candidate_rejected"
    assert len(script.calls) == 1  # Invalid candidate never reaches a sentinel.


@pytest.fixture
def archived_calls():
    base = Path(__file__).resolve().parents[1] / (
        "runs/repaired-harness-official-screen-20261104/evaluation/play_stacking_toy-l0-pos/episodes")
    paths = list(base.glob("*/harness_model_calls.jsonl"))
    if len(paths) != 1:
        pytest.skip("local historical public response archive is not bundled with source")
    return [json.loads(line) for line in paths[0].read_text().splitlines()]


def restore_input(h, row):
    content = row["messages"][1]["content"]
    text = content if isinstance(content, str) else next(
        item["text"] for item in content if item.get("type") == "text")
    packet, _ = json.JSONDecoder().raw_decode(text)
    h.options.update(packet["limits"])
    h.motion_epoch = packet["motion_epoch"]
    h.belief = copy.deepcopy(packet["previous_belief"])
    h.evidence = OrderedDict((e["id"], copy.deepcopy(e)) for e in packet["available_evidence"])
    return packet


def test_archived_first_candidate_reports_all_three_real_error_classes(tmp_path, archived_calls):
    h = harness(tmp_path)
    row = next(r for r in archived_calls if r["request"] == 1)
    restore_input(h, row)
    raw, _ = parse_harness_payload(row["response"])
    original = copy.deepcopy(raw)
    with pytest.raises(ProtocolIssues) as error:
        h._validate(raw)
    issues = error.value.issues
    assert len([e for e in issues if "missing required fields ['evidence']" in e]) == 14
    assert any("unexpected fields ['achievement', 'active_gap', 'epistemic']" in e for e in issues)
    assert any("total 33 exceeds max_records=24" in e for e in issues)
    assert raw == original and h.revision == 0


def test_archived_legal_belief_still_requires_sentinel_before_commit(tmp_path, archived_calls):
    h = harness(tmp_path)
    row = next(r for r in archived_calls if r["request"] == 4)
    restore_input(h, row)
    raw, _ = parse_harness_payload(row["response"])
    original, evidence = copy.deepcopy(raw), copy.deepcopy(h.evidence)
    assert h._protocol_issues(raw, "pos_belief") == []
    validated = h._validate(raw)
    assert validated["world"] == raw["world"]
    assert raw == original and h.evidence == evidence
    assert h.revision == 0  # structure validation is not a sentinel judgment or a commit
