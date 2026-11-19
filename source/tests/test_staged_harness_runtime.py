"""Phase contracts are exercised at the serialized transport, with a fake clock."""

import copy
import json
from types import SimpleNamespace

import pytest

from tests.test_auxiliary_scheduling import Clock, finish, is_aux, run
from tests.test_openai_runtime import _reply
from tests.test_pos_harness import audit, belief


class TwoStage:
    kind = "two_stage_test"

    def __init__(self, repair=False):
        self.repair = repair
        self.commits = 0

    def maintenance_requirements(self):
        return {
            "min_calls": 2,
            "role_followup_calls": {"pos_belief": 1, "pos_repair": 1, "pos_sentinel": 0},
        }

    def response_schema(self, role):
        key = "accepted" if role == "pos_sentinel" else "world"
        return {
            "type": "object",
            "properties": {key: {"type": "boolean"}},
            "required": [key],
            "additionalProperties": False,
        }

    def before_request(self, call):
        if call("pos_belief", "belief", {}) is None:
            return "Unavailable"
        audit = call("pos_sentinel", "audit", {})
        if audit is None:
            return "Unavailable"
        if self.repair:
            if call("pos_repair", "repair", {}) is None:
                return "Repair unavailable"
            audit = call("pos_sentinel", "audit", {})
            if audit is None:
                return "Repair audit unavailable"
        self.commits += 1
        return "Maintained"

    def after_tool(self, name, result, records):
        return result


def report():
    return _reply([("report", "submit_harness_state", {"payload": {"accepted": True}})])


def test_two_stages_with_historical_belief_latency_get_a_complete_round(
    robo, tmp_path, monkeypatch
):
    clock, sent, harness = Clock(), [], TwoStage()
    latency = iter([76.555, 70.0])

    def transport(url, body, timeout):
        sent.append((copy.deepcopy(body), timeout))
        if is_aux(body):
            elapsed = next(latency)
            assert elapsed <= timeout
            clock.advance(elapsed)
            return 200, report()
        return 200, finish()

    result, events = run(
        robo,
        tmp_path,
        monkeypatch,
        transport,
        clock,
        harness,
        budget_s=5400,
        auxiliary_stage_budget_s=120,
        request_timeout_s=900,
    )
    assert result["options"]["auxiliary_round_budget_s"] == 240
    assert harness.commits == 1 and result["auxiliary_wall_s"] == pytest.approx(146.555)
    assert result["status"] == "completed"
    schemas = [
        body["tools"][0]["function"]["parameters"]["properties"]["payload"]
        for body, _ in sent
        if is_aux(body)
    ]
    assert schemas == [
        harness.response_schema("pos_belief"),
        harness.response_schema("pos_sentinel"),
    ]
    assert any(
        e["type"] == "harness_stage_start" and e["reserved_followup_calls"] == 1 for e in events
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "episode/harness_model_calls.jsonl").read_text().splitlines()
    ]
    assert [row["response_schema"] for row in records] == schemas


def test_actual_pos_commits_only_after_both_schema_bound_stages(robo, tmp_path, monkeypatch):
    from pyrualean.pos_harness import PoSHarness

    harness = PoSHarness(
        SimpleNamespace(arm="cells", workspace=tmp_path / "workspace"),
        {"kind": "pos"},
        "Place the block",
    )
    candidate = belief("obs-0")
    candidate["world"]["states"][0]["status"] = "uncertain"
    clock, sent = Clock(), []
    replies = iter([(76.555, candidate), (70.0, audit())])

    def transport(url, body, timeout):
        sent.append(copy.deepcopy(body))
        if is_aux(body):
            assert harness.revision == 0  # No commit before independent audit returns.
            seconds, payload = next(replies)
            assert timeout >= seconds
            clock.advance(seconds)
            return 200, _reply([("report", "submit_harness_state", {"payload": payload})])
        return 200, finish()

    result, _ = run(
        robo, tmp_path, monkeypatch, transport, clock, harness, budget_s=5400, request_timeout_s=900
    )
    assert harness.revision == 1 and result["auxiliary_requests"] == 2
    assert result["auxiliary_wall_s"] == pytest.approx(146.555)
    assert result["options"]["auxiliary_budget_s"] == 1080
    assert result["options"]["planner_reserve_s"] == 540
    assert result["options"]["auxiliary_round_budget_s"] == 240
    for index, role in enumerate(["pos_belief", "pos_sentinel"]):
        schema = sent[index]["tools"][0]["function"]["parameters"]["properties"]["payload"]
        assert schema == harness.response_schema(role)
        assert schema["additionalProperties"] is False
    assert "POS_CONTEXT" in json.dumps(sent[-1]["messages"])


@pytest.mark.parametrize(
    "options",
    [
        {"auxiliary_round_budget_s": 1},
        {"auxiliary_budget_s": 1},
        {"max_requests": 2},
        {"auxiliary_stage_budget_s": 0},
    ],
)
def test_impossible_chain_is_rejected_before_any_request(robo, tmp_path, monkeypatch, options):
    def transport(*args):
        pytest.fail("invalid maintenance configuration reached the model")

    with pytest.raises(ValueError):
        run(robo, tmp_path, monkeypatch, transport, Clock(), TwoStage(), **options)


def test_remaining_auxiliary_tail_is_skipped_before_starting_belief(robo, tmp_path, monkeypatch):
    clock, harness = Clock(), TwoStage()
    aux, planners = 0, 0

    def transport(url, body, timeout):
        nonlocal aux, planners
        if is_aux(body):
            aux += 1
            clock.advance(0.4)
            return 200, report()
        planners += 1
        if planners == 1:
            return 200, _reply([("obs", "python", {"code": "print('raw evidence')"})])
        assert any(
            m.get("role") == "tool" and "raw evidence" in m["content"] for m in body["messages"]
        )
        return 200, finish()

    result, events = run(robo, tmp_path, monkeypatch, transport, clock, harness)
    assert aux == 2 and planners == 2 and harness.commits == 1
    assert result["auxiliary_wall_s"] == pytest.approx(0.8)
    assert any(
        e["type"] == "harness_maintenance_skipped" and e["reason"] == "incomplete_maintenance_chain"
        for e in events
    )


def test_retry_cannot_take_sentinel_and_main_request_slots(robo, tmp_path, monkeypatch):
    sent = []

    def transport(url, body, timeout):
        sent.append(is_aux(body))
        return (500, "temporary failure") if is_aux(body) else (200, finish())

    result, _ = run(
        robo, tmp_path, monkeypatch, transport, Clock(), TwoStage(), max_requests=3, retries=10
    )
    assert sent == [True, False]
    assert result["status"] == "completed" and result["requests"] == 2


@pytest.mark.parametrize("round_budget,expected_aux,commits", [(2, 2, 0), (4, 4, 1)])
def test_repair_chain_needs_its_complete_budget(
    robo, tmp_path, monkeypatch, round_budget, expected_aux, commits
):
    clock, harness, aux = Clock(), TwoStage(repair=True), 0

    def transport(url, body, timeout):
        nonlocal aux
        if is_aux(body):
            aux += 1
            clock.advance(0.4)
            return 200, report()
        return 200, finish()

    result, events = run(
        robo,
        tmp_path,
        monkeypatch,
        transport,
        clock,
        harness,
        auxiliary_budget_s=4,
        auxiliary_round_budget_s=round_budget,
    )
    assert aux == expected_aux and harness.commits == commits
    assert result["auxiliary_wall_s"] == pytest.approx(expected_aux * 0.4)
    assert any(e["type"] == "harness_stage_chain_skipped" for e in events) == (commits == 0)
