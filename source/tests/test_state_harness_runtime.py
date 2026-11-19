"""Auxiliary inference shares the policy budget and never executes robot tools."""

import json

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.openai_runtime import run_openai_agent
from tests.test_openai_runtime import ScriptedTransport, _reply


class StateHarness:
    kind = "test_state"

    def __init__(self, calls=1):
        self.calls = calls
        self.answers = []

    def before_request(self, model_call):
        if not self.answers:
            for _ in range(self.calls):
                self.answers.append(model_call("state_update", "Describe evidence", {}, []))
        return "Persisted state: " + json.dumps(self.answers)

    def after_tool(self, name, result, records):
        return result


def run(robo, tmp_path, responses, *, calls=1, **kwargs):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace", max_turns=5)
    kit.general_harness = StateHarness(calls)
    transport = ScriptedTransport(responses)
    result = run_openai_agent(
        "Task",
        [],
        kit,
        model="same-model",
        base_url="http://unused/v1",
        api_key=None,
        out_dir=tmp_path / "episode",
        transport=transport,
        budget_s=10,
        max_turns=5,
        sleep=lambda s: None,
        **kwargs,
    )
    return kit, transport, result


def finish():
    return _reply([("f", "finish", {"status": "stuck", "summary": "test"})])


def report(value):
    return _reply([("state", "submit_harness_state", {"payload": value})])


def test_auxiliary_usage_and_same_model_are_in_total(robo, tmp_path):
    kit, transport, result = run(robo, tmp_path, [report({"known": True}), finish()])
    assert result["requests"] == 2
    assert result["auxiliary_requests"] == 1
    assert result["input_tokens"] == 2000
    assert result["auxiliary_usage"]["prompt"] == 1000
    assert result["request_profile"] == [1000, 2000]
    assert not result["warnings"]
    assert kit.general_harness.answers == [{"known": True}]
    assert all(r["body"]["model"] == "same-model" for r in transport.requests)
    assert transport.requests[0]["body"]["tools"][0]["function"]["name"] == "submit_harness_state"
    assert {t["function"]["name"] for t in transport.requests[1]["body"]["tools"]} == {
        "python",
        "finish",
    }
    assert "Persisted state" in transport.requests[1]["body"]["messages"][-1]["content"]
    audit = json.loads((tmp_path / "episode/harness_model_calls.jsonl").read_text())
    assert audit["role"] == "state_update"


def test_request_ceiling_reserves_final_policy_request(robo, tmp_path):
    kit, transport, result = run(robo, tmp_path, [report({}), finish()], calls=3, max_requests=2)
    assert kit.general_harness.answers == [{}, None, None]
    assert result["requests"] == len(transport.requests) == 2
    assert kit.finish is not None


@pytest.mark.parametrize(
    "bad",
    [
        _reply([("malicious", "python", {"code": "raise RuntimeError('must not execute')"})]),
        _reply([("bad", "submit_harness_state", {"payload": []})]),
        _reply(text="not a structured report"),
    ],
)
def test_auxiliary_cannot_execute_policy_tools(robo, tmp_path, bad):
    kit, _, result = run(robo, tmp_path, [bad, finish()])
    assert kit.general_harness.answers == [None]
    assert kit._cell_index == 0
    assert result["auxiliary_usage"]["prompt"] == 1000


def test_auxiliary_transport_failure_is_visible_but_policy_can_continue(robo, tmp_path):
    kit, _, result = run(robo, tmp_path, [(400, "bad request"), finish()], retries=0)
    assert kit.general_harness.answers == [None]
    assert result["auxiliary_requests"] == 1
    assert result["warnings"] and not result["errors"]
    assert result["requests"] == 2 and kit.finish is not None


def test_auxiliary_retry_cannot_consume_reserved_planner_request(robo, tmp_path):
    kit, transport, result = run(
        robo, tmp_path, [(500, "try later"), finish()], retries=2, max_requests=2
    )
    assert kit.general_harness.answers == [None]
    assert result["requests"] == 2 and kit.finish is not None
    assert transport.requests[-1]["body"]["tools"][0]["function"]["name"] == "python"


def test_frozen_context_is_not_reinjected_each_policy_turn(robo, tmp_path):
    _, transport, result = run(
        robo,
        tmp_path,
        [
            report({"skill": "check feedback"}),
            _reply([("p", "python", {"code": "answer = 1"})]),
            finish(),
        ],
    )
    assert result["requests"] == 3 and result["auxiliary_requests"] == 1
    contexts = [
        m
        for m in transport.requests[-1]["body"]["messages"]
        if isinstance(m.get("content"), str) and m["content"].startswith("Persisted state")
    ]
    assert len(contexts) == 1


@pytest.mark.parametrize("value", [{"known": True}, {"payload": {"known": True}}])
def test_gateway_json_text_is_validated_as_state_not_executed(robo, tmp_path, value):
    kit, _, result = run(robo, tmp_path, [_reply(text=json.dumps(value)), finish()])
    assert kit.general_harness.answers == [{"known": True}]
    assert result["auxiliary_requests"] == 1 and result["input_tokens"] == 2000
    assert kit._cell_index == 0
