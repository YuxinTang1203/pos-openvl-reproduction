"""Exercise the real runtime tool boundary through scripted JSON responses."""

import pytest

from pyrualean.openai_runtime import run_openai_agent
from tests.test_openai_runtime import ScriptedTransport, _reply
from tests.test_reflact_harness import HISTORICAL_SHADOW, arguments, events, kit_for


def run(kit, tmp_path, transport, **options):
    return run_openai_agent(
        kit.general_harness.prompt(), [], kit, model="offline", base_url="http://unused/v1",
        api_key=None, max_turns=2, budget_s=10, out_dir=tmp_path / "episode",
        transport=transport, **options,
    )


def test_schema_and_rejection_feedback_reach_model_then_valid_decision_executes(robo, tmp_path):
    kit = kit_for(robo, tmp_path)
    transport = ScriptedTransport([
        _reply([("bad", "python", {"code": HISTORICAL_SHADOW + "bad_marker = 1"})]),
        _reply([("good", "python", arguments(HISTORICAL_SHADOW + "good_marker = 9"))]),
        _reply([("end", "finish", {"status": "stuck", "summary": "offline"})]),
    ])
    result = run(kit, tmp_path, transport)
    assert result["requests"] == 3 and result["auxiliary_requests"] == 0
    assert result["python_cells"] == 1 and result["status"] == "completed"
    assert "bad_marker" not in kit._globals and kit._globals["good_marker"] == 9
    for request in transport.requests:
        python = next(t["function"] for t in request["body"]["tools"]
                      if t["function"]["name"] == "python")
        assert python["parameters"]["required"] == ["observed_state", "goal_gap", "code"]
        assert python["parameters"]["additionalProperties"] is False
        assert list(python["parameters"]["properties"]) == [
            "observed_state", "goal_gap", "code"]
    bad_feedback = next(m for m in transport.requests[1]["body"]["messages"]
                        if m.get("tool_call_id") == "bad")
    assert "No code was executed" in bad_feedback["content"]
    assert "observed_state" in bad_feedback["content"] and "goal_gap" in bad_feedback["content"]
    assert len(events(kit, "reflection_recorded")) == 1
    assert result["feedback_delivery"]["bad"]["response_requests"] == [2]


@pytest.mark.parametrize("code_first", [True, False])
def test_serialization_order_is_audited_without_rejecting_valid_json(robo, tmp_path, code_first):
    kit = kit_for(robo, tmp_path)
    payload = arguments("marker = 7")
    if code_first:
        payload = {field: payload[field] for field in ("code", "observed_state", "goal_gap")}
    transport = ScriptedTransport([
        _reply([("act", "python", payload)]),
        _reply([("end", "finish", {"status": "stuck", "summary": "offline"})]),
    ])
    result = run(kit, tmp_path, transport)
    assert result["python_cells"] == 1 and kit._globals["marker"] == 7
    row = events(kit, "decision_serialization")[0]
    assert row["argument_order"] == list(payload)
    assert row["reflection_fields_precede_code"] is not code_first


@pytest.mark.parametrize(
    "field", ["model", "messages", "tools", "tool_choice", "parallel_tool_calls"])
def test_harness_schema_and_context_cannot_be_replaced_by_extra_body(robo, tmp_path, field):
    kit = kit_for(robo, tmp_path)
    transport = ScriptedTransport([])
    with pytest.raises(ValueError, match="cannot override"):
        run(kit, tmp_path, transport, extra_body={field: []})
    assert transport.requests == []
