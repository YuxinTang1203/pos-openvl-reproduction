"""Deterministic transport/clock regressions; no model or simulator requests."""

import copy
import json

import pytest

from pyrualean import openai_runtime
from pyrualean.arms import ArmToolkit
from tests.test_openai_runtime import _reply


class Clock:
    def __init__(self):
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds


class Maintenance:
    kind = "test_maintenance"

    def __init__(self, pending=False):
        self.pending = pending

    def after_tool(self, name, result, records):
        self.pending = True
        return result

    def before_request(self, model_call):
        if self.pending:
            self.pending = False
            model_call("test_evidence", "Review public evidence", {})
        return ""


def run(robo, tmp_path, monkeypatch, transport, clock, harness, **options):
    monkeypatch.setattr(openai_runtime.time, "perf_counter", lambda: clock.now)
    kit = ArmToolkit(
        robo,
        arm="cells",
        workspace=tmp_path / "workspace",
        max_turns=5,
        feedback="pure",
        images="on-demand",
    )
    kit.general_harness = harness
    result = openai_runtime.run_openai_agent(
        "Task",
        [],
        kit,
        model="offline",
        base_url="http://unused",
        api_key=None,
        max_turns=5,
        budget_s=options.pop("budget_s", 10),
        out_dir=tmp_path / "episode",
        transport=transport,
        sleep=clock.advance,
        request_timeout_s=options.pop("request_timeout_s", 10),
        **options,
    )
    events = [
        json.loads(x) for x in (tmp_path / "episode/openai_events.jsonl").read_text().splitlines()
    ]
    return result, events


def is_aux(body):
    return body["tools"][0]["function"]["name"] == "submit_harness_state"


def finish():
    return _reply([("finish", "finish", {"status": "stuck", "summary": "offline"})])


def test_aux_timeout_preserves_time_for_planner_to_receive_tool_text_and_images(
    robo, tmp_path, monkeypatch
):
    clock = Clock()
    sent = []

    def transport(url, body, timeout):
        sent.append((copy.deepcopy(body), timeout))
        if is_aux(body):
            clock.advance(timeout)
            return 0, "auxiliary timed out"
        clock.advance(1)
        if len(sent) == 1:
            return 200, _reply(
                [
                    (
                        "observation",
                        "python",
                        {"code": "print('latest evidence'); robo.show('agentview')"},
                    )
                ]
            )
        return 200, finish()

    result, events = run(
        robo,
        tmp_path,
        monkeypatch,
        transport,
        clock,
        Maintenance(),
        auxiliary_budget_s=9,
        auxiliary_round_budget_s=9,
        planner_reserve_s=3,
    )
    assert len(sent) == 3
    assert sent[1][1] == pytest.approx(6)
    assert result["status"] == "completed" and result["auxiliary_wall_s"] == 6
    messages = sent[-1][0]["messages"]
    assert any(m["role"] == "tool" and "latest evidence" in m["content"] for m in messages)
    assert any(
        m["role"] == "user"
        and isinstance(m.get("content"), list)
        and any(p.get("type") == "image_url" for p in m["content"])
        for m in messages
    )
    assert result["feedback_delivery"]["observation"] == {
        "submitted_requests": [3],
        "response_requests": [3],
    }
    # Normal finish replies are queued but do not require a further planner turn.
    assert result["feedback_delivery"]["finish"] == {
        "submitted_requests": [],
        "response_requests": [],
    }
    assert any(e["type"] == "planner_feedback_response" for e in events)


def test_aux_total_and_round_allowances_do_not_reset_after_tool(robo, tmp_path, monkeypatch):
    clock, sent = Clock(), []
    planners = 0

    def transport(url, body, timeout):
        nonlocal planners
        sent.append((is_aux(body), timeout))
        if is_aux(body):
            clock.advance(timeout)
            return 200, _reply([("report", "submit_harness_state", {"payload": {"known": True}})])
        planners += 1
        clock.advance(0.1)
        return 200, (
            finish()
            if planners == 3
            else _reply([(f"p{planners}", "python", {"code": "print('feedback')"})])
        )

    result, events = run(
        robo,
        tmp_path,
        monkeypatch,
        transport,
        clock,
        Maintenance(True),
        auxiliary_budget_s=2,
        auxiliary_round_budget_s=1,
        planner_reserve_s=1,
    )
    assert [timeout for aux, timeout in sent if aux] == pytest.approx([1, 1])
    assert result["auxiliary_requests"] == 2 and planners == 3
    assert result["auxiliary_wall_s"] == 2
    assert any(e["type"] == "harness_maintenance_skipped" for e in events)


@pytest.mark.parametrize("status_code", [429, 500])
def test_aux_retry_backoff_obeys_same_deadline(robo, tmp_path, monkeypatch, status_code):
    clock, sent = Clock(), []

    def transport(url, body, timeout):
        sent.append(is_aux(body))
        if is_aux(body):
            clock.advance(0.25)
            return status_code, "retry later"
        return 200, finish()

    result, _ = run(
        robo,
        tmp_path,
        monkeypatch,
        transport,
        clock,
        Maintenance(True),
        auxiliary_budget_s=8,
        auxiliary_round_budget_s=1,
        planner_reserve_s=1,
    )
    assert sent == [True, False]
    assert clock.now == pytest.approx(1)
    assert result["requests"] == 2 and result["status"] == "completed"


def test_zero_aux_allowance_keeps_raw_feedback_and_exception_events(robo, tmp_path, monkeypatch):
    clock, sent = Clock(), []

    def transport(url, body, timeout):
        assert not is_aux(body)
        sent.append(copy.deepcopy(body))
        if len(sent) == 1:
            return 200, _reply([("bad", "python", {"code": "raise ValueError('observed error')"})])
        return 200, finish()

    result, events = run(
        robo, tmp_path, monkeypatch, transport, clock, Maintenance(True), auxiliary_budget_s=0
    )
    assert result["auxiliary_requests"] == 0
    assert any(
        m["role"] == "tool" and "observed error" in m["content"] for m in sent[-1]["messages"]
    )
    assert next(e for e in events if e["type"] == "tool" and e["call_id"] == "bad")["ok"] is False


@pytest.mark.parametrize(
    "name",
    [
        "auxiliary_budget_s",
        "auxiliary_round_budget_s",
        "auxiliary_stage_budget_s",
        "planner_reserve_s",
    ],
)
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_aux_options_fail_before_transport(robo, tmp_path, monkeypatch, name, value):
    def transport(*args):
        pytest.fail("invalid configuration reached transport")

    with pytest.raises(ValueError, match=name):
        run(robo, tmp_path, monkeypatch, transport, Clock(), Maintenance(), **{name: value})


def test_maintenance_error_warns_planner_without_losing_feedback(robo, tmp_path, monkeypatch):
    class Broken(Maintenance):
        def before_request(self, model_call):
            if self.pending:
                raise ValueError("offline maintenance failure")
            return ""

    sent = []

    def transport(url, body, timeout):
        sent.append(copy.deepcopy(body))
        if len(sent) == 1:
            return 200, _reply([("obs", "python", {"code": "print('new evidence')"})])
        return 200, finish()

    result, events = run(robo, tmp_path, monkeypatch, transport, Clock(), Broken())
    messages = sent[-1]["messages"]
    assert any(m["role"] == "tool" and "new evidence" in m["content"] for m in messages)
    assert any(
        m["role"] == "user" and "Auxiliary maintenance failed" in m["content"] for m in messages
    )
    assert result["status"] == "completed"
    assert any(e["type"] == "harness_maintenance_error" for e in events)
