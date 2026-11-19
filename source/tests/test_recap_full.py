"""Exercise the recursive runtime with scripted providers and the real cell toolkit."""
import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from pyrualean import cli, play
from pyrualean.arms import ArmToolkit, MAX_TEXT_CHARS
from pyrualean.migrations import cli_migration, configure_migration, migration_prompt, validate_migration_runtime
from pyrualean.openai_runtime import run_openai_agent
from tests.test_openai_runtime import ScriptedTransport, _reply
from tests.test_openvl_cli_migrations import arguments, intercept_episode


def goal(text):
    return {"kind": "goal", "goal": text}


def leaf(text, code):
    return {"kind": "python", "goal": text, "code": code}


def decision(steps=(), *, thought="Use current observations", status="continue", evidence=""):
    return {"thought": thought, "status": status, "subtasks": list(steps), "evidence": evidence}


def reply(i, *steps, **kw):
    return _reply([(f"c{i}", "recap_decide", decision(steps, **kw))])


def make_kit(robo, tmp_path, *, max_turns=10, **options):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace",
                     images="on-demand", feedback="pure", max_turns=max_turns)
    configure_migration(kit, {"kind": "recap_full", **options}, task="Root task",
                        initial_images={}, runtime="openai")
    return kit


def run(kit, transport, tmp_path, **options):
    return run_openai_agent(migration_prompt("TASK AND API RULES", kit), [], kit,
                            model="test-model", base_url="http://unused/v1", api_key=None,
                            max_turns=kit.max_turns, budget_s=options.pop("budget_s", 30),
                            out_dir=tmp_path / "out", transport=transport, retries=0,
                            **options)


def assert_complete_tool_rounds(messages):
    pending = set()
    for m in messages:
        if m["role"] == "assistant":
            assert not pending
            pending = {c["id"] for c in m.get("tool_calls", [])}
        elif m["role"] == "tool":
            assert m["tool_call_id"] in pending
            pending.remove(m["tool_call_id"])
        else:
            assert not pending
    assert not pending


def test_recursive_execution_parent_refinement_and_window(robo, backend, tmp_path):
    kit = make_kit(robo, tmp_path, window_rounds=2)
    provider = ScriptedTransport([
        reply(1, goal("measure"), goal("final check"), thought="ROOT PLAN MUST SURVIVE"),
        reply(2, leaf("measure", "answer = 41; robo.show('agentview')"),
              leaf("obsolete action", "raise RuntimeError('must never run')"), thought="CHILD PLAN"),
        reply(3, leaf("correct the estimate", "answer += 1; print(answer)")),
        reply(4, status="complete", evidence="Cell 2 printed 42"),
        reply(5, leaf("final check", "print(answer)")),
        reply(6, status="complete", evidence="Cell 3 printed 42"),
    ])
    generation = run(kit, provider, tmp_path)
    assert generation["status"] == "completed" and generation["requests"] == 6
    assert generation["python_cells"] == 3 and kit._globals["answer"] == 42
    assert kit.finish["status"] == "success" and not backend.solved()
    assert kit.general_harness.closed and generation["compactions"] == 4
    assert kit.general_harness.frames[0]["id"] == 0
    for r in provider.requests:
        body = r["body"]
        assert body["tool_choice"] == "required" and body["parallel_tool_calls"] is False
        assert {s["function"]["name"] for s in body["tools"]} == {"recap_decide", "finish"}
        assert body["messages"][0]["content"].startswith("TASK AND API RULES")
        assert_complete_tool_rounds(body["messages"])
        assert sum(m["role"] == "assistant" for m in body["messages"]) <= 1
    parent_request = provider.requests[4]["body"]["messages"][-1]["content"]
    assert "ROOT PLAN MUST SURVIVE" in parent_request and "final check" in parent_request
    assert '"phase": "refine_parent"' in parent_request
    # The preceding round's camera image remains paired with its tool result.
    assert any(isinstance(m.get("content"), list) for m in provider.requests[2]["body"]["messages"])
    archive = json.loads((tmp_path / "out/messages.json").read_text())
    assert sum(m["role"] == "assistant" for m in archive) == 6
    assert any("image_ref" in json.dumps(m) for m in archive)
    history = [json.loads(s) for s in (kit.general_harness.directory / "events.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "execute_head" for e in history) == 3
    assert any(e["event"] == "return_to_parent" for e in history)


def test_leaf_failure_returns_to_parent_without_retry(robo, tmp_path):
    kit = make_kit(robo, tmp_path)
    provider = ScriptedTransport([
        reply(1, goal("grasp"), goal("place"), thought="Remember placement"),
        reply(2, leaf("attempt", "attempts = 1; raise RuntimeError('grasp failed')"), goal("lift")),
        reply(3, leaf("repair", "attempts += 1; print('different approach')")),
        reply(4, status="failed", evidence="The grasp is still uncertain"),
        _reply([("stop", "finish", {"status": "stuck", "summary": "Cannot verify grasp"})]),
    ])
    result = run(kit, provider, tmp_path)
    assert result["python_cells"] == 2 and kit._globals["attempts"] == 2
    repair_request = provider.requests[2]["body"]["messages"][-1]["content"]
    assert "repair_plan" in repair_request and "grasp failed" in repair_request
    assert "attempt" in repair_request and "lift" in repair_request
    parent_request = provider.requests[4]["body"]["messages"][-1]["content"]
    assert "agent_reported_failed" in parent_request and "Remember placement" in parent_request
    assert [s["goal"] for s in kit.general_harness.frames[0]["remaining"]] == ["grasp", "place"]
    assert kit.finish["status"] == "stuck"


def test_unprinted_primitive_feedback_requires_plan_repair(robo, tmp_path):
    kit = make_kit(robo, tmp_path)
    out = kit.execute_tool("recap_decide", decision([leaf("move", "robo.move_to([0, 0, 2])")]))
    assert "repair_plan" == kit.general_harness.phase
    assert out.result["recap_full"]["last_outcome"]["diagnostics"]
    assert kit.general_harness.frames[-1]["remaining"][0]["goal"] == "move"


def test_invalid_plan_rejected_atomically_and_no_bypass(robo, tmp_path):
    kit = make_kit(robo, tmp_path, max_depth=1)
    h = kit.general_harness
    before = copy.deepcopy(h.frames)
    for name, args in [
        ("recap_decide", decision([goal("go deeper")])),
        ("recap_decide", decision([leaf("bad", " ")])),
        ("recap_decide", decision([leaf("would move", "x = 1"), {"kind": "bad", "goal": "invalid"}])),
        ("recap_decide", decision(status="complete")),
        ("python", {"code": "x = 1"}),
        ("finish", {"status": "success", "summary": "skip verification"}),
    ]:
        out = kit.execute_tool(name, args)
        assert "error" in out.result
        assert h.frames == before and h.revision == 0
        assert kit._cell_index == 0 and kit.finish is None


def test_future_python_without_code_identifies_tail_and_never_executes_head(robo, tmp_path):
    """Kimi toy-l4 repair loop: valid home head, missing code in the next item."""
    kit = make_kit(robo, tmp_path)
    args = decision([
        leaf("Home left arm and re-locate pieces", "head_executed = True"),
        {"kind": "python", "goal": "Pick up yellow square with better approach"},
        goal("Verify remaining pieces and home both arms"),
    ])
    before = copy.deepcopy(kit.general_harness.frames)
    out = kit.execute_tool("recap_decide", args)
    assert "subtasks[1].code" in out.result["error"]
    assert "kind='goal'" in out.result["error"] and "No head was executed" in out.result["error"]
    assert kit._cell_index == 0 and "head_executed" not in kit._globals
    assert kit.general_harness.frames == before
    repaired = copy.deepcopy(args)
    repaired["subtasks"][1]["kind"] = "goal"
    _, _, steps, _ = kit.general_harness._validate_decision(repaired)
    assert steps[0]["code"] == args["subtasks"][0]["code"]
    assert kit._cell_index == 0  # validation replay is not action replay


def test_tool_schema_requires_code_for_every_python_item(robo, tmp_path):
    kit = make_kit(robo, tmp_path)
    schema = kit.get_tools_spec()[0]["input_schema"]["properties"]["subtasks"]["items"]
    branches = {item["properties"]["kind"]["enum"][0]: item for item in schema["anyOf"]}
    assert "code" in branches["python"]["required"]
    assert branches["python"]["properties"]["code"]["minLength"] == 1
    assert "code" not in branches["goal"]["properties"]
    assert not branches["goal"]["additionalProperties"]
    bad = decision([leaf("valid head", "x = 1"), {"kind": "goal", "goal": "later", "code": ""}])
    out = kit.execute_tool("recap_decide", bad)
    assert "subtasks[1].code" in out.result["error"] and kit._cell_index == 0


def test_multiple_decisions_in_one_response_execute_nothing(robo, tmp_path):
    kit = make_kit(robo, tmp_path)
    bad = decision([leaf("unexpected", "bad = True")])
    provider = ScriptedTransport([
        _reply([("a", "recap_decide", bad), ("b", "recap_decide", bad)]),
        reply(2, leaf("one leaf", "good = True")),
        reply(3, status="complete", evidence="good was set"),
    ])
    result = run(kit, provider, tmp_path)
    assert "bad" not in kit._globals and kit._globals["good"] is True
    assert result["python_cells"] == 1
    assert_complete_tool_rounds(provider.requests[1]["body"]["messages"])


def test_provider_context_overflow_trims_whole_round_and_does_not_repeat_action(robo, tmp_path):
    kit = make_kit(robo, tmp_path)
    provider = ScriptedTransport([
        reply(1, leaf("once", "n = 1")),
        (400, {"error": {"code": "context_length_exceeded"}}),
        reply(2, status="complete", evidence="n equals 1"),
    ])
    result = run(kit, provider, tmp_path)
    assert result["requests"] == 3 and result["compactions"] == 1
    assert result["python_cells"] == 1 and kit._globals["n"] == 1
    assert len(provider.requests[2]["body"]["messages"]) == 2
    assert "once" in provider.requests[2]["body"]["messages"][-1]["content"]


def test_cell_budget_and_request_budget_are_not_bypassed(robo, tmp_path):
    kit = make_kit(robo, tmp_path, max_turns=1)
    provider = ScriptedTransport([
        reply(1, leaf("first", "n = 1")),
        reply(2, leaf("second", "n += 1")),
    ])
    result = run(kit, provider, tmp_path, max_requests=2)
    assert result["max_requests_hit"] and result["python_cells"] == 1
    assert kit._globals["n"] == 1 and kit.turn_budget_exhausted


def test_wall_budget_stops_before_model_request(robo, tmp_path):
    kit = make_kit(robo, tmp_path)
    provider = ScriptedTransport([])
    stopped = []
    result = run(kit, provider, tmp_path, budget_s=0, on_timeout=lambda: stopped.append(True))
    assert result["status"] == "timeout" and not provider.requests and stopped == [True]


@pytest.mark.parametrize("runtime", ["codex", "claude", "openvlharness"])
def test_full_controller_requires_context_ownership(runtime):
    with pytest.raises(ValueError, match="requires runtime=openai"):
        validate_migration_runtime({"kind": "recap_full"}, runtime)


@pytest.mark.parametrize("options", [{"window_rounds": 0}, {"window_rounds": True}, {"max_depth": -1}, {"max_pending": 1000}])
def test_invalid_runtime_limits_rejected(options):
    with pytest.raises(ValueError):
        validate_migration_runtime({"kind": "recap_full", **options}, "openai")


def test_cli_configuration_and_no_model_protocol_override(robo, tmp_path):
    config = Path(__file__).resolve().parents[1] / "integrations/migrations/general_harness/recap_full.json"
    resolved = cli_migration(Namespace(general_harness_config=config, runtime="openai"))
    assert resolved["kind"] == "recap_full" and resolved["window_rounds"] == 16
    kit = make_kit(robo, tmp_path)
    with pytest.raises(ValueError, match="cannot override"):
        run(kit, ScriptedTransport([]), tmp_path, extra_body={"messages": []})


def test_both_cli_layers_and_deployment_use_full_controller(tmp_path, monkeypatch):
    config_dir = Path(__file__).resolve().parents[1] / "integrations/migrations/general_harness"
    captured = intercept_episode(monkeypatch)
    monkeypatch.setattr(cli, "run_host", lambda argv, **kw: play.main(argv[3:]))
    assert cli.main(["play", *arguments(tmp_path), "--general-harness-config",
                     str(config_dir / "recap_full.json"), "--openai-max-requests", "9",
                     "--openai-preserve-reasoning-content"]) == 0
    options = captured[0][2]
    assert options["migration"]["kind"] == "recap_full"
    assert options["runtime"] == "openai" and options["model"] == "unchanged-model"
    assert options["budget_s"] == 37 and options["max_turns"] == 7
    assert options["openai_options"]["max_requests"] == 9
    assert options["openai_options"]["preserve_reasoning_content"] is True
    deploy = yaml.safe_load((config_dir / "deploy_recap_full.yml").read_text())
    validate_migration_runtime(deploy["migration"], deploy["runtime"])
    assert deploy["max_attempts"] == 1 and deploy["openai_options"]["max_requests"] == 128


def test_context_recovery_respects_request_ceiling(robo, tmp_path):
    kit = make_kit(robo, tmp_path)
    provider = ScriptedTransport([
        reply(1, leaf("once", "n = 1")),
        (400, {"error": {"code": "context_length_exceeded"}}),
    ])
    result = run(kit, provider, tmp_path, max_requests=2)
    assert result["requests"] == 2 and result["max_requests_hit"]
    assert result["status"] == "completed" and not result["errors"]
    assert kit._globals["n"] == 1 and kit.finish is None


def test_provider_failure_at_request_ceiling_stays_an_error(robo, tmp_path):
    kit = make_kit(robo, tmp_path)
    result = run(kit, ScriptedTransport([(400, {"error": "unsupported model"})]),
                 tmp_path, max_requests=1)
    assert result["max_requests_hit"] and result["status"] == "error"
    assert result["errors"] and result["python_cells"] == 0


def test_large_stdout_does_not_hide_control_or_original_error(robo, tmp_path):
    kit = make_kit(robo, tmp_path)
    out = kit.execute_tool("recap_decide", decision([
        leaf("try action", "print('x' * 18000); raise RuntimeError('failure marker')")]))
    text = out.content_blocks[0]["text"]
    assert "repair_plan" in text and "failure marker" in text
    assert len(text) <= MAX_TEXT_CHARS


def test_pending_limit_rejects_plan_before_moving(robo, tmp_path):
    kit = make_kit(robo, tmp_path, max_pending=2)
    kit.execute_tool("recap_decide", decision([goal("child"), goal("sibling")]))
    before = copy.deepcopy(kit.general_harness.frames)
    out = kit.execute_tool("recap_decide", decision([leaf("oversized", "n = 1")]))
    assert "error" in out.result and kit.general_harness.frames == before
    assert kit._cell_index == 0 and "n" not in kit._globals
