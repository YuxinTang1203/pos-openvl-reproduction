"""Mechanism and actual shared-runtime checks; fake robot, never benchmark claims."""
from __future__ import annotations

import copy
import json
import time
from argparse import Namespace

import pytest

from pyrualean.analysis_repl import AnalysisREPL, validate_code
from pyrualean.arms import ArmToolkit
from pyrualean.evo_harness import load_development_episode
from pyrualean.migrations import (
    cli_migration,
    configure_migration,
    migration_prompt,
    validate_migration_runtime,
)
from pyrualean.openai_runtime import run_openai_agent
from pyrualean.reuseit_harness import compile_workflow, freeze_workflow, load_workflow
from tests.test_openai_runtime import ScriptedTransport, _reply


def development(tmp_path):
    records = []
    for name, success in [("failed", False), ("passed", True)]:
        root = tmp_path / name
        root.mkdir()
        (root / "result.json").write_text(json.dumps({
            "task": "development_" + name, "environment_success": success,
            "task_language": "Transfer a test object", "seed": 0, "status": "test_fixture"}))
        (root / "calls.jsonl").write_text(json.dumps({
            "tool": "move_to", "kwargs": {}, "summary": {"reached": success},
            "ok": True, "error": None, "stateful": True}) + "\n")
        records.append(load_development_episode(root, excluded_tasks=["unseen_task"], source_id=name))
    return records


def synthesis():
    return [
        {"guards": [
            {"id": "g-pre", "step": "Observe", "position": "pre", "condition": "Target visible",
             "evidence": ["failed:call0"]},
            {"id": "g-post", "step": "Observe", "position": "post", "condition": "Target still visible",
             "evidence": ["failed:call0"]}]},
        {"fallbacks": [{"id": "f-observe", "step": "Observe", "action": "Reinspect the scene",
                        "evidence": ["passed:call0"]}]},
        {"steps": [{"id": "observe", "action": "Observe target", "pre": ["g-pre"],
                    "post": ["g-post"], "fallbacks": ["f-observe"],
                    "evidence": ["passed:call0"]}]},
    ]


def compiled(tmp_path):
    reports = iter(synthesis())
    return compile_workflow(development(tmp_path), lambda *args: next(reports),
                            excluded_tasks=["unseen_task"])


@pytest.fixture
def workflow_config(tmp_path):
    book = compiled(tmp_path)
    path = tmp_path / "workflow.json"
    digest = freeze_workflow(book, path)
    return {"kind": "reuseit", "workflow": str(path), "sha256": digest}


def kit_for(robo, tmp_path, config):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace",
                     images="on-demand", feedback="pure", max_turns=12)
    configure_migration(kit, config, task="Synthetic protocol task, not benchmark data",
                        initial_images={}, runtime="openai")
    return kit


def run(kit, tmp_path, responses, **options):
    transport = ScriptedTransport(responses)
    result = run_openai_agent(
        migration_prompt("TASK AND API RULES", kit), [], kit, model="same-model",
        base_url="http://unused/v1", api_key=None, transport=transport,
        max_turns=12, budget_s=options.pop("budget_s", 60),
        out_dir=tmp_path / "out", retries=0, **options)
    return transport, result


def call(name, args, ident="c"):
    return _reply([(ident, name, args)])


def report(payload):
    return call("submit_harness_state", {"payload": payload})


def end(status="stuck"):
    return call("finish", {"status": status, "summary": "Synthetic protocol complete"})


def guard(harness, satisfied=True):
    def model_call(role, prompt, payload, images):
        assert role == "reuseit_guard" and images
        assert payload["task"] == harness.trace.task
        return {"satisfied": satisfied, "evidence": "Synthetic controlled observation",
                "evidence_ids": payload["eligible_evidence_ids"]}
    harness.before_request(model_call)


def test_compiler_runs_paper_stages_and_freezes(tmp_path):
    records = development(tmp_path)
    answers = iter(synthesis())
    calls = []

    def model(role, prompt, payload, images):
        calls.append((role, payload))
        return next(answers)

    book = compile_workflow(records, model, excluded_tasks=["unseen_task"])
    assert [c[0] for c in calls] == ["reuseit_conditions", "reuseit_fallbacks", "reuseit_workflow"]
    assert len(calls[0][1]["failed_traces"]) == 1
    assert len(calls[1][1]["successful_traces"]) == 1
    assert calls[2][1]["guards"] == book["guards"]
    path = tmp_path / "frozen.json"
    digest = freeze_workflow(book, path)
    assert load_workflow({"workflow": path, "sha256": digest}) == book
    with pytest.raises(FileExistsError):
        freeze_workflow(book, path)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_workflow({"workflow": path, "sha256": digest})


def test_compiler_refuses_unverifiable_sources_and_leakage(tmp_path):
    records = development(tmp_path)
    invoked = []
    model = lambda *args: invoked.append(args)
    with pytest.raises(ValueError, match="both successful and failed"):
        compile_workflow(records[:1], model, excluded_tasks=["unseen_task"])
    with pytest.raises(ValueError, match="leaked"):
        compile_workflow(records, model, excluded_tasks=[records[0]["task"]])
    path = tmp_path / "failed" / "calls.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        compile_workflow(records, model, excluded_tasks=["unseen_task"])
    assert not invoked


def test_frozen_workflow_cannot_silently_replace_model_checks(tmp_path):
    book = compiled(tmp_path)
    book["guards"][0]["condition"] = "A manually substituted condition"
    with pytest.raises(ValueError, match="synthesis provenance"):
        freeze_workflow(book, tmp_path / "tampered.json")


@pytest.mark.parametrize("mutation", ["false_ref", "position", "unattached", "wrong_source", "verdict_guard", "outcome_only"])
def test_compiler_rejects_invalid_guard_provenance(tmp_path, mutation):
    answers = synthesis()
    if mutation == "false_ref":
        answers[0]["guards"][0]["evidence"] = ["made-up:call0"]
    elif mutation == "position":
        answers[2]["steps"][0]["pre"] = ["g-post"]
    elif mutation == "unattached":
        answers[2]["steps"][0]["fallbacks"] = []
    elif mutation == "wrong_source":
        answers[0]["guards"][0]["evidence"] = ["passed:call0"]
    elif mutation == "verdict_guard":
        answers[0]["guards"][0]["condition"] = "Wait for the authoritative environment verdict"
    else:
        answers[0]["guards"][0]["evidence"] = ["failed:outcome"]
    reports = iter(answers)
    with pytest.raises(ValueError):
        compile_workflow(development(tmp_path), lambda *a: next(reports, answers[-1]), excluded_tasks=["unseen_task"])


def test_reuseit_checks_gate_actions_and_completion(robo, backend, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, workflow_config)
    harness = kit.general_harness
    args = {"step_id": "observe", "mode": "action", "code": "answer = 42"}
    assert "error" in kit.execute_tool("reuseit_step", args).result
    assert kit._cell_index == 0
    assert "error" in kit.execute_tool("python", {"code": "answer = 1"}).result
    assert "error" in kit.execute_tool("finish", {"status": "success", "summary": "skip"}).result
    guard(harness)
    assert harness.phase == "execute"
    assert "exception" not in kit.execute_tool("reuseit_step", args).result
    assert kit._globals["answer"] == 42 and harness.phase == "check_post"
    assert "error" in kit.execute_tool("reuseit_step", args).result
    guard(harness)
    assert harness.phase == "complete"
    kit.execute_tool("finish", {"status": "success", "summary": "model claim only"})
    assert kit.finish["status"] == "success" and not backend.solved()


def test_reuseit_bounded_fallback_rechecks_original_phase(robo, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, {**workflow_config, "max_retries": 1})
    h = kit.general_harness
    guard(h, False)
    assert h.phase == "fallback"
    action = {"step_id": "observe", "mode": "action", "code": "answer = 1"}
    assert "error" in kit.execute_tool("reuseit_step", action).result
    kit.execute_tool("reuseit_step", {**action, "mode": "fallback"})
    assert h.phase == "check_pre" and h.retries == 1
    guard(h, False)
    assert h.phase == "blocked"
    assert "error" in kit.execute_tool("reuseit_step", {**action, "mode": "fallback"}).result
    assert kit._cell_index == 1


def test_reuseit_exception_does_not_advance(robo, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, workflow_config)
    guard(kit.general_harness)
    kit.execute_tool("reuseit_step", {"step_id": "observe", "mode": "action",
                                      "code": "raise ValueError('execution failed')"})
    assert kit.general_harness.index == 0 and kit.general_harness.phase == "check_pre"
    assert kit.general_harness.retries == 0 and kit.general_harness.code_repairs == 1
    assert kit.general_harness.pending_repair["mode"] == "action"


def test_reuseit_post_failure_recovers_then_rechecks_post(robo, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, workflow_config)
    h = kit.general_harness
    guard(h)
    args = {"step_id": "observe", "mode": "action", "code": "x=1"}
    kit.execute_tool("reuseit_step", args)
    guard(h, False)
    assert h.failed_phase == "check_post" and h.phase == "fallback"
    kit.execute_tool("reuseit_step", {**args, "mode": "fallback"})
    assert h.phase == "check_post"
    guard(h)
    assert h.phase == "complete" and kit._cell_index == 2


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_reuseit_uncertain_or_mistyped_boolean_never_admits_action(robo, tmp_path, workflow_config, value):
    kit = kit_for(robo, tmp_path, {**workflow_config, "max_checks": 1})
    h = kit.general_harness
    guard(h, value)
    assert h.phase == "uncertain"
    assert "error" in kit.execute_tool("reuseit_step", {
        "step_id": "observe", "mode": "action", "code": "x=1"}).result
    kit.execute_tool("reuseit_observe", {})
    guard(h)
    assert h.phase == "blocked" and kit._cell_index == 0


def test_reuseit_stale_image_evidence_and_unknown_block(robo, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, workflow_config)
    h = kit.general_harness
    old_ids = []

    def check(role, prompt, payload, images):
        if not old_ids:
            old_ids.extend(payload["eligible_evidence_ids"])
        return {"satisfied": True, "evidence": "try reusing old image", "evidence_ids": old_ids}

    h.before_request(check)
    kit.execute_tool("reuseit_step", {"step_id": "observe", "mode": "action", "code": "x = 1"})
    h.before_request(check)
    assert h.phase == "uncertain" and h.last_guard["valid"] is False
    assert kit.finish is None
    kit.execute_tool("reuseit_observe", {})
    guard(h)
    assert h.phase == "complete"


def test_reuseit_runtime_auxiliary_budget_fail_closed(robo, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, workflow_config)
    transport, result = run(kit, tmp_path, [
        call("reuseit_step", {"step_id": "observe", "mode": "action", "code": "x=1"}), end()
    ], auxiliary_budget_s=0)
    assert kit._cell_index == 0 and result["auxiliary_requests"] == 0
    assert kit.general_harness.phase == "check_pre"
    assert "execution denied" in json.dumps(transport.requests[-1]["body"]["messages"])


def test_reuseit_real_runtime_checks_share_model_and_usage(robo, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, workflow_config)
    # guard observation IDs are independent of camera identities and tool replies.
    pre_ids = ["event-000002-image-0", "event-000002-image-1"]
    post_ids = ["event-000005-image-0", "event-000005-image-1"]
    yes = lambda ids: report({"satisfied": True, "evidence": "controlled protocol fixture", "evidence_ids": ids})
    transport, result = run(kit, tmp_path, [
        yes(pre_ids), call("reuseit_step", {"step_id": "observe", "mode": "action", "code": "answer=42"}),
        yes(post_ids), end("success")])
    assert result["requests"] == 4 and result["auxiliary_requests"] == 2
    assert result["input_tokens"] == 4000 and result["auxiliary_usage"]["prompt"] == 2000
    assert result["python_cells"] == 1 and kit.general_harness.phase == "complete"
    assert all(r["body"]["model"] == "same-model" for r in transport.requests)
    assert result["recap_full"] is None and result["controller"]["kind"] == "reuseit"


def test_analysis_repl_persistence_no_robot_and_hard_limits():
    kernel = AnalysisREPL(timeout_s=1)
    try:
        assert kernel.run("answer = sum(context); print(answer)", [1, 2], None)["stdout"] == "3\n"
        assert kernel.run("answer += 1; answer", [], None)["stdout"] == "4\n"
        assert "dict" in kernel.run("print(type(context)); print(repr(context))", {}, None)["stdout"]
        assert "NameError" in kernel.run("robo.home()", [], None)["error"]
        assert "NameError" in kernel.run("open('/etc/passwd')", [], None)["error"]
        assert "AttributeError" in kernel.run("re.enum.sys.modules['os'].getenv('HOME')", [], None)["error"]
        assert "AttributeError" in kernel.run("json.decoder", [], None)["error"]
        assert "error" in kernel.run("while True: pass", [], None)
        result = kernel.run("llm_query('q', context)", [1], lambda q, c, ids: "answer")
        assert result["stdout"] == "answer\n"
    finally:
        kernel.close()
    assert kernel.process is None


@pytest.mark.parametrize("code", ["import os", "context.__class__", "f.__globals__", "global context"])
def test_analysis_repl_refuses_private_and_imports(code):
    with pytest.raises(ValueError):
        validate_code(code)


def test_analysis_repl_output_and_context_are_bounded():
    kernel = AnalysisREPL(timeout_s=1, max_output=80, max_context_chars=100)
    try:
        result = kernel.run("print('x'*81)", [], None)
        assert "output limit" in result["error"]
        assert len(result["error"]) <= 80
        assert len(kernel.run("raise ValueError('x'*1000)", [], None)["error"]) <= 80
        assert "context too large" in kernel.run("llm_query('query', 'x'*101)", [], None)["error"]
    finally:
        kernel.close()


def test_analysis_timeout_excludes_delayed_parent_model_wait():
    kernel = AnalysisREPL(timeout_s=.15)
    def delayed_query(q, context, image_ids):
        time.sleep(.3)
        return "model reply"
    try:
        result = kernel.run("x=llm_query('query', context); print(x)", [1], delayed_query)
        assert result == {"stdout": "model reply\n"}
        assert "TimeoutError" in kernel.run("while True: pass", [], delayed_query)["error"]
    finally:
        kernel.close()


@pytest.mark.parametrize("arguments", [
    "question='sum', selected_context=[17,29]",
    "prompt='sum', context=[17,29]",
])
def test_rlm_documented_keyword_arguments_reach_parent(arguments):
    kernel = AnalysisREPL(timeout_s=1)
    calls = []
    def callback(question, context, image_ids):
        calls.append((question, context, image_ids))
        return str(sum(context))
    try:
        assert kernel.run(f"print(llm_query({arguments}))", [], callback) == {"stdout": "46\n"}
        assert calls == [("sum", [17, 29], [])]
    finally:
        kernel.close()


def test_rlm_external_history_survives_trimming_and_namespace_is_separate(robo, tmp_path):
    kit = kit_for(robo, tmp_path, {"kind": "rlm", "window_rounds": 1})
    transport, result = run(kit, tmp_path, [
        call("python", {"code": "robot_value=42"}, "p"),
        call("rlm", {"code": "print([e['payload']['arguments'] for e in context['events'] if e['kind']=='tool']); analysis_value=7"}, "a"),
        call("python", {"code": "print(robot_value); print('analysis_value' in locals())"}, "p2"), end()
    ])
    assert result["compactions"] >= 2 and result["auxiliary_requests"] == 0
    assert "robot_value=42" in kit.general_harness.last_analysis["stdout"]
    assert "analysis_value" not in kit._globals
    assert kit._cell_index == 2 and kit.general_harness.kernel.process is None
    assert len(json.loads((tmp_path / "out/messages.json").read_text())) > len(transport.requests[-1]["body"]["messages"])


def test_rlm_recursive_child_executes_selected_context_and_shares_budget(robo, tmp_path):
    kit = kit_for(robo, tmp_path, {"kind": "rlm"})
    transport, result = run(kit, tmp_path, [
        call("rlm", {"code": "answer = llm_query('sum values', [20,22]); print(answer)"}),
        report({"action": "execute", "code": "print(sum(context))"}),
        report({"action": "final", "answer": "42 from selected values"}), end()
    ])
    assert result["requests"] == 4 and result["auxiliary_requests"] == 2
    assert result["auxiliary_usage"]["prompt"] == 2000 and result["input_tokens"] == 4000
    assert kit.general_harness.last_analysis["stdout"] == "42 from selected values\n"
    second_child = transport.requests[2]["body"]["messages"][-1]["content"]
    assert "42" in second_child
    for index in [1, 2]:
        body = transport.requests[index]["body"]
        assert body["model"] == "same-model"
        assert [t["function"]["name"] for t in body["tools"]] == ["submit_harness_state"]
    assert kit._cell_index == 0


def test_rlm_nested_recursion_is_real(robo, tmp_path):
    kit = kit_for(robo, tmp_path, {"kind": "rlm", "max_depth": 2})
    _, result = run(kit, tmp_path, [
        call("rlm", {"code": "print(llm_query('analyze', [42]))"}),
        report({"action": "execute", "code": "x=llm_query('inspect element', context[0]); print(x)"}),
        report({"action": "execute", "code": "print(context)"}),
        report({"action": "final", "answer": "42 at depth two"}),
        report({"action": "final", "answer": "verified nested answer"}), end()
    ])
    assert result["auxiliary_requests"] == 4
    events = [json.loads(line) for line in (kit.general_harness.directory / "events.jsonl").read_text().splitlines()]
    assert {e["depth"] for e in events if e["event"] == "recursive_call"} == {1, 2}
    assert kit.general_harness.last_analysis["stdout"] == "verified nested answer\n"


def test_rlm_child_keeps_earlier_analysis_observations(robo, tmp_path):
    kit = kit_for(robo, tmp_path, {"kind": "rlm"})
    transport, result = run(kit, tmp_path, [
        call("rlm", {"code": "print(llm_query('inspect then answer', [42]))"}),
        report({"action": "execute", "code": "print(type(context)); print(sum(context))"}),
        report({"action": "execute", "code": "print('second observation')"}),
        report({"action": "final", "answer": "42"}), end()])
    last_child_prompt = transport.requests[3]["body"]["messages"][-1]["content"]
    assert "analysis_history" in last_child_prompt and "42" in last_child_prompt
    assert "second observation" in last_child_prompt
    assert result["auxiliary_requests"] == 3


@pytest.mark.parametrize("options", [{"max_subcalls": 1}, {"max_depth": 1}, {"recursive": False}])
def test_rlm_bounds_and_disabled_recursion_are_visible(robo, tmp_path, options):
    kit = kit_for(robo, tmp_path, {"kind": "rlm", **options})
    responses = [call("rlm", {"code": "print(llm_query('first', [1]))"})]
    if options.get("recursive") is not False:
        responses.append(report({"action": "execute", "code": "print(llm_query('nested', context))"}))
    if "max_depth" in options:
        responses.append(report({"action": "final", "answer": "nested call rejected; uncertainty retained"}))
    responses.append(end())
    _, result = run(kit, tmp_path, responses)
    assert kit._cell_index == 0 and result["errors"] == []
    logs = (kit.general_harness.directory / "events.jsonl").read_text()
    if "max_depth" in options:
        assert "recursion depth exhausted" in logs
    else:
        assert "budget exhausted" in logs or "disabled" in logs


def test_rlm_request_reserve_and_provider_failure(robo, tmp_path):
    kit = kit_for(robo, tmp_path, {"kind": "rlm"})
    _, result = run(kit, tmp_path, [
        call("rlm", {"code": "print(llm_query('query', []))"}),
        (400, "provider failed"), end()], max_requests=3)
    assert result["requests"] == 3 and result["auxiliary_requests"] == 1
    assert kit.finish is not None and kit._cell_index == 0
    assert "unavailable" in kit.general_harness.last_analysis["error"]


def test_rlm_unknown_images_never_reach_provider(robo, tmp_path):
    kit = kit_for(robo, tmp_path, {"kind": "rlm"})
    _, result = run(kit, tmp_path, [
        call("rlm", {"code": "llm_query('inspect', [], image_ids=['/etc/passwd'])"}), end()])
    assert result["auxiliary_requests"] == 0
    assert "registered image" in kit.general_harness.last_analysis["error"]


def test_new_controllers_reject_parallel_action_batch(robo, tmp_path):
    kit = kit_for(robo, tmp_path, {"kind": "rlm"})
    _, result = run(kit, tmp_path, [
        _reply([("a", "python", {"code": "bad=1"}), ("b", "python", {"code": "bad=2"})]), end()])
    assert kit._cell_index == 0 and "bad" not in kit._globals
    assert result["python_cells"] == 0


def test_cli_resolves_workflow_and_requires_api_runtime(tmp_path, workflow_config):
    path = tmp_path / "reuseit.json"
    config = {**workflow_config, "workflow": "workflow.json"}
    path.write_text(json.dumps(config))
    parsed = cli_migration(Namespace(general_harness_config=path, runtime="openai"))
    assert parsed["workflow"] == workflow_config["workflow"]
    for conf in [parsed, {"kind": "rlm"}]:
        with pytest.raises(ValueError, match="runtime=openai"):
            validate_migration_runtime(conf, "codex")


def colour_robot(tmp_path):
    from pyrualean.hosts import robodojo_native as native
    from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
    from tests.test_robodojo_colour import BOWL, CUBE, Scene, Segmenter
    scene = Scene([CUBE, BOWL])
    backend = native.NativeBackend(native.NativeToolkit(
        scene, Segmenter(scene), tmp_path / "native", colour_only=True))
    return RobodojoOfficialColourRobot(backend, primitives="no-vla")


def test_reuseit_actual_robodojo_rgb_adapter_with_public_observation(robo, tmp_path, workflow_config):
    robot = colour_robot(tmp_path)
    kit = kit_for(robot, tmp_path, workflow_config)
    guard(kit.general_harness)
    result = kit.execute_tool("reuseit_step", {"step_id": "observe", "mode": "action",
        "code": "robo.home('right'); robo.show('cam_head')"})
    assert not result.result.get("exception")
    assert "general_harness_error" not in result.result
    assert result.images
    guard(kit.general_harness)
    assert kit.general_harness.phase == "complete"
    assert kit.general_harness.trace.epoch == 1
    assert robot._reported_success() is False


def test_rlm_actual_robodojo_rgb_history_and_registered_image_subcall(tmp_path):
    robot = colour_robot(tmp_path)
    kit = kit_for(robot, tmp_path, {"kind": "rlm"})
    transport, result = run(kit, tmp_path, [
        call("python", {"code": "robo.home('right'); robo.show('cam_head')"}),
        call("rlm", {"code": "e=[e for e in context['events'] if e['kind']=='tool'][0]; "
             "print(llm_query('inspect this past frame', e, image_ids=[e['images'][0]['id']]))"}),
        report({"action": "final", "answer": "Past head-camera observation; current object state unknown"}), end()
    ])
    assert result["auxiliary_requests"] == 1
    assert kit.general_harness.trace.epoch == 1
    assert "general_harness_error" not in json.dumps(kit.general_harness.trace.context())
    body = transport.requests[2]["body"]
    user = body["messages"][-1]["content"]
    assert isinstance(user, list) and any(p["type"] == "image_url" for p in user)
    assert "image_metadata" in json.dumps(user) and "epoch" in json.dumps(user)
    assert robot._reported_success() is False


def test_compiler_repairs_invalid_evidence_with_model_and_preserves_audit(tmp_path):
    answers = synthesis()
    invalid = copy.deepcopy(answers[0])
    invalid["guards"][0]["evidence"] = ["failed:call999999"]
    replies = iter([invalid, *answers])
    calls = []

    def model(role, prompt, payload, images):
        calls.append((role, payload))
        return next(replies)

    book = compile_workflow(development(tmp_path), model, excluded_tasks=["unseen_task"])
    assert len(calls) == 4
    assert calls[0][0] == calls[1][0] == "reuseit_conditions"
    assert calls[1][1]["previous_response"] == invalid
    assert "failed:call999999" not in calls[1][1]["allowed_evidence_ids"]
    assert len(book["compilation_audit"][0]["attempts"]) == 2
    assert book["compilation_audit"][0]["attempts"][0]["response"] == invalid
    assert book["guards"] == answers[0]["guards"]


def test_compiler_repairs_step_guard_position_with_model(tmp_path):
    answers = synthesis()
    bad = copy.deepcopy(answers[2])
    bad["steps"][0]["pre"] = ["g-post"]
    replies = iter([answers[0], answers[1], bad, answers[2]])
    book = compile_workflow(development(tmp_path), lambda *a: next(replies),
                            excluded_tasks=["unseen_task"])
    assert book["steps"] == answers[2]["steps"]
    assert len(book["compilation_audit"][2]["attempts"]) == 2


def test_reuseit_checker_receives_public_initial_reference_not_as_current_evidence(robo, tmp_path, workflow_config):
    h = kit_for(robo, tmp_path, workflow_config).general_harness
    initial = copy.deepcopy(h.task_reference)

    def check(role, prompt, payload, images):
        assert payload["task_reference"] == initial
        refs = {i["id"] for i in initial["images"]}
        current = {i["id"] for i in payload["observation"]["images"]}
        assert refs and current and refs.isdisjoint(current)
        assert refs.issubset({iid for iid, _ in images})
        assert refs.isdisjoint(payload["eligible_evidence_ids"])
        return {"satisfied": True, "evidence": "Controlled current RGB",
                "evidence_ids": list(current)}

    h.before_request(check)
    assert h.phase == "execute" and h.task_reference == initial
