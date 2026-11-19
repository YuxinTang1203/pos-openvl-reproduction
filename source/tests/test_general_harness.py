"""Offline integration tests: no real model requests or simulator startup."""

import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pyrualean import cli
from pyrualean import play as play_module
from pyrualean.arms import MAX_TEXT_CHARS, ArmResult, ArmToolkit
from pyrualean.general_harness import curate_playbook, load_playbook
from pyrualean.migrations import (
    cli_migration,
    configure_migration,
    migration_prompt,
    validate_migration_runtime,
)
from pyrualean.openai_runtime import run_openai_agent
from tests.test_openai_runtime import ScriptedTransport, _reply
from tests.test_openvl_cli_migrations import arguments, intercept_episode

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "integrations/migrations/general_harness"


def config(kind, runtime="codex"):
    return cli_migration(Namespace(general_harness_config=CONFIG / f"{kind}.json", runtime=runtime))


def kit_for(robo, tmp_path, kind):
    kit = ArmToolkit(
        robo,
        arm="cells",
        workspace=tmp_path / "workspace",
        feedback="pure",
        images="on-demand",
        max_turns=100,
    )
    record = configure_migration(
        kit, config(kind), task="Move the bowl", initial_images={}, runtime="codex"
    )
    assert record["agent_loop"] == "codex exec"
    return kit


@pytest.mark.parametrize("kind", ["baseline", "recap", "ace", "failure_memory"])
def test_independent_toolkit_and_same_scripted_model(robo, tmp_path, kind):
    kit = kit_for(robo, tmp_path, kind)
    assert {s["name"] for s in kit.get_tools_spec()} == {"python", "finish"}
    assert kit.robo._action_feedback is None and kit.evidence_adapter is None
    assert kit.stage_verifier is None and kit.harnesspro_hook is None
    transport = ScriptedTransport(
        [
            _reply([("a", "python", {"code": "answer = 41"})]),
            _reply([("b", "python", {"code": "print(answer + 1)"})]),
            _reply([("c", "finish", {"status": "stuck", "summary": "offline smoke"})]),
        ]
    )
    result = run_openai_agent(
        migration_prompt("Baseline prompt", kit),
        [],
        kit,
        model="gpt-6-astra",
        base_url="http://unused/v1",
        api_key=None,
        max_turns=100,
        budget_s=10,
        out_dir=tmp_path / "episode",
        transport=transport,
    )
    assert result["requests"] == 3 and result["status"] == "completed"
    assert all(r["body"]["model"] == "gpt-6-astra" for r in transport.requests)
    assert kit._globals["answer"] == 41
    if kind == "baseline":
        assert migration_prompt("unchanged", kit) == "unchanged"
    else:
        with pytest.raises(ValueError, match="clean toolkit"):
            configure_migration(kit, {"kind": "recap"}, task="x", initial_images={})


def test_recursive_parent_restore_and_recovery_siblings(robo, tmp_path):
    kit = kit_for(robo, tmp_path, "recap")
    out = kit.execute_tool(
        "python",
        {
            "code": "recap_plan(['transfer', 'verify final']); recap_refine(['grasp', 'place']); "
            "recap_done('current wrist view shows a lifted object')"
        },
    )
    assert "error" not in out.result
    h = kit.general_harness
    assert h.frames[-1]["remaining"] == ["place"]
    h.done("after release, object visible in target")
    assert len(h.frames) == 1 and h.frames[0]["remaining"] == ["transfer", "verify final"]
    h.refine(["inspect", "check alignment"])
    h.backtrack("object shifted in fresh view", ["reobserve", "correct transfer"])
    assert h.frames[0]["remaining"] == ["reobserve", "correct transfer", "verify final"]
    with pytest.raises(ValueError):
        h.done("")
    with pytest.raises(ValueError):
        h.plan(["erase history"])
    assert kit.finish is None


def test_recursive_memory_bound_and_rejected_update_is_atomic(robo, tmp_path):
    kit = kit_for(robo, tmp_path, "recap")
    h = kit.general_harness
    h.plan([str(i) for i in range(12)])
    h.refine([str(i) for i in range(12)])
    before = copy.deepcopy(h.frames)
    with pytest.raises(ValueError):
        h.refine(["more"])
    assert h.frames == before
    for _ in range(12):
        h.done("observed in current view")
    assert len(h.completed) == 4


def record(index, *, summary, xyz=2, tool="move_to", kwargs=None, error=None):
    return SimpleNamespace(
        index=index, tool=tool, kwargs=kwargs or {"xyz": [0, 0, xyz]}, summary=summary, error=error
    )


def test_failures_survive_cells_and_resolve_only_same_command(robo, tmp_path):
    h = kit_for(robo, tmp_path, "failure_memory").general_harness
    bad = {"reached": False, "ik_error": "out of reach 0.02 m"}
    assert not h.observe([record(0, summary=bad)])["replan_recommended"]
    h.observe([record(1, summary={"reached": True}, xyz=0.6)])
    again = h.observe([record(2, summary=bad)])
    assert again["replan_recommended"] and again["unresolved"][0]["first_call"] == 0
    h.observe([record(3, summary={"reached": True})])
    assert not h.observe([])["unresolved"]
    assert not h.observe([record(4, summary=bad)])["replan_recommended"]
    for i in range(50):
        h.observe([record(i + 5, summary=bad, xyz=i)])
    assert len(h.failures) == 32


def test_contact_and_intentional_empty_gripper_do_not_trigger_retry(robo, tmp_path):
    h = kit_for(robo, tmp_path, "failure_memory").general_harness
    for i in range(3):
        context = h.observe(
            [
                record(i, summary={"reached": False, "blocked": True}),
                record(i, summary={"holding": False}, tool="set_gripper", kwargs={"gripper": 1}),
                record(i, summary={"holding": False}, tool="set_gripper", kwargs={"gripper": -1}),
            ]
        )
    assert not context["replan_recommended"] and not context["unresolved"]
    assert context["recent"]


def test_unprinted_primitive_failure_survives_large_output(robo, tmp_path):
    kit = kit_for(robo, tmp_path, "failure_memory")
    kit.execute_tool("python", {"code": "robo.move_to([0,0,2]); print('x'*18000)"})
    out = kit.execute_tool("python", {"code": "robo.move_to([0,0,2]); print('x'*18000)"})
    context = out.result["general_harness"]
    assert context["replan_recommended"]
    text = out.content_blocks[0]["text"]
    assert "target_not_reached" in text and len(text) < MAX_TEXT_CHARS + 30
    assert not kit.turn_budget_exhausted


def test_ace_delta_provenance_dedup_and_exclusion():
    sources = {"dev": {"task": "development"}}
    delta = {"op": "add", "id": "a", "text": "Inspect the outcome.", "sources": ["dev"]}
    book = curate_playbook(
        [],
        [delta, {"op": "tag", "id": "a", "tag": "helpful"}],
        sources=sources,
        excluded_tasks=["heldout"],
    )
    assert book["bullets"][0]["helpful"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        curate_playbook(
            book["bullets"], [{**delta, "id": "b"}], sources=sources, excluded_tasks=["heldout"]
        )
    with pytest.raises(ValueError, match="leaked"):
        curate_playbook([], [delta], sources=sources, excluded_tasks=["development"])


def test_ace_hash_and_runtime_freeze(robo, tmp_path):
    frozen = config("ace")
    before = Path(frozen["playbook"]).read_bytes()
    kit = kit_for(robo, tmp_path, "ace")
    kit.execute_tool("python", {"code": "print('test observation must not train playbook')"})
    assert Path(frozen["playbook"]).read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == frozen["sha256"]
    with pytest.raises(ValueError, match="hash mismatch"):
        load_playbook({**frozen, "sha256": "0" * 64})
    book = json.loads(before)
    assert not set(book["excluded_tasks"]) & {s["task"] for s in book["sources"].values()}
    assert all(b["helpful"] == b["harmful"] == 0 for b in book["bullets"])


@pytest.mark.parametrize("kind", ["baseline", "recap", "ace", "failure_memory"])
@pytest.mark.parametrize("runtime", ["codex", "openai"])
def test_both_cli_layers_forward_config_and_preserve_budgets(tmp_path, monkeypatch, kind, runtime):
    captured = intercept_episode(monkeypatch)
    monkeypatch.setattr(cli, "run_host", lambda argv, **kwargs: play_module.main(argv[3:]))
    assert (
        cli.main(
            [
                "play",
                *arguments(tmp_path, runtime),
                "--general-harness-config",
                str(CONFIG / f"{kind}.json"),
            ]
        )
        == 0
    )
    opts = captured[0][2]
    assert opts["migration"]["kind"] == kind and opts["runtime"] == runtime
    assert opts["model"] == "unchanged-model" and opts["max_turns"] == 7
    assert opts["budget_s"] == 37 and captured[0][0].max_episode_steps == 123
    if kind == "ace":
        assert Path(opts["migration"]["playbook"]).is_absolute()


def test_other_migrations_still_reject_codex():
    with pytest.raises(ValueError, match="runtime=openai"):
        validate_migration_runtime({"kind": "openvl_evidence"}, "codex")
    with pytest.raises(ValueError):
        validate_migration_runtime({"kind": "recap", "recovery": True}, "codex")


def test_deploy_variants_differ_only_by_migration_and_output():
    configs = [
        yaml.safe_load((CONFIG / f"deploy_{k}.yml").read_text())
        for k in ["baseline", "recap", "ace", "failure_memory"]
    ]
    controls = [
        {k: v for k, v in c.items() if k not in {"migration", "result_dir"}} for c in configs
    ]
    assert all(c == controls[0] for c in controls)
    assert controls[0]["runtime"] == "codex" and controls[0]["reasoning_effort"] == "high"


def test_feedback_rendering_preserves_error_tail_and_images(robo, tmp_path):
    h = kit_for(robo, tmp_path, "failure_memory").general_harness
    out = ArmResult({"stdout": "x" * 35000, "error": "important error at tail"}, [b"image"])
    h.after_tool("python", out, [record(0, summary={"reached": False})])
    text = out.content_blocks[0]["text"]
    assert out.rendered_text is not None
    assert "target_not_reached" in text and "important error at tail" in text
    assert len(text) <= MAX_TEXT_CHARS and out.images == [b"image"]
