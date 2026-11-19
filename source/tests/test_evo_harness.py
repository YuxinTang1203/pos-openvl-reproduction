"""Evo compilation isolation, evidence contracts, and frozen selection behavior."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyrualean.arms import ArmResult
from pyrualean.evo_harness import (
    EvoHarness,
    apply_edits,
    compile_harness,
    freeze_book,
    load_book,
    load_development_episode,
    validate_book,
    validate_options,
)

EXCLUDED = ["general_pickup", "stack_blocks", "play_stacking_toy"]


def episode(tmp_path, task="insert_tubes", success=False):
    directory = tmp_path / task
    directory.mkdir()
    (directory / "result.json").write_text(json.dumps({
        "task": task, "seed": 3, "environment_success": success,
        "finish": {"status": "success"}, "status": "completed", "task_language": task,
    }))
    (directory / "calls.jsonl").write_text(json.dumps({
        "tool": "move_to", "kwargs": {"xyz": [0, 0, 1]},
        "summary": {"reached": False, "blocked": True}, "ok": True, "stateful": True,
    }) + "\n")
    return directory


class Compiler:
    def __init__(self):
        self.calls = []

    def __call__(self, role, prompt, payload, images):
        self.calls.append(role)
        if role == "evo_reflect":
            return {"candidates": [{
                "lesson": "Inspect the failed precondition before dependent motion.",
                "trigger": "A commanded motion did not reach its target.",
                "topic": "manipulation", "scope_hint": "task_type",
                "evidence": [payload["evidence"][1]["id"]],
            }]}
        level = payload["level"]
        candidates = payload["candidates"]
        return {"edits": [{"op": "add", "skill": {
            "id": "skill-" + level, "level": level, "topic": "manipulation",
            "trigger": "The preceding action failed.",
            "procedure": ("Check the return value before a dependent action."
                          if level == "task_type"
                          else "Resolve a failed prerequisite before continuing the procedure."),
            "evidence": [c["evidence"][0] for c in candidates],
            "candidate_ids": [c["id"] for c in candidates],
        }}]}


def make_book(tmp_path):
    records = [load_development_episode(episode(tmp_path, task), excluded_tasks=EXCLUDED)
               for task in ("insert_tubes", "swap_blocks")]
    model = Compiler()
    return compile_harness(records, model, excluded_tasks=EXCLUDED), model, records


def frozen_config(tmp_path):
    book, model, records = make_book(tmp_path)
    path = tmp_path / "book.json"
    digest = freeze_book(book, path)
    return {"kind": "evoharness", "book": str(path), "sha256": digest}, book


def test_failed_environment_overrides_agent_success_claim_and_two_level_compilation(tmp_path):
    book, model, records = make_book(tmp_path)
    assert model.calls == ["evo_reflect", "evo_reflect", "evo_compile_task_type",
                           "evo_compile_cross_task"]
    assert {s["level"] for s in book["skills"]} == {"task_type", "cross_task"}
    assert records[0]["feedback"]["success"] is False
    assert book["measured_effectiveness"] is None
    assert all(s["split"] == "development" for s in book["sources"].values())


def test_reject_heldout_before_any_model_call(tmp_path):
    path = episode(tmp_path, "stack_blocks")
    with pytest.raises(ValueError, match="held-out"):
        load_development_episode(path, excluded_tasks=EXCLUDED)
    record = load_development_episode(path, excluded_tasks=["play_stacking_toy"])
    model = Compiler()
    with pytest.raises(ValueError, match="leaked"):
        compile_harness([record], model, excluded_tasks=EXCLUDED)
    assert model.calls == []


def test_changed_source_and_spoofed_task_or_outcome_are_rejected(tmp_path):
    path = episode(tmp_path)
    record = load_development_episode(path, excluded_tasks=EXCLUDED)
    spoofed = copy.deepcopy(record)
    spoofed["task"] = "swap_blocks"
    with pytest.raises(ValueError, match="metadata disagrees"):
        compile_harness([spoofed], Compiler(), excluded_tasks=EXCLUDED)
    forged = copy.deepcopy(record)
    forged["evidence"][1]["text"] = "A fabricated action succeeded."
    with pytest.raises(ValueError, match="evidence text differs"):
        compile_harness([forged], Compiler(), excluded_tasks=EXCLUDED)
    (path / "calls.jsonl").write_text("[]\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        compile_harness([record], Compiler(), excluded_tasks=EXCLUDED)


def test_successful_episodes_do_not_create_memories(tmp_path):
    record = load_development_episode(episode(tmp_path, success=True), excluded_tasks=EXCLUDED)
    model = Compiler()
    with pytest.raises(ValueError, match="no grounded"):
        compile_harness([record], model, excluded_tasks=EXCLUDED)
    assert model.calls == []


def test_single_task_cannot_claim_cross_task_support(tmp_path):
    record = load_development_episode(episode(tmp_path), excluded_tasks=EXCLUDED)
    model = Compiler()
    book = compile_harness([record], model, excluded_tasks=EXCLUDED)
    assert model.calls == ["evo_reflect", "evo_compile_task_type"]
    assert book["compilation_audit"][-1]["status"] == "skip_insufficient_task_diversity"
    skill = copy.deepcopy(book["skills"][0])
    skill["level"] = "cross_task"
    with pytest.raises(ValueError, match="two tasks"):
        apply_edits([], [{"op": "add", "skill": skill}],
                    evidence=book["evidence_registry"],
                    candidate_ids={c["id"] for c in book["candidate_memories"]}, level="cross_task")


def test_unknown_evidence_and_no_model_result_abort_without_frozen_artifact(tmp_path):
    record = load_development_episode(episode(tmp_path), excluded_tasks=EXCLUDED)
    with pytest.raises(ValueError, match="unavailable"):
        compile_harness([record], lambda *args: None, excluded_tasks=EXCLUDED)
    model = Compiler()

    def bad(*args):
        reply = model(*args)
        reply["candidates"][0]["evidence"] = ["invented:call0"]
        return reply

    with pytest.raises(ValueError, match="unknown evidence"):
        compile_harness([record], bad, excluded_tasks=EXCLUDED)
    assert not (tmp_path / "book.json").exists()


def test_duplicate_procedures_are_rejected_and_merge_is_atomic(tmp_path):
    book, _, _ = make_book(tmp_path)
    before = copy.deepcopy(book["skills"])
    duplicate = {**book["skills"][0], "id": "duplicated"}
    with pytest.raises(ValueError, match="duplicate procedure"):
        apply_edits(before, [{"op": "add", "skill": duplicate}],
                    evidence=book["evidence_registry"],
                    candidate_ids={c["id"] for c in book["candidate_memories"]}, level="task_type")
    assert before == book["skills"]
    with pytest.raises(ValueError, match="another level"):
        apply_edits(before, [{"op": "merge", "targets": [s["id"] for s in before],
                              "skill": duplicate}], evidence=book["evidence_registry"],
                    candidate_ids={c["id"] for c in book["candidate_memories"]}, level="task_type")
    assert before == book["skills"]


def test_frozen_hash_and_evidence_chain_are_verified(tmp_path):
    config, book = frozen_config(tmp_path)
    assert load_book(config) == book
    with pytest.raises(FileExistsError):
        freeze_book(book, config["book"])
    bad = copy.deepcopy(book)
    bad["sources"][next(iter(bad["sources"]))]["task"] = "stack_blocks"
    with pytest.raises(ValueError, match="leakage"):
        validate_book(bad)
    Path(config["book"]).write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_options(config)


def test_selector_once_budgeted_and_eval_feedback_never_updates_book(tmp_path):
    config, _ = frozen_config(tmp_path)
    config.update(max_skills=1, max_chars=7000)
    kit = SimpleNamespace(arm="cells", workspace=tmp_path / "episode" / "workspace")
    harness = EvoHarness(kit, config, "Pick up the object")
    ids = [s["id"] for s in harness.book["skills"]]
    calls = []

    def select(*args):
        calls.append(args)
        return {"selected_ids": ids, "reason": "relevant"}

    before = Path(config["book"]).read_bytes()
    context = harness.before_request(select)
    assert ids[0] in context and ids[1] not in context
    result = ArmResult({"error": "new evaluation failure", "success": False})
    assert harness.after_tool("python", result, []) is result
    assert harness.before_request(select) == context
    assert len(calls) == 1 and Path(config["book"]).read_bytes() == before
    assert harness.provenance["evaluation_updates"] is False


@pytest.mark.parametrize("reply,status", [
    (None, "unavailable"), ({"selected_ids": ["imaginary"]}, "invalid_selection"),
    ({"selected_ids": []}, "no_applicable_skills"),
])
def test_selector_failure_is_audited_without_hidden_fallback(tmp_path, reply, status):
    config, _ = frozen_config(tmp_path)
    kit = SimpleNamespace(arm="cells", workspace=tmp_path / "episode" / "workspace")
    harness = EvoHarness(kit, config, "Pick up the object")
    calls = []

    def select(*args):
        calls.append(args)
        return reply

    assert "no skills injected" in harness.before_request(select)
    harness.before_request(select)
    assert len(calls) == 1
    assert harness.context()["selection_status"] == status
    assert status in (harness.directory / "events.jsonl").read_text()


def test_bounded_development_input_discloses_omissions(tmp_path):
    record = load_development_episode(episode(tmp_path), excluded_tasks=EXCLUDED,
                                      max_evidence=1)
    assert len(record["evidence"]) == 1
    assert record["coverage"]["omitted_records"] == 1
