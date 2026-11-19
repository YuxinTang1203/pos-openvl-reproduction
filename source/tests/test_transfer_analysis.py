"""Offline analysis must preserve denominators, official verdicts, and provenance."""

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / (
    "integrations/migrations/general_harness/transfer_analysis.py"
)
SPEC = importlib.util.spec_from_file_location("transfer_analysis", MODULE)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def save_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def manifest(tmp_path):
    jobs = {}
    for task, layout in [("general_pickup", 4), ("stack_blocks", 1)]:
        for kind in ["baseline", "pos", "longharness", "evoharness"]:
            key = f"{task}-l{layout}-{kind}"
            jobs[key] = {
                "kind": kind,
                "task": task,
                "layout": layout,
                "folder": key,
                "official_result": f"{key}/official.json",
            }
    plan = {
        "jobs": jobs,
        "planned_episodes": 8,
        "model": "qwen3.8-flash",
        "max_turns": 60,
        "budget_s": 1200,
        "openai_options": {"max_requests": 96},
    }
    save(tmp_path / "manifest.json", plan)
    return plan


def completed(tmp_path, plan, job_id, success=False, kind=None):
    job = plan["jobs"][job_id]
    episode = tmp_path / job["folder"] / "episodes/episode-1"
    save(episode / "card.json", {"task": job["task"], "seed": job["layout"]})
    save(episode / "migration.json", {"kind": kind or job["kind"]})
    save(
        episode / "result.json",
        {
            "task": job["task"],
            "seed": job["layout"],
            "model": plan["model"],
            "status": "completed",
            "policy_wall_s": 80,
            "environment_success": success,
            "finish": {"status": "success", "summary": "agent self-report"},
            "generation": {
                "python_cells": 2,
                "requests": 4,
                "max_requests_hit": False,
                "max_turns_hit": False,
            },
        },
    )
    save(
        tmp_path / job["official_result"],
        {
            "task_name": job["task"],
            "details": {
                "layout": {
                    "layout_id": job["layout"],
                    "success": success,
                    "score": 1 if success else 0,
                }
            },
        },
    )
    return episode


def test_pending_eight_cases_keep_fixed_denominator_and_unknowns(tmp_path):
    manifest(tmp_path)
    report = analysis.write_analysis(tmp_path)
    assert len(report["jobs"]) == report["planned_episodes"] == 8
    assert report["official_scored_episodes"] == 0
    assert all(v["planned"] == 2 and v["unknown"] == 2 for v in report["methods"].values())
    assert all(v["possible_success_fraction_range"] == [0, 1] for v in report["methods"].values())
    assert (tmp_path / "ANALYSIS.json").exists() and (tmp_path / "ANALYSIS.txt").exists()
    assert "未知结果不当作失败" in (tmp_path / "ANALYSIS.txt").read_text()


def test_official_verdict_overrides_finish_and_summary_claims(tmp_path):
    plan = manifest(tmp_path)
    job_id = "general_pickup-l4-baseline"
    completed(tmp_path, plan, job_id, success=False)
    save(tmp_path / "RESULTS.json", {"jobs": {job_id: {"status": "scored", "success": True}}})
    report = analysis.analyze(tmp_path)
    row = report["jobs"][job_id]
    assert row["official"]["success"] is False
    assert row["termination_clues"]["agent_claimed_success_but_official_failure"] is True
    assert row["termination_clues"]["unsuccessful_finish_before_budget_limits"] is True
    assert report["methods"]["baseline"]["confirmed_failures"] == 1
    assert report["methods"]["baseline"]["unknown"] == 1
    assert row["official"]["evidence"]["json_pointer"] == "/details/layout"


def test_missing_official_never_inferred_from_episode_success(tmp_path):
    plan = manifest(tmp_path)
    job_id = "general_pickup-l4-pos"
    completed(tmp_path, plan, job_id, success=True)
    (tmp_path / plan["jobs"][job_id]["official_result"]).unlink()
    report = analysis.analyze(tmp_path)
    assert report["jobs"][job_id]["official"]["success"] is None
    assert report["methods"]["pos"]["unknown"] == 2


def test_paired_outcomes_and_identity_mismatch_retained(tmp_path):
    plan = manifest(tmp_path)
    completed(tmp_path, plan, "general_pickup-l4-baseline", success=False)
    completed(tmp_path, plan, "general_pickup-l4-pos", success=True)
    completed(tmp_path, plan, "general_pickup-l4-longharness", success=True, kind="pos")
    report = analysis.analyze(tmp_path)
    pair = next(x for x in report["paired_cases"] if x["case"] == "general_pickup/l4")
    assert pair["comparisons"]["pos"] == "gain"
    assert pair["comparisons"]["longharness"] == "episode_identity_unverified_or_mismatched"
    assert pair["comparisons"]["evoharness"] == "incomplete_or_multiple_repeats"
    assert report["planned_episodes"] == 8


def test_planner_auxiliary_usage_retries_and_mechanism_rejections(tmp_path):
    plan = manifest(tmp_path)
    job_id = "general_pickup-l4-longharness"
    episode = completed(tmp_path, plan, job_id)
    save_lines(
        episode / "openai_events.jsonl",
        [
            {
                "type": "request",
                "role": "longharness_evidence",
                "ok": False,
                "status_code": 429,
                "latency_s": 1,
            },
            {"type": "request", "role": "longharness_evidence", "ok": True, "latency_s": 2},
            {
                "type": "harness_assistant",
                "role": "longharness_evidence",
                "usage": {"prompt": 100, "completion": 20, "cached": 40},
            },
            {"type": "request", "role": "planner", "ok": True, "latency_s": 3},
            {"type": "assistant", "usage": {"prompt": 200, "completion": 30, "cached": 50}},
            {
                "type": "harness_model_skipped",
                "role": "longharness_conclusions",
                "reason": "shared_budget_or_stop",
            },
        ],
    )
    save_lines(
        episode / "general_harness/events.jsonl",
        [
            {
                "event": "evidence_update",
                "receipt": {"accepted": False, "reason": "unknown source"},
            },
            {"event": "evidence_update", "receipt": {"accepted": True, "id": "e1"}},
            {"event": "maintenance_complete", "context": {"revision": 2}},
            {"event": "maintenance_failed", "reason": "bad schema"},
        ],
    )
    row = analysis.analyze(tmp_path)["jobs"][job_id]
    assert row["requests"]["planner"]["input_tokens"] == 200
    assert row["requests"]["auxiliary"]["attempts"] == 2
    assert row["requests"]["auxiliary"]["request_latency_s"] == 3
    assert row["requests"]["auxiliary"]["output_tokens"] == 20
    assert row["transport_failure_observed"] is True and row["infrastructure_failure"] is False
    assert row["mechanism"]["rejected_updates"] == 1
    assert row["mechanism"]["degraded_cycles"] == 1
    assert row["mechanism"]["activation"] == "evidence_accepted"
    assert row["mechanism"]["rejection_evidence"][0]["evidence"]["line"] == 1


def test_public_repeat_and_observation_clues_have_concrete_evidence(tmp_path):
    plan = manifest(tmp_path)
    job_id = "general_pickup-l4-pos"
    episode = completed(tmp_path, plan, job_id)
    save_lines(
        episode / "calls.jsonl",
        [
            {
                "index": 10,
                "tool": "move_to",
                "kwargs": {"xyz": [1, 2, 3]},
                "summary": {"reached": False, "blocked": True},
                "ok": True,
            },
            {
                "index": 11,
                "tool": "move_to",
                "kwargs": {"xyz": [1, 2, 3]},
                "summary": {"reached": False},
                "ok": True,
            },
            {
                "index": 12,
                "tool": "move_to",
                "kwargs": {"xyz": [1, 2, 4]},
                "summary": {"reached": False},
                "ok": True,
            },
            {
                "index": 13,
                "tool": "set_gripper",
                "kwargs": {"gripper": 1},
                "summary": {"holding": False},
                "ok": True,
            },
            {"index": 14, "tool": "artifact", "kwargs": {}, "summary": {}, "ok": True},
        ],
    )
    save(
        episode / "tool_turns.json",
        [
            {"tool": "python", "moved": False, "elapsed_s": 2},
            {"tool": "python", "moved": True, "elapsed_s": 5},
        ],
    )
    save_lines(
        episode / "openai_events.jsonl",
        [
            {"type": "tool", "tool": "python", "images": 1},
            {"type": "tool", "tool": "python", "images": 1},
        ],
    )
    execution = analysis.analyze(tmp_path)["jobs"][job_id]["execution"]
    assert execution["failure_signal_calls"] == 3  # opening gripper is not a failed grasp
    assert execution["repeat_failure_excess"] == 1
    repeat = execution["repeated_identical_failed_commands"][0]
    assert [x["evidence"]["call_index"] for x in repeat["occurrences"]] == [10, 11]
    assert repeat["occurrences"][1]["evidence"]["line"] == 2
    assert execution["image_cell_fraction"] == 1
    assert execution["observation_only_cell_fraction"] == 0.5
    assert execution["observation_only_cell_evidence"][0]["turn"]["json_pointer"] == "/0"


def test_partial_jsonl_and_bad_verdict_remain_unknown(tmp_path):
    plan = manifest(tmp_path)
    job_id = "stack_blocks-l1-evoharness"
    episode = completed(tmp_path, plan, job_id)
    (episode / "openai_events.jsonl").write_text('{"type":"request","ok":true}\n{"type":')
    save(
        tmp_path / plan["jobs"][job_id]["official_result"],
        {"details": {"wrong-layout": {"layout_id": 99, "success": True}}},
    )
    row = analysis.analyze(tmp_path)["jobs"][job_id]
    assert row["official"]["success"] is None
    assert row["requests"]["planner"]["attempts"] == 1
    assert any(x["problem"] == "invalid_jsonl_record" for x in row["data_issues"])


def test_missing_manifest_fails_without_writing_false_results(tmp_path):
    with pytest.raises(ValueError, match="full planned jobs"):
        analysis.write_analysis(tmp_path)
    assert not (tmp_path / "ANALYSIS.json").exists()
