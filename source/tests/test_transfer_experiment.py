"""Screening reports must distinguish method failures from official scores."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "transfer_experiment", ROOT / "integrations/migrations/general_harness/transfer_experiment.py"
)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_job(tmp_path, kind="longharness"):
    episode = tmp_path / "episodes/episode"
    write(episode / "card.json", {"task": "general_pickup", "seed": 4})
    write(episode / "migration.json", {"kind": kind})
    write(
        episode / "result.json",
        {
            "task": "general_pickup",
            "seed": 4,
            "model": experiment.MODEL,
            "max_turns": 60,
            "reasoning_effort": "default",
            "status": "completed",
            "generation": {"runtime": "openai", "python_cells": 3},
        },
    )
    write(
        tmp_path / "official.json",
        {"details": {"4": {"layout_id": 4, "success": True, "score": 1.0}}},
    )
    return {
        "folder": str(tmp_path),
        "official_result": str(tmp_path / "official.json"),
        "task": "general_pickup",
        "layout": 4,
        "kind": kind,
    }, episode


def test_official_success_survives_but_harness_failure_is_disclosed(tmp_path):
    job, episode = fixture_job(tmp_path)
    events = episode / "general_harness/events.jsonl"
    events.parent.mkdir()
    events.write_text(json.dumps({"event": "maintenance_failed", "reason": "bad source"}) + "\n")
    result = experiment.score(job)
    assert result["success"] is True and result["score"] == 1.0
    assert result["mechanism_updates"] == 0 and result["mechanism_used"] is False
    assert result["mechanism_failures"][0]["reason"] == "bad source"
    assert result["comparison_eligible"]


def test_empty_evo_selection_does_not_count_as_used(tmp_path):
    job, episode = fixture_job(tmp_path, "evoharness")
    events = episode / "general_harness/events.jsonl"
    events.parent.mkdir()
    events.write_text(json.dumps({"event": "selected", "context": {"selected_ids": []}}) + "\n")
    result = experiment.score(job)
    assert result["mechanism_updates"] == 1 and result["mechanism_used"] is False


def test_wrong_layout_is_not_silently_scored(tmp_path):
    job, _ = fixture_job(tmp_path)
    job["layout"] = 9
    with pytest.raises(ValueError, match="layout identity"):
        experiment.score(job)


def test_missing_result_is_infrastructure_and_not_comparable(tmp_path):
    job, episode = fixture_job(tmp_path)
    (episode / "result.json").unlink()
    result = experiment.score(job)
    assert result["infrastructure_issue"] and not result["comparison_eligible"]
    assert result["success"] is True  # preserve the existing official verdict


@pytest.mark.parametrize("gpus", [[], [3, 3], [-1]])
def test_prepare_rejects_unsafe_gpu_lane_mapping_before_copy(tmp_path, gpus):
    with pytest.raises(ValueError, match="unique nonnegative"):
        experiment.prepare(tmp_path, gpus)
    assert not (tmp_path / "sources").exists()
