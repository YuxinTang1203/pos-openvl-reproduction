"""Budget-limited attempts must not be reported as successful-task speedups."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("baseline_success", [False, True])
def test_paired_latency_requires_successful_completion(tmp_path, monkeypatch, baseline_success):
    path = Path(__file__).resolve().parents[1] / "integrations/latency/experiment.py"
    spec = importlib.util.spec_from_file_location("phase_latency_experiment", path)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    monkeypatch.setattr(experiment, "load_module", lambda *a: SimpleNamespace(summarize=lambda paths: {}))
    manifest = {"arms": {"high": {}, "phase": {}}, "schedule": [["high", "case"], ["phase", "case"]],
                "cases": [{"id": "case"}], "planned": 2, "baseline_reused": False, "limitations": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for arm in manifest["arms"]:
        directory = tmp_path / arm / "cases/case"
        directory.mkdir(parents=True)
        success = arm == "phase" or baseline_success
        (directory / "job.json").write_text(json.dumps({
            "arm": arm, "case": "case", "status": "scored", "success": success,
            "budget_exhausted": not success, "termination_category": "task_success" if success else "budget_exhausted",
            "policy_wall_s": 900 if arm == "high" else 300, "model_requests": 17 if arm == "high" else 8}))
    result = experiment.summarize(tmp_path)
    pair = result["paired"][0]
    assert pair["completion_latency_comparable"] is baseline_success
    assert result["successful_completion_latency_comparable"] is baseline_success
    if baseline_success:
        assert pair["candidate_wall_change_fraction"] == pytest.approx(-2/3)
    else:
        assert pair["candidate_wall_change_fraction"] is None
        assert pair["high_termination"] == "budget_exhausted"


def test_request_limit_is_reported_even_when_runtime_status_is_completed(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "integrations/latency/experiment.py"
    spec = importlib.util.spec_from_file_location("phase_latency_limits", path)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    monkeypatch.setattr(experiment, "load_module", lambda *a: SimpleNamespace(summarize=lambda paths: {}))
    manifest = {"variant": "phase", "arms": {"high": {}, "phase": {}},
                "arm_options": {"high": {"phase_tools": False}, "phase": {"phase_tools": True}},
                "schedule": [["high", "case"], ["phase", "case"]], "cases": [{"id": "case"}],
                "planned": 2, "baseline_reused": False, "limitations": []}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for arm in manifest["arms"]:
        directory = tmp_path / arm / "cases/case"
        directory.mkdir(parents=True)
        episode = directory / "result.json"
        episode.write_text(json.dumps({"generation": {"status": "completed", "max_requests_hit": arm == "phase",
                                         "options": {"phase_tools": arm == "phase"}}}))
        (directory / "job.json").write_text(json.dumps({
            "arm": arm, "case": "case", "status": "scored", "success": arm == "high",
            "episode_result": str(episode), "policy_wall_s": 500, "model_requests": 24,
            "budget_exhausted": False, "termination_category": "task_success" if arm == "high" else "task_not_completed"}))
    result = experiment.summarize(tmp_path)
    candidate = result["jobs"][1]
    assert candidate["budget_exhausted"] and not candidate["comparison_eligible"]
    assert candidate["termination_category"] == "request_budget_exhausted"
    assert candidate["raw_runner_termination_category"] == "task_not_completed"
    assert result["paired"][0]["candidate_wall_change_fraction"] is None
