"""Offline preparation must preserve sources, budgets, denominators and verdicts."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "expansion_experiment", ROOT / "integrations/migrations/general_harness/expansion_experiment.py"
)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    here = project / "integrations/migrations/general_harness"
    here.mkdir(parents=True)
    for name in ("transfer_experiment.py", "expansion_experiment.py", "expansion_launch.sh"):
        (here / name).write_bytes((ROOT / "integrations/migrations/general_harness" / name).read_bytes())
    for kind in experiment.VARIANTS:
        write(here / f"{kind}.json", {"kind": kind})
    write(project / "src/pyrualean/fixture.json", {"user_change": "preserve this current resource"})
    for name in ("model.py", "deploy.py"):
        p = project / "integrations/xpolicylab/PyRUA_Lean" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# new adapter\n")
    prior = project / "runs/trusted"
    bench = prior / "baseline/robodojo"
    for name in ("env", "env_cfg", "scripts", "src", "task", "utils", "XPolicyLab/client_server", "XPolicyLab/utils", "third_party"):
        p = bench / name / "fixture.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# frozen benchmark\n")
    for task in {task for task, _ in experiment.CASES}:
        for layout in range(5):
            write(bench / "Assets/Eval_Layout/RoboDojo/arx_x5/0" / f"{task}_{layout}.json", {"id": layout})
    policy = bench / "XPolicyLab/policy/FixturePolicy"
    policy.mkdir(parents=True)
    (policy / "deploy.yml").write_text("max_episode_steps: 1200\nrobot: arx_x5\n")
    hashes = {str(p.relative_to(prior)): experiment.sha(p) for p in bench.rglob("*") if p.is_file() and "Assets" not in p.parts}
    manifest = prior / "manifest.json"
    write(manifest, {"arms": {"baseline": {"bench": str(bench), "policy_dir": str(policy), "policy": "FixturePolicy"}}, "sha256": hashes})
    monkeypatch.setattr(experiment, "PROJECT", project)
    monkeypatch.setattr(experiment, "HERE", here)
    monkeypatch.setattr(experiment, "validate_config", lambda config: None)
    # Prove offline preparation does not invoke the reused online interfaces.
    def forbidden(*args, **kwargs):
        pytest.fail("prepare attempted model/GPU/simulator access")
    monkeypatch.setattr(experiment.base, "catalog", forbidden)
    monkeypatch.setattr(experiment.base, "run_all", forbidden)
    monkeypatch.setattr(experiment.base, "gpu_inventory", forbidden)
    return project, manifest, bench


def test_prepare_freezes_six_isolated_jobs_without_mutating_inputs(prepared_inputs):
    project, prior, bench = prepared_inputs
    root = project / "runs/new"
    before = prior.read_bytes(), (bench / "task/fixture.py").read_bytes()
    plan = experiment.prepare(root, [2, 7], prior)
    assert plan["queue_order"] == [
        "cover_blocks-l0-baseline", "cover_blocks-l0-reflact", "swap_blocks-l2-reflact",
        "swap_blocks-l2-baseline", "cover_blocks-l4-baseline", "cover_blocks-l4-reflact",
    ]
    assert plan["planned_episodes"] == 6 and plan["held_out"] is False
    assert plan["budget_s"] == 2400 and plan["max_attempts"] == 1
    assert "sources/pyrualean/fixture.json" in plan["sha256"]
    assert before == (prior.read_bytes(), (bench / "task/fixture.py").read_bytes())
    assert len({j["bench"] for j in plan["jobs"].values()}) == 6
    for job in plan["jobs"].values():
        config = yaml.safe_load((Path(job["bench"]) / "XPolicyLab/policy/FixturePolicy/deploy.yml").read_text())
        assert config["result_dir"] == str(Path(job["folder"]) / "episodes")
        assert config["migration"]["kind"] == job["kind"]
        assert config["openai_options"] == experiment.OPTIONS
        assert config["reasoning_effort"] == "default"
        assert config["budget_s"] == 2400 and config["max_turns"] == 60
        resume = next((Path(job["bench"]) / "eval_result").rglob("_resume_*.json"))
        state = json.loads(resume.read_text())
        assert state["eval_seed"] == 0 and state["completed_layout_ids"] == []
        assert state["abandoned_layout_ids"] == [x for x in range(5) if x != job["layout"]]
        assert state["save_dir"] == str(Path(job["official_result"]).parent)
    assert experiment.check(root)["planned"] == 6
    (root / "sources/pyrualean/fixture.json").write_text("changed")
    with pytest.raises(ValueError, match="Frozen file changed"):
        experiment.check(root)


@pytest.mark.parametrize("gpus", [[], [1, 1], [-1], [True], ["1"]])
def test_bad_gpu_spec_rejected_before_source_copy(tmp_path, gpus):
    with pytest.raises(ValueError, match="unique nonnegative"):
        experiment.prepare(tmp_path / "new", gpus)
    assert not (tmp_path / "new").exists()


def test_existing_root_and_trusted_drift_are_never_overwritten(prepared_inputs):
    project, prior, bench = prepared_inputs
    root = project / "runs/old"
    write(root / "manifest.json", {"preserve": True})
    with pytest.raises(FileExistsError):
        experiment.prepare(root, [2], prior)
    assert json.loads((root / "manifest.json").read_text()) == {"preserve": True}
    (bench / "task/fixture.py").write_text("drift")
    with pytest.raises(ValueError, match="Trusted benchmark changed"):
        experiment.prepare(project / "runs/new", [2], prior)
    assert not (project / "runs/new").exists()


def fixture_job(tmp_path, kind="reflact"):
    episode = tmp_path / "episodes/episode"
    write(episode / "card.json", {"task": "cover_blocks", "seed": 0})
    write(episode / "migration.json", {"kind": kind})
    write(episode / "result.json", {
        "task": "cover_blocks", "seed": 0, "model": experiment.MODEL,
        "max_turns": 60, "reasoning_effort": "default", "status": "completed", "timeout_s": 2400,
        "generation": {"runtime": "openai", "python_cells": 2, "options": experiment.OPTIONS},
    })
    write(tmp_path / "official.json", {"details": {"0": {"layout_id": 0, "success": False, "score": 0.2}}})
    return {"folder": str(tmp_path), "official_result": str(tmp_path / "official.json"),
            "task": "cover_blocks", "layout": 0, "kind": kind}, episode


def log_events(episode, events):
    path = episode / "general_harness/events.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(json.dumps({"kind": "reflact", **event}) + "\n" for event in events))


def test_reflact_uses_real_helper_events_not_committed_or_model_success(tmp_path):
    job, episode = fixture_job(tmp_path)
    log_events(episode, [{"event": "initialized"}, {"event": "committed"},
                         {"event": "reflection_rejected", "reason": "invalid"}])
    result = experiment.score(job)
    assert result["success"] is False and result["score"] == 0.2
    assert not result["mechanism_used"] and result["mechanism_updates"] == 0
    log_events(episode, [{"event": "reflection_recorded", "id": 1},
                        {"event": "after_tool", "tool": "python", "adoption": "recorded",
                         "reflection_usage": [{"id": 1, "before_first_primitive": False,
                                               "before_first_stateful": False}]}])
    result = experiment.score(job)
    assert result["mechanism_used"] and result["mechanism_updates"] == 1
    assert result["reflact"]["adoption_counts"] == {"recorded": 1}
    assert result["reflact"]["ordering"]["reflections_before_first_primitive"] == 0
    assert result["reflact"]["ordering"]["reflections_after_first_primitive"] == 1
    assert result["comparison_eligible"]


def test_real_reflact_schema_preserves_no_motion_null_and_counts_cells_separately(tmp_path):
    from types import SimpleNamespace
    from pyrualean.arms import ArmResult
    from pyrualean.reflact_harness import ReflActHarness

    job, episode = fixture_job(tmp_path)
    kit = SimpleNamespace(arm="cells", workspace=episode / "workspace", _globals={},
                          _cell_index=1, turns=[], robo=SimpleNamespace(_ledger=[]))
    harness = ReflActHarness(kit, {"kind": "reflact"}, "public task")
    harness.reflect("Item observed", "Goal incomplete")
    record = SimpleNamespace(index=0, stateful=True, error=None)
    kit.robo._ledger.append(record)
    harness.after_tool("python", ArmResult({"cell": 1}), [record])
    kit.turns.append({})
    kit._cell_index = 2
    harness.reflect("Only reasoning this cell", "No new observation")
    harness.after_tool("python", ArmResult({"cell": 2}), [])
    result = experiment.score(job)
    audit = result["reflact"]
    assert audit["recorded_reflections"] == audit["matched_cell_reflections"] == 2
    assert result["mechanism_used"] and result["mechanism_updates"] == 2
    for boundary in ("primitive", "stateful"):
        assert audit["ordering"][f"reflections_before_first_{boundary}"] == 1
        assert audit["ordering"][f"reflections_after_first_{boundary}"] == 0
        assert audit["ordering"][f"reflections_with_no_{boundary}"] == 1
        assert audit["ordering"][f"cells_with_pre_{boundary}_reflection"] == 1


def test_unmatched_recording_is_not_completed_cell_adoption(tmp_path):
    job, episode = fixture_job(tmp_path)
    log_events(episode, [{"event": "reflection_recorded", "id": "orphan"},
                        {"event": "after_tool", "tool": "python", "adoption": "missing",
                         "reflection_usage": [], "orphaned_reflection_count": 1}])
    result = experiment.score(job)
    assert result["reflact"]["recorded_reflections"] == 1
    assert result["reflact"]["orphaned_reflections"] == 1
    assert not result["mechanism_used"] and result["mechanism_updates"] == 0


def test_unexpected_auxiliary_call_is_disclosed_without_changing_official_score(tmp_path):
    job, episode = fixture_job(tmp_path)
    log_events(episode, [{"event": "reflection_recorded"}])
    (episode / "openai_events.jsonl").write_text(json.dumps({"type": "request", "role": "aux", "ok": True}) + "\n")
    result = experiment.score(job)
    assert result["score"] == 0.2 and not result["comparison_eligible"]
    assert "unexpected auxiliary requests" in " ".join(result["identity_errors"])


def test_summary_keeps_all_planned_cases_when_only_one_is_finished(tmp_path):
    jobs = {f"job{i}": {} for i in range(6)}
    write(tmp_path / "manifest.json", {"jobs": jobs, "limitations": experiment.LIMITATIONS})
    write(tmp_path / "STATUS.json", {"status": "incomplete", "jobs": {"job0": {"status": "scored", "success": True}}})
    result = experiment.summarize(tmp_path)
    assert len(result["jobs"]) == result["planned"] == 6
    assert result["official_successes"] == 1 and result["official_outcomes_missing"] == 5
