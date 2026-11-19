"""Import completed PyRUA evaluations into HarnessPro's released rho runtime.

This adapter calls HarnessPro's own stores, trajectory types, and observability
functions. It does not substitute PyRUA's experimental harness optimizer for an
unreleased HarnessPro optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _rho(harnesspro_root: Path):
    source = harnesspro_root.resolve(strict=True) / "src" / "rho" / "protocols.py"
    if not source.is_file():
        raise ValueError(f"HarnessPro source missing: {source}")
    source_root = str(source.parents[1])
    if "rho" in sys.modules:
        loaded = Path(sys.modules["rho"].__file__).resolve()
        if loaded.parent != source.parent:
            raise RuntimeError(f"rho already loaded from a different checkout: {loaded}")
    elif source_root not in sys.path:
        sys.path.insert(0, source_root)
    from rho.observability import trajectory_cost_summary, trajectory_metrics, usage_summary
    from rho.protocols import Grade, Trajectory
    from rho.stores.harness import FilesystemHarnessStore
    from rho.stores.trajectory import FilesystemTrajectoryStore

    return (Grade, Trajectory, FilesystemHarnessStore, FilesystemTrajectoryStore,
            trajectory_metrics, trajectory_cost_summary, usage_summary)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing event stream: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def ingest(experiments: list[Path], out_dir: Path, harnesspro_root: Path) -> dict[str, Any]:
    """Store actual PyRUA trajectories in native rho stores and grade officially."""
    (Grade, Trajectory, HarnessStore, TrajectoryStore, trajectory_metrics,
     trajectory_cost_summary, usage_summary) = _rho(harnesspro_root)
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise ValueError(f"refusing to overwrite {out_dir}")
    # Validate the complete input set before creating a native store.
    prepared = []
    for experiment_path in experiments:
        experiment_path = experiment_path.resolve(strict=True)
        report = _json(experiment_path)
        if report.get("schema") != "pyrualean-harness-experiment-v1":
            raise ValueError(f"unsupported experiment: {experiment_path}")
        manifest = _json(Path(report["source_manifest"]))
        for row in report["runs"]:
            if row.get("status") != "completed":
                raise ValueError(f"only completed rows can be ingested: {row}")
            episode = Path(row["episode_dir"]).resolve(strict=True)
            official = Path(row["official_result"]).resolve(strict=True)
            result = _json(episode / "result.json")
            verdict = _json(official)
            success = verdict.get("success_rate")
            if success not in (0, 0.0, 1, 1.0):
                raise ValueError(f"official success missing or nonbinary: {official}")
            if bool(success) != (result.get("environment_success") is True):
                raise ValueError(f"official verdict and episode disagree: {episode}")
            profile = row["profile"]
            candidate = None if profile == "baseline" else manifest["candidates"][profile]
            harness_path = (Path(candidate.get("runtime_policy") or candidate["prompt"])
                            .resolve(strict=True).parent if candidate else None)
            if candidate is not None:
                if candidate.get("runtime_policy"):
                    from .harness_policy import load_policy
                    _, identity, _ = load_policy(candidate["runtime_policy"])
                else:
                    from .harness_opt import load_harness_addendum
                    _, identity = load_harness_addendum(candidate["prompt"])
                observed = (result.get("harness") or {}).get("sha256")
                if identity["sha256"] != candidate["sha256"] or observed != identity["sha256"]:
                    raise ValueError(f"harness identity changed or mismatched: {episode}")
            elif result.get("harness") is not None:
                raise ValueError(f"baseline unexpectedly used a harness: {episode}")
            events = _jsonl(episode / "codex_events.jsonl")
            prepared.append((row, episode, official, result, bool(success),
                             profile, harness_path, events))
    out_dir.mkdir(parents=True)
    harness_store = HarnessStore(out_dir / "harness")
    trajectory_store = TrajectoryStore(out_dir / "trajectories")
    baseline = harness_store.empty()
    captured: dict[Path, Any] = {}
    trajectories = []
    rows = []
    for row, episode, official, result, success, profile, harness_path, events in prepared:
        if harness_path is None:
            harness = baseline
        else:
            if harness_path not in captured:
                captured[harness_path] = harness_store.capture(harness_path)
            harness = captured[harness_path]
        key = f"{row['task']}:{row['seed']}:{profile}:{episode}"
        trajectory_id = "pyr_" + hashlib.sha256(key.encode()).hexdigest()[:20]
        trajectory = Trajectory(
            id=trajectory_id, kind="evaluate",
            task_id=f"{row['task']}:seed{row['seed']}", harness_id=harness.id,
            instructions=(episode / "prompt.txt").read_text(encoding="utf-8"),
            events=events,
            final_message=(episode / "last_message.md").read_text(encoding="utf-8")
            if (episode / "last_message.md").is_file() else "",
            stdout="",
            stderr=(episode / "codex_stderr.txt").read_text(encoding="utf-8")
            if (episode / "codex_stderr.txt").is_file() else "",
            workspace_diff={}, workspace_deletions=frozenset(),
            exit_code=int(result.get("codex_returncode") or 0),
            wall_time_s=float(result["policy_wall_s"]),
            timed_out=result.get("status") == "timeout",
            stage=profile, model=result.get("model"),
            reasoning_effort=result.get("reasoning_effort"),
        )
        trajectory_store.put(trajectory)
        reread = trajectory_store.get(trajectory_id)
        grade = Grade(passed=success, score=float(success),
                      details={"official_result": str(official)})
        metric = trajectory_metrics(reread)
        trajectories.append(reread)
        rows.append({"task": row["task"], "seed": row["seed"], "profile": profile,
                     "episode": str(episode), "trajectory_id": trajectory_id,
                     "harness_id": harness.id, "grade": {"passed": grade.passed,
                     "score": grade.score, "details": grade.details}, "metrics": metric})
    summary = {
        "schema": "pyrualean-harnesspro-native-v1",
        "harnesspro_source": str(harnesspro_root.resolve(strict=True)),
        "native_components": ["rho.protocols.Trajectory", "rho.protocols.Grade",
                              "rho.stores.harness.FilesystemHarnessStore",
                              "rho.stores.trajectory.FilesystemTrajectoryStore",
                              "rho.observability.trajectory_metrics",
                              "rho.observability.trajectory_cost_summary",
                              "rho.observability.usage_summary"],
        "rows": rows,
        "usage": usage_summary(trajectories),
        "cost": trajectory_cost_summary(trajectories),
        "scope": "Native HarnessPro artifact, grading type, and accounting integration; "
                 "PyRUA/RoboDojo still executes the robot and supplies official verdicts. "
                 "The public HarnessPro release has no automatic optimizer source.",
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)
                                         + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--harnesspro-root", type=Path,
                        default=Path(__file__).resolve().parents[3] / "HarnessPro")
    args = parser.parse_args(argv)
    report = ingest(args.experiment, args.out_dir, args.harnesspro_root)
    print(json.dumps({"report": str(args.out_dir.resolve() / "report.json"),
                      "trajectories": len(report["rows"]),
                      "native_components": report["native_components"]}, indent=2))


if __name__ == "__main__":
    main()
