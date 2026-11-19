"""Run HarnessPro's released runtime policy and hook on PyRUA MCP tool calls."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .harnesspro_native import _rho


class NativeHarnessProHook:
    def __init__(self, *, root: Path, policy_file: Path, workspace: Path,
                 task_description: str, base_url: str | None = None,
                 api_key: str | None = None, classifier_transport=None):
        _rho(root)
        from rho.harness_runtime import HarnessRuntimePolicy

        self.policy = HarnessRuntimePolicy.load(policy_file)
        self.workspace = workspace.resolve()
        self.root = root.resolve()
        self.hook = (self.root / "harnesses" / "terminal-bench-2" / "medium" /
                     "runtime" / "hook.py")
        if not self.hook.is_file():
            raise ValueError(f"native HarnessPro hook missing: {self.hook}")
        self.state = workspace.parent / "harnesspro_runtime"
        self.state.mkdir(parents=True, exist_ok=False)
        self.robot_options = json.loads(policy_file.read_text()).get("pyrualean", {})
        self.classification = None
        self.counters = dict(model_requests=0, python_cells=0, robot_primitives=0,
                             environment_steps=0, denied_cells=0)
        self.primitive_feedback = []
        if self.policy.classifier is not None:
            if not base_url:
                raise ValueError("HarnessPro model classifier requires base_url")
            from .harnesspro_classifier import classify

            self.selection, self.classification = classify(
                root, self.policy, task_description, self.state / "classification",
                base_url=base_url, api_key=api_key, transport=classifier_transport,
            )
        else:
            self.selection = self.policy.select(task_description)
        self.selection_path = self.state / "selection.json"
        self.selection_path.write_text(
            json.dumps(self.policy.workspace_selection_payload(self.selection), indent=2)
            + "\n", encoding="utf-8")
        self.audit_path = self.state / "hook_audit.jsonl"
        self.session_context = self._call("SessionStart", "", {})

    def _call(self, event: str, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = {"hook_event_name": event, "cwd": str(self.workspace),
                   "tool_name": f"mcp__pyrualean__{name}", **payload}
        env = dict(os.environ)
        env.update(RHO_RUNTIME_SELECTION_PATH=str(self.selection_path),
                   RHO_RUNTIME_STATE_ROOT=str(self.state / "control"),
                   RHO_RUNTIME_AUDIT_PATH=str(self.audit_path), RHO_RTK_BIN="",
                   RHO_RUNTIME_INTERFACE="codex")
        proc = subprocess.run([sys.executable, str(self.hook)], input=json.dumps(request),
                              text=True, capture_output=True, env=env, timeout=10,
                              check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"HarnessPro hook {event} failed: {proc.stderr[-1000:]}")
        return json.loads(proc.stdout) if proc.stdout.strip() else {}

    @staticmethod
    def _robot_context(text: str) -> str:
        if "Economy phase" in text:
            return (
                "HarnessPro economy phase: batch independent observations and avoid "
                "repeated unchanged perception. Continue necessary robot actions and verification."
            )
        if "Convergence phase" in text:
            return (
                "HarnessPro convergence phase: focus on the current robot failure or "
                "remaining placement. Use measured feedback, then verify the task state."
            )
        return text

    def before(self, name: str, args: dict[str, Any]) -> tuple[str | None, str | None]:
        self.primitive_feedback = []
        # The robot's finish tool must remain available after the action budget.
        if name == "finish":
            return None, None
        counted = self.robot_options.get("count_tools")
        if counted is not None and name not in counted:
            with self.audit_path.open("a") as log:
                log.write(json.dumps({"event": "robot_observation_tool", "tool": name,
                                      "consumes_action_budget": False}) + "\n")
            return None, None
        response = self._call("PreToolUse", name, {"tool_input": args})
        specific = response.get("hookSpecificOutput") or {}
        if specific.get("permissionDecision") == "deny":
            self.counters["denied_cells"] += int(name in {"python", "run_program"})
            self._save_counters()
            return "HarnessPro tool budget exhausted. Call finish with the observed outcome.", None
        self.counters["python_cells"] += int(name in {"python", "run_program"})
        self._save_counters()
        context = specific.get("additionalContext")
        return None, self._robot_context(str(context)) if context else None

    def after(self, name: str, result: dict[str, Any], turn: int) -> str | None:
        if name == "finish":
            return None
        protected = {k: result[k] for k in (
            "error", "exception", "recovery_required", "action_feedback", "evidence_update",
            "vista_evidence", "openvl_vista_link", "robot_execution", "images",
            "state", "episode_done", "harnesspro_context") if k in result}
        # Compress only ordinary output; otherwise the hook's head/tail can
        # contain the same action feedback that we preserve in full below.
        ordinary = {k: v for k, v in result.items() if k not in protected}
        response = self._call("PostToolUse", name,
                              {"tool_response": ordinary, "tool_use_id": f"robot-{turn:04d}"})
        original = json.dumps(result, indent=2, ensure_ascii=False, default=str)
        final = original
        candidate = None
        if response.get("decision") == "block":
            candidate = str(response["reason"]).replace(
                "Full raw result:", "Full unprotected output:")
            if protected:
                candidate += ("\nProtected robot feedback (not a task verdict):\n"
                              + json.dumps(protected, ensure_ascii=False, default=str))
            if len(candidate) < len(original):
                final = candidate
        # Native hook counters describe its intermediate text, not delivery.
        with self.audit_path.open("a") as log:
            log.write(json.dumps({
                "event": "robot_output_presentation", "tool": name, "turn": turn,
                "original_chars": len(original), "final_chars": len(final),
                "candidate_chars": len(candidate) if candidate is not None else None,
                "protected_chars": len(json.dumps(protected, ensure_ascii=False, default=str)),
                "hook_requested_compaction": response.get("decision") == "block",
                "compacted": final != original,
                "saved_chars": len(original) - len(final),
            }) + "\n")
        # Explicit rendering also prevents the general text limit from cutting
        # an uncompressed but large protected action record in half.
        return final

    def _save_counters(self):
        (self.state / "counters.json").write_text(json.dumps({
            "phase_unit": "accepted_python_cells", **self.counters}, indent=2) + "\n")

    def note_model_request(self):
        self.counters["model_requests"] += 1
        self._save_counters()

    def observe_primitive(self, tool, arguments, result, index):
        from dataclasses import asdict, is_dataclass
        data = asdict(result) if is_dataclass(result) else result
        if not isinstance(data, dict):
            return
        # Preserve decision-relevant public measurements, excluding image/mask arrays.
        # Include read-only perception outcomes as well as action outcomes: a failed
        # detection is not an exception, but must survive stdout compression.
        keys = ("reached", "holding", "found", "blocked", "ik_error", "dist_to_target_m",
                "final_dist_m", "final_xyz", "final_eef_pos", "target_xyz", "arm",
                "orientation_error_deg", "pitch", "yaw", "roll", "final_pitch", "final_err",
                "commanded", "opening", "width_m", "terminated", "truncated", "note",
                "steps_used", "max_steps", "step", "step_limit", "error", "geometry_error",
                "world_error", "world_xyz", "center_xyz", "camera", "score", "views",
                "height_guess", "budget_exhausted", "success")
        feedback = {k: data[k] for k in keys if k in data}
        if feedback:
            self.primitive_feedback.append(dict(call_index=index, tool=tool, feedback=feedback))

    def execution_summary(self, records, steps):
        self.counters["robot_primitives"] += sum(r.stateful for r in records)
        self.counters["environment_steps"] += max(0, steps or 0)
        self._save_counters()
        return {"phase_unit": "accepted_python_cells", "counts": dict(self.counters),
                "primitive_feedback": list(self.primitive_feedback),
                "execution_errors": [dict(tool=r.tool, error=r.error) for r in records if r.error],
                "task_outcome": "unknown"}

    def record(self) -> dict[str, Any]:
        return {"source": str(self.root), "policy": str(self.policy.source),
                "hook": str(self.hook), "profile": self.selection.profile.name,
                "policy_sha256": hashlib.sha256(self.policy.source.read_bytes()).hexdigest(),
                "hook_sha256": hashlib.sha256(self.hook.read_bytes()).hexdigest(),
                "selection": str(self.selection_path), "audit": str(self.audit_path),
                "classification": self.classification, "robot_options": self.robot_options,
                "counter_file": str(self.state / "counters.json"), "phase_unit": "accepted_python_cells"}
