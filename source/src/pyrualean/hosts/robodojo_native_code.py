"""Experimental code cells over RoboDojo's public observation and native joint action."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from ..arms import ArmResult, _to_png
from ..openai_runtime import run_openai_agent
from ..sandbox import audit_source, blocked_names, make_builtins
from .robodojo_kinematics import JOINT_LIMITS
from .xpolicylab_bridge import Bridge, EpisodeOver

ARMS = ("left", "right")
JOINT_STEP_RAD = 0.15


def validate_action(action: Any, state: dict[str, Any], previous: dict[str, list[float]] | None = None) -> dict[str, list[float]]:
    """Check one complete action against limits and the preceding command."""
    keys = {f"{arm}_{part}_joint_state" for arm in ARMS for part in ("arm", "ee")}
    if not isinstance(action, dict) or set(action) != keys:
        raise ValueError(f"action must contain exactly {sorted(keys)}")
    checked = {}
    for arm in ARMS:
        key = f"{arm}_arm_joint_state"
        current = state.get(key)
        values = action[key]
        if len(values) != 6 or current is None or len(current) != 6:
            raise ValueError(f"{key} must have six joints")
        q = [float(v) for v in values]
        if not all(math.isfinite(v) for v in q):
            raise ValueError(f"{key} has a non-finite joint target")
        if any(not lo <= v <= hi for v, (lo, hi) in zip(q, JOINT_LIMITS)):
            raise ValueError(f"{key} exceeds the ARX-X5 joint limits")
        reference = previous[key] if previous is not None else current
        if any(abs(a - float(b)) > JOINT_STEP_RAD + 1e-9 for a, b in zip(q, reference)):
            raise ValueError(f"{key} exceeds {JOINT_STEP_RAD} rad per native action")
        checked[key] = q
        key = f"{arm}_ee_joint_state"
        values = action[key]
        if len(values) != 1:
            raise ValueError(f"{key} must have one target")
        grip = float(values[0])
        if not math.isfinite(grip) or not 0 <= grip <= 1:
            raise ValueError(f"{key} must be within [0, 1]")
        checked[key] = [grip]
    return checked


class NativeCodeToolkit:
    """Persistent Python cells with only observe, camera, display and step."""

    def __init__(self, bridge: Bridge, workspace: Path, max_turns: int):
        self.bridge = bridge
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_turns = max_turns
        self.turn_budget_exhausted = False
        self.finish: dict[str, str] | None = None
        self.turns: list[dict[str, Any]] = []
        self.harnesspro_hook = None
        self._images: list[tuple[str, bytes]] = []
        self._actions: list[dict[str, list[float]]] = []
        self._globals = {
            "__name__": "__policy__",
            "__builtins__": make_builtins(workspace),
            "np": np,
            "math": math,
            "observe": self.observe,
            "camera": self.camera,
            "display": self.display,
            "step": self.step,
        }
        # The shared OpenAI loop asks this read-only predicate after each tool call.
        self.robo = self

    def _reported_success(self) -> bool:
        # The official verdict arrives only after the evaluation client ends the trial.
        return False

    def observe(self) -> dict[str, Any]:
        obs = self.bridge.get_obs()
        return {
            "instruction": str(obs.get("instruction") or obs.get("instructions") or ""),
            "state": json.loads(json.dumps(obs.get("state") or {}, default=lambda x: x.tolist())),
            "cameras": sorted((obs.get("vision") or {}).keys()),
            "status": self.bridge.status(),
        }

    def camera(self, name: str) -> np.ndarray:
        cam = (self.bridge.get_obs().get("vision") or {}).get(name)
        if not isinstance(cam, dict) or cam.get("color") is None:
            raise ValueError(f"camera {name!r} is unavailable")
        return np.asarray(cam["color"]).copy()

    def display(self, image: Any) -> None:
        self._images.append((f"display_{len(self._images)}", _to_png(image)))

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        status = self.bridge.status()
        if status.get("over") or int(status.get("step", 0)) >= int(status.get("step_limit", 1 << 30)):
            raise EpisodeOver("RoboDojo episode or native action budget has ended")
        checked = validate_action(
            action, self.bridge.get_obs().get("state") or {},
            self._actions[-1] if self._actions else None,
        )
        self._actions.append(checked)
        self.bridge.step(checked)
        return self.observe()

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return [
            {"name": "python", "description": "Run Python in a persistent session with observe(), camera(name), display(image), and step(action).", "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
            {"name": "finish", "description": "End the episode after inspecting the public observations.", "input_schema": {"type": "object", "properties": {"status": {"type": "string"}, "summary": {"type": "string"}}, "required": ["status", "summary"]}},
        ]

    def execute_tool(self, name: str, args: dict[str, Any]) -> ArmResult:
        if name == "finish":
            self.finish = {"status": str(args.get("status", "")), "summary": str(args.get("summary", ""))}
            return ArmResult({"ok": True, **self.finish})
        if name != "python":
            return ArmResult({"error": f"unknown tool: {name}"})
        if len(self.turns) >= self.max_turns:
            self.turn_budget_exhausted = True
            return ArmResult({"error": "code cell budget exhausted; call finish"})
        code = str(args.get("code", ""))
        index = len(self.turns) + 1
        (self.workspace / f"cell_{index:03d}.py").write_text(code, encoding="utf-8")
        flags = audit_source(code, workspace=self.workspace)
        result: dict[str, Any] = {"cell": index, "sandbox_flags": flags}
        before = len(self._actions)
        self._images = []
        stdout, stderr = io.StringIO(), io.StringIO()
        started = time.perf_counter()
        try:
            blocked = blocked_names(code)
            if blocked:
                raise ValueError(f"sandbox blocked introspection names: {blocked}")
            tree = ast.parse(code, filename=f"<cell {index}>")
            tail = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(tree, f"<cell {index}>", "exec"), self._globals)
                if tail is not None:
                    value = eval(compile(ast.Expression(tail.value), f"<cell {index}>", "eval"), self._globals)
                    if value is not None:
                        result["value"] = repr(value)[:4000]
        except EpisodeOver as exc:
            result["episode_finished"] = str(exc)
        except Exception:
            result["exception"] = traceback.format_exc()[-4000:]
        result.update(stdout=stdout.getvalue()[-8000:], stderr=stderr.getvalue()[-8000:], elapsed_s=round(time.perf_counter() - started, 2), native_actions=len(self._actions) - before)
        images = self._images
        if images:
            result["images"] = [name for name, _ in images]
        self.turns.append({"index": index, "native_actions": len(self._actions) - before, "sandbox_flags": flags})
        if len(self.turns) >= self.max_turns:
            self.turn_budget_exhausted = True
        return ArmResult(result, [data for _, data in images])


def run_native_code(bridge: Bridge, out: Path, *, task: str, model: str, max_turns: int, budget_s: float, openai_options: dict[str, Any]) -> dict[str, Any]:
    """Run one official RoboDojo trial through the existing XPolicyLab bridge."""
    out.mkdir(parents=True, exist_ok=True)
    workspace = out / "workspace"
    toolkit = NativeCodeToolkit(bridge, workspace, max_turns)
    initial = bridge.first_obs(30)
    images = []
    for name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        cam = (initial.get("vision") or {}).get(name)
        if isinstance(cam, dict) and cam.get("color") is not None:
            path = out / f"initial_{name}.png"
            path.write_bytes(_to_png(cam["color"]))
            images.append(path)
    prompt = (
        f"RoboDojo task: {task}\n"
        "Control two ARX-X5 arms through persistent Python cells. Available functions: "
        "observe() returns the public instruction, measured joint state, camera names and action budget; "
        "camera(name) returns a current RGB numpy array; display(image) shows it; "
        "step(action) submits one native joint action and returns the next public observation. "
        "Use no task-level or motion primitives. Every action must contain exactly "
        "left_arm_joint_state, right_arm_joint_state (six radians each), and "
        "left_ee_joint_state, right_ee_joint_state (one gripper target each, 0 closed to 1 open). "
        "Include both arms on every action, holding unchanged joints where needed. "
        "To start, run `s = observe(); a = {k: s['state'][k] for k in "
        "('left_arm_joint_state', 'right_arm_joint_state', 'left_ee_joint_state', "
        "'right_ee_joint_state')}` and modify a joint target before `step(a)`. "
        f"Each joint target may change at most {JOINT_STEP_RAD} rad from the preceding commanded target. "
        "Read observations after motion; images may be delayed by one step on older clients. "
        "The official evaluator decides success after the episode; do not assume a submitted action succeeded. "
        "Call finish(status, summary) when done."
    )
    (out / "prompt.txt").write_text(prompt, encoding="utf-8")
    opts = dict(openai_options)
    opts.pop("max_turns", None)
    opts.pop("budget_s", None)
    opts.pop("out_dir", None)
    generation = run_openai_agent(
        prompt, images, toolkit, model=model,
        base_url=os.environ.get("PYRUALEAN_BASE_URL") or os.environ["OPENAI_BASE_URL"],
        api_key=os.environ.get(os.environ.get("PYRUALEAN_API_KEY_ENV") or "OPENAI_API_KEY"),
        max_turns=max_turns, budget_s=budget_s, out_dir=out,
        on_timeout=bridge.finish, should_stop=lambda: bridge.over is not None, **opts,
    )
    bridge.finish()
    verdict = bridge.verdict(timeout_s=25)
    (out / "native_actions.json").write_text(json.dumps(toolkit._actions, indent=2) + "\n")
    result = {"status": generation["status"], "task": task, "robot": "robodojo_native_code", "native_actions": len(toolkit._actions), "generation": generation, "verdict": verdict, "environment_success": verdict.get("success") if verdict else None}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
