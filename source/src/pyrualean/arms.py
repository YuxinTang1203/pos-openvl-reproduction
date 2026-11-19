# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Tool surfaces for the code arms, served to an agent loop through MCP.

Two arms, one contract.  Each is a tiny toolkit (``get_tools_spec`` /
``execute_tool``) that an MCP server can expose to Codex:

* ``cells``  - one tool ``python(code)``: the code runs in a *persistent*
  namespace where ``robo`` lives, like a notebook or OpenAI's python tool.
  Variables and functions defined in one cell are available in the next.
* ``program`` - one tool ``run_program(code | path)``: the code is a complete
  file defining ``run(robo)``; every program starts from a fresh namespace
  (files in the workspace persist and can be imported).  ``max_programs=1``
  is the single-shot protocol.

Both arms also get ``finish(status, summary)``.  Every tool result is text
(stdout, errors, the ledger of primitive calls, the robot state) plus the two
current camera images whenever the robot moved, so the agent's per-turn
perception matches what a tool-calling agent receives after each primitive.
"""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import math
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._backend import EpisodeFinished
from ._robot import RobotBase
from .runner import PolicyLoadError, load_policy, run_policy
from .sandbox import audit_source, blocked_names, make_builtins

class CellDeadlineExceeded(BaseException):
    """Enforce the existing whole-episode deadline in generated Python."""


ARMS = ("cells", "program")
#: When camera images are attached to a tool result: after every turn that
#: moved the robot (CUA-style parity), only when the code called
#: ``robo.show()`` (the author decides when to look), or never.
IMAGE_POLICIES = ("on-motion", "on-demand", "none")
#: What a tool result contains besides the program's own output.  ``pure`` is
#: python-tool semantics: stdout, stderr, the exception, the trailing value and
#: whatever ``robo.show()`` asked for, nothing else.  ``rich`` adds the robot
#: state, the ledger of primitive calls and step counts.
FEEDBACK_MODES = ("pure", "rich")
MAX_TEXT_CHARS = 20_000


@dataclass
class ArmResult:
    """What the MCP server hands back: ``result`` (dict) and content blocks."""

    result: dict[str, Any]
    images: list[bytes] = field(default_factory=list)
    rendered_text: str | None = None

    @property
    def content_blocks(self) -> list[dict[str, Any]]:
        text = json.dumps(self.result, indent=2, ensure_ascii=False, default=str)
        if self.rendered_text is not None:
            text = self.rendered_text
        elif len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "\n[truncated]"
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for data in self.images:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(data).decode("ascii"),
                    },
                }
            )
        return blocks


def _to_png(item: Any) -> bytes:
    """Encode a numpy array, PIL image or PNG bytes as PNG bytes."""
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)
    from PIL import Image

    if hasattr(item, "save") and hasattr(item, "mode"):
        image = item
    else:
        import numpy as np

        array = np.asarray(item)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _trim(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return (
        text[: limit // 2]
        + f"\n... [{len(text) - limit} chars omitted] ...\n"
        + text[-limit // 2 :]
    )


def _state_dict(robo: RobotBase) -> dict[str, Any]:
    try:
        return robo._summary_dict()
    except Exception as exc:  # noqa: BLE001 - reported to the agent
        return {"error": f"state unavailable: {exc}"}


def _blocked_message(names: list[str], what: str) -> str:
    """Why a cell or program was not run (it named :data:`BLOCKED_NAMES` members)."""

    return (
        f"SandboxError: {', '.join(names)} may not be used by policy code; the {what} was not "
        "run.  Stay within the documented `robo` API."
    )


class ArmToolkit:
    """MCP-servable tool surface around one robot (a :class:`RobotBase`)."""

    def __init__(
        self,
        robo: RobotBase,
        *,
        arm: str,
        workspace: Path,
        max_programs: int | None = None,
        program_timeout_s: float | None = None,
        images: bool | str = True,
        feedback: str = "rich",
        max_turns: int | None = None,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        if images is True:
            images = "on-motion"
        elif images is False:
            images = "none"
        if images not in IMAGE_POLICIES:
            raise ValueError(f"images must be one of {IMAGE_POLICIES}, got {images!r}")
        if feedback not in FEEDBACK_MODES:
            raise ValueError(f"feedback must be one of {FEEDBACK_MODES}, got {feedback!r}")
        self.feedback = feedback
        self.robo = robo
        self.arm = arm
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_programs = max_programs
        self.program_timeout_s = program_timeout_s
        #: Decision budget: at most this many `python` / `run_program` calls.
        self.max_turns = max_turns
        self.turn_budget_exhausted = False
        self.images = images
        self.finish: dict[str, Any] | None = None
        self.stage_verifier = None
        self.guardian_verifier = None
        self.evidence_adapter = None
        self.harnesspro_hook = None
        self.native_vista = None
        self.general_harness = None
        self.turns: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._displayed: list[Any] = []
        #: Audit flags of every cell / program that raised any (host-side only).
        self.sandbox_flags: list[dict[str, Any]] = []
        self._last_flags: list[str] = []
        self._builtins = make_builtins(self.workspace)
        self._globals: dict[str, Any] = {
            "__name__": "__policy__",
            "__builtins__": self._builtins,
            "robo": robo,
            "math": math,
            "time": time,
            "display": self._display,
            "log": print,
        }
        try:
            import numpy as np

            self._globals["np"] = np
            self._globals["numpy"] = np
        except ImportError:  # pragma: no cover - numpy is expected at runtime
            pass
        self._cell_index = 0
        self._program_index = 0

    def enable_guardian(self, *, client=None, task=None, cameras=None, **options):
        """Opt in to the independent Guardian verifier; baseline stays unchanged."""
        from .guardian import GuardianVerification, LocalGuardian

        if self.guardian_verifier is not None:
            raise ValueError("Guardian already enabled")
        client = client if client is not None else LocalGuardian(**options)
        verifier = GuardianVerification(self, client, task=task, cameras=cameras)
        self.guardian_verifier = verifier
        self.robo._guardian_verifier = verifier
        return verifier

    def enable_openvl_evidence(self, **options):
        """Use upstream memory/capabilities without replacing the agent loop."""
        from .openvl_adapter import OpenVLEvidenceAdapter

        if self.evidence_adapter is not None:
            raise ValueError("OpenVL evidence is already enabled")
        self.evidence_adapter = OpenVLEvidenceAdapter(self, **options)
        return self.evidence_adapter

    def enable_stage_verification(self):
        """Observe stage outcomes through public camera evidence and native memory."""
        from .stage_verification import StageVerification

        if self.stage_verifier is not None:
            raise ValueError("Stage verification already enabled")
        # A review must reach the model before a later code call can assess it.
        # Validate before installing the recovery guard or namespace helpers.
        used = self._cell_index if self.arm == "cells" else self._program_index
        limits = [self.max_turns - used] if self.max_turns is not None else []
        if self.arm == "program" and self.max_programs is not None:
            limits.append(self.max_programs - self._program_index)
        if limits and min(limits) < 2:
            raise ValueError(
                "Stage verification requires at least two remaining code calls across "
                "separate model responses (review, then record); increase max_turns "
                "or max_programs before enabling it"
            )
        verifier = StageVerification(self)
        if self.evidence_adapter:
            verifier.memory = self.evidence_adapter.memory
        self.enable_execution_recovery()
        self.stage_verifier = verifier
        self.robo._stage_verifier = self.stage_verifier
        for name in ("begin", "review", "record", "retry"):
            self._globals["stage_" + name] = getattr(self.stage_verifier, name)

    def enable_execution_recovery(self):
        """Enable primitive checks without changing the persistent Python loop."""
        from .action_feedback import ExecutionFeedback

        self.enable_action_feedback()
        if not isinstance(self.robo._action_feedback, ExecutionFeedback):
            self.robo._action_feedback = ExecutionFeedback()
        self._globals["action_scope"] = self.robo.action_scope

    def enable_action_feedback(self):
        """Opt in for harness runs; plain baseline feedback remains unchanged."""
        from .action_feedback import ActionFeedback, checked

        if self.robo._action_feedback is None:
            self.robo._action_feedback = ActionFeedback()
        self._globals["checked"] = checked

    def _display(self, image: Any) -> None:
        """Attach an image (numpy array, PIL image or PNG bytes) to this turn's result."""
        self._displayed.append(image)

    # -- toolkit contract ---------------------------------------------------

    def get_tools_spec(self) -> list[dict[str, Any]]:
        finish = {
            "name": "finish",
            "description": (
                "End the episode. Call it once the task predicate has fired "
                "(a tool result showed terminated=true) or when you cannot make "
                "progress, then stop."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success", "failure", "stuck"]},
                    "summary": {"type": "string", "description": "one or two sentences"},
                },
                "required": ["status", "summary"],
            },
        }
        if self.arm == "cells":
            main = {
                "name": "python",
                "description": (
                    "Execute Python code in a persistent session where `robo` "
                    "(the robot), `np`, `math` and `time` are defined. "
                    "Variables, functions and imports persist between calls. "
                    "Returns stdout, stderr, the exception if any, the value of a "
                    "trailing expression, the robot state, and the current camera "
                    "images whenever the robot moved."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python source"}},
                    "required": ["code"],
                },
            }
        else:
            budget = (
                f" You may call it at most {self.max_programs} time(s)."
                if self.max_programs
                else ""
            )
            main = {
                "name": "run_program",
                "description": (
                    "Run a complete Python program that defines `run(robo)`: pass "
                    "the full source as `code`, or `path` of a .py file you wrote "
                    "in the workspace. Each program starts from a fresh namespace "
                    "(workspace files persist and can be imported). Returns "
                    "stdout, stderr, the exception if any, the ledger of primitive "
                    "calls, the robot state and the current camera images." + budget
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "full program source"},
                        "path": {"type": "string", "description": "workspace-relative .py file"},
                    },
                },
            }
        from .prompt import enabled_helper_reference

        helper_reference = enabled_helper_reference(self.robo)
        if helper_reference:
            main["description"] += "\n\nEnabled robot helpers:\n" + helper_reference
        if getattr(self.general_harness, "controls_loop", False):
            return self.general_harness.tools([main, finish])
        specs = ([main, finish]
                 + (self.evidence_adapter.specs() if self.evidence_adapter else [])
                 + (self.native_vista.specs() if self.native_vista else []))
        if hasattr(self.general_harness, "prepare_tools"):
            specs = self.general_harness.prepare_tools(specs)
        return specs

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ArmResult:
        with self._lock:
            started = time.perf_counter()
            calls_before = len(self.robo._ledger)
            action_feedback = self.robo._action_feedback
            feedback_start = len(action_feedback.events) if action_feedback else 0
            if name in {"python", "run_program"} and hasattr(action_feedback, "begin_turn"):
                action_feedback.begin_turn()
            if self.stage_verifier:
                self.stage_verifier.begin_turn()
            steps_before = self.robo._backend.env_steps()
            if self.evidence_adapter:
                self.evidence_adapter.before_tool(name=name, args=input_dict)
            hook_context = None
            hook_denial = None
            try:
                if self.harnesspro_hook:
                    hook_denial, hook_context = self.harnesspro_hook.before(name, input_dict)
                if hook_denial:
                    result = ArmResult({"error": hook_denial})
                elif getattr(self.general_harness, "controls_loop", False):
                    result = self.general_harness.execute(name, input_dict, self._python, self._finish)
                elif self.native_vista and name in self.native_vista.bindings:
                    payload, pictures = self.native_vista.execute(name, input_dict)
                    result = ArmResult(payload, pictures)
                elif name == "finish":
                    if hasattr(self.general_harness, "execute_finish"):
                        result = self.general_harness.execute_finish(input_dict, self._finish)
                    else:
                        result = self._finish(input_dict)
                elif name == "python" and self.arm == "cells":
                    if hasattr(self.general_harness, "execute_python"):
                        result = self.general_harness.execute_python(input_dict, self._python)
                    else:
                        result = self._python(str(input_dict.get("code", "")))
                elif name == "run_program" and self.arm == "program":
                    result = self._run_program(input_dict)
                elif self.evidence_adapter and name in self.evidence_adapter.tools:
                    result = self.evidence_adapter.invoke(name, input_dict)
                else:
                    result = ArmResult({"error": f"unknown tool: {name}"})
            except Exception as exc:  # noqa: BLE001 - never crash the server
                result = ArmResult({"error": f"{type(exc).__name__}: {exc}"})
            if action_feedback:
                evidence = action_feedback.since(feedback_start)
                if evidence:
                    result.result = {"action_feedback": evidence, **result.result}
            if self.guardian_verifier:
                result.result["guardian"] = self.guardian_verifier.status()
            if self.stage_verifier:
                result.result = {"stage_verification": self.stage_verifier.status(),
                                 **result.result}
            if self.evidence_adapter:
                try:
                    result = self.evidence_adapter.after_tool(result)
                except Exception as exc:
                    # Optional evidence must never hide the original execution feedback.
                    result.result["evidence_error"] = f"{type(exc).__name__}: {exc}"
                    self.evidence_adapter.pending_images.clear()
            if hook_context:
                result.result["harnesspro_context"] = hook_context
            combined_memory = self.native_vista and self.evidence_adapter
            archive_tool = name in {"python", "run_program", "finish"} or (
                combined_memory and name in self.evidence_adapter.tools)
            if self.native_vista and archive_tool:
                try:
                    if combined_memory:
                        # Present identical image bytes once, preserving label/image alignment.
                        import hashlib
                        seen, labels, pictures, aliases = {}, [], [], {}
                        for label, data in zip(result.result.get("images", []), result.images, strict=True):
                            digest = hashlib.sha256(data).hexdigest()
                            if digest not in seen:
                                seen[digest] = label
                                labels.append(label)
                                pictures.append(data)
                            else:
                                aliases[label] = seen[digest]
                        result.images = pictures
                        if labels or "images" in result.result:
                            result.result["images"] = labels
                        if aliases:
                            result.result["image_aliases"] = aliases
                    result.result["vista_evidence"] = self.native_vista.record(
                        len(self.turns), name, result.result, result.images)
                    if combined_memory:
                        recovery = result.result.get("evidence_update", {}).get("recovery_evidence")
                        if recovery:
                            for item in recovery.get("recent_evidence_context", []):
                                item["history"] = self.evidence_adapter.binding_history.get(item["reference"])
                            recovery["current_vista_event"] = dict(result.result["vista_evidence"])
                            recovery["image_aliases"] = aliases
                        link = {
                            "image_aliases": aliases,
                            "vista_event": result.result["vista_evidence"]["event"],
                            "robot_step": self.robo._backend.env_steps(),
                            "tool": name,
                            "action_targets": result.result.get("evidence_update", {}).get("action_targets", []),
                            "failure_observation": result.result.get("evidence_update", {}).get("recovery_evidence", {}).get("failure"),
                            "new_bindings": list(self.evidence_adapter.bridge.refs[
                                self.evidence_adapter.ref_start:]),
                            "binding_history": {
                                ref: self.evidence_adapter.binding_history.get(ref)
                                for ref in self.evidence_adapter.bridge.refs[
                                    self.evidence_adapter.ref_start:]},
                            "images": [{"view": label, "sha256": hashlib.sha256(data).hexdigest(),
                                        "derived": label.startswith(("evidence_", "display_"))}
                                       for label, data in zip(labels, pictures, strict=True)],
                        }
                        result.result["openvl_vista_link"] = link
                        with (self.workspace.parent / "openvl_vista_links.jsonl").open("a") as f:
                            f.write(json.dumps(link) + "\n")
                except Exception as exc:
                    # The robot may already have moved. Return its original feedback
                    # even when optional history storage/rendering is unavailable.
                    result.result["vista_error"] = f"{type(exc).__name__}: {exc}"
            if self.harnesspro_hook and not hook_denial:
                try:
                    current_steps = self.robo._backend.env_steps()
                    delta = (current_steps - steps_before if current_steps is not None and steps_before is not None else None)
                    result.result["robot_execution"] = self.harnesspro_hook.execution_summary(
                        self.robo._ledger.records[calls_before:], delta)
                    compacted = self.harnesspro_hook.after(name, result.result, len(self.turns))
                    if compacted:
                        result.rendered_text = compacted
                except Exception as exc:
                    result.result["harnesspro_hook_error"] = f"{type(exc).__name__}: {exc}"
            if self.general_harness:
                try:
                    result = self.general_harness.after_tool(
                        name, result, self.robo._ledger.records[calls_before:])
                except Exception as exc:
                    result.result["general_harness_error"] = f"{type(exc).__name__}: {exc}"
            moved_flag = any(r.stateful for r in self.robo._ledger.records[calls_before:])
            steps_after = self.robo._backend.env_steps()
            steps_delta = (
                steps_after - steps_before
                if steps_before is not None and steps_after is not None
                else None
            )
            flags, self._last_flags = self._last_flags, []
            self.turns.append(
                {
                    "index": len(self.turns),
                    "tool": name,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "ok": "error" not in result.result and "exception" not in result.result,
                    "moved": bool(result.result.get("moved", moved_flag)),
                    "env_steps": result.result.get("env_steps", steps_delta),
                    "terminated": self.robo._reported_success(),
                    "sandbox_flags": flags,
                }
            )
            if flags:
                self.sandbox_flags.append({"turn": len(self.turns) - 1, "flags": flags})
            return result

    # -- tools ----------------------------------------------------------------

    def _finish(self, args: dict[str, Any]) -> ArmResult:
        guard = self.robo._action_feedback
        if (getattr(guard, "persistent", False) and guard.blocked
                and args.get("status") == "success"):
            return ArmResult({"error": "Resolve pending recovery before finish(success)",
                              "recovery_required": guard.blocked})
        if (self.guardian_verifier and args.get("status") == "success"
                and not self.guardian_verifier.final_supported):
            return ArmResult({"error": "Guardian requires a passing final checkpoint "
                              "before finish(success); otherwise finish as stuck.",
                              "guardian": self.guardian_verifier.status()})
        if (self.stage_verifier and args.get("status") == "success"
                and not self.stage_verifier.can_finish()):
            self.stage_verifier.audit("finish_rejected", reason="missing_final_review")
            return ArmResult({"error": "Complete a final stage review and record observed "
                              "task outcomes before claiming success; otherwise finish as stuck.",
                              "stage_verification": self.stage_verifier.status()})
        completion_review = None
        reviewer = getattr(self.evidence_adapter, "completion_reviewer", None)
        if reviewer and args.get("status") == "success":
            try:
                completion_review = reviewer.review()
            except Exception as exc:
                completion_review = {"verdict": "unknown", "checks": [],
                                     "explanation": f"Completion reviewer unavailable: {type(exc).__name__}"}
            if reviewer.mode == "gate" and completion_review.get("verdict") != "satisfied":
                if completion_review.get("exhausted"):
                    self.finish = {"status": "stuck", "summary": "Completion could not be verified within review budget."}
                    return ArmResult({"ok": True, **self.finish, "completion_review": completion_review})
                return ArmResult({"error": "Completion unconfirmed. Review the evidence, obtain missing observations or repair the task; finish as stuck if unresolved.",
                                  "completion_review": completion_review})
        self.finish = {
            "status": str(args.get("status", "")),
            "summary": str(args.get("summary", "")),
        }
        if self._live_success():
            self.finish["environment_terminated"] = self.robo._reported_success()
        return ArmResult({"ok": True, **self.finish, **({"completion_review": completion_review} if completion_review else {})})

    def _live_success(self) -> bool:
        """Whether the robot reports task success during the episode (it has ``done``).

        Without it the feedback carries no success field at all: a constant
        False would read as "not solved yet".
        """
        return "done" not in getattr(self.robo, "ABSENT_MEMBERS", ())

    def _snapshot(self) -> tuple[int, int | None]:
        return len(self.robo._ledger), self.robo._backend.env_steps()

    def _feedback(self, before: tuple[int, int | None], extra: dict[str, Any]) -> ArmResult:
        calls_before, steps_before = before
        records = self.robo._ledger.records[calls_before:]
        moved = any(r.stateful for r in records)
        steps_after = self.robo._backend.env_steps()
        result: dict[str, Any] = dict(extra)
        if self.feedback == "pure":
            return self._attach_images(result, moved, {})
        result["primitive_calls"] = [
            {
                "tool": r.tool,
                "kwargs": r.kwargs,
                "ok": r.ok,
                "error": r.error,
                **{k: v for k, v in r.summary.items() if k != "eef_pos"},
            }
            for r in records
            if r.tool not in ("artifact",)
        ][-40:]
        result["moved"] = moved
        result["env_steps"] = (
            steps_after - steps_before
            if steps_before is not None and steps_after is not None
            else None
        )
        result["state"] = _state_dict(self.robo)
        if self._live_success():
            result["episode_done"] = self.robo._reported_success()
        return self._attach_images(result, moved, result)

    def _attach_images(
        self, result: dict[str, Any], moved: bool, _unused: dict[str, Any]
    ) -> ArmResult:
        images: list[bytes] = []
        recovery = getattr(self.robo._action_feedback, "blocked", None)
        if recovery:
            result["recovery_required"] = recovery
            if self.images != "none" and getattr(self.robo._action_feedback, "capture_failure_images", True):
                for camera in self.robo.CAMERAS:
                    self.robo.show(camera)
        requested = self.robo._take_show_requests()
        if self.images == "on-motion" and moved:
            wanted = list(type(self.robo).ON_MOTION_IMAGES)
        elif self.images == "on-demand":
            wanted = self.robo._show_artifacts(requested)
        else:
            wanted = []
        attached: list[str] = []
        for name in wanted:
            try:
                images.append(self.robo.artifact(name))
                attached.append(name)
            except Exception as exc:  # noqa: BLE001 - image is optional
                if recovery:
                    result.setdefault("recovery_image_errors", []).append(
                        {"image": name, "error": f"{type(exc).__name__}: {exc}"}
                    )
                continue
        displayed, self._displayed = self._displayed, []
        for index, item in enumerate(displayed):
            try:
                images.append(_to_png(item))
                attached.append(f"display_{index}")
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                result.setdefault("display_errors", []).append(f"{type(exc).__name__}: {exc}")
        if self.stage_verifier:
            for name, data in self.stage_verifier.pending_images:
                images.append(data)
                attached.append(name)
            self.stage_verifier.pending_images.clear()
        if attached:
            result["images"] = attached
        return ArmResult(result, images)

    def _budget_left(self, used: int) -> bool:
        """False once the decision budget is spent; ends the episode on the robot."""
        if self.max_turns is None or used < self.max_turns:
            return True
        if not self.turn_budget_exhausted:
            self.turn_budget_exhausted = True
            self.robo._stop_episode("call_budget")
        return False

    def _python(
        self, code: str, *, before_execute: Callable[[], None] | None = None
    ) -> ArmResult:
        if not self._budget_left(self._cell_index):
            return ArmResult(
                {
                    "error": "call budget exhausted: the episode is over, call finish",
                    "cell": self._cell_index,
                }
            )
        self._cell_index += 1
        index = self._cell_index
        (self.workspace / f"cell_{index:03d}.py").write_text(code, encoding="utf-8")
        self._last_flags = audit_source(code, workspace=self.workspace)
        before = self._snapshot()
        stdout, stderr = io.StringIO(), io.StringIO()
        extra: dict[str, Any] = {"cell": index}
        if before_execute is not None:
            before_execute()
        blocked = blocked_names(code)
        if blocked:
            extra["exception"] = _blocked_message(blocked, "cell")
            return self._feedback(before, extra)
        started = time.perf_counter()
        try:
            tree = ast.parse(code, filename=f"<cell {index}>")
        except SyntaxError as exc:
            extra["exception"] = f"SyntaxError: {exc}"
            return self._feedback(before, extra)
        tail: ast.Expr | None = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tail = tree.body.pop()  # type: ignore[assignment]
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(  # noqa: S102 - the whole point of this arm
                    compile(tree, f"<cell {index}>", "exec"), self._globals
                )
                if tail is not None:
                    value = eval(  # noqa: S307 - trailing expression echo
                        compile(ast.Expression(tail.value), f"<cell {index}>", "eval"),
                        self._globals,
                    )
                    if value is not None:
                        extra["value"] = _trim(repr(value), 4000)
        except CellDeadlineExceeded as exc:
            extra["episode_timeout"] = True
            extra["exception"] = str(exc)
        except EpisodeFinished as exc:
            extra["episode_finished"] = exc.reason
        except SystemExit as exc:  # exit()/quit() are removed, sys is not importable
            extra["exception"] = f"SystemExit: {exc}"
        except Exception:  # noqa: BLE001 - reported to the agent
            extra["exception"] = _trim(traceback.format_exc(), 4000)
        extra["stdout"] = _trim(stdout.getvalue())
        extra["stderr"] = _trim(stderr.getvalue())
        extra["elapsed_s"] = round(time.perf_counter() - started, 2)
        return self._feedback(before, extra)

    def _run_program(self, args: dict[str, Any]) -> ArmResult:
        if not self._budget_left(self._program_index):
            return ArmResult({"error": "call budget exhausted: the episode is over, call finish"})
        if self.max_programs is not None and self._program_index >= self.max_programs:
            return ArmResult(
                {
                    "error": f"program budget exhausted ({self.max_programs}); call finish",
                    "state": _state_dict(self.robo),
                }
            )
        code = args.get("code")
        path = args.get("path")
        if not code and not path:
            return ArmResult({"error": "run_program needs `code` or `path`"})
        self._program_index += 1
        index = self._program_index
        if code:
            target = self.workspace / f"program_{index:02d}.py"
            target.write_text(str(code), encoding="utf-8")
        else:
            target = (self.workspace / str(path)).resolve()
            if (
                self.workspace.resolve() not in target.parents
                and target != self.workspace.resolve()
            ):
                return ArmResult({"error": "path must be inside the workspace"})
            if not target.is_file():
                return ArmResult({"error": f"no such file in workspace: {path}"})
        try:
            source = target.read_text(encoding="utf-8")
        except OSError:
            source = ""
        self._last_flags = audit_source(source, workspace=self.workspace)
        before = self._snapshot()
        extra: dict[str, Any] = {"program": index, "file": target.name}
        blocked = blocked_names(source)
        if blocked:  # before load_policy: importing the file runs its module code
            extra["exception"] = _blocked_message(blocked, "program")
            return self._feedback(before, extra)
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            policy = load_policy(target, builtins=self._builtins)
        except PolicyLoadError as exc:
            extra["exception"] = str(exc)
            return self._feedback(before, extra)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            outcome = run_policy(policy, self.robo, timeout_s=self.program_timeout_s)
        extra["status"] = outcome.status
        if outcome.reason:
            extra["reason"] = outcome.reason
        if outcome.error_type:
            extra["exception"] = _trim(
                outcome.traceback or f"{outcome.error_type}: {outcome.error_message}", 4000
            )
        extra["stdout"] = _trim(stdout.getvalue())
        extra["stderr"] = _trim(stderr.getvalue())
        extra["elapsed_s"] = round(outcome.elapsed_s, 2)
        extra["programs_left"] = (
            None if self.max_programs is None else self.max_programs - self._program_index
        )
        return self._feedback(before, extra)


CELLS_CONTRACT = """\
# How you control the robot

You control {robot_blurb} through ONE tool,
`python(code)`.  Each call executes your code in a persistent Python session
(like a notebook): `robo` (a `{cls}` bound to the live simulator{svc_nl}), `np`, `math`, `time`,
`log(value)` and `display(image)` are predefined, and every variable,
function or import you create stays
available in later calls: helpers you define once can be called directly in
any later call, without re-defining or re-importing them.  {feedback_note}
{image_note}
The session is sandboxed: you may import `numpy`, `math`, `time`, `json`,
`re`, `PIL` and other pure-computation modules, but not `os`, `sys`,
`subprocess`, `pathlib` or networking, and `open` only works inside your
workspace; code that reaches for Python internals (`__globals__`,
`__subclasses__`, ...) is refused.

# How the episode is scored

{scoring_head}
{ending_note}
{scoring_tail}

Your working directory is a scratch workspace: you may keep notes or helper
files there.  Do not read files outside it and do not use the shell to reach
the simulator; every action goes through `python`.
"""

PROGRAM_CONTRACT = """\
# How you control the robot

You control {robot_blurb} through ONE tool,
`run_program`.  Each call runs a complete Python program that defines
`run(robo)`: pass the whole source as `code`, or write a `.py` file into your
working directory and pass its `path`.  `robo` is a `{cls}` bound to the
live simulator{svc}.  Every
program starts from a fresh namespace (files in the workspace persist and can
be imported); no language model runs while a program executes, so every
decision inside it must be made in code from the return values documented
below.  The tool returns the program's stdout, stderr, the exception if any,
{feedback_note}  {image_note}
{program_budget}

# How the episode is scored

{scoring_head}
{ending_note}
  A program that raises stops there; the episode continues with the robot
  where it stopped.
{scoring_tail}

Programs run in a sandbox: they may import `math`, `time`, `numpy`, `json`,
`re`, `PIL` and other pure-computation modules, but not `os`, `sys`,
`subprocess`, `pathlib` or networking, and `open` only works inside the
workspace; code that reaches for Python internals (`__globals__`,
`__subclasses__`, ...) is refused; keep them self-contained.
Do not read files outside the workspace and do not use the shell to reach the
simulator; every action goes through `run_program`.
"""


IMAGE_NOTES = {
    "on-motion": (
        "Whenever the robot moved during the call, the two current camera "
        "images (agentview and wrist) are attached as well.  Write short cells "
        "when you need to look before deciding, and loops when you do not."
    ),
    "on-demand": (
        "Images are NOT attached automatically: call {show_examples} "
        "to see a current camera view, or "
        "`display(image)` to show any image array you built (a crop, an "
        "overlay, a picture you kept in memory), when you need to look before "
        "writing the next turn.  Prefer cells that carry out several steps with "
        "in-code checks (segment, back_project, gripper opening, Move.reached) "
        "and look only when a decision depends on the picture."
    ),
    "none": ("Camera images are never attached; rely on the returned state and results."),
}


FEEDBACK_NOTES = {
    "cells": {
        "pure": (
            "The tool returns exactly what your code produced: stdout, stderr, "
            "the exception if any, and the value of a trailing expression.  "
            "Nothing else is added: print what you want to know (for example "
            "`print(robo.state())` or the result of a motion call)."
        ),
        "rich": (
            "The tool returns stdout, stderr, the exception if any, the value of "
            "a trailing expression, the ledger of primitive calls made during "
            "the call and the robot state."
        ),
    },
    "program": {
        "pure": (
            "The tool returns exactly what the program produced: stdout, stderr "
            "and the exception if any.  Nothing else is added: print what you "
            "want to know."
        ),
        "rich": (
            "The tool returns the program's stdout, stderr, the exception if any, "
            "the ledger of primitive calls it made and the robot state."
        ),
    },
}


#: The scoring bullets around ``{ending_note}``, per arm.  ``live``: the
#: benchmark reports success while the episode runs.  ``final``: it reports
#: the score afterwards, possibly using execution history as well as the scene (a robot
#: without ``done``, i.e. RoboDojo, whose success RPent keeps from its own
#: planner too).
SCORING_NOTES = {
    "live": {
        "cells": (
            "- Success = the benchmark's own task predicate fires (`terminated`) at any\n"
            "  moment.  The simulator checks it, not you; motion results carry a\n"
            "  `terminated` field and `robo.done` reports it.",
            "- Once the task terminates, motion calls raise `EpisodeFinished`: call\n"
            '  `finish(status="success", ...)` and stop.  If you cannot make progress, call\n'
            "  `finish` with `failure` or `stuck`.",
        ),
        "program": (
            "- Success = the benchmark's own task predicate fires (`terminated`) at any\n"
            "  moment while a program runs.  The simulator checks it, not you.",
            "- Once the task terminates, motion calls raise `EpisodeFinished`: call\n"
            '  `finish(status="success", ...)` and stop.  If you cannot make progress, call\n'
            "  `finish` with `failure` or `stuck`.",
        ),
    },
    "final": {
        "cells": (
            "- The benchmark evaluates the task's required outcome and execution history,\n"
            "  including any order specified in the instruction. Follow that order even\n"
            "  when a different sequence could produce the same final scene. No result\n"
            "  carries a task-success flag; judge progress from observations and actions.\n"
            "  The absence of live success feedback does not mean only the final scene counts.",
            "- When observations and completed actions support that the instruction is satisfied, call\n"
            '  `finish(status="success", ...)` and stop.  If you cannot make progress, call\n'
            "  `finish` with `failure` or `stuck`.",
        ),
        "program": (
            "- The benchmark evaluates the task's required outcome and execution history,\n"
            "  including any order specified in the instruction. Follow that order even\n"
            "  when a different sequence could produce the same final scene. No result\n"
            "  carries a task-success flag while a program runs; judge progress from\n"
            "  observations and actions. The absence of live success feedback does not\n"
            "  mean only the final scene counts.",
            "- When observations and completed actions support that the instruction is satisfied, call\n"
            '  `finish(status="success", ...)` and stop.  If you cannot make progress, call\n'
            "  `finish` with `failure` or `stuck`.",
        ),
    },
}


ENDING_NOTES = {
    "wall": (
        "- The episode ends after {budget_s} s of wall clock in total (your thinking\n"
        "  included) or {max_env_steps} simulator steps; that counts as a failure."
    ),
    "calls": (
        "- The number of `{tool}` calls in an episode is limited (every call counts,\n"
        "  including calls that only look, print or display); the episode also ends\n"
        "  after {max_env_steps} simulator steps.  Reaching either limit counts as a\n"
        "  failure, so make every call do purposeful work and verify results in code."
    ),
}
BUDGET_MODES = tuple(ENDING_NOTES)
#: The same limits when success is not exposed live: the benchmark evaluates
#: the resulting scene and execution history, so reaching a limit alone need
#: not determine the score.
ENDING_NOTES_FINAL = {
    "wall": (
        "- The episode ends after {budget_s} s of wall clock in total (your thinking\n"
        "  included) or {max_env_steps} simulator steps; the benchmark then evaluates\n"
        "  the resulting scene and execution history."
    ),
    "calls": (
        "- The number of `{tool}` calls in an episode is limited (every call counts,\n"
        "  including calls that only look, print or display); the episode also ends\n"
        "  after {max_env_steps} simulator steps. At either limit the benchmark evaluates\n"
        "  the resulting scene and execution history, so make every call do purposeful\n"
        "  work and verify results in code."
    ),
}


GUIDES_NOTE = (
    "Reference guides: the directory `guides/` in your working directory holds "
    "the three operating guides that the tool-calling version of this system "
    "reads at run time, written for its tool interface: "
    "`strict_hybrid_guide.md` (rules, localisation, command vocabulary, common "
    "failure modes), `pro_hybrid_guide.md` (LIBERO-PRO scene frames and "
    "swap-variant gotchas) and `env_calibration.md` (reachable workspace and "
    "reference heights).  The tool names used there are the `robo` methods of "
    "the same name; remarks about the runner, MCP or servers do not apply "
    "here.  Reading them is optional."
)


#: Wording of the LIBERO contract (the other robots pass their adapter's).
LIBERO_ROBOT = {
    "robot_blurb": "a simulated Franka arm in the LIBERO benchmark",
    "cls": "LiberoRobot",
    "svc": ", a frozen Pi0.5 VLA and a SAM3 segmentation service",
    "svc_nl": ", a\nfrozen Pi0.5 VLA and a SAM3 segmentation service",
    "show_examples": "`robo.show('agentview')` or `robo.show('wrist')`",
    "scoring": "live",
}


def arm_contract(
    arm: str,
    *,
    budget_s: int,
    max_env_steps: int,
    max_programs: int | None,
    images: str = "on-motion",
    feedback: str = "rich",
    guides: bool = False,
    budget_mode: str = "wall",
    robot: dict[str, str] | None = None,
) -> str:
    """Render the arm's contract; ``guides=True`` appends :data:`GUIDES_NOTE`.

    ``budget_mode="wall"`` states the wall-clock budget; ``"calls"`` states
    that the number of tool calls is limited (without the number, which the
    tool-calling arm's own prompt does not state either).  ``robot`` names
    the robot for the contract (keys of :data:`LIBERO_ROBOT`; default LIBERO);
    its ``scoring`` key picks the scoring bullets (:data:`SCORING_NOTES`).
    """

    if budget_mode not in BUDGET_MODES:
        raise ValueError(f"budget_mode must be one of {BUDGET_MODES}, got {budget_mode!r}")
    text = _arm_contract(
        arm,
        budget_s=budget_s,
        max_env_steps=max_env_steps,
        max_programs=max_programs,
        images=images,
        feedback=feedback,
        budget_mode=budget_mode,
        robot=robot,
    )
    if guides:
        text = text.rstrip() + "\n\n" + GUIDES_NOTE + "\n"
    return text


def robot_words(adapter: Any, primitives: str = "full") -> dict[str, str]:
    """Contract wording for a :class:`pyrualean.robots.RobotAdapter`.

    ``primitives="no-vla"`` names the adapter's ``services_novla`` (the
    services left once the VLA is removed) instead of ``services``.
    """

    key = "services" if primitives == "full" else "services_novla"
    phrase = getattr(adapter, key, "") or ""
    head, _, tail = phrase.partition(" ")
    return {
        "robot_blurb": adapter.blurb,
        "cls": adapter.robot_cls.__name__,
        "svc": f", {phrase}" if phrase else "",
        # the cells contract breaks the line after the first word of the phrase
        "svc_nl": f", {head}\n{tail}" if tail else (f", {phrase}" if phrase else ""),
        "show_examples": adapter.show_examples,
        "scoring": "live" if getattr(adapter, "live_success", True) else "final",
    }


def _arm_contract(
    arm: str,
    *,
    budget_s: int,
    max_env_steps: int,
    max_programs: int | None,
    images: str = "on-motion",
    feedback: str = "rich",
    budget_mode: str = "wall",
    robot: dict[str, str] | None = None,
) -> str:
    words = dict(LIBERO_ROBOT, **(robot or {}))
    image_note = IMAGE_NOTES[images].format(show_examples=words["show_examples"])
    feedback_note = FEEDBACK_NOTES[arm][feedback]
    ending_notes = ENDING_NOTES if words["scoring"] == "live" else ENDING_NOTES_FINAL
    ending_note = ending_notes[budget_mode].format(
        budget_s=budget_s,
        max_env_steps=max_env_steps,
        tool="python" if arm == "cells" else "run_program",
    )
    scoring_head, scoring_tail = SCORING_NOTES[words["scoring"]][arm]
    if arm == "cells":
        return CELLS_CONTRACT.format(
            budget_s=budget_s,
            max_env_steps=max_env_steps,
            image_note=image_note,
            feedback_note=feedback_note,
            ending_note=ending_note,
            scoring_head=scoring_head,
            scoring_tail=scoring_tail,
            **words,
        )
    if max_programs == 1:
        budget = (
            "You may call it EXACTLY ONCE: there is no second attempt and no "
            "feedback before it runs, so the program must localise, verify and "
            "recover on its own."
        )
    elif max_programs:
        budget = f"You may call it at most {max_programs} times; use the feedback between calls."
    else:
        budget = "You may call it as many times as the budget allows."
    return PROGRAM_CONTRACT.format(
        budget_s=budget_s,
        max_env_steps=max_env_steps,
        program_budget=budget,
        image_note=image_note,
        feedback_note=feedback_note,
        ending_note=ending_note,
        scoring_head=scoring_head,
        scoring_tail=scoring_tail,
        **words,
    )


__all__ = [
    "ARMS",
    "LIBERO_ROBOT",
    "BUDGET_MODES",
    "FEEDBACK_MODES",
    "GUIDES_NOTE",
    "IMAGE_POLICIES",
    "ArmResult",
    "ArmToolkit",
    "arm_contract",
    "robot_words",
]
