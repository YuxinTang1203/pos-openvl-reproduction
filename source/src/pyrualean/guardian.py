"""Optional Guardian verification at explicit Python action boundaries."""
from __future__ import annotations

import contextlib
import importlib
import json
import sys
import time
from pathlib import Path

PROMPT = """
Guardian independently checks key action outcomes using public RGB observations.
All state-changing robot calls MUST run inside a guardian_checkpoint, including
move_to, home, gripper changes and motions used to obtain a better view. Read-only
observations remain available outside checkpoints. This is enforced at runtime.
Use `with robo.guardian_checkpoint(subtask, expected_state, kind='grasp'):` around
one grasp, placement (kind='place'), or other meaningful subtask, possibly with
multiple primitives. State the object, destination and visible expected state
before acting. Do not put dependent subtasks inside the same checkpoint. The
context captures before/after views and verifies on exit; a pass continues Python.
A failure or uncertain result interrupts the cell and blocks further motion even
if caught. Observe with robo.image/show, then call robo.guardian_retry(change=...)
with a concrete revised approach before a new checkpoint. Never blindly replay.
guardian_retry only permits starting a revised checkpoint; it never permits bare
motion. A passed plan also does not authorize motion outside a checkpoint.
Before finish(success), use a kind='final' checkpoint whose expected state covers
the whole task (an empty body is allowed). Subsequent motion invalidates it.
Optional: robo.guardian_plan(plan) verifies a proposed plan against one initial view.
Guardian is fallible; its verdict is separate from official benchmark success.
"""


class GuardianRejected(RuntimeError):
    """Verification did not support continuing dependent actions."""


class LocalGuardian:
    """Load the official wrapper only when this optional integration is enabled."""

    def __init__(self, *, repo, model_path, variant="vanilla"):
        if variant not in {"thinking", "vanilla"}:
            raise ValueError("variant must be thinking or vanilla")
        root = Path(repo).expanduser().resolve()
        if not (root / "examples/guardian.py").is_file():
            raise ValueError(f"Guardian checkout missing examples/guardian.py: {root}")
        model_path = str(Path(model_path).expanduser().resolve())
        other = "thinking" if variant == "vanilla" else "vanilla"
        if Path(model_path).name == f"guardian-{other}":
            raise ValueError("Checkpoint name and Guardian variant disagree")
        # Upstream uses absolute `examples.utils` imports.
        sys.path.insert(0, str(root))
        try:
            module = importlib.import_module("examples.guardian")
            if Path(module.__file__).resolve() != root / "examples/guardian.py":
                raise RuntimeError("Another examples.guardian module is already loaded")
            self.model = module.Guardian(model_path=model_path, thinking=variant == "thinking")
        finally:
            sys.path.remove(str(root))
        # Upstream's six-image prompt calls the views front/left/right; RoboDojo
        # actually has a head camera and two wrist cameras. Keep their identity.
        self.model.execution_instruction = (
            "For these observations, Front means head camera, Left means left wrist "
            "camera, and Right means right wrist camera. " + self.model.execution_instruction
        )

    def verify_subtask(self, **kwargs):
        return self.model.verify_subtask(**kwargs)

    def verify_plan(self, **kwargs):
        return self.model.verify_plan(**kwargs)


class GuardianVerification:
    def __init__(self, toolkit, client, *, task=None, cameras=None):
        self.robot = toolkit.robo
        self.client = client
        self.task = task if task is not None else self.robot.task
        self.cameras = tuple(cameras or self.robot.CAMERAS)
        if len(self.cameras) not in (1, 3, 4) or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("Guardian needs 1, 3 or 4 distinct camera views")
        if any(c not in self.robot.CAMERAS for c in self.cameras):
            raise ValueError("Unknown Guardian camera")
        self.root = toolkit.workspace.parent / "guardian"
        self.root.mkdir(parents=True, exist_ok=True)
        self.blocked = False
        self.active = False
        self.final_supported = False
        self.sequence = 0
        self.last = None
        self.history = []

    def status(self):
        return dict(blocked=self.blocked, final_supported=self.final_supported,
                    checks=len(self.history), last=self.last, requires_checkpoint=True)

    def _audit(self, event):
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def _capture(self, phase, cameras=None):
        paths = []
        for index, camera in enumerate(cameras or self.cameras):
            names = self.robot._show_artifacts([camera])
            if len(names) != 1:
                raise ValueError(f"Missing RGB artifact for {camera}")
            data = self.robot.artifact(names[0])
            # Reject corrupt/empty images before handing them to the model.
            import io

            from PIL import Image
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            path = self.root / f"{self.sequence:04d}_{phase}_{index}.png"
            path.write_bytes(data)
            paths.append(str(path))
        return paths

    def before_motion(self, *, tool=None):
        if self.blocked:
            raise GuardianRejected("Guardian requires observation and guardian_retry(change=...)")
        if not self.active:
            self._audit(dict(event="unguarded_motion_rejected", previous=self.sequence, tool=tool))
            raise GuardianRejected(
                "Motion requires an active guardian_checkpoint(subtask, expected_state, kind=...). "
                "Declare the expected outcome and wrap this action in a checkpoint; "
                "guardian_retry and a passed plan do not authorize bare motion."
            )
        self.final_supported = False
        self._audit(dict(event="motion_authorized", id=self.sequence, tool=tool))

    def _check(self, mode, paths, instruction, *, kind, started):
        event = dict(id=self.sequence, mode=mode, kind=kind, task=self.task,
                     instruction=instruction, cameras=self.cameras, images=paths)
        inference_started = time.perf_counter()
        try:
            method = getattr(self.client, "verify_" + mode)
            key = "plan" if mode == "plan" else "subtask_instruction"
            answer, category = method(img_paths_list=paths, task_instruction=self.task,
                                      **{key: instruction})
            # Official wrapper returns bool or None. Never use bool('False').
            if answer is True and isinstance(category, str) and category.strip() == "success":
                verdict = "pass"
            elif answer is False and isinstance(category, str) and category.strip() != "success":
                verdict = "fail"
            else:
                verdict = "uncertain"
            event.update(answer=answer, category=category, verdict=verdict)
        except Exception as exc:
            event.update(verdict="uncertain", error=f"{type(exc).__name__}: {exc}")
        event["inference_s"] = time.perf_counter() - inference_started
        event["elapsed_s"] = time.perf_counter() - started
        self.last = event
        self.history.append(event)
        self.blocked = event["verdict"] != "pass"
        self.final_supported = not self.blocked and kind == "final"
        self._audit(event)
        if self.blocked:
            raise GuardianRejected(json.dumps(event, ensure_ascii=False, default=str))
        return dict(event)

    @contextlib.contextmanager
    def checkpoint(self, subtask, expected_state, *, kind="subtask"):
        if self.active or self.blocked:
            raise GuardianRejected("Finish current checkpoint or guardian_retry before starting")
        if kind not in {"grasp", "place", "subtask", "final"}:
            raise ValueError("kind must be grasp, place, subtask or final")
        if not isinstance(subtask, str) or not subtask.strip():
            raise ValueError("subtask is required")
        if not isinstance(expected_state, str) or not expected_state.strip():
            raise ValueError("expected_state is required before execution")
        self.active = True
        self.final_supported = False
        self.sequence += 1
        started = time.perf_counter()
        instruction = f"{subtask}\nExpected observable outcome: {expected_state}"
        if kind == "final":
            instruction += f"\nVerify completion of the entire task: {self.task}"
        try:
            before = self._capture("before")
            self._audit(dict(event="begin", id=self.sequence, instruction=instruction,
                             kind=kind, images=before))
            yield
            after = self._capture("after")
            self._check("subtask", before + after, instruction, kind=kind, started=started)
        except BaseException as exc:
            self.blocked = True
            self.final_supported = False
            if not isinstance(exc, GuardianRejected):
                self.last = dict(id=self.sequence, verdict="uncertain", kind=kind,
                                 instruction=instruction, error=f"{type(exc).__name__}: {exc}")
            self._audit(dict(event="checkpoint_interrupted", id=self.sequence,
                             error=f"{type(exc).__name__}: {exc}"))
            raise
        finally:
            self.active = False

    def retry(self, *, change):
        if self.active or not self.blocked:
            raise ValueError("guardian_retry requires an interrupted/failed checkpoint or plan")
        if not isinstance(change, str) or not change.strip():
            raise ValueError("Describe the changed recovery approach")
        self._audit(dict(event="retry", change=change, previous=self.sequence))
        self.blocked = False
        self.final_supported = False
        return {"recovery": change, "next": "Observe and execute a revised checkpoint"}

    def plan(self, plan):
        if self.active or self.blocked:
            raise GuardianRejected("Finish checkpoint or guardian_retry before plan verification")
        if not isinstance(plan, str) or not plan.strip():
            raise ValueError("plan must be a nonempty string")
        self.sequence += 1
        self.final_supported = False
        started = time.perf_counter()
        try:
            paths = self._capture("plan", self.cameras[:1])
            return self._check("plan", paths, plan, kind="plan", started=started)
        except BaseException:
            self.blocked = True
            raise
