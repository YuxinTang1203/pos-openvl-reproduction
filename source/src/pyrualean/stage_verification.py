"""Public-observation stage reviews; model assessments are never task verdicts."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

PROMPT = """
Stage verification is enabled. Python stays persistent and supports multi-action cells.
For run_program, use the same helpers as robo.stage_begin/review/record/retry.
Before moving, call stage_begin(goal, criteria, kind='motion', camera=None).
criteria is a list of concrete visible outcomes, not commands or reached/holding flags.
kind is placement, button, insertion, motion, or final. Use one stage per meaningful
outcome: each transfer then each button press/release; each tube seated after release.
Keep approach/grasp/transfer/release/retreat in one Python cell when possible.
At the stage boundary call stage_review(). It returns labeled before/contact/release/
after images through native memory when available. You must inspect those images in
THE NEXT code call, then call stage_record(assessments), one item per criterion:
{'criterion': 0, 'status': 'observed'|'uncertain'|'not_met',
 'observation': 'describe actual visible change', 'evidence': ['exact image id', ...]}.
Use uncertain for occluded, indistinguishable or missing evidence. Do not infer a
button press from an endpoint: show button travel during contact and release after.
Do not infer insertion from placing a tube above the rack: inspect upright alignment,
seating depth and stability after releasing and retreating. Use wrist views if needed.
Wrap intended pressing/insertion motions in `with robo.action_scope('contact'):`;
a frame is captured at scope exit, so exit the scope BEFORE retracting from contact.
For failed/uncertain review call stage_retry(change='specific changed parameter or
new observation strategy') and recover the SAME goal, then review and record again.
A repeated unchanged retry is rejected. stage_begin cannot discard an unresolved goal.
After all task stages, stage_begin(..., kind='final') must list ALL task requirements,
including final object positions/orientations and required arm state. Review/record it
before finish(status='success'). Evidence-supported model assessments are not official
success; the benchmark independently judges the task. If uncertain, finish as stuck.
"""


class StageVerification:
    def __init__(self, toolkit):
        if toolkit.images == "none":
            raise ValueError("Stage verification requires camera images")
        self.kit = toolkit
        self.robot = toolkit.robo
        self.root = toolkit.workspace.parent / "stage_verification"
        self.root.mkdir(exist_ok=True)
        self.current = None
        self.stages = []
        self.turn = 0
        self.motion_revision = 0
        self.memory = None
        self.pending_images = []
        self.holding = {}
        self.snapshots = {}
        self.sequence = 0

    def audit(self, event, **data):
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps({"event": event, "turn": self.turn, **data}, default=str) + "\n")

    @property
    def model_epoch(self):
        # Direct toolkit tests/manual calls have no model transport. Runtimes
        # set model_turn once per request, shared by all calls in its response.
        return getattr(self.kit, "model_turn", self.turn)

    def begin_turn(self):
        self.turn += 1

    def begin(self, goal, criteria, *, kind="motion", camera=None):
        if self.current and self.current["status"] != "supported":
            raise ValueError("Resolve the current stage with review/record/retry first")
        if kind not in {"motion", "placement", "button", "insertion", "final"}:
            raise ValueError("Unknown stage kind")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("A stage needs a concrete goal")
        if (
            not isinstance(criteria, list)
            or not criteria
            or any(not isinstance(c, str) or not c.strip() for c in criteria)
        ):
            raise ValueError("criteria must be a nonempty list of visible outcomes")
        camera = camera or self.robot.CAMERAS[0]
        if camera not in self.robot.CAMERAS:
            raise ValueError(f"camera must be one of {self.robot.CAMERAS}")
        self.current = dict(
            id=len(self.stages) + 1,
            goal=goal,
            criteria=criteria,
            kind=kind,
            camera=camera,
            status="acting",
            attempt=1,
            evidence=[],
            changes=[],
        )
        self.stages.append(self.current)
        initial = next(
            (
                v
                for v in self.snapshots.values()
                if v["phase"] == "before" and v["view"] in self.robot._show_artifacts([camera])
            ),
            None,
        )
        if kind == "final" and initial:
            self.current["evidence"].append(initial["id"])
        else:
            self.capture("before")
        self.audit("stage_begin", stage=self.current)
        return self.status()

    def before_motion(self):
        if not self.current or self.current["status"] != "acting":
            raise RuntimeError(
                "Stage review required: stage_begin before motion; "
                "review/record a pending comparison or stage_retry to recover"
            )
        self.motion_revision += 1  # even rejected backend actions invalidate a review

    def observe(self, tool, args, result):
        data = asdict(result) if is_dataclass(result) else result
        if not isinstance(data, dict):
            return
        self.audit(
            "primitive_result",
            tool=tool,
            args=args,
            feedback=data,
            stage_id=self.current["id"] if self.current else None,
        )
        arm = args.get("arm", "default")
        previous = self.holding.get(arm, False)
        opened = args.get("gripper") == self.robot.OPEN
        if "holding" in data:
            self.holding[arm] = data["holding"] is True
        elif opened:
            self.holding[arm] = False
        if previous and opened:
            self.capture("release")

    def capture(self, phase):
        if not self.current or self.current["status"] != "acting":
            return
        stage = self.current
        self.sequence += 1
        identifier = f"stage_{stage['id']}_a{stage['attempt']}_{phase}_{self.sequence}"
        name = self.robot._show_artifacts([stage["camera"]])[0]
        try:
            data = self.robot.artifact(name)
            path = self.root / (identifier + ".png")
            path.write_bytes(data)
            snapshot = dict(
                id=identifier, phase=phase, view=name, revision=self.motion_revision, path=str(path)
            )
            vista = getattr(self.kit, "native_vista", None)
            if vista is not None:
                ref = vista.record(self.turn, "stage_snapshot", {"images": [name]}, [data])
                snapshot.update(native="vista", event=ref["event"])
            elif self.memory is not None:
                url = "data:image/png;base64," + base64.b64encode(data).decode()
                ref = self.memory.register_image(url, provenance=identifier)
                if not ref:
                    raise ValueError("Native OpenVL image registration failed")
                snapshot.update(native="openvlharness", image_ref=ref)
            else:
                snapshot["native"] = "pyrua_public_camera"
            self.snapshots[identifier] = snapshot
            stage["evidence"].append(identifier)
            self.audit("snapshot", snapshot=snapshot)
        except Exception as exc:
            self.audit("capture_error", phase=phase, error=f"{type(exc).__name__}: {exc}")

    def retrieve(self, ids):
        pictures = []
        vista = getattr(self.kit, "native_vista", None)
        for ident in ids:
            snap = self.snapshots[ident]
            if snap["native"] == "vista":
                payload, imgs = vista.execute(
                    "inspect",
                    {
                        "question": "Observe this recorded stage view; assess the stage criteria.",
                        "views": [{"event": snap["event"], "view": snap["view"], "label": ident}],
                    },
                )
                if not payload.get("ok") or len(imgs) != 1:
                    raise RuntimeError(f"Native VISTA retrieval failed: {payload}")
                data = imgs[0]
            elif snap["native"] == "openvlharness":
                url, _, _ = self.memory.resolve_image(snap["image_ref"])
                if not url or not url.startswith("data:image/"):
                    raise RuntimeError("Native OpenVL retrieval returned no image")
                data = base64.b64decode(url.split(",", 1)[1], validate=True)
            else:
                data = Path(snap["path"]).read_bytes()
            pictures.append((ident + ".png", data))
            self.audit("evidence_retrieved", id=ident, native=snap["native"])
        return pictures

    def review(self):
        if not self.current or self.current["status"] != "acting":
            raise ValueError("Begin or retry a stage before reviewing")
        self.capture("after")
        stage = self.current
        # Keep first before and last of each other phase in the current attempt.
        selected = {}
        for ident in stage["evidence"]:
            phase = self.snapshots[ident]["phase"]
            if phase != "before" or phase not in selected:
                selected[phase] = ident
        if not {"before", "after"} <= selected.keys():
            raise RuntimeError("Missing before/after image; stage remains unverified")
        pics = self.retrieve(list(selected.values()))
        self.pending_images.extend(pics)
        stage.update(
            status="awaiting_assessment",
            review_turn=self.turn,
            review_epoch=self.model_epoch,
            review_revision=self.motion_revision,
            reviewed=selected,
            review_camera=stage["camera"],
        )
        self.audit("review", stage=stage)
        return {
            "stage": stage,
            "instruction": "Inspect attached images in the next code call. "
            "Record observations for every criterion; missing evidence means uncertain.",
        }

    def record(self, assessments):
        s = self.current
        if not s or s["status"] != "awaiting_assessment":
            raise ValueError("stage_review must deliver images before stage_record")
        if self.model_epoch <= s["review_epoch"]:
            raise ValueError("Images must reach the model: record in the NEXT code call")
        if self.motion_revision != s["review_revision"]:
            raise ValueError("Evidence is stale after motion")
        if not isinstance(assessments, list) or len(assessments) != len(s["criteria"]):
            raise ValueError("Assess every criterion exactly once")
        seen = set()
        refs = set(s["reviewed"].values())
        for a in assessments:
            i = a.get("criterion")
            if type(i) is not int or not 0 <= i < len(s["criteria"]) or i in seen:
                raise ValueError("criterion must be a unique zero-based index")
            seen.add(i)
            if a.get("status") not in {"observed", "uncertain", "not_met"}:
                raise ValueError("Unknown criterion status")
            if not isinstance(a.get("observation"), str) or not a["observation"].strip():
                raise ValueError("Describe the observed outcome")
            if (
                not isinstance(a.get("evidence"), list)
                or not a["evidence"]
                or not set(a["evidence"]) <= refs
            ):
                raise ValueError("Cite the exact evidence ids returned by stage_review")
        supported = all(a["status"] == "observed" for a in assessments)
        needed = {"after"}
        if s["kind"] == "button":
            needed |= {"before", "contact"}
        if s["kind"] in {"placement", "insertion"}:
            needed.add("release")
        cited = {ref for a in assessments for ref in a["evidence"]}
        if supported and any(s["reviewed"].get(phase) not in cited for phase in needed):
            raise ValueError(f"Evidence missing for {sorted(needed)}; record uncertain and retry")
        s.update(status="supported" if supported else "needs_recovery", assessments=assessments)
        self.audit("assessment", stage=s, source="model_assessment_not_benchmark_verdict")
        return self.status()

    def retry(self, *, change, camera=None):
        s = self.current
        if not s or s["status"] not in {"needs_recovery", "acting"}:
            raise ValueError("Record an uncertain/failed assessment before retrying")
        if not isinstance(change, str) or not change.strip() or change in s["changes"]:
            raise ValueError("State a new concrete changed parameter or observation strategy")
        if camera is not None and camera not in self.robot.CAMERAS:
            raise ValueError("Unknown camera")
        s["changes"].append(change)
        s.update(
            status="acting", attempt=s["attempt"] + 1, evidence=[], camera=camera or s["camera"]
        )
        self.capture("before")
        self.audit("retry", stage=s)
        return self.status()

    def can_finish(self):
        s = self.current
        return bool(
            s
            and s["kind"] == "final"
            and s["status"] == "supported"
            and s.get("review_revision") == self.motion_revision
        )

    def status(self):
        return {
            "stage_id": self.current["id"] if self.current else None,
            "status": self.current["status"] if self.current else "not_started",
            "final_supported": self.can_finish(),
            "assessment_source": "model; not an official task verdict",
            "goal": self.current["goal"] if self.current else None,
            "criteria": self.current["criteria"] if self.current else [],
            "reviewed": self.current.get("reviewed", {}) if self.current else {},
            "attempt": self.current["attempt"] if self.current else 0,
        }
