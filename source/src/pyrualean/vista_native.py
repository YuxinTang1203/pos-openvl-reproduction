"""Adapt PyRUA observations to VISTA's original visual and memory tool handlers.

TaskContext, SessionController, GameToolDispatcher and ArtifactStore are imported
from the upstream checkout. PyRUA retains its agent loop and Python execution.
The runtime port below exposes public event bookkeeping, not simulator state.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROMPT = """
# VISTA visual memory
The available model tools are python/run_program for robot actions, finish for
ending the episode, and the VISTA tools below for inspecting archived evidence.
The visual tools are independent tool calls; they do not move the robot or reset
the persistent Python session. A task may use both kinds of tool.
The inspect, read_pixels, history, read_guide, write_guide, read_working and
write_working tools use VISTA's native visual memory. Initial camera views are
event 0. Subsequent robot feedback includes vista_evidence event/view references;
Always specify a view when an event has multiple cameras. Camera names with and
without the image extension are accepted when unambiguous. On a failed lookup,
use available_references in the error to correct the event/moment/view selection.
history lists the available events. Inspect uses original image pixel coordinates
and does not move the robot. Historical images describe their recorded scene,
not necessarily the current scene. Compare archived views when useful and keep
stage goals, measured progress, obstacles and next steps in WORKING.md. GUIDE.md
holds reusable observations within this episode. These notes are your hypotheses,
not official success labels. Robot actions still run as multi-step Python programs
in the persistent PyRUA session; get fresh camera evidence via robo.show when needed.

"""


def adapt_robot_prompt(prompt: str, arm: str) -> str:
    """Fix the base robot-only tool inventory when native VISTA tools are enabled."""
    tool = "python(code)" if arm == "cells" else "run_program"
    candidates = [f"through ONE tool,\n`{tool}`.",
                  f"through the primary execution tool,\n`{tool}`."]
    matches = [text for text in candidates if text in prompt]
    if len(matches) != 1 or prompt.count(matches[0]) != 1:
        raise ValueError(f"expected exactly one base {tool} tool-inventory sentence")
    original = matches[0]
    expanded = (
        f"through the `{tool}` robot tool and VISTA visual-memory tools "
        "(`inspect`, `read_pixels`, `history`, `read_guide`, `write_guide`, "
        "`read_working`, `write_working`). Robot actions still use "
        f"`{tool}`."
    )
    return prompt.replace(original, expanded, 1)


class NativeVistaMemory:
    def __init__(self, *, source: Path, directory: Path, task: str, initial_images: dict):
        source = source.resolve(strict=True)
        package = source / "src" / "vista"
        if not (package / "core/support.py").is_file():
            raise ValueError(f"VISTA source missing: {package}")
        if "vista" in sys.modules:
            if Path(sys.modules["vista"].__file__).resolve().parent != package:
                raise RuntimeError("VISTA already imported from a different source checkout")
        elif str(package.parent) not in sys.path:
            sys.path.insert(0, str(package.parent))
        from jsonschema import Draft202012Validator
        from vista.core.artifacts import ArtifactStore
        from vista.core.contracts import PublicTask
        from vista.core.json import thaw
        from vista.core.support import TaskContext

        self.source, self.directory = source, directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.artifacts = ArtifactStore(self.directory / "audit")
        self.port = SimpleNamespace(
            artifacts=self.artifacts,
            state_revision=0,
            terminal=False,
            experiment=SimpleNamespace(budgets={"tool_calls": 10000}),
            specs=(),
        )
        bundle = self._bundle("initial", initial_images)
        self.context = TaskContext(PublicTask("pyrualean", task, observation=bundle), "openai")
        self.context.bind(self.port, self.directory / "context")
        self.artifacts.evidence(bundle)
        self.bindings = {binding.spec.name: binding for binding in self.context.tools}
        self.validators = {
            name: Draft202012Validator(thaw(binding.spec.input_schema))
            for name, binding in self.bindings.items()
        }
        self.artifacts.manifest(self.provenance())

    @staticmethod
    def _bundle(moment, images):
        from PIL import Image
        from vista.core.contracts import Observation, ObservationBundle, Visual

        views = []
        for name, data in images.items():
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
            views.append(
                Visual(
                    name,
                    f"{moment}:{name}:{hashlib.sha256(data).hexdigest()}",
                    data,
                    width,
                    height,
                )
            )
        return (
            ObservationBundle((Observation(moment, tuple(views)),))
            if views
            else ObservationBundle()
        )

    def specs(self):
        return [
            {
                "name": definition["name"],
                "description": definition["description"],
                "input_schema": definition["inputSchema"],
            }
            for binding in self.bindings.values()
            for definition in [binding.spec.public_definition()]
        ]

    def references(self):
        """Only public observation metadata; no simulator state or image contents."""
        return [
            {"event": event["event"], **observation}
            for event in self.context.public_events
            for observation in event["observations"]
        ]

    def prompt(self):
        return PROMPT + "\nInitial VISTA references: " + json.dumps(self.references()) + "\n"

    def _resolve_views(self, arguments):
        resolved = copy.deepcopy(arguments)
        references = self.references()
        for selection in resolved.get("views", []):
            candidates = [r for r in references if r["event"] == selection.get("event")]
            if not candidates:
                continue
            moment = selection.get("moment", candidates[-1]["moment"])
            reference = next((r for r in candidates if r["moment"] == moment), None)
            if reference is None:
                continue
            names = [v["view"] for v in reference["views"]]
            requested = selection.get("view")
            if requested in names or not isinstance(requested, str):
                continue
            matches = [n for n in names if Path(n).stem == requested or Path(requested).stem == n]
            if len(matches) == 1:
                selection["view"] = matches[0]
        return resolved

    def record_binding(self, reference, measurement, source, image):
        """Archive the actual perception frame, independently of end-of-cell images.

        These are existing public camera bytes captured by OpenVL, not a new
        observation request. Do not replace VISTA's current view with this history.
        """
        view = f"source_{reference}.png"
        supplied = {view: image} if image is not None else {}
        bundle = self._bundle(f"evidence-{reference}", supplied)
        event = self.context.record_event(
            bundle, action={"tool": source["tool"], "binding": reference,
                            "source_step": source.get("step"),
                            "call_index": source.get("call_index")},
            result=json.dumps(measurement, ensure_ascii=False, default=str),
            update_current=False, environment=False,
        )
        self.artifacts.evidence(bundle)
        return {"event": event, "views": list(supplied),
                "source_step": source.get("step"), "camera": source.get("camera"),
                "image_sha256": hashlib.sha256(image).hexdigest() if image is not None else None}

    def record(self, turn, name, result, images):
        # Only already agent-visible camera pixels and feedback are supplied.
        # Agent-produced display arrays stay in the PyRUA reply, not in VISTA's
        # archive of original camera observations.
        labels = result.get("images") or []
        supplied = {
            label: data
            for label, data in zip(labels, images, strict=True)
            if not label.startswith("display_")
        }
        bundle = self._bundle(f"robot-{turn}", supplied)
        self.port.state_revision += 1
        event = self.context.record_event(
            bundle,
            action={"tool": name, "pyrualean_turn": turn},
            result=json.dumps(result, ensure_ascii=False, default=str),
            success="error" not in result and "exception" not in result,
            environment=True,
        )
        self.artifacts.evidence(bundle)
        if name == "finish":
            self.port.terminal = True
        return {"event": event, "views": list(supplied)}

    def execute(self, name, arguments):
        from jsonschema import ValidationError

        try:
            self.validators[name].validate(arguments)
        except ValidationError as exc:
            payload = {
                "ok": False,
                "error": exc.message,
                "text": exc.message,
                "available_references": self.references()[-3:],
            }
            self.artifacts.record(
                "tool",
                {"name": name, "arguments": arguments, "success": False, "text": exc.message},
            )
            return payload, []
        resolved = (
            self._resolve_views(arguments) if name in {"inspect", "read_pixels"} else arguments
        )
        reply = self.bindings[name].handler(resolved)
        pictures = [
            visual.data for obs in reply.observation.displayed_observations for visual in obs.views
        ]
        payload = {"ok": reply.success, "text": reply.text}
        if not reply.success:
            payload["error"] = reply.text
            if name in {"inspect", "read_pixels"}:
                requested = {v.get("event") for v in arguments.get("views", [])}
                references = self.references()
                payload["available_references"] = [
                    r for r in references if r["event"] in requested
                ] or references[-3:]
                payload["recovery"] = (
                    "Select an exact event, moment and view from available_references; "
                    "use history for other events. Do not repeat the unchanged failing selection."
                )
        if resolved != arguments:
            payload["resolved_views"] = resolved["views"]
        if pictures:
            payload["images"] = list(self.context.labels(reply.observation))
        self.artifacts.record(
            "tool",
            {
                "name": name,
                "arguments": arguments,
                "resolved_arguments": resolved,
                "success": reply.success,
                "text": reply.text,
            },
        )
        self.artifacts.evidence(reply.observation)
        return payload, pictures

    def provenance(self):
        from vista.core.artifacts import ArtifactStore
        from vista.core.dispatcher import GameToolDispatcher
        from vista.core.session import SessionController
        from vista.core.support import TaskContext

        files = {}
        for cls in (TaskContext, SessionController, GameToolDispatcher, ArtifactStore):
            path = Path(inspect.getfile(cls)).resolve()
            files[cls.__name__] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        commit = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        return {
            "source": str(self.source),
            "commit": commit,
            "files": files,
            "tools": list(self.bindings),
            "artifacts": str(self.directory),
            "scope": "Native VISTA visual/memory tools; PyRUA robot and Qwen agent loop",
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
