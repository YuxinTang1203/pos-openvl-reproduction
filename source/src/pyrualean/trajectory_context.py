"""Append-only public observations for the ReUseIt and RLM mechanism ports."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path


def text(value, name="text", limit=16000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} requires nonempty text of at most {limit} characters")
    return value.strip()


def integer(config, key, default, low=1, high=256):
    value = config.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} must be an integer in {low}..{high}")
    return value


class PublicTrajectory:
    """Never reads archived evaluator results or the simulator's hidden state.

    Derived display images are historical/unknown-time. Only fresh camera reads
    made by capture() qualify for the ReUseIt current-observation guard.
    """

    def __init__(self, directory, task, initial_images=()):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.images = {}
        self.epoch = 0
        self.task = str(task)
        self.append("initial", {"instruction": self.task}, initial_images)

    def append(self, kind, payload, images=(), *, current=False):
        eid = f"event-{len(self.events):06d}"
        event = {"id": eid, "kind": kind, "epoch": self.epoch,
                 "time_monotonic": time.monotonic(),
                 "payload": json.loads(json.dumps(payload, default=str)), "images": []}
        pairs = list(images.items()) if isinstance(images, dict) else list(images)
        for index, (label, data) in enumerate(pairs):
            if not isinstance(data, bytes):
                raise ValueError("trajectory image must be bytes")
            iid = f"{eid}-image-{index}"
            target = self.directory / f"{iid}.png"
            target.write_bytes(data)
            self.images[iid] = (str(label), data)
            event["images"].append({"id": iid, "label": str(label),
                                    "sha256": hashlib.sha256(data).hexdigest(),
                                    "path": str(target),
                                    "current_capture": bool(current)})
        self.events.append(event)
        with (self.directory / "trajectory.jsonl").open("a") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return copy.deepcopy(event)

    def record_tool(self, name, args, result, records):
        records = list(records)
        if any(r.stateful for r in records):
            self.epoch += 1
        public = [{"tool": r.tool, "kwargs": r.kwargs, "summary": r.summary,
                   "ok": r.ok, "error": r.error, "stateful": r.stateful} for r in records]
        labels = result.result.get("images", [])
        pairs = [(labels[i] if i < len(labels) else f"display_{i}", data)
                 for i, data in enumerate(result.images)]
        return self.append("tool", {"tool": name, "arguments": args,
                                    "result": result.result, "public_calls": public}, pairs)

    def capture(self, robot, max_images=2):
        from .arms import _to_png
        pictures, errors = [], []
        for camera in tuple(robot.CAMERAS)[:max_images]:
            try:
                pictures.append((camera, _to_png(robot.image(camera))))
            except Exception as exc:
                errors.append(f"{camera}: {type(exc).__name__}: {exc}")
        return self.append("guard_observation", {"errors": errors}, pictures, current=True)

    def context(self):
        # Do not expose filesystem paths to the analysis REPL: images use IDs.
        data = copy.deepcopy({"task": self.task, "epoch": self.epoch, "events": self.events})
        for event in data["events"]:
            for item in event["images"]:
                item.pop("path", None)
        return data


class Controller:
    controls_loop = True

    def __init__(self, kit, config, task, initial_images=()):
        if kit.arm != "cells":
            raise ValueError(f"{self.kind} requires arm=cells")
        self.kit = kit
        self.options = {"window_rounds": integer(config, "window_rounds", 8)}
        self.directory = kit.workspace.parent / "general_harness"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trace = PublicTrajectory(self.directory / "trace", task, initial_images)
        self.revision = 0
        self.phase = "initial"
        self.closed = False
        self.last_input = {}

    def log(self, event, **fields):
        self.revision += 1
        with (self.directory / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"event": event, "kind": self.kind,
                                     "revision": self.revision, **fields}, default=str) + "\n")
        target = self.directory / "state.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.context(), default=str, indent=2) + "\n")
        temporary.replace(target)

    def after_tool(self, name, result, records):
        event = self.trace.record_tool(name, self.last_input, result, records)
        result.result["trajectory_event"] = event["id"]
        return result

    def request_prompt(self):
        return f"{self.kind} controller state:\n" + json.dumps(self.context(), ensure_ascii=False)

    def record_assistant(self, message):
        # Public planner text/calls, excluding private provider reasoning fields.
        self.trace.append("assistant", {k: copy.deepcopy(message[k]) for k in
                          ("content", "tool_calls") if k in message})
