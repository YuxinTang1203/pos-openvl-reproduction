"""OpenVL evidence modules inside PyRUA; no upstream model adapter or agent loop."""

from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
from openvlharness.capabilities._base import BaseCapability
from openvlharness.contracts import VirtualCall
from openvlharness.memory.store import MemoryStore
from PIL import Image, ImageDraw

from .arms import ArmResult, _to_png
from .openvl_evidence import PerceptionMemory, RobotPerceptionCapability

PROMPT = """
OpenVL evidence modules are enabled INSIDE the existing PyRUA agent loop.
Keep multi-action Python cells, helpers and variables. Existing robot perception
is unchanged; segment/locate/back_project results become native bindings. Masks
stay in memory instead of being repeated in tool replies.
Python helpers (also on robo):
- evidence_find(query=None, camera=None, limit=3) returns summary DICTIONARIES
  with reference, measurement, validity, camera. If matches is nonempty, use
  ref = matches[0]['reference'], not matches[0]. Empty means no stored match.
- evidence(ref=None, fields=None, allow_historical=False) reads copied values;
  only omitted/None ref defaults to latest. Stale reads raise by default.
- evidence_history(ref, fields=None) returns {reference, purpose:'history',
  current_target:False, measurement, validity} for inspection after actions.
- evidence_current(ref, fields=None) returns {reference, usable, measurement,
  validity, next_step}. Stale => usable=False, measurement=None; it never moves
  or automatically re-localizes. Branch on usable before reading measurement:
    current = evidence_current(ref)
    if current['usable']:
        measurement = current['measurement']  # also check found/geometry_error
    else:
        print(current['next_step'])  # re-localize using its query/camera
- evidence_check(ref) checks visible region/calibration/public depth WITHOUT SAM.
  RGB-only visual_unchanged cannot refresh stale 3D geometry. Unknown, missing or
  changed evidence requires re-localization. Never check then unconditionally read.
- evidence_inspect(ref) shows the stored image/box.
- failure_evidence(query=None, reference=None) retrieves latest-failure context;
  associations are query/read history, not object identity or tracking.
The Python global evidence_refs contains ID strings. The tool-feedback field
evidence_refs contains dictionaries: use row['reference']. image_ref is an image ID; VISTA event/view is
not a robot evidence ID. Use evidence_history for historical review. For targeting,
check freshness after intervening actions; ordinary copied Python coordinates
cannot be automatically invalidated. Same-step evidence can be reused, but
freshness does not prove detection success, identity, geometry or task success.
Robot_Perception_Tool / Robot_Evidence_Tool expose these services; Python helpers
can combine checks and actions without extra model rounds. Do not invent IDs.
"""

REFERENCE_ID_HELP = (
    "Expected a non-empty robot evidence ID string. evidence_find() returns summary "
    "dictionaries: after checking matches is nonempty, use ref = matches[0]['reference'], "
    "not matches[0]. image_ref and VISTA event/view are not robot evidence IDs."
)


def _dict(value):
    return asdict(value) if is_dataclass(value) else value


def _query(args):
    return str(args.get("text_prompt") or args.get("prompt") or args.get("query") or "")


class EvidenceMemory(PerceptionMemory):
    """Immutable native bindings plus explicit validity proofs and query lineage."""

    def __init__(self, toolkit, event, *, validate_geometry=True):
        super().__init__(toolkit, event)
        self.values = {}
        self.entries = {}
        self.proofs = {}
        self.revision = 0
        self.suspended = False
        self.last_read = None
        self.validate_geometry = validate_geometry

    def capture(self, tool, arguments, payload):
        try:
            self._index_capture(tool, arguments, payload)
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            self.event("robot_evidence_error", tool=tool, error=self.errors[-1])

    def _index_capture(self, tool, arguments, payload):
        if self.suspended:
            return
        before = len(self.refs)
        super().capture(tool, arguments, payload)
        if len(self.refs) == before:
            return
        ref = self.refs[-1]
        value = self.memory.copy_bindings_for_execution()[ref]
        self.values[ref] = value
        query = _query(arguments)
        previous = next(
            (
                r
                for r in reversed(self.refs[:-1])
                if r in self.entries
                and self.entries[r]["query"].casefold() == query.casefold()
                and self.entries[r]["camera"] == value["source"].get("camera")
            ),
            None,
        )
        self.entries[ref] = dict(
            query=query,
            camera=value["source"].get("camera"),
            revision=self.revision,
            previous_ref=previous,
        )
        if (
            self.validate_geometry
            and value.get("found", True)
            and value["source"].get("step") is not None
        ):
            try:
                self.proofs[ref] = self.fingerprint(ref, initial=True)
            except Exception as exc:
                self.entries[ref]["validation_unavailable"] = str(exc)
        self.event("binding_indexed", **self.summary(ref))
        adapter = getattr(self.toolkit.robo, "_evidence_adapter", None)
        if adapter:
            adapter.archive(ref, value)
        if payload.get("found") is False and adapter:
            adapter.observe(tool, arguments, payload, value["source"]["call_index"])

    def require_reference(self, ref):
        """Validate an explicit ID without changing evidence or robot state."""
        if not isinstance(ref, str):
            raise TypeError(f"Received {type(ref).__name__} as evidence reference. {REFERENCE_ID_HELP}")
        if not ref.strip():
            raise ValueError(f"Empty evidence reference. {REFERENCE_ID_HELP}")
        if ref not in self.entries:
            raise ValueError(f"Unknown robot evidence ID {ref[:120]!r}. {REFERENCE_ID_HELP}")
        return ref

    def raw(self, ref):
        self.require_reference(ref)
        # Use the public upstream copy API. Callers cannot mutate stored evidence.
        return self.memory.copy_bindings_for_execution()[ref]

    def validity(self, ref):
        self.require_reference(ref)
        value = self.values[ref]
        source, entry = value["source"], self.entries[ref]
        step = self.toolkit.robo._backend.env_steps()
        direct = (
            source.get("step") is not None
            and source["step"] == step
            and entry["revision"] == self.revision
        )
        validated = entry.get("validated_at") == (self.revision, step) and step is not None
        return dict(
            usable=bool(direct or validated),
            status="fresh"
            if direct
            else "revalidated_visible_region"
            if validated
            else "historical_or_unknown",
            current_step=step,
            source_step=source.get("step"),
            caution="Visible-region evidence only; no guarantee of identity or hidden state.",
        )

    def summary(self, ref):
        self.require_reference(ref)
        value, entry = self.values[ref], self.entries[ref]
        keys = (
            "found",
            "center_xyz",
            "world_xyz",
            "top_z",
            "grasp_yaw",
            "score",
            "box_px",
            "box",
            "centroid_rc",
            "views",
            "geometry_error",
            "height_guess",
        )
        return dict(
            reference=ref,
            query=entry["query"],
            camera=entry["camera"],
            previous_ref=entry["previous_ref"],
            association="same query/view; not object identity",
            measurement={k: value[k] for k in keys if k in value},
            image_ref=value["source"].get("image_ref"),
            timestamp_utc=value["source"].get("timestamp_utc"),
            validity=self.validity(ref),
        )

    def find(self, query=None, camera=None, limit=3):
        """Return summary dicts; select row['reference'] for an evidence ID string."""
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("limit must be 1..10")
        refs = [
            r
            for r in reversed(self.refs)
            if r in self.entries
            and (query is None or str(query).casefold() in self.entries[r]["query"].casefold())
            and (camera is None or self.entries[r]["camera"] == camera)
        ]
        return [self.summary(r) for r in refs[:limit]]

    def read(self, reference=None, *, fields=None, allow_historical=False):
        """Read an ID string; only None/omission selects the latest binding."""
        if reference is None:
            if not self.refs:
                raise ValueError("No evidence yet; call locate or segment")
            reference = self.refs[-1]
        value = self.raw(reference)
        validity = self.validity(reference)
        self.event(
            "robot_evidence_read",
            binding=reference,
            allowed=validity["usable"] or allow_historical,
            historical=not validity["usable"],
        )
        if not validity["usable"] and not allow_historical:
            raise ValueError(
                "Historical geometry: use evidence_history(ref) for inspection/comparison, "
                "or evidence_current(ref) and branch on usable before accessing measurement. "
                "If unusable, re-localize; evidence_check alone does not refresh RGB-only geometry. "
                "allow_historical=True is only for comparison"
            )
        self.last_read = reference
        if fields is not None:
            if not isinstance(fields, list) or any(k not in value for k in fields):
                raise ValueError("fields must name existing measurement fields")
            value = {k: value[k] for k in fields}
        value["evidence_validity"] = validity
        return value

    def history(self, reference, *, fields=None):
        """Explicit historical read; never advertise the snapshot as a current target."""
        self.require_reference(reference)
        value = self.read(reference, fields=fields, allow_historical=True)
        return dict(reference=reference, purpose="history", current_target=False,
                    measurement=value, validity=value["evidence_validity"])

    def current(self, reference, *, fields=None):
        """Return no coordinates when stale; callers must choose whether to re-localize."""
        validity = self.check(reference)
        entry = self.entries[reference]
        usable = validity["usable"]
        return dict(reference=reference, usable=usable, validity=validity,
                    measurement=self.read(reference, fields=fields) if usable else None,
                    next_step=None if usable else dict(
                        operation="relocalize", query=entry["query"], camera=entry["camera"],
                        reason=validity.get("reason", "Historical or unknown geometry")))

    def image_bytes(self, ref):
        self.require_reference(ref)
        image_ref = self.values[ref]["source"].get("image_ref")
        if not image_ref:
            raise ValueError("No source image recorded for this evidence")
        url, _, _ = self.memory.resolve_image(image_ref)
        if not url or not url.startswith("data:image/"):
            raise ValueError("Native image reference is unavailable")
        return base64.b64decode(url.split(",", 1)[1], validate=True)

    def fingerprint(self, ref, *, initial=False):
        self.require_reference(ref)
        value = self.values[ref]
        robo = self.toolkit.robo
        camera = self.entries[ref]["camera"]
        step = robo._backend.env_steps()
        if step is None or not camera:
            raise ValueError("No tracked step/camera for geometric validation")
        if initial and value["source"].get("step") != step:
            raise ValueError("Capture already historical; no initial depth proof")
        # All probes are read-only public APIs. Suppress recursive binding capture.
        self.suspended = True
        try:
            calibration = robo.camera_meta(camera)
            keys = ("intrinsic_matrix", "extrinsic_matrix", "shape")
            if not all(k in calibration for k in keys):
                raise ValueError("Camera calibration unavailable: reacquire current perception")
            matrix = json.dumps({k: calibration[k] for k in keys}, sort_keys=True)
            data = self.image_bytes(ref) if initial else _to_png(robo.image(camera))
            with Image.open(io.BytesIO(data)) as image:
                rgb = np.asarray(image.convert("RGB"))
            box = value.get("box_px", value.get("box"))
            if box is None or len(box) != 4:
                raise ValueError("No object ROI: cannot validate old geometry")
            x0, y0, x1, y1 = map(int, box)
            h, w = rgb.shape[:2]
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError("Invalid full-frame pixel box")
            pixels = [(int(y0 + (y1 - y0) * f), int(x0 + (x1 - x0) * f)) for f in (0.25, 0.5, 0.75)]
            pixels = list(dict.fromkeys(pixels))
            if len(pixels) < 3:
                raise ValueError("ROI too small for independent depth probes")
            points = []
            for row, col in pixels if calibration.get("has_depth") else []:
                sample = _dict(robo.back_project(row, col, camera=camera))
                point = sample.get("world_xyz") if isinstance(sample, dict) else None
                if point is None or np.shape(point) != (3,) or not np.isfinite(point).all():
                    raise ValueError("Invalid current depth probe")
                points.append(list(point))
            if robo._backend.env_steps() != step:
                raise ValueError("Environment advanced during validation")
            return dict(
                calibration=matrix,
                rgb_hash=hashlib.sha256(rgb[y0:y1, x0:x1].tobytes()).hexdigest(),
                pixels=pixels,
                points=points,
                step=step,
            )
        finally:
            self.suspended = False

    def check(self, ref):
        self.raw(ref)  # validate id
        state = self.validity(ref)
        if state["usable"]:
            self.event("evidence_validation", binding=ref, **state)
            return state
        reason = "No initial visual/depth proof; reacquire perception"
        # Any failure revokes an earlier proof before work begins.
        self.entries[ref].pop("validated_at", None)
        try:
            if not self.validate_geometry or ref not in self.proofs:
                raise ValueError(reason)
            before, after = self.proofs[ref], self.fingerprint(ref)
            if before["calibration"] != after["calibration"]:
                raise ValueError("Camera calibration/viewpoint changed")
            if before["rgb_hash"] != after["rgb_hash"]:
                raise ValueError("Object image region changed or is occluded")
            if not before["points"] or not after["points"]:
                state = {
                    **self.validity(ref),
                    "visual_unchanged": True,
                    "reason": "RGB region unchanged; no measured depth to revalidate 3D",
                    "next_step": "Use as visual comparison; re-localize before reusing stale 3D",
                }
                self.event("evidence_validation", binding=ref, **state)
                return state
            if not np.allclose(before["points"], after["points"], rtol=0, atol=0.001):
                raise ValueError("Depth probes changed by more than 1 mm")
            self.entries[ref]["validated_at"] = (self.revision, after["step"])
            state = self.validity(ref)
        except Exception as exc:
            state = {
                **self.validity(ref),
                "reason": str(exc),
                "next_step": "Re-localize the object",
            }
        self.event("evidence_validation", binding=ref, **state)
        return state


class CompactPerceptionCapability(RobotPerceptionCapability):
    def execute(self, call, args, memory):
        result = super().execute(call, args, memory)
        ref = result["result"]["binding"]
        result["result"]["measurement"] = self.bridge.summary(ref)
        return result


class EvidenceCapability(BaseCapability):
    name = "Robot_Evidence_Tool"
    backends = {"pyrualean": "pyrualean"}
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Find, validate or inspect native evidence; retrieve failure context. "
                "Use history to review past measurements. current returns usable=false and "
                "measurement=null for stale geometry; branch before using a measurement. "
                "No robot motion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["find", "check", "inspect", "failure", "history", "current"],
                    },
                    "reference": {
                        "type": "string", "minLength": 1,
                        "description": REFERENCE_ID_HELP + " Required for check, inspect, history and current.",
                    },
                    "query": {"type": "string"},
                    "camera": {"type": "string"},
                },
                "required": ["operation"],
                "allOf": [{
                    "if": {"properties": {"operation": {"enum": ["check", "inspect", "history", "current"]}}},
                    "then": {"required": ["reference"]},
                }],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, adapter):
        self.adapter = adapter
        super().__init__(backend="pyrualean", client_config={}, client=self)

    def execute(self, call, args, memory):
        operation = args["operation"]
        if operation == "find":
            result = self.adapter.bridge.find(args.get("query"), args.get("camera"))
        elif operation == "check":
            result = self.adapter.bridge.check(args["reference"])
        elif operation == "inspect":
            result = self.adapter.inspect(args["reference"])
        elif operation == "history":
            result = self.adapter.bridge.history(args["reference"])
        elif operation == "current":
            result = self.adapter.bridge.current(args["reference"])
        else:
            result = self.adapter.failure_evidence(
                query=args.get("query"), reference=args.get("reference")
            )
        return {"ok": True, "result": result}

    def normalize_result(self, call, result, prepared, binding_refs):
        return result

    @staticmethod
    def process_output(payload, prepared):
        return None


class OpenVLEvidenceAdapter:
    def __init__(
        self,
        toolkit,
        *,
        validate_geometry=True,
        max_failure_images=2,
        execution_recovery=False,
        stage_verification=False,
        temporal_feedback=False,
    ):
        if type(max_failure_images) is not int or not 0 <= max_failure_images <= 2:
            raise ValueError("max_failure_images must be 0..2")
        self.kit = toolkit
        self.root = toolkit.workspace.parent / "openvl_evidence"
        self.root.mkdir(exist_ok=True)
        self.memory = MemoryStore(modules=("image", "tool_context"), emit_policy="on_change")
        self.memory.start([])
        self.bridge = EvidenceMemory(toolkit, self.event, validate_geometry=validate_geometry)
        self.bridge.bind(self.memory)
        self.tools = {
            c.name: c for c in (CompactPerceptionCapability(self.bridge), EvidenceCapability(self))
        }
        self.failures = []
        self.pending_images = []
        self.max_failure_images = max_failure_images
        self.serial = 0
        self.temporal_feedback = temporal_feedback
        self.active_target = None
        self.action_links = []
        self.action_start = 0
        self.binding_history = {}
        self.event_counts = {}
        toolkit.robo._evidence_adapter = self
        enable_capture = getattr(toolkit.robo._backend, "enable_perception_evidence", None)
        if callable(enable_capture):
            enable_capture()
        # Instance-only helpers avoid changing the API prompt for baseline runs.
        for name, method in dict(
            evidence=self.bridge.read,
            evidence_find=self.bridge.find,
            evidence_check=self.bridge.check,
            evidence_history=self.bridge.history,
            evidence_current=self.bridge.current,
            evidence_inspect=self.inspect,
            failure_evidence=self.failure_evidence,
        ).items():
            setattr(toolkit.robo, name, method)
        toolkit._globals.update(
            evidence=self.bridge.read,
            evidence_find=self.bridge.find,
            evidence_check=self.bridge.check,
            evidence_history=self.bridge.history,
            evidence_current=self.bridge.current,
            evidence_inspect=self.inspect,
            failure_evidence=self.failure_evidence,
        )
        if temporal_feedback:
            toolkit.robo.evidence_target = self.target_scope
            toolkit._globals["evidence_target"] = self.target_scope
        if execution_recovery or stage_verification:
            toolkit.enable_execution_recovery()
        if stage_verification:
            toolkit.enable_stage_verification()
            toolkit.stage_verifier.memory = self.memory
        files = {}
        for cls in (MemoryStore, BaseCapability):
            path = Path(inspect.getfile(cls)).resolve()
            files[cls.__name__] = dict(
                path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        self.provenance = dict(
            mode="evidence_only",
            agent_loop="unchanged PyRUA runtime",
            native_components=files,
            options=dict(
                validate_geometry=validate_geometry,
                max_failure_images=max_failure_images,
                execution_recovery=execution_recovery,
                stage_verification=stage_verification,
                temporal_feedback=temporal_feedback,
            ),
        )
        (self.root / "provenance.json").write_text(json.dumps(self.provenance, indent=2) + "\n")

    def event(self, kind, **data):
        counts = getattr(self, "event_counts", None)
        if counts is not None:
            counts[kind] = counts.get(kind, 0) + 1
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps({"type": kind, **data}, default=str) + "\n")

    def archive(self, reference, value):
        """Save original measurements/images for post-run audits, never prompt dumps."""

        def plain(item):
            if isinstance(item, np.ndarray):
                return item.tolist()
            if isinstance(item, np.generic):
                return item.item()
            raise TypeError(f"Unsupported evidence value: {type(item).__name__}")

        (self.root / f"{reference}.json").write_text(json.dumps(value, default=plain) + "\n")
        image = None
        if value["source"].get("image_ref"):
            image = self.bridge.image_bytes(reference)
            (self.root / f"{reference}.png").write_bytes(image)
        if self.kit.native_vista:
            self.binding_history[reference] = self.kit.native_vista.record_binding(
                reference, self.bridge.summary(reference), value["source"], image)

    def specs(self):
        return [
            dict(
                name=c.name,
                description=c.schema["function"]["description"],
                input_schema=c.schema["function"]["parameters"],
            )
            for c in self.tools.values()
        ]

    def invoke(self, name, args):
        self.serial += 1
        result = self.tools[name].invoke(
            VirtualCall(f"evidence_{self.serial}", name, args), self.memory
        )
        self.event("native_capability", name=name, ok=result.ok, processor=result.processor_info)
        # Original BaseCapability validation, binding and rendering already ran.
        text = [part["text"] for part in result.evidence if part.get("type") == "text"]
        payload = {"native_evidence": text, "data": result.data} if result.ok else {"error": result.error}
        if not result.ok and name == EvidenceCapability.name and (
            args.get("operation") in {"check", "inspect"} or "reference" in args
        ):
            payload["reference_hint"] = REFERENCE_ID_HELP
        return ArmResult(payload)

    def before_tool(self, *, name=None, args=None):
        self.ref_start = len(self.bridge.refs)
        self.failure_start = len(self.failures)
        self.action_start = len(self.action_links)

    def target_scope(self, reference, *, intent="move"):
        """Declare a target by ID string (row['reference']); validate at scope entry."""
        from contextlib import contextmanager

        @contextmanager
        def scoped():
            if not isinstance(intent, str) or intent not in {"move", "grasp", "contact", "release"}:
                raise ValueError("intent must be move, grasp, contact or release")
            self.bridge.require_reference(reference)  # No implicit latest target for None.
            self.bridge.read(reference)  # Historical geometry must be explicit elsewhere.
            previous = self.active_target
            self.active_target = {"reference": reference, "intent": intent}
            try:
                yield
            finally:
                self.active_target = previous
        return scoped()

    def before_action(self, *, tool=None, arguments=None):
        if self.temporal_feedback:
            target = dict(self.active_target) if self.active_target else None
            self.action_links.append({
                "tool": tool, "arguments": dict(arguments or {}),
                "target": target,
                "association": "model_declared" if target else "unassigned",
                "step_before": self.kit.robo._backend.env_steps(),
                "validity_before": self.bridge.validity(target["reference"]) if target else None,
            })
        self.bridge.revision += 1

    def observe(self, tool, args, result, index):
        data = _dict(result)
        if not isinstance(data, dict):
            return
        reason = None
        if data.get("ik_error"):
            reason = "ik_failure"
        elif data.get("reached") is False or data.get("blocked"):
            reason = "motion_or_contact_observation"
        elif data.get("found") is False:
            reason = "object_not_found"
        elif data.get("holding") is False and args.get("gripper") == self.kit.robo.CLOSE:
            reason = "empty_close_observation"
        if reason:
            event = dict(reason=reason, tool=tool, args=args, feedback=data, call_index=index)
            if self.temporal_feedback:
                event.update(target=dict(self.active_target) if self.active_target else None,
                             observed_step=self.kit.robo._backend.env_steps())
                # Capture the first anomaly before later primitives can change the scene.
                if len(self.failures) == self.failure_start and self.kit.images != "none" and self.max_failure_images:
                    target = event["target"]
                    camera = (self.bridge.summary(target["reference"])["camera"] if target else
                              args.get("camera")) or self.kit.robo.CAMERAS[0]
                    try:
                        data_png = _to_png(self.kit.robo.image(camera))
                        name = f"failure_at_{len(self.failures)}_{camera}.png"
                        (self.root / name).write_bytes(data_png)
                        event["snapshot"] = {"image": name, "camera": camera,
                                             "step": event["observed_step"],
                                             "sha256": hashlib.sha256(data_png).hexdigest()}
                    except Exception as exc:
                        event["snapshot_error"] = f"{type(exc).__name__}: {exc}"
            self.failures.append(event)
            self.event("primitive_observation", **event)

    def inspect(self, reference):
        value = self.bridge.raw(reference)
        data = self.bridge.image_bytes(reference)
        with Image.open(io.BytesIO(data)) as im:
            picture = im.convert("RGB")
        box = value.get("box_px", value.get("box"))
        if box is not None and len(box) == 4:
            draw = ImageDraw.Draw(picture)
            draw.rectangle(tuple(box), outline="yellow", width=2)
            draw.text(
                (4, 4),
                f"Source evidence: {reference}, step {value['source'].get('step')}",
                fill="yellow",
            )
        self.pending_images.append((f"evidence_{reference}.png", _to_png(picture)))
        self.event("evidence_image_read", binding=reference, native="openvlharness")
        return self.bridge.summary(reference)

    def failure_evidence(self, query=None, reference=None):
        if reference is not None:
            self.bridge.require_reference(reference)
        if not self.failures:
            return {"observation": "No failed primitive recorded"}
        failure = (self.failures[self.failure_start] if self.temporal_feedback and
                   len(self.failures) > self.failure_start else self.failures[-1])
        query = query or (
            _query(failure["args"]) if failure["reason"] == "object_not_found" else None
        )
        candidates = self.bridge.find(query=query, limit=3)
        declared = (failure.get("target") or {}).get("reference")
        preferred = reference or declared or (self.bridge.last_read if query is None and not self.temporal_feedback else None)
        if self.temporal_feedback and not preferred and not query:
            candidates = []  # Do not present the latest unrelated object as the failed target.
        if preferred:
            candidates = [self.bridge.summary(preferred)]
        reason = failure["reason"]
        guidance = {
            "object_not_found": (
                "The query returned no object. Check the camera, visibility and query; "
                "do not reuse a prior positive localization as a current detection."
            ),
            "ik_failure": (
                "Use the full IK feedback to revise arm/orientation/path. Validate "
                "current geometry before reusing the target."
            ),
            "motion_or_contact_observation": (
                "Distinguish intended contact from obstruction. Compare source and "
                "current scene before replanning."
            ),
            "empty_close_observation": (
                "Empty CLOSE can be intentional pressing. For acquisition, check "
                "whether the object moved; revalidate or re-localize before retrying."
            ),
        }
        snapshot = failure.get("snapshot")
        if snapshot and self.kit.images != "none" and self.max_failure_images:
            self.pending_images.append((snapshot["image"], (self.root / snapshot["image"]).read_bytes()))
            if self.max_failure_images > 1:
                camera = snapshot["camera"]
                try:
                    self.pending_images.append((f"current_{camera}.png", _to_png(self.kit.robo.image(camera))))
                except Exception as exc:
                    self.event("evidence_render_error", error=str(exc))
        elif candidates and self.kit.images != "none" and self.max_failure_images:
            ref = candidates[0]["reference"]
            try:
                self.inspect(ref)
                camera = candidates[0]["camera"]
                if camera and self.max_failure_images > 1:
                    self.pending_images.append(
                        (f"current_{camera}.png", _to_png(self.kit.robo.image(camera)))
                    )
            except Exception as exc:
                self.event("evidence_render_error", error=str(exc))
        if not snapshot and not candidates and self.kit.images != "none" and self.max_failure_images:
            camera = failure["args"].get("camera") or self.kit.robo.CAMERAS[0]
            try:
                self.pending_images.append(
                    (f"current_{camera}.png", _to_png(self.kit.robo.image(camera)))
                )
            except Exception as exc:
                self.event("evidence_render_error", error=str(exc))
        context = dict(
            failure=failure,
            candidates=candidates,
            guidance=guidance[reason],
            association=("Explicit reference or model-declared target; identity is not verified."
                         if preferred and self.temporal_feedback else
                         "Unassigned target; query matches only." if self.temporal_feedback else
                         "Recent query/read history; candidate identity is not verified."),
        )
        if self.temporal_feedback:
            context.update(current_step=self.kit.robo._backend.env_steps(),
                           snapshot_role="first_anomaly_in_code_call" if snapshot else "unavailable",
                           task_outcome="unknown")
            # Context candidates are not asserted to be the action's target.
            refs = ([preferred] if preferred else
                    [self.bridge.last_read] if self.bridge.last_read else
                    list(reversed(self.bridge.refs[-3:])))
            refs = [ref for ref in refs if ref in self.bridge.entries]
            context["recent_evidence_context"] = [
                {"reference": ref, "query": self.bridge.entries[ref]["query"],
                 "validity": self.bridge.validity(ref),
                 "history": self.binding_history.get(ref),
                 "association": "declared" if ref == declared else "unverified_context"}
                for ref in refs]
            context["target_review"] = {
                "commanded_target": failure["args"].get("xyz", failure["args"].get("target_xyz")),
                "declared_reference": declared,
                "current_target_geometry": (
                    "available" if declared and self.bridge.validity(declared)["usable"] else "unknown"),
                "copied_python_coordinates": "not tracked; their source may now be stale",
                "next_decision": "Compare current evidence with the linked history. If target geometry is no longer supported, obtain a fresh observation before deciding whether to reuse or revise the code. Intended contact alone is not failure.",
            }
            # When anomaly/current pixels are identical, spend the spare image
            # slot on historical context, clearly marked as unverified context.
            if self.kit.images != "none" and self.max_failure_images and refs:
                unique = {hashlib.sha256(data).hexdigest() for _, data in self.pending_images}
                if len(unique) < self.max_failure_images:
                    try:
                        ref = refs[0]
                        self.pending_images.append((f"history_context_{ref}.png", self.bridge.image_bytes(ref)))
                        context["historical_image_role"] = "unverified recent context; not a current target"
                    except Exception as exc:
                        self.event("evidence_render_error", error=str(exc))
        self.event(
            "failure_evidence", reason=reason, references=[c["reference"] for c in candidates]
        )
        return context

    def after_tool(self, result):
        from .evidence_temporal import TEMPORAL_GUIDANCE

        update = {}
        # Refresh labels even on turns with no new perception. Old transcript
        # messages remain immutable; this snapshot supersedes their freshness.
        known = [r for r in self.bridge.refs if r in self.bridge.entries]
        result.result['evidence_temporal_context'] = {
            'assessment_step': self.kit.robo._backend.env_steps(),
            'assessment_revision': self.bridge.revision,
            'guidance': TEMPORAL_GUIDANCE,
            'perception': [dict(
                reference=r, source={k: self.bridge.values[r]['source'].get(k) for k in
                                     ('step', 'call_index', 'timestamp_utc', 'camera', 'image_ref')},
                validity=self.bridge.validity(r),
            ) for r in known[-16:]],
            'omitted_older_bindings': max(0, len(known) - 16),
            'omitted_policy': 'Not certified current; check reference or re-localize.',
        }
        if self.temporal_feedback and len(self.action_links) > self.action_start:
            links = self.action_links[self.action_start:]
            update["action_targets"] = links[-8:]
            update["omitted_action_targets"] = max(0, len(links) - 8)
        refs = [r for r in self.bridge.refs[self.ref_start :] if r in self.bridge.entries]
        if refs:
            update["new_evidence"] = [self.bridge.summary(r) for r in refs[-3:]]
            update["additional_bindings"] = max(0, len(refs) - 3)
        if getattr(self, "automatic_feedback", True) and len(self.failures) > self.failure_start:
            update["recovery_evidence"] = self.failure_evidence()
            update["primitive_observations"] = len(self.failures) - self.failure_start
        if update:
            result.result = {"evidence_update": update, **result.result}
        if self.kit.images != "none" and self.pending_images:
            names = result.result.setdefault("images", [])
            for name, data in self.pending_images[-4:]:
                result.images.append(data)
                names.append(name)
        self.pending_images.clear()
        return result

    def summary(self):
        return {
            **self.provenance,
            "bindings": len(self.bridge.refs),
            "events": dict(self.event_counts),
        }


def adapt_prompt(prompt, toolkit):
    """Keep the existing robot contract while accurately listing additional tools."""
    prompt = prompt.replace("through ONE tool,", "through the primary execution tool,")
    if getattr(toolkit.evidence_adapter, "prompt_mode", None) == "visual":
        return prompt + "\n" + toolkit.evidence_adapter.prompt()
    evidence_prompt = PROMPT
    if not getattr(toolkit.evidence_adapter, 'expose_robot_tools', True):
        evidence_prompt = evidence_prompt.replace(
            'Robot_Perception_Tool / Robot_Evidence_Tool expose these services; Python helpers\n'
            'can combine checks and actions without extra model rounds. Do not invent IDs.',
            'Use the existing robo.locate/segment and Python evidence helpers; they can\n'
            'combine checks and actions without extra model rounds. Do not invent IDs.')
    prompt += "\n" + evidence_prompt
    if hasattr(toolkit.evidence_adapter, "prompt"):
        prompt += toolkit.evidence_adapter.prompt()
    if toolkit.arm == "program":
        prompt += (
            "\nInside run(robo), use robo.evidence/evidence_find/evidence_check/"
            "evidence_history/evidence_current/evidence_inspect/failure_evidence "
            "with the same arguments.\n"
        )
    if (
        toolkit.evidence_adapter.provenance["options"]["execution_recovery"]
        or toolkit.stage_verifier
    ):
        from .action_feedback import RECOVERY_INSTRUCTIONS

        prompt += "\n" + RECOVERY_INSTRUCTIONS
    if getattr(toolkit.robo._action_feedback, "persistent", False):
        from .openvl_recovery import PROMPT as CROSS_TURN_PROMPT
        prompt += "\n" + CROSS_TURN_PROMPT
    if toolkit.stage_verifier:
        from .stage_verification import PROMPT as STAGE_PROMPT

        prompt += "\n" + STAGE_PROMPT
    return prompt
