"""PyRUA public perception -> original OpenVL memory -> persistent Python.

Uses the upstream BaseCapability lifecycle and binding store. No hidden scene
state, alternate robot executor, or replacement perception model is introduced.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone

from openvlharness.capabilities._base import BaseCapability

from .arms import _to_png
from .openai_runtime import image_part

INSTRUCTIONS = """
Robot_Perception_Tool runs the existing PyRUA segment/locate service on a CURRENT
camera, returning an original OpenVL structured binding. Calls to robo.segment,
robo.locate and robo.back_project inside python are archived the same way.
Use evidence('binding_id') in persistent Python to read a copy of its measured
fields, source image ID, camera calibration and simulator step. evidence() reads
the latest robot binding; evidence_refs lists the available IDs. Reuse a binding
without re-segmenting when no action has intervened. After movement, reacquire
geometry: stale reads raise unless allow_historical=True is explicitly supplied
for comparison. Historical evidence is not a current action target.
Coordinates are full camera pixels (centroid_rc is row,column; box_px is xyxy),
and center_xyz is world metres. Never feed Zoom's 0-1000 or cropped coordinates
directly into back_project; map back to the original image and camera first.
Use checked(move_result) and checked(grip_result, require='holding') inside a
multi-step cell when subsequent actions require success. Print full diagnostics.
Contact operations need outcome verification; reached alone is not task success.
"""

# Only values returned by the public perception service are eligible for storage.
FIELDS = {
    "found",
    "center_xyz",
    "top_z",
    "height_m",
    "length_m",
    "width_m",
    "long_axis_deg",
    "grasp_yaw",
    "score",
    "centroid_rc",
    "mask_moments",
    "box_px",
    "camera",
    "views",
    "geometry_error",
    "height_guess",
    "note",
    "world_xyz",
    "pixel",
    "depth_m",
    "mask_rle",
    "mask_shape",
    "robot_px_fraction",
    "pred_masks",
    "box",
    "world_error",
    "area_px",
}


class PerceptionMemory:
    def __init__(self, toolkit, event):
        self.toolkit, self.event = toolkit, event
        self.memory = None
        self.refs = []
        self.errors = []
        toolkit.robo._perception_sink = self.capture
        toolkit._globals.update(evidence=self.read, evidence_refs=())

    def bind(self, memory):
        self.memory = memory

    def capture(self, tool, arguments, payload):
        if self.memory is None:
            return
        try:
            self._capture(tool, arguments, payload)
        except Exception as exc:
            # A memory failure must not discard a valid public perception return.
            message = f"{type(exc).__name__}: {exc}"
            self.errors.append(message)
            self.event("robot_evidence_error", tool=tool, error=message)

    def _capture(self, tool, arguments, payload):
        robo = self.toolkit.robo
        step = robo._backend.env_steps()
        requested_step = arguments.get("step", -1)
        camera = payload.get("camera") or arguments.get("camera")
        record = {key: copy.deepcopy(payload[key]) for key in FIELDS if key in payload}
        source = {
            "tool": tool,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "arguments": dict(arguments),
            "step": step if requested_step == -1 else None,
            "requested_step": requested_step,
            "call_index": len(robo._ledger) - 1,
            "camera": camera,
            "coordinates": ("full camera pixels; centroid_rc=row,column; "
                            "box_px=xyxy; xyz=world metres"),
        }
        if camera:
            # Same locked tool turn, with no intervening robot action.
            png = _to_png(robo.image(camera, step=requested_step))
            source["image_ref"] = self.memory.register_image(
                image_part(png)["image_url"]["url"], provenance=f"{camera}:step={step}:{tool}"
            )
            source["calibration"] = copy.deepcopy(robo.camera_meta(camera, step=requested_step))
        record["source"] = source
        key = self.memory.store_binding(
            prefix="robot",
            query=arguments.get("text_prompt") or arguments.get("prompt") or tool,
            value=record,
            source_tool=f"pyrualean.{tool}",
            source_args=dict(arguments),
            descriptions={
                "source": "Original image/camera/step and calibration. Historical after motion.",
                "center_xyz": "Measured world metres; honor views/height_guess uncertainty.",
                "views": "Number of camera views supporting this measurement.",
            },
        )
        self.refs.append(key)
        self.toolkit._globals["evidence_refs"] = tuple(self.refs)
        self.event("robot_evidence_stored", binding=key, source=source, found=record.get("found"))

    def read(self, reference=None, *, allow_historical=False):
        if self.memory is None:
            raise RuntimeError("OpenVL memory is not active")
        if reference is None:
            if not self.refs:
                raise ValueError("No perception binding yet; call locate or segment first")
            reference = self.refs[-1]
        values = self.memory.copy_bindings_for_execution()
        if reference not in values:
            raise ValueError(f"Unknown binding {reference!r}; available: {list(values)}")
        value = values[reference]
        source = value.get("source", {})
        step = self.toolkit.robo._backend.env_steps()
        fresh = source.get("step") is not None and source["step"] == step
        self.event(
            "robot_evidence_read",
            binding=reference,
            current_step=step,
            source_step=source.get("step"),
            historical=not fresh,
            allowed=fresh or allow_historical,
        )
        if not fresh and not allow_historical:
            raise ValueError(
                "Historical or untracked geometry: reacquire current perception; "
                "use allow_historical=True only for comparison"
            )
        return value


class RobotPerceptionCapability(BaseCapability):
    name = "Robot_Perception_Tool"
    backends = {"pyrualean": "pyrualean"}
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": "Locate or segment an object in a CURRENT robot camera; "
            "store measurements "
            "in OpenVL memory for evidence(binding_id) in persistent Python. No motion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["locate", "segment"]},
                    "query": {"type": "string", "minLength": 1},
                    "camera": {"type": "string"},
                },
                "required": ["operation", "query", "camera"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, bridge):
        self.bridge = bridge
        super().__init__(backend="pyrualean", client_config={}, client=self)

    def execute(self, call, args, memory):
        self.bridge.bind(memory)
        before = len(self.bridge.refs)
        method = getattr(self.bridge.toolkit.robo, args["operation"], None)
        if not callable(method):
            raise ValueError(f"This robot does not expose {args['operation']}")
        method(args["query"], camera=args["camera"])
        if len(self.bridge.refs) == before:
            raise ValueError(
                "Perception returned but no binding was stored; "
                + "; ".join(self.bridge.errors[-1:])
            )
        reference = self.bridge.refs[-1]
        return {
            "ok": True,
            "result": {"binding": reference, "measurement": self.bridge.read(reference)},
            "meta": {"backend": "PyRUA"},
        }

    def store_result(self, call, result, memory):
        # _readonly already stored the exact same measurement with native store_binding.
        return [result["binding"]]

    def normalize_result(self, call, result, prepared, binding_refs):
        return result

    @staticmethod
    def process_output(payload, prepared):
        return None  # Native BaseCapability renders evidence and binding artifact refs.
