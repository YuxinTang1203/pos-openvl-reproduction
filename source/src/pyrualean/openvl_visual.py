"""OpenVL's visual tools and addressable images inside the original PyRUA loop.

This adapter adds multi-instance visual grounding over RGB, native Zoom and
direct image recall. Native RGB camera-metadata requests can add read-only
coordinate context without changing their return. Passive history copies existing
action feedback without extra sensing or control; robot execution remains PyRUA's.
"""
import base64
import copy
import functools
import inspect
import io
import json
from pathlib import Path

from PIL import Image
from openvlharness.bootstrap import build_capabilities, load_agent_config, memory_factory
from openvlharness.contracts import VirtualCall

from .arms import ArmResult, _to_png
from .openai_runtime import image_part
from .openvl_feedback import zoom_arguments
from .openvl_grounding import LocalGroundingClient
from .openvl_native import LocalZoomClient
from .openvl_visual_history import PassiveVisualHistory


class ProjectionResult(dict):
    """Public projection fields with mapping and attribute access.

    Values and uncertainty are unchanged; this bridges ObjectSight-style reads.
    Missing attributes raise AttributeError rather than inventing a value.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class OpenVLVisualAdapter:
    prompt_mode = "visual"

    def __init__(self, kit, *, config, initial_images, segmenter=None):
        self.kit = kit
        self.root = kit.workspace.parent / "openvl_visual"
        self.root.mkdir(parents=True, exist_ok=True)
        settings = load_agent_config(config)
        self.memory = memory_factory(settings["memory"])()
        self.memory.start([])
        self.tools = build_capabilities(settings)
        if set(self.tools) != {"Visual_Grounding_Tool", "Zoom_In_Tool"}:
            raise ValueError("openvl_visual requires exactly native grounding and zoom")
        self.tools["Visual_Grounding_Tool"].client = LocalGroundingClient(segmenter=segmenter)
        self.tools["Zoom_In_Tool"].client = LocalZoomClient()
        self.tools["Recall_Images"] = None
        self.native_names = set(self.tools)
        self.pending_images = []
        self.frames = {}
        self.groundings = {}
        self.calls = 0
        self.counts = {}
        self.remaining = None
        self.stopped = None
        self._tool = None
        self._geometry_context_active = False
        self._geometry_contexts = {}
        self.provenance = {
            "mode": "native_visual_workspace", "agent_loop": "pyrualean.openai_runtime",
            "config": str(Path(config).resolve()),
            "native_components": ["MemoryStore", "Grounding capability/schema/renderer", "Zoom capability/backend"],
            "local_backend": "Existing shared PyRUA SAM3; all valid candidates, not just top one",
            "options": {"execution_recovery": False, "stage_verification": False},
            "robot_action_hook": "passive_history_only", "automatic_action_capture": False,
            "passive_history": "Existing public results/states only; on-demand reads, no sensing, action policy or automatic directory",
        }
        self.history = PassiveVisualHistory(self)
        self.history.attach()
        for name, data in initial_images.items():
            self.register_frame(data, camera=self.camera_for_name(name), phase="initial")
        for name, method in (("visual_find", self.history.find), ("visual_context", self.history.context)):
            kit._globals[name] = method
            setattr(kit.robo, name, method)
        kit._globals["visual_data"] = self.read_data
        kit.robo.visual_data = self.read_data
        kit._globals["visual_candidate"] = self.visual_candidate
        kit.robo.visual_candidate = self.visual_candidate
        kit._globals["locate_candidate"] = self.locate_candidate
        kit.robo.locate_candidate = self.locate_candidate
        kit._globals["visual_project"] = self.visual_project
        kit.robo.visual_project = self.visual_project
        original_meta = getattr(kit.robo, "camera_meta", None)
        self._original_camera_meta = getattr(original_meta, "__visual_geometry_original__", original_meta)
        if callable(original_meta) and self._supports_locate_candidate():
            kit.robo.camera_meta = self._camera_meta_wrapper(self._original_camera_meta)
        self.provenance["camera_metadata_context"] = {
            "enabled": callable(original_meta) and self._supports_locate_candidate(),
            "trigger": "Python camera_meta request; bypassed when routing is disabled",
            "original_return_unchanged": True, "extra_image_payload": False,
        }
        self.persist()

    def event(self, kind, **values):
        self.counts[kind] = self.counts.get(kind, 0) + 1
        try:
            with (self.root / "events.jsonl").open("a") as f:
                f.write(json.dumps({"type": kind, **values}, default=str) + "\n")
        except Exception as exc:
            self.history.trace_error(exc)
        self.history.safely(self.history.event, kind, values)

    def camera_for_name(self, name):
        for (camera, _), artifact in self.kit.robo.IMAGE_ARTIFACTS.items():
            if name == artifact or Path(str(name)).name == Path(artifact).name:
                return camera
        for camera in self.kit.robo.CAMERAS:
            if camera in str(name):
                return camera
        return None

    def action_revision(self):
        return sum(bool(r.stateful) for r in self.kit.robo._ledger.records)

    def register_frame(self, data, *, camera=None, phase, parent=None):
        """A new capture alias keeps time immutable even when pixels deduplicate."""
        data = _to_png(data)
        ref = f"frame_{len(self.frames) + 1:04d}"
        native = self.memory.register_image(image_part(data)["image_url"]["url"], aliases=(ref,), provenance=phase)
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
        source = self.frames[parent] if parent else None
        unknown_capture = not source and phase == "python_image_output" and camera is None
        self.frames[ref] = {
            "image_id": ref, "native_image_id": native, "camera": camera if not source else source["camera"],
            "phase": phase, "size_hw": [height, width], "parent_image_id": parent,
            "source_step": source["source_step"] if source else None if unknown_capture else self.kit.robo._backend.env_steps(),
            "action_revision": source["action_revision"] if source else None if unknown_capture else self.action_revision(),
            "delivery_step": self.kit.robo._backend.env_steps(),
            "is_original_camera_frame": camera is not None and parent is None,
            "file": f"{ref}.png",
        }
        try:
            (self.root / f"{ref}.png").write_bytes(data)
        except Exception as exc:
            self.history.trace_error(exc)
        self.history.safely(self.history.frame, ref, self.frames[ref])
        return ref

    def _frame_bytes(self, ref):
        """Read registered original bytes even if optional disk archival failed."""
        native = self.frames[ref]["native_image_id"]
        url, _, _ = self.memory.resolve_image(native)
        if not url or not url.startswith("data:"):
            raise ValueError("Registered image pixels are unavailable")
        return base64.b64decode(url.split(",", 1)[1])

    def frame_context(self, ref):
        if ref not in self.frames:
            raise ValueError(f"Unknown image_id {ref!r}; use a returned frame ID")
        frame = copy.deepcopy(self.frames[ref])
        frame.pop("file", None)
        frame["actions_since_capture"] = (self.action_revision() - frame["action_revision"]
                                          if frame["action_revision"] is not None else None)
        frame["current_camera_pixels"] = bool(
            frame["is_original_camera_frame"] and frame["actions_since_capture"] == 0
            and frame["source_step"] == self.kit.robo._backend.env_steps())
        return frame

    def specs(self):
        out = []
        for name in ("Visual_Grounding_Tool", "Zoom_In_Tool"):
            spec = copy.deepcopy(self.tools[name].schema["function"])
            params = spec["parameters"]
            props = params["properties"]
            props["image_id"]["description"] = "An exact returned frame ID. Historical images stay historical."
            if name == "Visual_Grounding_Tool":
                spec["description"] = (
                    "Find ALL matching candidates in an RGB image using SAM3; returns boxes, scores and "
                    "a numbered mask image. Use to distinguish/count similar objects, check which object "
                    "a query selected, or inspect visibility before choosing an action. Candidates are "
                    "hypotheses, not identities or proof of contact. With no image_id, captures the current camera.")
                props["camera"] = {"type": "string", "enum": list(self.kit.robo.CAMERAS),
                    "description": "Current camera, default first overview camera; exclusive with image_id."}
            else:
                spec["description"] = (
                    "Inspect an image region. Prefer binding_id + candidate_number from Grounding to avoid "
                    "coordinate conversion. Alternatively use image_id + coords in the declared coord_space. "
                    "A crop retains its source time and does not measure world geometry.")
                props.update({
                    "binding_id": {"type": "string", "description": "Grounding binding ID; use with candidate_number."},
                    "candidate_number": {"type": "integer", "minimum": 1, "description": "The one-based number printed on the Grounding mask image."},
                    "coord_space": {"type": "string", "enum": ["normalized_1000", "pixels"], "default": "normalized_1000"},
                })
                props["coords"].update(minItems=4, maxItems=4,
                    description="[x1,y1,x2,y2] in coord_space (default normalized_1000); no nested list.")
                props["padding"].update(default=50, minimum=0,
                    description="Normalized 0–1000 margin; default 50, smaller values explicitly raised to 50.")
                params.pop("required", None)
                params["oneOf"] = [{"required": ["binding_id", "candidate_number"]}, {"required": ["image_id", "coords"]}]
            params["additionalProperties"] = False
            out.append(dict(name=name, description=spec["description"], input_schema=params))
        out.append({"name": "Recall_Images", "description":
            "Actually view 1–4 saved images together, e.g. initial and current frames to compare changes. "
            "Returns image pixels and source times; does not move, capture new frames or judge task completion.",
            "input_schema": {"type": "object", "properties": {"image_ids": {"type": "array",
                "items": {"type": "string"}, "minItems": 1, "maxItems": 4}},
                "required": ["image_ids"], "additionalProperties": False}})
        return out

    def prompt(self):
        return """
OpenVL visual workspace:
When a decision depends on object identity, count, visibility, contact/alignment or
what changed, state the concrete visual question and use the available visual tools
when the existing view is insufficient. Visual_Grounding_Tool returns all matching
candidates and a numbered mask image. Inspect the candidates; try a different query
or view if detections are missing/conflicting. Zoom can take binding_id and
candidate_number directly, without manual coordinate conversion. Recall_Images
returns actual saved image pixels together for before/after comparison.
The task instruction still determines which objects, order and outcome are required.
Detections are hypotheses; pixels and motion arrival do not certify a grasp, contact
or task success. After selecting a candidate, preserve that choice for geometry:
p = visual_candidate(binding_id, candidate_number)
point = robo.back_project(p["row"], p["col"], camera=p["camera"])
visual_candidate (also robo.visual_candidate) returns an interior pixel of that
candidate's mask only when the original camera frame is still current. It rejects
old/cropped frames; obtain fresh visual evidence after motion. Do not re-run
locate(text) to project an already selected candidate: it may choose a different
top-scoring object. Check back_project's surface and uncertainty notes: with RGB,
the object point is estimated; a table result may also mean an unmeasurable object.
This is one visible surface point, not an object centre, grasp pose or proven contact.
On the native RGB-only RoboDojo backend, use
o = robo.locate_candidate(binding_id, candidate_number)
when you need the selected object's full ObjectSight geometry: center_xyz, top_z,
height_m, length_m, width_m, long_axis_deg, grasp_yaw, views and note. This retains
the exact selected source mask; other cameras use point-seed matching, not another
text top-one selection or prompt-keyed height memory. Check found, center_xyz,
views and note before acting. It retains the original RGB geometry's limitations
(including table-supported-object assumptions and uncertain one-view heights),
so it does not prove a grasp or task success. Unsupported backends raise explicitly.
For a pixel believed to lie on an explicitly chosen world z plane, use
p = robo.visual_project(image_id, row, col, plane_z)
(also visual_project). This uses public camera calibration with the correct
camera-to-world, -z optical/+y-up convention. It accepts current original camera
frames or same-size Grounding annotations of them; crops and stale frames fail.
The result is a mapping; attribute reads of its fields are also supported:
if p["status"] == "ok":
    xyz = p["xyz_m"]  # equivalently p.xyz_m; do not use an unknown projection
else:
    print(p["reason"])
An ok status means the conditional projection was computed, not that plane_z
was measured. The point is conditional on your plane_z,
not a measured surface or object pose. Pixel-noise uncertainty is reported;
plane-height uncertainty is unknown unless you supply plane_sigma_m. No grasp,
contact or success follows from this projection. Do not substitute a crop's pixels.
2D evidence is not a world target. Crops use local pixels and must not be passed
directly to camera back_project. Robot motion remains
in PyRUA Python. Multi-action cells and persistent variables remain available.
visual_data(binding_id), also on robo, reads copied native pred_boxes/pred_scores /
pred_masks (COCO RLE counts, size orig_img_h/orig_img_w), unlike the tool response's
canonical boxes/scores/masks. Historical data never becomes fresh by reading it.
visual_find(kind=None, tool=None, cell=None, call_index=None, limit=10) lists
historical record summaries; visual_context(reference, limit=10) reads one record
and bounded related references and same_cell_observations (temporal context only,
not necessarily observations of that action). Both are also on robo. Use the returned
reference string, frame ID or Grounding binding ID. These copy already available
public results, cached/returned states and exceptions; they capture nothing new.
Missing observations stay unknown. Links are provenance or temporal context,
not proof that an action used a target, that an object stayed held, or of success.
Nested current/status fields in history describe the recorded instant, not now.
Initial images: """ + json.dumps([self.frame_context(r) for r in self.frames], ensure_ascii=False)

    def before_tool(self, *, name=None, args=None):
        self._tool = name
        self._geometry_context_active = name == "python"
        self._geometry_contexts = {}
        self.history.safely(self.history.begin, name, args)

    def _camera_meta_wrapper(self, original):
        """Keep the public return untouched; add request-time context to Python feedback."""
        @functools.wraps(original)
        def with_context(*args, **kwargs):
            active = self._geometry_context_active and getattr(self, "routing_enabled", True)
            before, capture_error = None, None
            if active:
                try:
                    before = self.kit.robo._backend.env_steps(), self.action_revision()
                except Exception as exc:
                    capture_error = exc
            result = self._original_camera_meta(*args, **kwargs)
            if not active:
                return result
            camera = self.kit.robo.CAMERAS[0]
            try:
                if capture_error is not None:
                    raise capture_error
                bound = inspect.signature(original).bind(*args, **kwargs)
                bound.apply_defaults()
                camera, step = bound.arguments["camera"], bound.arguments.get("step", -1)
                context = {"camera": camera, "requested_step": step, "image_id": None,
                           "coordinate_contract": "camera-to-world E; optical -z, +y up; ray_camera=[(col-cx)/fx, -(row-cy)/fy, -1]; world_ray=E[:3,:3] @ ray_camera",
                           "projection_hint": "visual_project(image_id, row, col, plane_z) returns a point conditional on your supplied plane height, not a measured surface/contact."}
                if step != -1:
                    context.update(status="historical", current_camera_pixels=False,
                                   note="Historical calibration request; no current image was captured.")
                else:
                    previous = self._geometry_contexts.get(camera, {})
                    ref = previous.get("image_id")
                    pixels = self.kit.robo.image(camera)
                    if before != (self.kit.robo._backend.env_steps(), self.action_revision()):
                        raise ValueError("Scene changed during camera context capture")
                    if ref and self.frame_context(ref)["current_camera_pixels"]:
                        import numpy as np
                        with Image.open(self.root / self.frames[ref]["file"]) as saved:
                            same = np.array_equal(np.asarray(pixels)[..., :3], np.asarray(saved.convert("RGB")))
                        if same:
                            return result  # Reuse one memory capture for this unchanged scene.
                    ref = self.register_frame(pixels, camera=camera, phase="camera_meta_request")
                    context.update(status="current", image_id=ref, **{
                        "source": self.frame_context(ref), "current_camera_pixels": True})
                self._geometry_contexts[camera] = context
                self.event("visual_geometry_context_capture", **context)
            except Exception as exc:
                # Context is auxiliary; camera_meta's original successful result must survive.
                self._geometry_contexts[camera] = {
                    "camera": camera, "status": "unavailable", "image_id": None,
                    "current_camera_pixels": False,
                    "note": f"Camera context unavailable: {type(exc).__name__}: {exc}",
                }
            return result
        with_context.__visual_geometry_original__ = original
        return with_context

    def read_data(self, binding_id):
        if binding_id not in self.groundings:
            raise ValueError("Expected a returned Grounding binding ID")
        row = self.groundings[binding_id]
        self.event("visual_data_read", binding_id=binding_id)
        return {"source": self.frame_context(row["source_image_id"]),
                "data": copy.deepcopy(self.memory.state.bindings.bindings[binding_id])}

    def _current_candidate(self, binding_id, candidate_number):
        """Validate a selected full-frame mask against the current observation."""
        import numpy as np
        from pycocotools import mask as mask_utils

        if binding_id not in self.groundings:
            raise ValueError("Expected a returned Grounding binding ID")
        if isinstance(candidate_number, bool) or not isinstance(candidate_number, int):
            raise ValueError("candidate_number must be the one-based integer on the mask image")
        record = self.groundings[binding_id]
        source = self.frame_context(record["source_image_id"])
        if not source["current_camera_pixels"] or source["camera"] not in self.kit.robo.CAMERAS:
            raise ValueError("Candidate geometry requires a current original camera frame; capture fresh Grounding evidence")
        data = record["data"]
        masks, boxes = data.get("masks") or [], data.get("boxes") or []
        if not 1 <= candidate_number <= len(boxes) or len(masks) != len(boxes):
            raise ValueError("candidate_number must identify an available aligned Grounding mask")
        if data.get("size") != source["size_hw"]:
            raise ValueError("Grounding mask dimensions differ from the original camera frame")
        mask = mask_utils.decode({"size": data["size"], "counts": masks[candidate_number - 1]})
        if mask.shape != tuple(source["size_hw"]) or not mask.any():
            raise ValueError("Selected Grounding mask is empty or malformed")
        # Capture metadata can be incomplete for Python image outputs. Compare
        # pixels too, without advancing the simulator or calling the segmenter.
        current = np.asarray(self.kit.robo.image(source["camera"]))
        with Image.open(io.BytesIO(self._frame_bytes(record["source_image_id"]))) as image:
            original = np.asarray(image.convert("RGB"))
        if current.ndim != 3 or not np.array_equal(current[..., :3], original):
            raise ValueError("Saved candidate pixels differ from the current camera; capture fresh Grounding evidence")
        return source, data, mask, original

    def visual_candidate(self, binding_id, candidate_number):
        """Return a current selected-mask pixel without measuring robot geometry."""
        import cv2
        import numpy as np

        source, data, mask, _ = self._current_candidate(binding_id, candidate_number)
        binary = (mask > 0).astype(np.uint8)
        # Padding treats image edges as boundaries even for a full-frame mask.
        distance = cv2.distanceTransform(np.pad(binary, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        choices = np.argwhere(distance == distance.max())
        centre = np.argwhere(binary).mean(axis=0)
        row, col = (int(v) for v in choices[np.argmin(np.sum((choices - centre) ** 2, axis=1))])
        self.event("visual_candidate_read", binding_id=binding_id, candidate_number=candidate_number,
                   row=row, col=col, source_image_id=source["image_id"])
        return {"row": row, "col": col, "camera": source["camera"], "source": source,
                "candidate_number": candidate_number, "mask_area_px": int(binary.sum()),
                "interpretation": "Interior source-mask pixel for back_project before motion; not world geometry or a grasp pose. Check the returned surface and measurement uncertainty."}

    def _supports_locate_candidate(self):
        from .hosts.robodojo_native import NativeBackend
        from .robodojo_xpl_rgb import RobodojoOfficialColourRobot
        return (isinstance(self.kit.robo, RobodojoOfficialColourRobot)
                and isinstance(self.kit.robo._backend, NativeBackend)
                and self.kit.robo._backend.colour_only)

    def visual_project(self, image_id, row, col, plane_z, *, pixel_sigma_px=1.0,
                       plane_sigma_m=None):
        """Project a current original-image pixel onto a caller-supplied z plane."""
        import hashlib
        import numpy as np
        from .openvl_geometry import calibrated_view, project_to_plane

        if not self._supports_locate_candidate():
            raise NotImplementedError("visual_project requires the native RGB-only RoboDojo backend")
        requested, seen = self.frame_context(image_id), set()
        source = requested
        while source["phase"] == "grounding_annotation":
            if source["image_id"] in seen or not source["parent_image_id"]:
                raise ValueError("Incomplete or cyclic annotation source lineage")
            seen.add(source["image_id"])
            parent = self.frame_context(source["parent_image_id"])
            if parent["size_hw"] != source["size_hw"]:
                raise ValueError("Annotation dimensions differ from its original camera image")
            source = parent
        if not source["current_camera_pixels"] or source["camera"] not in self.kit.robo.CAMERAS:
            raise ValueError("visual_project requires a current original camera frame; crops/stale frames are not supported")
        robo = self.kit.robo
        current = np.asarray(robo.image(source["camera"]))
        with Image.open(io.BytesIO(self._frame_bytes(source["image_id"]))) as image:
            original = np.asarray(image.convert("RGB"))
        if current.ndim != 3 or not np.array_equal(current[..., :3], original):
            raise ValueError("Saved image pixels differ from the current camera; capture a fresh frame")
        metadata = self._original_camera_meta(source["camera"])
        view = calibrated_view(metadata)
        if list(view.shape) != source["size_hw"]:
            raise ValueError("Camera calibration shape differs from the original image")
        # Public calibration reads must not turn a historical capture into current evidence.
        if not self.frame_context(source["image_id"])["current_camera_pixels"]:
            raise ValueError("Scene changed while reading camera calibration")
        result = project_to_plane(view, row, col, plane_z, pixel_sigma_px=pixel_sigma_px,
                                  plane_sigma_m=plane_sigma_m)
        result.update(source=source, requested_image_id=image_id, calibration=copy.deepcopy(metadata),
                      calibration_convention="camera_to_world; optical -z; +y up; original row,col pixels",
                      source_rgb_sha256=hashlib.sha256(np.ascontiguousarray(original).tobytes()).hexdigest())
        self.event("visual_plane_projection", **copy.deepcopy(result))
        return ProjectionResult(result)

    def locate_candidate(self, binding_id, candidate_number):
        """Measure selected-instance ObjectSight on the native RGB-only backend.

        No source text re-detection, simulator object pose, extra depth or robot
        action is used. All uncertainty comes from the original RGB geometry.
        """
        import hashlib
        import numpy as np
        from .robodojo_xpl_rgb import object_sight_from_payload

        robo = self.kit.robo
        if not self._supports_locate_candidate():
            raise NotImplementedError("locate_candidate requires the native RGB-only RoboDojo backend; no geometry fallback is used")
        source, data, mask, pixels = self._current_candidate(binding_id, candidate_number)
        index = candidate_number - 1
        scores = data.get("scores") or []
        if len(scores) != len(data.get("masks") or []):
            raise ValueError("Selected candidate scores and masks are not aligned")
        payload = robo._readonly("locate_mask", {
            "camera": source["camera"], "mask_rle": {"size": data["size"], "counts": data["masks"][index]},
            "score": scores[index], "source_step": source["source_step"],
            "source_rgb_sha256": hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest(),
        })
        self.event("selected_mask_geometry", binding_id=binding_id, candidate_number=candidate_number,
                   source_image_id=source["image_id"], geometry_source=payload.get("geometry_source"),
                   source_step=source["source_step"], action_revision=source["action_revision"],
                   primitive_ledger_index=robo._ledger.records[-1].index,
                   geometry_payload=copy.deepcopy(payload),
                   found=payload.get("found"), views=payload.get("views"),
                   uncertainty=payload.get("geometry_error") or payload.get("height_guess"))
        return object_sight_from_payload(payload, source["camera"])

    def invoke(self, name, args):
        if (self.stopped and self.stopped()) or (self.remaining and self.remaining() <= 0):
            return ArmResult({"error": "Episode ended before visual tool execution"})
        from jsonschema import validate
        validate(args, next(s["input_schema"] for s in self.specs() if s["name"] == name))
        args = copy.deepcopy(args)
        if name == "Recall_Images":
            refs = args["image_ids"]
            contexts = [self.frame_context(r) for r in refs]
            self.event("image_recall", image_ids=refs)
            pictures = ([self._frame_bytes(r) for r in refs]
                        if self.kit.images != "none" else [])
            return ArmResult({"images": refs, "image_contexts": contexts,
                              "pixels_delivered": bool(pictures)}, pictures)
        if name == "Visual_Grounding_Tool":
            if "camera" in args and "image_id" in args:
                raise ValueError("Choose a current camera OR a stored image_id")
            camera = args.pop("camera", self.kit.robo.CAMERAS[0])
            if "image_id" not in args:
                args["image_id"] = self.register_frame(_to_png(self.kit.robo.image(camera)), camera=camera, phase="explicit_grounding_capture")
        else:
            if "binding_id" in args:
                if "image_id" in args or "coords" in args or "coord_space" in args:
                    raise ValueError("Use binding_id+candidate_number OR image_id+coords")
                binding, index = args.pop("binding_id"), args.pop("candidate_number")
                if binding not in self.groundings:
                    raise ValueError("Unknown Grounding binding_id")
                row = self.groundings[binding]
                boxes = row["data"].get("boxes_normalized") or []
                if not 1 <= index <= len(boxes):
                    raise ValueError(f"candidate_number must be from 1 to {len(boxes)}")
                args.update(image_id=row["source_image_id"], coords=boxes[index - 1])
            else:
                if "candidate_number" in args:
                    raise ValueError("candidate_number requires binding_id")
                space = args.pop("coord_space", "normalized_1000")
                if space == "pixels":
                    height, width = self.frame_context(args["image_id"])["size_hw"]
                    box = args["coords"]
                    if any(v < 0 or v > (width if i % 2 == 0 else height) for i, v in enumerate(box)):
                        raise ValueError("Pixel bbox is outside the source image")
                    args["coords"] = [v / (width if i % 2 == 0 else height) * 1000 for i, v in enumerate(box)]
            args, adjustment = zoom_arguments(args)
        source = args["image_id"]
        context = self.frame_context(source)
        native_args = {**args, "image_id": context["native_image_id"]}
        self.calls += 1
        result = self.tools[name].invoke(VirtualCall(f"visual_{self.calls}", name, native_args), self.memory)
        self.event("native_capability", name=name, ok=result.ok, source_image_id=source)
        payload = {"source": context, "data": copy.deepcopy(result.data),
                   "artifact_refs": [{"kind": r.kind, "id": r.id} for r in result.artifact_refs],
                   "evidence": [], "images": [], "image_contexts": []}
        if not result.ok:
            payload["error"] = result.error
        pictures = []
        aliases = {}
        for part in result.evidence:
            if part.get("type") == "text":
                payload["evidence"].append(part["text"])
            elif part.get("type") == "image_url":
                url = part["image_url"]
                url = url["url"] if isinstance(url, dict) else url
                data_url, _, _ = self.memory.resolve_image(url)
                if data_url:
                    data = base64.b64decode(data_url.split(",", 1)[1])
                    ref = self.register_frame(data, phase="grounding_annotation" if name == "Visual_Grounding_Tool" else "crop", parent=source)
                    aliases[self.frames[ref]["native_image_id"]] = ref
                    if self.kit.images != "none":
                        pictures.append(data)
                    payload["images"].append(ref)
                    payload["image_contexts"].append(self.frame_context(ref))
        # Native renderers name native IDs. Expose the immutable capture aliases
        # consistently in their text, canonical image field and artifact list.
        for native_ref, frame_ref in aliases.items():
            payload["evidence"] = [text.replace(native_ref, frame_ref) for text in payload["evidence"]]
            for field in ("image", "rendered_image"):
                if isinstance(payload["data"], dict) and payload["data"].get(field) == native_ref:
                    payload["data"][field] = frame_ref
            for artifact in payload["artifact_refs"]:
                if artifact["id"] == native_ref:
                    artifact["id"] = frame_ref
        if name == "Visual_Grounding_Tool" and result.ok:
            refs = [r.id for r in result.artifact_refs if r.id in self.memory.state.bindings.bindings]
            payload["binding_ids"] = refs
            data = result.data
            payload["candidates"] = [{"number": i + 1, "box_px": box,
                "score": (data.get("scores") or [])[i]}
                for i, box in enumerate(data.get("boxes") or [])]
            payload["interpretation"] = "Candidate regions may overlap or include false positives. Select by visual evidence; their count is not a verified object count."
            payload["evidence"] = [t.replace(f"grounded {len(payload['candidates'])} objects", f"returned {len(payload['candidates'])} candidate regions") for t in payload["evidence"]]
            for ref in refs:
                self.groundings[ref] = {"source_image_id": source, "data": copy.deepcopy(result.data)}
                self.history.safely(self.history.grounding, ref, source)
            # Mask arrays/RLE stay in memory; canonical boxes/scores remain visible.
            if isinstance(payload["data"], dict):
                payload["data"].pop("masks", None)
        elif name == "Zoom_In_Tool" and adjustment:
            payload["argument_adjustment"] = adjustment
        self.persist()
        return ArmResult(payload, pictures)

    def after_tool(self, result):
        try:
            return self._after_visual_tool(result)
        except Exception as exc:
            # Optional image/history metadata must not replace an already
            # returned Python value, robot result, or original exception.
            self.history.trace_error(exc)
            self._geometry_context_active = False
            self._geometry_contexts = {}
            return result
        finally:
            self.history.safely(self.history.finish, result)

    def _after_visual_tool(self, result):
        self._geometry_context_active = False
        if self._tool not in self.tools and result.images:
            labels = result.result.get("images", [])
            refs = []
            for i, data in enumerate(result.images):
                label = labels[i] if i < len(labels) else "display"
                ref = self.register_frame(data, camera=self.camera_for_name(label), phase="python_image_output")
                refs.append(self.frame_context(ref))
            result.result["visual_images"] = refs
            self.persist()
        if self._geometry_contexts:
            contexts = list(self._geometry_contexts.values())
            for context in contexts:
                if context.get("image_id"):
                    source = self.frame_context(context["image_id"])
                    context.update(source=source, current_camera_pixels=source["current_camera_pixels"],
                                   status="current" if source["current_camera_pixels"] else "historical")
            result.result["visual_geometry_context"] = copy.deepcopy(contexts)
            self.event("visual_geometry_context_delivery", contexts=contexts)
            self._geometry_contexts = {}
            self.persist()
        return result

    def persist(self):
        self.history.safely(self._persist)

    def _persist(self):
        for ref, value in self.memory.state.bindings.bindings.items():
            path = self.root / f"{ref}.json"
            if not path.exists():
                path.write_text(json.dumps(value, default=str) + "\n")
        (self.root / "manifest.json").write_text(json.dumps({
            "frames": self.frames, "groundings": self.groundings, "provenance": self.provenance,
            "history": {"references": list(self.history.rows), "aliases": self.history.aliases,
                        **self.history.summary()},
            "binding_files": {ref: f"{ref}.json" for ref in self.memory.state.bindings.bindings}}, default=str, indent=2) + "\n")

    def summary(self):
        return {**self.provenance, "native_calls": self.calls, "image_captures": len(self.frames),
                "grounding_records": len(self.groundings), "events": self.counts,
                **self.history.summary()}
