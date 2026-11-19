"""Opt-in observation phases using the existing RGB-only robot API.

No simulator object state, depth, reward, task predicate, or language model is
read here. The ledger is used only to invalidate observations after actions.
"""
import copy
import hashlib
import json
from dataclasses import asdict, is_dataclass

import numpy as np

from .openvl_geometry import calibrated_view
from .phase_geometry import triangulate_point


PROMPT = """
Observation phases (optional host helpers; also available on robo):
- print(observe_scene(cameras=['cam_head', 'cam_right_wrist'])) captures and
  displays original RGB images together with public robot state/calibration.
  It returns a snapshot_id and camera-to-displayed-image mapping. It moves nothing.
- print(move_and_observe([x,y,z], arm='right', pitch=90, yaw=0,
    cameras=['cam_head','cam_right_wrist'], locate_query='the red block'))
  performs exactly one ordinary move_to (gripper unchanged), checks its result,
  and captures the requested views in this same cell. locate_query is optional:
  only after an unblocked, reached motion does it call the existing locate once.
  Its return is {'status': 'observed' or 'motion_incomplete', 'motion': {...},
  'observation': {'status': 'observed', 'snapshot_id': ..., 'step': ...,
  'action_revision': ..., 'state': {...}, 'frames': {...}, 'interpretation': ...},
  'perception': None or the locate result, 'interpretation': ...}; when locate
  runs it also includes perception_assumption. The top level has no snapshot_id
  or images key: images are attached separately, mapped by observation.frames.
  Use r = move_and_observe([x,y,z], arm='right', cameras=['cam_head','cam_right_wrist']);
  print(r); snapshot_id = r['observation']['snapshot_id'].
  Inspect r['status'], r['motion'], r['perception'] and the attached images.
  Do not repeat the movement just because later Python code failed to read r:
  the motion and image capture may already have happened. It never retries or grasps.
  locate retains its TABLE-SUPPORTED-object assumption; omit it for held/stacked
  objects. Motion arrival is not verification of contact or task success.
- print(measure_points(snapshot_id, {'corner': {
    'cam_head': [row,col], 'cam_right_wrist': [row,col]}})) triangulates a point
  selected in two or three views. Use original-image pixels of the SAME physical
  corner/feature, not mask centroids or different silhouette edges. Up to 16
  named points can be measured together. No guessed height or table plane is used.
  Check each point's status, xyz_m, std_xyz_m and limitations. Correspondence is
  your hypothesis; small reprojection error does not prove it. Occluded or
  ambiguous correspondences need another view/decision, not invented pixels.
  All frames must be from this snapshot and still current; movement, changed
  pixels or calibration invalidates it. Unknown results carry no usable position.
  After move_and_observe, call measure_points(r['observation']['snapshot_id'], matches).
Keep each cell focused on the next missing observation or executable subgoal.
Use the helpers when they replace work; do not add observation calls without a
concrete information need. Preserve outcome verification before finish(success).
""".strip()


def supported(robo):
    from .hosts.robodojo_native import NativeBackend
    from .robodojo_xpl_rgb import RobodojoOfficialColourRobot
    return (isinstance(robo, RobodojoOfficialColourRobot)
            and isinstance(robo._backend, NativeBackend) and robo._backend.colour_only)


def plain(value):
    return asdict(value) if is_dataclass(value) else copy.deepcopy(value)


def rgb_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


class ObservationPhases:
    def __init__(self, kit):
        if not supported(kit.robo):
            raise ValueError("phase_tools requires native RGB-only RoboDojo")
        self.kit, self.robo = kit, kit.robo
        self.root = kit.workspace.parent / "observation_phases"
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots = {}
        self.sequence = 0
        # Bypass only the visual adapter's extra context capture, not the API.
        original = self.robo.camera_meta
        self.camera_meta = getattr(original, "__visual_geometry_original__", original)

    def event(self, kind, **values):
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"type": kind, **values}, ensure_ascii=False) + "\n")

    def stamp(self):
        return (self.robo.state().step,
                sum(bool(r.stateful) for r in self.robo._ledger.records))

    def cameras(self, cameras):
        cameras = [self.robo.CAMERAS[0]] if cameras is None else list(cameras)
        if (not cameras or len(cameras) > 3 or len(set(cameras)) != len(cameras)
                or any(c not in self.robo.CAMERAS for c in cameras)):
            raise ValueError("Select one to three distinct public RGB cameras")
        return cameras

    def observe_scene(self, *, cameras=None):
        cameras = self.cameras(cameras)
        before = self.stamp()
        captured = {}
        for camera in cameras:
            image = np.array(self.robo.image(camera), copy=True)
            metadata = self.camera_meta(camera)
            view = calibrated_view(metadata)
            if image.ndim != 3 or image.shape[2] != 3 or tuple(image.shape[:2]) != tuple(view.shape):
                raise ValueError("Expected original RGB pixels matching public calibration")
            captured[camera] = {"image": image, "calibration": plain(metadata),
                                "rgb_sha256": rgb_hash(image)}
        state = plain(self.robo.state())
        if self.stamp() != before:
            raise ValueError("Scene changed during capture; no synchronized snapshot produced")
        self.sequence += 1
        ref = f"observation_{self.sequence:04d}"
        frames = {}
        from .arms import _to_png
        for camera, data in captured.items():
            name = f"{ref}_{camera}.png"
            (self.root / name).write_bytes(_to_png(data["image"]))
            slot = len(self.kit._displayed)
            self.kit._display(data["image"].copy())
            frames[camera] = {"file": name, "display_label": f"display_{slot}",
                              "shape": list(data["image"].shape),
                              "rgb_sha256": data["rgb_sha256"], "calibration": plain(data["calibration"])}
        result = {"status": "observed", "snapshot_id": ref, "step": before[0],
                  "action_revision": before[1], "state": state, "frames": frames,
                  "interpretation": "Synchronized public observations, not a task verdict."}
        self.snapshots[ref] = {"stamp": before, "captured": captured, "summary": plain(result)}
        while len(self.snapshots) > 16:
            del self.snapshots[next(iter(self.snapshots))]
        self.event("observation", **result)
        return result

    def move_and_observe(self, position, *, arm="right", pitch=None, yaw=None, roll=None,
                         path="auto", cameras=None, locate_query=None, locate_camera=None):
        cameras = self.cameras(cameras)  # Reject bad options before any motion.
        locate_camera = cameras[0] if locate_camera is None else locate_camera
        if locate_camera not in cameras:
            raise ValueError("locate_camera must be one of the observed cameras")
        if locate_query is not None and (not isinstance(locate_query, str) or not locate_query.strip()):
            raise ValueError("locate_query must be a nonempty string or None")
        # Existing action/evidence/recovery hooks and all native budgets remain active.
        motion = plain(self.robo.move_to(position, arm=arm, pitch=pitch, yaw=yaw,
                                         roll=roll, path=path, gripper=0))
        reached = (motion.get("reached") is True and not motion.get("blocked")
                   and not motion.get("ik_error") and not motion.get("truncated"))
        observation = self.observe_scene(cameras=cameras)
        result = {"status": "observed" if reached else "motion_incomplete", "motion": motion,
                  "observation": observation, "perception": None,
                  "interpretation": "One move and observations; no grasp or task-success judgement."}
        # Exceptions propagate: they must stop dependent actions, not invent success.
        if reached and locate_query is not None:
            result["perception"] = plain(self.robo.locate(locate_query, camera=locate_camera))
            result["perception_assumption"] = "Existing locate assumes a table-supported object."
            if self.stamp() != self.snapshots[observation["snapshot_id"]]["stamp"]:
                raise ValueError("Scene changed during localization; observations are stale")
        self.event("move_and_observe", **result)
        return result

    def measure_points(self, snapshot_id, matches, *, pixel_sigma_px=1.0):
        if snapshot_id not in self.snapshots:
            raise ValueError("Unknown/evicted snapshot_id; call observe_scene first")
        snapshot = self.snapshots[snapshot_id]
        if self.stamp() != snapshot["stamp"]:
            raise ValueError("Stale snapshot after motion; capture fresh observations")
        if not isinstance(matches, dict) or not 1 <= len(matches) <= 16:
            raise ValueError("Provide one to sixteen named point correspondences")
        captured = snapshot["captured"]
        used = set()
        for name, pixels in matches.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(pixels, dict):
                raise ValueError("Each named point maps camera names to original row,col pixels")
            if not set(pixels) <= set(captured):
                raise ValueError("Every camera must belong to the same snapshot")
            used.update(pixels)
        for camera in used:
            current = np.asarray(self.robo.image(camera))
            if (rgb_hash(current) != captured[camera]["rgb_sha256"]
                    or plain(self.camera_meta(camera)) != captured[camera]["calibration"]):
                raise ValueError("Pixels or calibration changed; capture fresh observations")
        points = {}
        for name, pixels in matches.items():
            cameras = list(pixels)
            views = [calibrated_view(captured[c]["calibration"]) for c in cameras]
            points[name] = triangulate_point(views, list(pixels.values()), pixel_sigma_px=pixel_sigma_px)
            points[name]["correspondence"] = plain(pixels)
        if self.stamp() != snapshot["stamp"]:
            raise ValueError("Scene changed during measurement; no current position returned")
        result = {"snapshot_id": snapshot_id, "points": points,
                  "sources": {c: plain(snapshot["summary"]["frames"][c]) for c in sorted(used)},
                  "position_source": "Matched RGB rays; no depth, assumed height or hidden object pose."}
        self.event("measurement", **result)
        return result


def install(kit):
    if getattr(kit, "observation_phases", None) is not None:
        raise ValueError("Observation phases already installed")
    helper = ObservationPhases(kit)
    kit.observation_phases = helper
    for name in ("observe_scene", "move_and_observe", "measure_points"):
        kit._globals[name] = getattr(helper, name)
        setattr(kit.robo, name, getattr(helper, name))
    return helper
