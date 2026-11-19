"""Selected OpenVL masks enter real RGB geometry without text re-selection.

The existing CPU ray-cast test fixture supplies masks as a fake SAM response.
No simulator, GPU model, network service or hidden live object pose is queried.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.hosts import robodojo_native as native
from pyrualean.hosts import robodojo_tools as tools
from pyrualean.hosts.sam3_local import Segmented
from pyrualean.openvl_visual import OpenVLVisualAdapter
from pyrualean.openvl_routed import OpenVLRoutedAdapter
from pyrualean.robodojo_xpl_rgb import ObjectSight, RobodojoOfficialColourRobot


spec = importlib.util.spec_from_file_location("candidate_colour_fixture", Path(__file__).with_name("test_robodojo_colour.py"))
colour = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colour)
CONFIG = Path(__file__).resolve().parents[1] / "integrations/migrations/openvl/capabilities_visual.yml"


class AllMasks:
    def __init__(self, scene):
        self.scene = scene

    def segment_all(self, image, *, text_prompt, min_score):
        camera = colour.CAMERAS[int(np.asarray(image)[0, 0, 0])]
        owner = colour.render(self.scene.view(camera), self.scene.shapes)
        masks = []
        for index in range(len(self.scene.shapes)):
            mask = owner == index
            rows, cols = np.nonzero(mask)
            if len(rows):
                masks.append(Segmented(True, .9, [float(cols.min()), float(rows.min()),
                    float(cols.max() + 1), float(rows.max() + 1)], mask))
        return masks


@pytest.fixture
def configured(tmp_path):
    scene = colour.Scene([colour.CUBE, colour.BOWL])
    sam = colour.Segmenter(scene)
    toolkit = native.NativeToolkit(scene, sam, tmp_path / "native", colour_only=True)
    backend = native.NativeBackend(toolkit)
    robo = RobodojoOfficialColourRobot(backend, primitives="no-vla")
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace", images="on-demand", feedback="pure")
    adapter = OpenVLVisualAdapter(kit, config=CONFIG, initial_images={}, segmenter=AllMasks(scene))
    kit.evidence_adapter = adapter
    grounded = kit.execute_tool("Visual_Grounding_Tool", {"query": "objects", "camera": "cam_head"})
    assert not grounded.result.get("error")
    binding = grounded.result["binding_ids"][0]
    return scene, sam, toolkit, robo, kit, adapter, binding


def test_selected_second_mask_returns_full_original_object_sight_without_text_calls(configured, monkeypatch):
    scene, sam, toolkit, robo, kit, adapter, binding = configured
    masks = adapter.groundings[binding]["data"]["boxes"]
    assert masks[0][0] < masks[1][0]  # Left bowl, then right cube.
    def forbidden_depth(*args, **kwargs):
        raise AssertionError("RGB-only selected geometry must not read depth")
    monkeypatch.setattr(tools, "world_points", forbidden_depth)
    original = robo.locate("the cube")
    assert isinstance(original, ObjectSight)
    sam.calls.clear()
    point_thresholds = []
    original_segment = sam.segment
    def record_point_threshold(image, **kwargs):
        assert kwargs.get("text_prompt") is None and kwargs.get("point") is not None
        point_thresholds.append(kwargs.get("min_score"))
        return original_segment(image, **kwargs)
    monkeypatch.setattr(sam, "segment", record_point_threshold)
    before_records, before_steps = len(toolkit.state.records), scene.step_count
    # Reads or writes of prompt-keyed heights would mix identities; reject both.
    class NoHeightAccess(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("Selected instance must not read prompt-keyed heights")
        def __setitem__(self, *args, **kwargs):
            raise AssertionError("Selected instance must not write prompt-keyed heights")
    cache = NoHeightAccess({"the cube": (.4, "wrong_other_object", .99)})
    toolkit._primitives._measured_heights = cache
    selected = robo.locate_candidate(binding, 2)
    assert isinstance(selected, ObjectSight) and selected.found
    assert colour._off(selected.center_xyz, colour.CUBE) < .004
    assert selected.height_m == pytest.approx(.035, abs=.005)
    assert selected.top_z is not None and selected.length_m is not None
    assert selected.width_m is not None and selected.long_axis_deg is not None
    assert selected.grasp_yaw is not None and selected.views >= 2 and selected.note is None
    assert selected.centroid == original.centroid and selected.area_px == original.area_px
    assert sam.calls and all(isinstance(prompt, tuple) for _, prompt in sam.calls)
    assert point_thresholds and all(value == .2 for value in point_thresholds)
    assert all(camera != "cam_head" for camera, _ in sam.calls)  # Source mask is used verbatim.
    assert toolkit._primitives._measured_heights is cache
    assert len(toolkit.state.records) == before_records and scene.step_count == before_steps
    assert robo._ledger.records[-1].tool == "locate_mask" and not robo._ledger.records[-1].stateful
    assert kit._globals["locate_candidate"].__self__ is adapter
    event = json.loads((adapter.root / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "selected_mask_geometry"
    assert event["primitive_ledger_index"] == robo._ledger.records[-1].index
    assert event["geometry_payload"]["center_xyz"] == list(selected.center_xyz)
    assert event["geometry_payload"]["grasp_yaw"] == selected.grasp_yaw
    assert event["geometry_payload"]["text_prompt"] is None
    assert event["geometry_payload"]["cross_view_min_score"] == .2


def test_other_candidate_remains_a_different_measured_object(configured):
    _, sam, _, robo, _, _, binding = configured
    first = robo.locate_candidate(binding, 1)
    second = robo.locate_candidate(binding, 2)
    assert first.center_xyz[0] < 0 < second.center_xyz[0]
    assert colour._off(first.center_xyz, colour.BOWL) < .005
    assert first.length_m > second.length_m * 2
    assert all(isinstance(prompt, tuple) for _, prompt in sam.calls)


def test_one_view_uncertainty_survives_selected_mask_geometry(configured):
    scene, _, _, robo, _, _, binding = configured
    scene.blind.update({"cam_left_wrist", "cam_right_wrist"})
    result = robo.locate_candidate(binding, 2)
    assert result.found and result.views == 1 and result.note
    assert "guess" in result.note


def test_geometry_failure_is_not_replaced_by_a_pixel_or_invented_target(configured, monkeypatch):
    _, _, _, robo, _, _, binding = configured
    monkeypatch.setattr(tools, "_measure_in_colour", lambda *args, **kwargs: (None, {"views": 1, "hidden": "target occluded"}))
    result = robo.locate_candidate(binding, 2)
    assert result.found and result.center_xyz is None
    assert result.top_z is None and result.grasp_yaw is None and result.note == "target occluded"


def test_stale_step_rejected_before_geometry_and_crop_is_not_a_camera_mask(configured):
    scene, sam, _, robo, kit, _, binding = configured
    zoom = kit.execute_tool("Zoom_In_Tool", {"binding_id": binding, "candidate_number": 1})
    class CropMask:
        def segment_all(self, image, **kwargs):
            height, width = np.asarray(image).shape[:2]
            return [Segmented(True, .9, [0, 0, width, height], np.ones((height, width), dtype=bool))]
    kit.evidence_adapter.tools["Visual_Grounding_Tool"].client.segmenter = CropMask()
    crop_ground = kit.execute_tool("Visual_Grounding_Tool", {"query": "objects", "image_id": zoom.result["images"][0]})
    assert crop_ground.result.get("binding_ids")
    with pytest.raises(ValueError, match="current original camera"):
        robo.locate_candidate(crop_ground.result["binding_ids"][0], 1)
    scene.step_count += 1
    with pytest.raises(ValueError, match="current original camera"):
        robo.locate_candidate(binding, 2)
    assert not sam.calls


def mask_request(adapter, binding, number):
    source, data, _, pixels = adapter._current_candidate(binding, number)
    return {"camera": source["camera"], "mask_rle": {"size": data["size"], "counts": data["masks"][number - 1]},
            "score": data["scores"][number - 1], "source_step": source["source_step"],
            "source_rgb_sha256": hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()}


def test_backend_rechecks_step_pixels_and_mask_shape_without_silent_fallback(configured):
    scene, sam, toolkit, _, _, adapter, binding = configured
    original = mask_request(adapter, binding, 2)
    for changed, message in [({"source_step": 999}, "step"),
                             ({"source_rgb_sha256": "not-current"}, "RGB"),
                             ({"mask_rle": {"size": [1, 1], "counts": "bad"}}, "dimensions")]:
        result = toolkit.execute_tool("locate_mask", {**original, **changed})
        assert message in result["error"]
    assert not sam.calls and scene.step_count == 0 and len(toolkit.state.records) == 1


def test_backend_rejects_depth_mode_even_when_images_and_step_match(configured):
    _, sam, toolkit, _, _, adapter, binding = configured
    request = mask_request(adapter, binding, 2)
    toolkit._primitives.colour_only = False
    result = toolkit.execute_tool("locate_mask", request)
    assert "RGB-only" in result["error"] and not sam.calls
    with pytest.raises(NotImplementedError, match="RGB-only"):
        adapter.locate_candidate(binding, 2)


def test_unrelated_robot_backend_gets_explicit_unavailable_error(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace", images="on-demand", feedback="pure")
    adapter = OpenVLVisualAdapter(kit, config=CONFIG, initial_images={}, segmenter=None)
    with pytest.raises(NotImplementedError, match="native RGB-only RoboDojo"):
        adapter.locate_candidate("unavailable", 1)


def test_actual_routed_choice_recommends_complete_geometry_on_supported_backend(configured):
    scene, _, _, robo, kit, _, _ = configured
    adapter = OpenVLRoutedAdapter(kit, config=CONFIG, initial_images={}, segmenter=AllMasks(scene))
    kit.evidence_adapter = adapter
    result = kit.execute_tool("python", {"code": "robo.locate('objects')"})
    message = result.result["exception"]
    assert "PerceptionChoiceRequired" in message
    assert "prefer robo.locate_candidate(binding_id, candidate_number)" in message
    assert "found, center_xyz, views and note" in message
    assert "alternate single-surface-pixel path" in message
    binding = result.result["routed_perception"][0]["binding_ids"][0]
    continued = kit.execute_tool("python", {"code": f"selected = robo.locate_candidate({binding!r}, 2)\nprint(selected)"})
    assert not continued.result.get("exception")
    assert isinstance(kit._globals["selected"], ObjectSight)
