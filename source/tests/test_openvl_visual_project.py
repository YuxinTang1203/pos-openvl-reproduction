"""Conditional projections preserve current original-image provenance."""

import json

import numpy as np
import pytest

from tests.test_openvl_locate_candidate import configured as configured


def raw_frame(adapter, binding):
    return adapter.groundings[binding]["source_image_id"]


def test_public_helper_and_global_use_calibration_without_segmentation_or_actions(configured):
    scene, sam, toolkit, robo, kit, adapter, binding = configured
    sam.calls.clear()
    frame = raw_frame(adapter, binding)
    expected = np.array([.12, -.03, .81])
    row, col, depth = scene.view("cam_head").project(expected)
    before = scene.step_count, len(toolkit.state.records), adapter.action_revision()
    result = robo.visual_project(frame, row, col, .81)
    assert result["status"] == "ok"
    np.testing.assert_allclose(result["xyz_m"], expected, atol=1e-10)
    assert result["forward_depth_m"] == pytest.approx(depth)
    assert kit._globals["visual_project"].__self__ is adapter
    assert (scene.step_count, len(toolkit.state.records), adapter.action_revision()) == before
    assert not sam.calls
    assert result["source"]["current_camera_pixels"]
    assert result["calibration"]["extrinsic_matrix"] == robo.camera_meta("cam_head")["extrinsic_matrix"]
    assert result["plane_sigma_m"] is None and result["conditional_on_plane_z_m"] == .81
    event = json.loads((adapter.root / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "visual_plane_projection" and event["source_rgb_sha256"]


def test_same_size_grounding_annotation_resolves_only_to_its_current_raw_parent(configured):
    _, _, _, robo, _, adapter, binding = configured
    original = raw_frame(adapter, binding)
    annotation = next(ref for ref, row in adapter.frames.items()
                      if row["phase"] == "grounding_annotation"
                      and row["parent_image_id"] == original)
    a = robo.visual_project(original, 200, 380, .81)
    b = robo.visual_project(annotation, 200, 380, .81)
    assert a["xyz_m"] == b["xyz_m"]
    assert b["requested_image_id"] == annotation and b["source"]["image_id"] == original


def test_crop_and_annotation_of_crop_rejected(configured):
    _, _, _, robo, kit, adapter, binding = configured
    zoom = kit.execute_tool("Zoom_In_Tool", {"binding_id": binding, "candidate_number": 1})
    crop = zoom.result["images"][0]
    with pytest.raises(ValueError, match="current original"):
        robo.visual_project(crop, 10, 10, .81)
    data = (adapter.root / adapter.frames[crop]["file"]).read_bytes()
    marked_crop = adapter.register_frame(data, phase="grounding_annotation", parent=crop)
    with pytest.raises(ValueError, match="current original"):
        robo.visual_project(marked_crop, 10, 10, .81)


def test_stale_or_unknown_frames_rejected(configured):
    scene, _, _, robo, _, adapter, binding = configured
    frame = raw_frame(adapter, binding)
    with pytest.raises(ValueError, match="Unknown image_id"):
        robo.visual_project("missing", 200, 380, .81)
    scene.step_count += 1
    with pytest.raises(ValueError, match="current original"):
        robo.visual_project(frame, 200, 380, .81)


def test_pixel_mismatch_rejected_even_without_step_change(configured, monkeypatch):
    _, _, _, robo, _, adapter, binding = configured
    changed = np.array(robo.image("cam_head"), copy=True)
    changed[1, 1, 1] ^= 1
    monkeypatch.setattr(robo, "image", lambda camera: changed)
    with pytest.raises(ValueError, match="pixels differ"):
        robo.visual_project(raw_frame(adapter, binding), 200, 380, .81)


def test_current_wrist_calibration_and_bad_shape_rejection(configured, monkeypatch):
    scene, _, _, robo, _, adapter, _ = configured
    frame = adapter.register_frame(robo.image("cam_left_wrist"), camera="cam_left_wrist", phase="test")
    view = scene.view("cam_left_wrist")
    ray = view.rays(np.asarray(220.), np.asarray(340.))
    point = view.pos + .25*ray
    result = robo.visual_project(frame, 220, 340, point[2], plane_sigma_m=.005)
    np.testing.assert_allclose(result["xyz_m"], point, atol=1e-9)
    assert result["plane_sigma_m"] == .005
    meta = robo.camera_meta("cam_left_wrist")
    monkeypatch.setattr(adapter, "_original_camera_meta", lambda camera: {**meta, "shape": [240, 320, 3]})
    with pytest.raises(ValueError, match="calibration shape"):
        robo.visual_project(frame, 220, 340, point[2])


def test_depth_backend_and_scene_change_during_calibration_rejected(configured, monkeypatch):
    scene, _, toolkit, robo, _, adapter, binding = configured
    frame = raw_frame(adapter, binding)
    toolkit._primitives.colour_only = False
    with pytest.raises(NotImplementedError, match="RGB-only"):
        robo.visual_project(frame, 200, 380, .81)
    toolkit._primitives.colour_only = True
    original = adapter._original_camera_meta
    def changing(camera):
        meta = original(camera)
        scene.step_count += 1
        return meta
    monkeypatch.setattr(adapter, "_original_camera_meta", changing)
    with pytest.raises(ValueError, match="Scene changed"):
        robo.visual_project(frame, 200, 380, .81)


def test_prompt_describes_conditional_projection(configured):
    _, _, _, _, _, adapter, _ = configured
    prompt = adapter.prompt()
    assert "robo.visual_project(image_id, row, col, plane_z)" in prompt
    assert "not a measured surface" in prompt
