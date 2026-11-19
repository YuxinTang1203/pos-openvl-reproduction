"""Camera request cues preserve the public method and prefix replay behavior."""

import copy

import pytest

from pyrualean.openvl_routed import OpenVLRoutedAdapter
from tests.test_openvl_locate_candidate import AllMasks, CONFIG
from tests.test_openvl_locate_candidate import configured as configured


def test_return_identity_preserved_and_one_context_per_camera_per_cell(configured, monkeypatch):
    _, _, _, robo, kit, adapter, _ = configured
    sentinel = robo.camera_meta("cam_head")
    before = copy.deepcopy(sentinel)
    monkeypatch.setattr(adapter, "_original_camera_meta", lambda *a, **kw: sentinel)
    frame_count = len(adapter.frames)
    out = kit.execute_tool("python", {"code": "m1 = robo.camera_meta('cam_head')\nm2 = robo.camera_meta('cam_head')\nprint(m1['intrinsic_matrix'])"})
    assert not out.result.get("exception")
    assert kit._globals["m1"] is sentinel and kit._globals["m2"] is sentinel
    assert sentinel == before
    contexts = out.result["visual_geometry_context"]
    assert len(contexts) == 1 and len(adapter.frames) == frame_count + 1
    assert contexts[0]["status"] == "current" and contexts[0]["image_id"] in adapter.frames
    assert "[(col-cx)/fx, -(row-cy)/fy, -1]" in contexts[0]["coordinate_contract"]
    assert "conditional" in contexts[0]["projection_hint"]
    assert not out.images  # Stored in memory; not additional model image payload.
    later = kit.execute_tool("python", {"code": "print('no camera request')"})
    assert "visual_geometry_context" not in later.result
    assert len(adapter.frames) == frame_count + 1


def test_disabled_prefix_does_not_capture_or_hint(configured, monkeypatch):
    scene, _, _, robo, kit, _, _ = configured
    adapter = OpenVLRoutedAdapter(kit, config=CONFIG, initial_images={}, segmenter=AllMasks(scene))
    kit.evidence_adapter = adapter
    adapter.disable_routing()
    original_image = robo.image
    def forbidden(*a, **kw):
        raise AssertionError("Disabled prefix must not capture RGB")
    monkeypatch.setattr(robo, "image", forbidden)
    out = kit.execute_tool("python", {"code": "m = robo.camera_meta('cam_head')"})
    assert not out.result.get("exception") and "visual_geometry_context" not in out.result
    assert not adapter.frames
    monkeypatch.setattr(robo, "image", original_image)
    adapter.enable_routing()
    enabled = kit.execute_tool("python", {"code": "m = robo.camera_meta('cam_head')"})
    assert enabled.result["visual_geometry_context"][0]["status"] == "current"
    assert len(adapter.frames) == 1  # Reinstalling adapters did not nest context wrappers.


def test_projection_internal_calibration_does_not_generate_context(configured):
    _, _, _, _, kit, adapter, binding = configured
    frame = adapter.groundings[binding]["source_image_id"]
    count = len(adapter.frames)
    out = kit.execute_tool("python", {"code": f"p = visual_project({frame!r}, 200, 380, .81)\nprint(p['status'])"})
    assert not out.result.get("exception") and kit._globals["p"]["status"] == "ok"
    assert "visual_geometry_context" not in out.result and len(adapter.frames) == count


def test_historical_metadata_has_no_fresh_frame_or_capture(configured, monkeypatch):
    _, _, _, robo, kit, adapter, _ = configured
    count = len(adapter.frames)
    def forbidden(*a, **kw):
        raise AssertionError("Historical request must not capture a current frame")
    monkeypatch.setattr(robo, "image", forbidden)
    out = kit.execute_tool("python", {"code": "past = robo.camera_meta('cam_head', step=0)"})
    assert not out.result.get("exception")
    context = out.result["visual_geometry_context"][0]
    assert context["status"] == "historical" and context["image_id"] is None
    assert context["requested_step"] == 0 and not context["current_camera_pixels"]
    assert len(adapter.frames) == count


def test_same_cell_scene_change_marks_capture_historical(configured):
    scene, _, _, robo, kit, adapter, _ = configured
    def advance():
        scene.step_count += 1
    kit._globals["advance_test_scene"] = advance
    out = kit.execute_tool("python", {"code": "m = robo.camera_meta('cam_head')\nadvance_test_scene()"})
    context = out.result["visual_geometry_context"][0]
    assert context["status"] == "historical" and not context["current_camera_pixels"]
    assert context["source"]["source_step"] < scene.step_count
    with pytest.raises(ValueError, match="current original"):
        robo.visual_project(context["image_id"], 200, 380, .81)
    # A later explicit request captures a current frame, instead of repeating history.
    later = kit.execute_tool("python", {"code": "m = robo.camera_meta('cam_head')"})
    assert later.result["visual_geometry_context"][0]["status"] == "current"


def test_capture_failure_reports_unavailable_without_breaking_original_method(configured, monkeypatch):
    _, _, _, robo, kit, adapter, _ = configured
    sentinel = robo.camera_meta("cam_head")
    monkeypatch.setattr(adapter, "_original_camera_meta", lambda *a, **kw: sentinel)
    def fail(*a, **kw):
        raise RuntimeError("image unavailable")
    monkeypatch.setattr(robo, "image", fail)
    out = kit.execute_tool("python", {"code": "m = robo.camera_meta('cam_head')\nprint('continued')"})
    assert not out.result.get("exception") and kit._globals["m"] is sentinel
    context = out.result["visual_geometry_context"][0]
    assert context["status"] == "unavailable" and context["image_id"] is None
    assert "image unavailable" in context["note"]


def test_two_cameras_get_only_their_own_current_cell_context(configured):
    _, _, _, _, kit, _, _ = configured
    out = kit.execute_tool("python", {"code": "a = robo.camera_meta('cam_head')\nb = robo.camera_meta('cam_right_wrist')"})
    assert {c["camera"] for c in out.result["visual_geometry_context"]} == {"cam_head", "cam_right_wrist"}
    later = kit.execute_tool("python", {"code": "a = robo.camera_meta('cam_head')"})
    assert [c["camera"] for c in later.result["visual_geometry_context"]] == ["cam_head"]
