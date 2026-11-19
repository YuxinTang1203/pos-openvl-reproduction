"""Selected visual instances keep their identity at the existing geometry API."""
from pathlib import Path

import numpy as np
import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.hosts.sam3_local import Segmented
from pyrualean.openvl_visual import OpenVLVisualAdapter


CONFIG = Path(__file__).resolve().parents[1] / "integrations/migrations/openvl/capabilities_visual.yml"


class RingCandidates:
    def __init__(self):
        self.calls = 0

    def segment_all(self, image, *, text_prompt, min_score):
        self.calls += 1
        h, w = np.asarray(image).shape[:2]
        out = []
        for left, score in ((w // 8, .95), (w * 5 // 8, .7)):
            top, side = h // 4, min(h, w) // 4
            mask = np.zeros((h, w), dtype=bool)
            mask[top:top + side, left:left + side] = True
            quarter = side // 4
            mask[top + quarter:top + side - quarter, left + quarter:left + side - quarter] = False
            out.append(Segmented(True, score, [left, top, left + side, top + side], mask))
        return out


def grounded(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace", images="on-demand", feedback="pure")
    segmenter = RingCandidates()
    adapter = OpenVLVisualAdapter(kit, config=CONFIG, initial_images={}, segmenter=segmenter)
    kit.evidence_adapter = adapter
    result = kit.execute_tool("Visual_Grounding_Tool", {"query": "objects"})
    assert "error" not in result.result
    return kit, adapter, segmenter, result.result["binding_ids"][0]


def test_selected_mask_pixel_preserves_number_and_calls_no_geometry_or_motion(robo, backend, tmp_path):
    kit, adapter, segmenter, binding = grounded(robo, tmp_path)
    calls_before, step_before = list(backend.calls), backend.env_steps()
    point = kit._globals["visual_candidate"](binding, 2)
    assert point == robo.visual_candidate(binding, 2)
    assert point["col"] > adapter.groundings[binding]["data"]["size"][1] // 2
    mask = segmenter.segment_all(robo.image(), text_prompt="objects", min_score=.2)[1].mask
    assert mask[point["row"], point["col"]]
    ys, xs = np.nonzero(mask)
    assert not mask[int(ys.mean()), int(xs.mean())]  # Bounding-box centre is the hole.
    assert segmenter.calls == 2  # One grounding + explicit test reconstruction; helper never resegments.
    assert backend.calls == calls_before and backend.env_steps() == step_before
    assert not {"xyz", "world_xyz", "grasp_yaw", "top_z"} & point.keys()
    assert point["source"]["current_camera_pixels"]
    robo.back_project(point["row"], point["col"], camera=point["camera"])
    assert backend.calls[-1][0] == "back_project"
    assert backend.calls[-1][1]["row"] == point["row"]
    assert backend.calls[-1][1]["col"] == point["col"]
    assert all(name not in {"locate", "segment", "move_to"} for name, _ in backend.calls)


def test_motion_rejects_stale_candidate_but_keeps_historical_data_readable(robo, tmp_path):
    kit, adapter, segmenter, binding = grounded(robo, tmp_path)
    kit.execute_tool("python", {"code": "robo.move_to([0, 0, .6])"})
    with pytest.raises(ValueError, match="current original camera"):
        adapter.visual_candidate(binding, 1)
    assert adapter.read_data(binding)["source"]["actions_since_capture"] == 1
    assert segmenter.calls == 1


def test_pixel_drift_is_rejected_even_without_recorded_action(robo, tmp_path, monkeypatch):
    _, adapter, segmenter, binding = grounded(robo, tmp_path)
    pixels = np.array(robo.image(), copy=True)
    pixels[0, 0, 0] ^= 1
    monkeypatch.setattr(robo, "image", lambda *args, **kwargs: pixels)
    with pytest.raises(ValueError, match="differ from the current camera"):
        adapter.visual_candidate(binding, 1)
    assert segmenter.calls == 1


def test_crop_candidate_cannot_be_used_as_camera_pixel(robo, tmp_path):
    kit, adapter, _, binding = grounded(robo, tmp_path)
    zoom = kit.execute_tool("Zoom_In_Tool", {"binding_id": binding, "candidate_number": 1})
    crop = zoom.result["images"][0]
    out = kit.execute_tool("Visual_Grounding_Tool", {"query": "objects", "image_id": crop})
    crop_binding = out.result["binding_ids"][0]
    with pytest.raises(ValueError, match="current original camera"):
        adapter.visual_candidate(crop_binding, 1)


@pytest.mark.parametrize("number", [0, 3, -1, 1.0, True, "1"])
def test_invalid_number_is_explicitly_rejected(robo, tmp_path, number):
    _, adapter, segmenter, binding = grounded(robo, tmp_path)
    with pytest.raises(ValueError, match="candidate_number"):
        adapter.visual_candidate(binding, number)
    assert segmenter.calls == 1


def test_corrupt_mask_dimensions_fail_without_geometry(robo, tmp_path):
    _, adapter, _, binding = grounded(robo, tmp_path)
    adapter.groundings[binding]["data"]["size"] = [2, 2]
    with pytest.raises(ValueError, match="dimensions"):
        adapter.visual_candidate(binding, 1)


def test_prompt_explains_selected_candidate_geometry_and_schema(robo, tmp_path):
    _, adapter, _, _ = grounded(robo, tmp_path)
    prompt = adapter.prompt()
    assert "visual_candidate(binding_id, candidate_number)" in prompt
    assert 'robo.back_project(p["row"], p["col"], camera=p["camera"])' in prompt
    assert "surface point, not an object centre, grasp pose" in prompt
    assert "COCO RLE" in prompt and "pred_masks" in prompt
