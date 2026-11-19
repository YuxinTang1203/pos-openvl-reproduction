"""Automatic routing stops ambiguous geometry before dependent robot actions."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.hosts.sam3_local import Segmented
from pyrualean.migrations import configure_migration, validate_migration
from pyrualean.openvl_routed import OpenVLRoutedAdapter, PerceptionChoiceRequired, PerceptionRoutingError, assess_ambiguity


CONFIG = Path(__file__).resolve().parents[1] / "integrations/migrations/openvl/capabilities_visual.yml"


class Candidates:
    def __init__(self, count=2):
        self.count, self.calls = count, []
        self.scores = (.95, .9)

    def segment_all(self, image, *, text_prompt, min_score):
        self.calls.append((text_prompt, min_score))
        h, w = np.asarray(image).shape[:2]
        result = []
        for i, score in enumerate(self.scores[:self.count]):
            if score < min_score:
                continue
            x, y = w // 8 + i * w // 2, h // 4
            mask = np.zeros((h, w), dtype=bool)
            mask[y:y + h // 4, x:x + w // 4] = True
            result.append(Segmented(True, score, [x, y, x + w // 4, y + h // 4], mask))
        return result


def setup(robo, tmp_path, count=2, **options):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace", images="on-demand", feedback="pure")
    sam = Candidates(count)
    adapter = OpenVLRoutedAdapter(kit, config=CONFIG, initial_images={}, segmenter=sam, **options)
    kit.evidence_adapter = adapter
    return kit, adapter, sam


def last_archived_route(adapter):
    rows = [json.loads(line) for line in (adapter.root / "events.jsonl").read_text().splitlines()]
    return next(row["payload"] for row in reversed(rows) if row["type"] == "automatic_perception_result")


def test_choice_stops_later_motion_delivers_numbered_image_and_keeps_prior_actions(robo, backend, tmp_path):
    kit, adapter, sam = setup(robo, tmp_path)
    out = kit.execute_tool("python", {"code": "visits = 1\nrobo.move_to([0, 0, .6])\np = robo.segment('blocks')\nrobo.move_to([.1, 0, .6])\nvisits += 1"})
    assert "PerceptionChoiceRequired" in out.result["exception"]
    assert kit._globals["visits"] == 1
    assert sum(name == "move_to" for name, _ in backend.calls) == 1
    assert not any(name == "segment" for name, _ in backend.calls)
    evidence = out.result["routed_perception"][0]
    assert evidence["route_status"] == "choice_required" and len(evidence["candidates"]) == 2
    assert out.images and evidence["images"]
    binding = evidence["binding_ids"][0]
    followup = kit.execute_tool("python", {"code": f"p = visual_candidate({binding!r}, 2)\npoint = robo.back_project(p['row'], p['col'], camera=p['camera'])\nprint(point)"})
    assert not followup.result.get("exception")
    assert sum(name == "move_to" for name, _ in backend.calls) == 1  # Nothing replayed.
    assert len(sam.calls) == 1 and adapter.summary()["autonomous_tool_calls"] == 0
    assert adapter.summary()["automatic_grounding_calls"] == 1


@pytest.mark.parametrize("count", [0, 1])
def test_zero_and_one_keep_exact_original_locate_return_and_arguments(robo, tmp_path, count):
    calls = []
    original_result = SimpleNamespace(found=bool(count), center_xyz=(1, 2, 3) if count else None)

    def locate(text_prompt, camera="agentview", min_score=.2):
        calls.append((text_prompt, camera, min_score))
        return original_result

    robo.locate = locate
    kit, adapter, sam = setup(robo, tmp_path, count)
    out = kit.execute_tool("python", {"code": "located = robo.locate('objects', 'wrist', .65)"})
    assert not out.result.get("exception")
    assert kit._globals["located"] is original_result
    assert calls == [("objects", "wrist", .65)] and sam.calls == [("objects", .65)]
    assert "routed_perception" not in out.result and not out.images
    archived = last_archived_route(adapter)
    assert archived["source"]["camera"] == "wrist"
    assert archived["route_status"] == "original_perception"
    assert adapter.tools["Visual_Grounding_Tool"].client.min_score == .2


def test_requested_min_score_changes_candidates_without_changing_native_default(robo, backend, tmp_path):
    kit, adapter, sam = setup(robo, tmp_path)
    out = kit.execute_tool("python", {"code": "s = robo.segment('objects', min_score=.92)"})
    assert not out.result.get("exception")
    assert "routed_perception" not in out.result and not out.images
    assert last_archived_route(adapter)["data"]["num_masks"] == 1
    assert sam.calls == [("objects", .92)]
    assert backend.calls[-1][1]["min_score"] == .92
    assert adapter.tools["Visual_Grounding_Tool"].client.min_score == .2


def test_point_segmentation_and_disabled_prefix_keep_original_path(robo, backend, tmp_path):
    kit, adapter, sam = setup(robo, tmp_path)
    point = kit.execute_tool("python", {"code": "s = robo.segment(point=(20, 30), min_score=.4)"})
    assert not point.result.get("exception") and "routed_perception" not in point.result
    assert backend.calls[-1][1]["point"] == [20, 30]
    adapter.disable_routing()
    prefix = kit.execute_tool("python", {"code": "s = robo.segment('objects')"})
    assert not prefix.result.get("exception") and "routed_perception" not in prefix.result
    assert not sam.calls
    adapter.enable_routing()
    resumed = kit.execute_tool("python", {"code": "s = robo.segment('objects')"})
    assert "PerceptionChoiceRequired" in resumed.result["exception"] and len(sam.calls) == 1


def test_grounding_failure_never_falls_back_to_original_geometry(robo, backend, tmp_path):
    kit, adapter, _ = setup(robo, tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("SAM unavailable")

    adapter.tools["Visual_Grounding_Tool"].client.segmenter.segment_all = fail
    out = kit.execute_tool("python", {"code": "robo.segment('objects')\nrobo.move_to([0, 0, .6])"})
    assert "PerceptionRoutingError" in out.result["exception"]
    assert "SAM unavailable" in str(out.result)
    assert out.result["routed_perception"][0]["route_status"] == "grounding_error"
    assert not any(name in {"segment", "move_to"} for name, _ in backend.calls)


def test_invalid_grounding_result_schema_never_falls_back(robo, backend, tmp_path):
    kit, adapter, _ = setup(robo, tmp_path)
    adapter.tools["Visual_Grounding_Tool"].client.execute = lambda *args, **kwargs: {
        "ok": True, "tool_name": "Visual_Grounding_Tool", "result": {
            "pred_boxes": [[float("nan"), 0, 1, 1]], "pred_scores": [.9], "orig_img_h": 256, "orig_img_w": 256}}
    out = kit.execute_tool("python", {"code": "robo.segment('objects')"})
    assert "PerceptionRoutingError" in out.result["exception"]
    assert not any(name == "segment" for name, _ in backend.calls)


def test_selection_binding_becomes_stale_after_motion(robo, tmp_path):
    kit, adapter, _ = setup(robo, tmp_path)
    first = kit.execute_tool("python", {"code": "robo.segment('objects')"})
    binding = first.result["routed_perception"][0]["binding_ids"][0]
    kit.execute_tool("python", {"code": "robo.move_to([0, 0, .6])"})
    with pytest.raises(ValueError, match="current original camera"):
        adapter.visual_candidate(binding, 1)


def test_same_cell_motion_updates_feedback_freshness_without_refreshing_capture(robo, tmp_path):
    kit, adapter, _ = setup(robo, tmp_path, count=1)
    out = kit.execute_tool("python", {"code": "robo.segment('objects')\nrobo.move_to([0, 0, .6])"})
    assert not out.result.get("exception")
    assert "routed_perception" not in out.result
    evidence = last_archived_route(adapter)
    assert evidence["source"]["actions_since_capture"] == 1
    assert not evidence["source"]["current_camera_pixels"]
    assert evidence["image_contexts"][0]["actions_since_capture"] == 1
    with pytest.raises(ValueError, match="current original camera"):
        adapter.visual_candidate(evidence["binding_ids"][0], 1)


def test_autonomous_tool_calls_are_separate_from_automatic_routes(robo, tmp_path):
    kit, adapter, _ = setup(robo, tmp_path)
    kit.execute_tool("Visual_Grounding_Tool", {"query": "objects"})
    assert adapter.summary()["autonomous_tool_calls"] == 1
    assert adapter.summary()["automatic_grounding_calls"] == 0
    kit.execute_tool("python", {"code": "robo.segment('objects')"})
    assert adapter.summary()["autonomous_tool_calls"] == 1
    assert adapter.summary()["automatic_grounding_calls"] == 1


def test_ordinary_python_may_catch_choice_exception(robo, tmp_path):
    kit, adapter, _ = setup(robo, tmp_path)
    out = kit.execute_tool("python", {"code": "try:\n    robo.segment('objects')\nexcept RuntimeError:\n    caught = True\ncompleted = True"})
    assert not out.result.get("exception")
    assert kit._globals["caught"] and kit._globals["completed"] and out.images
    assert adapter.choice_boundaries == 1


def test_caught_choice_then_motion_keeps_delivered_evidence_historical(robo, tmp_path):
    kit, adapter, _ = setup(robo, tmp_path)
    out = kit.execute_tool("python", {"code": "try:\n    robo.segment('objects')\nexcept RuntimeError:\n    pass\nrobo.move_to([0, 0, .6])"})
    assert not out.result.get("exception")  # An ordinary exception is not a global action interlock.
    evidence = out.result["routed_perception"][0]
    assert evidence["route_status"] == "choice_required" and out.images
    assert evidence["source"]["actions_since_capture"] == 1
    assert not evidence["source"]["current_camera_pixels"]
    assert evidence["source"]["source_step"] < robo._backend.env_steps()
    assert evidence["image_contexts"][0]["source_step"] == evidence["source"]["source_step"]
    assert evidence["image_contexts"][0]["actions_since_capture"] == 1
    with pytest.raises(ValueError, match="current original camera"):
        adapter.visual_candidate(evidence["binding_ids"][0], 1)


def test_invalid_signature_and_historical_text_query_do_not_route_current_rgb(robo, tmp_path):
    _, adapter, sam = setup(robo, tmp_path)
    with pytest.raises(TypeError):
        robo.segment("objects", invented_parameter=True)
    with pytest.raises(PerceptionRoutingError, match="current observation"):
        robo.segment("objects", step=0)
    with pytest.raises(ValueError, match="min_score"):
        robo.segment("objects", min_score=float("nan"))
    assert not sam.calls


def test_new_migration_is_separate_and_baseline_remains_unwrapped(robo, tmp_path):
    validate_migration({"kind": "openvl_routed", "config": str(CONFIG)})
    with pytest.raises(ValueError):
        validate_migration({"kind": "openvl_routed"})
    original = robo.segment
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "baseline", images="on-demand", feedback="pure")
    configure_migration(kit, {"kind": "baseline"}, task="fixture", initial_images={})
    assert kit.evidence_adapter is None and robo.segment == original


def test_single_candidate_preserves_original_segment_return_type(robo, tmp_path):
    original_type = type(robo.segment("objects"))
    kit, _, _ = setup(robo, tmp_path, count=1)
    out = kit.execute_tool("python", {"code": "segment_result = robo.segment('objects')"})
    assert not out.result.get("exception")
    assert type(kit._globals["segment_result"]) is original_type


def canonical(scores, masks):
    from pycocotools import mask as mask_utils
    return {"boxes": [[0, 0, m.shape[1], m.shape[0]] for m in masks],
            "scores": list(scores), "size": list(masks[0].shape) if masks else [30, 40],
            "masks": [mask_utils.encode(np.asfortranarray(m, dtype=np.uint8))["counts"].decode() for m in masks]}


def distinct_masks():
    a = np.zeros((30, 40), dtype=bool)
    b = a.copy()
    a[5:15, 5:15] = True
    b[5:15, 25:35] = True
    return a, b


def test_low_score_alternatives_no_longer_interrupt_or_disappear(robo, backend, tmp_path):
    kit, adapter, sam = setup(robo, tmp_path)
    sam.scores = (.9296875, .462890625)
    out = kit.execute_tool("python", {"code": "s = robo.segment('objects')\nrobo.move_to([0, 0, .6])"})
    assert not out.result.get("exception") and "routed_perception" not in out.result
    assert sum(name == "move_to" for name, _ in backend.calls) == 1
    archived = last_archived_route(adapter)
    assert len(archived["candidates"]) == 2  # Scores affect the choice gate, not stored proposals.
    assert archived["ambiguity_assessment"]["competitive_candidate_numbers"] == [1]
    assert not archived["ambiguity_assessment"]["choice_required"]


def test_same_score_distinct_masks_require_selection_and_preserve_numbers():
    a, b = distinct_masks()
    data = canonical([.97265625, .97265625], [a, b])
    out = assess_ambiguity(data)
    assert out["choice_required"] and out["competitive_candidate_numbers"] == [1, 2]
    assert out["hypothesis_groups"] == [
        {"representative_number": 1, "candidate_numbers": [1]},
        {"representative_number": 2, "candidate_numbers": [2]}]
    assert len(data["masks"]) == 2 and data["scores"] == [.97265625, .97265625]


def test_duplicate_masks_do_not_create_false_instance_choice():
    a, _ = distinct_masks()
    data = canonical([.94, .95], [a, a])
    out = assess_ambiguity(data)
    assert not out["choice_required"]
    assert out["hypothesis_groups"] == [{"representative_number": 2, "candidate_numbers": [2, 1]}]
    assert len(data["masks"]) == 2  # No renumbering or data deletion.


def test_score_margin_is_inclusive_and_configurable():
    a, b = distinct_masks()
    assert assess_ambiguity(canonical([.9, .8], [a, b]))["choice_required"]
    assert not assess_ambiguity(canonical([.9, .799], [a, b]))["choice_required"]
    assert assess_ambiguity(canonical([.9, .75], [a, b]), score_margin=.2)["choice_required"]


def test_iou_margin_is_inclusive_and_configurable():
    a, _ = distinct_masks()
    b = a.copy()
    b[5, 5:15] = False  # Exactly 90 shared pixels over 100 union pixels.
    data = canonical([.9, .9], [a, b])
    assert not assess_ambiguity(data)["choice_required"]
    assert assess_ambiguity(data, duplicate_mask_iou=.91)["choice_required"]


def test_empty_masks_mean_no_effective_candidates():
    out = assess_ambiguity(canonical([.9], [np.zeros((30, 40), dtype=bool)]))
    assert not out["choice_required"] and out["reason"] == "no_valid_candidates"
    assert out["empty_candidate_numbers"] == [1]


@pytest.mark.parametrize("values", [
    {"ambiguity_score_margin": -1}, {"ambiguity_score_margin": float("nan")},
    {"ambiguity_score_margin": True}, {"duplicate_mask_iou": 0}, {"duplicate_mask_iou": 1.1},
    {"duplicate_mask_iou": "0.9"},
])
def test_heuristic_parameters_validate_at_migration_and_constructor(robo, tmp_path, values):
    with pytest.raises(ValueError):
        validate_migration({"kind": "openvl_routed", "config": str(CONFIG), **values})
    with pytest.raises(ValueError):
        setup(robo, tmp_path, **values)


def test_custom_thresholds_are_used_and_recorded(robo, tmp_path):
    kit, adapter, sam = setup(robo, tmp_path, ambiguity_score_margin=.3, duplicate_mask_iou=.95)
    sam.scores = (.95, .7)
    out = kit.execute_tool("python", {"code": "robo.segment('objects')"})
    assert "PerceptionChoiceRequired" in out.result["exception"]
    assert adapter.provenance["ambiguity_heuristic"]["score_margin"] == .3
    assert adapter.provenance["ambiguity_heuristic"]["duplicate_mask_iou"] == .95
    assert not adapter.provenance["ambiguity_heuristic"]["calibrated_probability_guarantee"]


def test_missing_mask_or_score_fails_assessment_explicitly():
    a, b = distinct_masks()
    data = canonical([.9, .9], [a, b])
    data["masks"] = None
    with pytest.raises(ValueError, match="aligned"):
        assess_ambiguity(data)
    data = canonical([.9, .9], [a, b])
    data["scores"] = [float("nan"), .9]
    with pytest.raises(ValueError, match="finite"):
        assess_ambiguity(data)
