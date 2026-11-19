"""Real native capability/renderer with scripted multi-candidate SAM outputs."""

import base64
import contextlib
import io
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("openvlharness")
mask_utils = pytest.importorskip("pycocotools.mask")

from openvlharness.capabilities.grounding import Capability
from openvlharness.contracts import VirtualCall
from openvlharness.memory.store import MemoryStore

from pyrualean.hosts import sam3_local
from pyrualean.openvl_grounding import LocalGroundingClient


def scene():
    pixels = np.zeros((60, 100, 3), dtype=np.uint8)
    pixels[:] = [30, 70, 140]
    masks = np.zeros((3, 60, 100), dtype=bool)
    masks[0, 10:35, 60:90] = True
    masks[1, 20:45, 10:30] = True
    masks[2, 0:5, 0:5] = True
    boxes = np.array([[60, 10, 90, 35], [10, 20, 30, 45], [0, 0, 5, 5]])
    scores = np.array([0.95, 0.8, 0.1])
    return pixels, masks, boxes, scores


def data_url(pixels):
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii")


class FakeSam:
    def __init__(self, candidates=None):
        pixels, masks, boxes, scores = scene()
        self.candidates = candidates if candidates is not None else sam3_local.select_all(masks, scores, boxes, .2)
        self.calls = []

    def segment_all(self, image, *, text_prompt, min_score):
        self.calls.append((np.array(image, copy=True), text_prompt, min_score))
        return self.candidates


def test_native_capability_keeps_candidates_rle_alignment_and_visible_evidence():
    pixels, masks, boxes, scores = scene()
    sam = FakeSam()
    client = LocalGroundingClient(sam)
    memory = MemoryStore()
    memory.start([])
    ref = memory.register_image(data_url(pixels), provenance="fixture:camera:step=7")
    cap = Capability(backend="sam3_segment", client_config={}, client=client)
    result = cap.invoke(VirtualCall("ground-1", "Visual_Grounding_Tool", {
        "query": "blocks", "image_id": ref,
    }), memory)
    assert result.ok, result.error
    assert result.data["num_masks"] == 2
    assert result.data["boxes"] == [boxes[1].tolist(), boxes[0].tolist()]
    assert result.data["scores"] == [scores[1], scores[0]]
    assert result.data["boxes_normalized"] == [[100, 333, 300, 750], [600, 167, 900, 583]]
    assert len(sam.calls) == 1 and sam.calls[0][1:] == ("blocks", .2)
    np.testing.assert_array_equal(sam.calls[0][0], pixels)  # RGB remains RGB.
    for encoded, expected in zip(result.data["masks"], (masks[1], masks[0])):
        decoded = mask_utils.decode({"size": [60, 100], "counts": encoded})
        np.testing.assert_array_equal(decoded, expected)
    binding = next(a.id for a in result.artifact_refs if a.kind == "binding")
    stored = memory.copy_bindings_for_execution()[binding]
    assert stored["pred_boxes"] == result.data["boxes"]
    assert len(stored["pred_masks"]) == 2
    image_ref = result.data["rendered_image"]
    assert image_ref and image_ref != ref
    assert any(p["type"] == "image_url" for p in result.evidence)
    assert "number" in " ".join(p.get("text", "") for p in result.evidence)
    rendered_url, _, _ = memory.resolve_image(image_ref)
    overlay = np.array(Image.open(io.BytesIO(base64.b64decode(rendered_url.split(",", 1)[1]))))
    assert overlay.shape[:2] == pixels.shape[:2]
    assert not np.array_equal(overlay[..., :3], pixels)
    assert "world" not in result.data and "center_xyz" not in result.data


def test_empty_detection_is_available_evidence_not_execution_failure():
    pixels, *_ = scene()
    client = LocalGroundingClient(FakeSam([]))
    memory = MemoryStore()
    memory.start([])
    ref = memory.register_image(data_url(pixels))
    cap = Capability(backend="sam3_segment", client_config={}, client=client)
    result = cap.invoke(VirtualCall("empty", "Visual_Grounding_Tool", {
        "query": "unseen object", "image_id": ref,
    }), memory)
    assert result.ok and result.data["num_masks"] == 0
    assert result.data["boxes"] == result.data["masks"] == []
    assert result.data["rendered_image"] is None
    assert "alternative / simpler prompt" in result.evidence[0]["text"]


def test_shared_segmenter_is_lazy_and_reused(monkeypatch):
    pixels, *_ = scene()
    calls = []
    sam = FakeSam([])

    def shared(checkpoint, cuda_device):
        calls.append((checkpoint, cuda_device))
        return sam

    monkeypatch.setattr(sam3_local, "shared", shared)
    client = LocalGroundingClient(cuda_device=5, checkpoint="existing.pt")
    assert not calls
    for i in range(2):
        client.execute("Visual_Grounding_Tool", {"query": "blocks", "image_base64": data_url(pixels)}, tool_call_id=str(i))
    assert calls == [("existing.pt", 5)]
    assert len(sam.calls) == 2


def test_segment_all_keeps_distinct_and_overlapping_masks_without_changing_segment():
    pixels, masks, boxes, scores = scene()
    # Both existing robot segment and new candidate analysis use this one processor.
    model = sam3_local.LocalSam3.__new__(sam3_local.LocalSam3)
    model._lock = threading.Lock()
    model._torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    model._device = "cuda:0"
    model._autocast = contextlib.nullcontext
    cached_state = {}
    model._state_for = lambda image: cached_state
    seen = []

    def infer(*, prompt, state):
        assert state is cached_state
        seen.append(prompt)
        return {"masks": masks, "boxes": boxes, "scores": scores}

    model._processor = SimpleNamespace(set_text_prompt=infer)
    all_candidates = model.segment_all(pixels, text_prompt="blocks")
    top = model.segment(pixels, text_prompt="blocks")
    assert len(all_candidates) == 2 and seen == ["blocks", "blocks"]
    assert top.score == .95 and top.box == boxes[0].tolist()
    np.testing.assert_array_equal(top.mask, masks[0])
    all_candidates[0].mask[:] = False
    assert masks[0].any()  # Results don't alias the processor cache.
    duplicated = sam3_local.select_all(np.stack([masks[1], masks[1]]), np.array([.8, .7]), None, .2)
    assert len(duplicated) == 2  # No silent count-changing overlap suppression.


@pytest.mark.parametrize("masks,scores,boxes", [
    (np.ones((2, 5, 5)), np.array([.9]), None),
    (np.ones((1, 5, 5)), np.array([float("nan")]), None),
    (np.ones((1, 5, 5)), np.array([.9]), np.array([[0, 0, 5, float("inf")]])),
])
def test_malformed_candidates_fail_explicitly(masks, scores, boxes):
    with pytest.raises(RuntimeError):
        sam3_local.select_all(masks, scores, boxes, .2)


def test_invalid_image_query_and_unsupported_backend_do_not_fall_back():
    pixels, *_ = scene()
    sam = FakeSam([])
    client = LocalGroundingClient(sam)
    for args in ({"query": "", "image_base64": data_url(pixels)},
                 {"query": "blocks", "image_base64": "bad image"},
                 {"query": "blocks", "image_path": "/invented.png"}):
        with pytest.raises(ValueError):
            client.execute("Visual_Grounding_Tool", args, tool_call_id="invalid")
    assert not sam.calls
    client = LocalGroundingClient(SimpleNamespace(segment=lambda *a, **k: None))
    with pytest.raises(RuntimeError, match="segment_all"):
        client.execute("Visual_Grounding_Tool", {"query": "blocks", "image_base64": data_url(pixels)}, tool_call_id="unsupported")
