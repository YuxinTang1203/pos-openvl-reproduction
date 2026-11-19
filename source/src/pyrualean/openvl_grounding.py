"""Original OpenVL grounding capability backed by PyRUA's shared local SAM3.

The native capability still owns schema validation, memory registration and
model-visible evidence. This client replaces only the HTTP segmentation backend:
it returns every qualifying mask, pixel box and score plus the upstream numbered
visualization. No robot API, model-generation request or world geometry is used.

Unlike OpenVL's standalone SAM server, this shared backend preserves PyRUA's
processor scoring and does not install a presence-token monkey patch or remove
overlapping proposals. Outputs are candidate regions, not verified object counts.
"""

import base64
import io

import numpy as np
from PIL import Image


def _render_candidates(image, boxes, masks, rles):
    """Use the original CPU Visualizer without importing its GPU-loading server."""
    from openvlharness.services.tools.sam3_segment.viz.visualizer import Visualizer

    visualizer = Visualizer(
        np.asarray(image), font_size_multiplier=1.2, border_width_multiplier=0.0,
    )
    palette = [(0.15, 0.75, 0.95), (0.95, 0.4, 0.2), (0.5, 0.9, 0.3),
               (0.85, 0.35, 0.9), (0.95, 0.85, 0.25), (0.3, 0.5, 0.95)]
    visualizer.overlay_instances(
        boxes=np.asarray(boxes), masks=rles, binary_masks=masks,
        assigned_colors=[palette[i % len(palette)] for i in range(len(boxes))],
        alpha=0.15, label_mode="1",
    )
    out = io.BytesIO()
    Image.fromarray(visualizer.output.get_image()).save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


class LocalGroundingClient:
    """Inject into native ``Visual_Grounding_Tool`` as its tool client.

    ``segmenter`` may be an injected object implementing ``segment_all`` for
    CPU-only tests. Otherwise the same ``sam3_local.shared`` instance used by
    the robot is obtained lazily, on its existing checkpoint and CUDA device.
    """

    def __init__(self, segmenter=None, cuda_device=None, *, checkpoint=None, min_score=0.2):
        if isinstance(min_score, bool) or not np.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("min_score must be a finite number between 0 and 1")
        self.segmenter = segmenter
        self.cuda_device = cuda_device
        self.checkpoint = checkpoint
        self.min_score = float(min_score)

    def execute(self, tool_name, args, *, tool_call_id):
        from openvlharness.services.envelope import _build_success_payload
        from openvlharness.services.tools.image_utils import load_image_base64
        import pycocotools.mask as mask_utils

        if tool_name != "Visual_Grounding_Tool":
            raise ValueError(f"Unsupported local grounding tool: {tool_name!r}")
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty object name or noun phrase")
        # Native capability resolves registered image IDs into this inline field.
        # No fallback to a live camera or invented local image path is performed.
        image = load_image_base64(args.get("image_base64"))
        pixels = np.array(image, dtype=np.uint8, copy=True)
        if self.segmenter is None:
            from .hosts.sam3_local import shared
            self.segmenter = shared(self.checkpoint, cuda_device=self.cuda_device)
        if not callable(getattr(self.segmenter, "segment_all", None)):
            raise RuntimeError("The configured SAM3 backend lacks segment_all; local shared SAM3 is required")
        candidates = self.segmenter.segment_all(
            pixels, text_prompt=query.strip(), min_score=self.min_score,
        )
        width, height = image.size
        rows = []
        for candidate in candidates:
            mask = np.asarray(candidate.mask)
            box = candidate.box
            score = candidate.score
            if (not candidate.found or mask.shape != (height, width) or not mask.any()
                    or box is None or len(box) != 4 or not np.isfinite(box).all()
                    or score is None or not np.isfinite(score) or not 0 <= score <= 1
                    or not 0 <= box[0] < box[2] <= width
                    or not 0 <= box[1] < box[3] <= height):
                raise ValueError("SAM3 returned an invalid or unaligned pixel candidate")
            binary = np.asfortranarray(mask > 0, dtype=np.uint8)
            rle = mask_utils.encode(binary)
            rows.append(([float(v) for v in box], float(score), binary,
                         {"size": [height, width], "counts": rle["counts"].decode("ascii")}))
        # A stable left-to-right order keeps the visible numbers aligned with
        # boxes, scores and masks; it does not establish cross-frame identity.
        rows.sort(key=lambda item: (item[0][0], item[0][1]))
        boxes = [row[0] for row in rows]
        scores = [row[1] for row in rows]
        masks = [row[2] for row in rows]
        rles = [row[3] for row in rows]
        result = {
            "model": "sam3-image-pyrualean-shared", "prompt": query.strip(),
            "pred_boxes": boxes, "pred_scores": scores,
            "pred_masks": [rle["counts"] for rle in rles],
            "orig_img_h": height, "orig_img_w": width, "num_masks": len(rows),
            "rendered_image_b64": _render_candidates(image, boxes, masks, rles) if rows else "",
        }
        return _build_success_payload(tool_name, tool_call_id, result, meta={
            "execution": "local_shared_sam3", "min_score": self.min_score,
            "coordinates": "source-image pixel xyxy; no world geometry",
            "candidate_policy": "all nonempty candidates above threshold; overlaps retained",
            "processor_scoring": "unchanged PyRUA shared processor",
        })
