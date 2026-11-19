# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""SAM3 in this process, behind the interface the RoboDojo tools segment through.

RPent runs SAM3 as a server of its own and hands its tools a client.  A host without RPent's
runtime (:mod:`.xpolicylab_robodojo`) loads the same model here: the official ``sam3``
package and its interactive point head, with the calls and settings RPent's server makes
(every candidate kept, bfloat16 autocast, the best-scoring instance returned), so one picture
and one prompt give one mask whichever of the two runs it.  ``segment`` takes exactly one of
a text prompt and a ``[row, col]`` pixel and answers like
``rpent.robots.components.sam3_client.Sam3Client``.
"""

from __future__ import annotations

import hashlib
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Segmented:
    """One mask, or none: what RPent's client returns."""

    found: bool
    score: float | None = None
    box: list[float] | None = None
    mask: np.ndarray | None = None
    reason: str | None = None


def _array(value: Any) -> np.ndarray:
    return value if isinstance(value, np.ndarray) else value.detach().float().cpu().numpy()


def select_top(masks: Any, scores: Any, boxes: Any, min_score: float) -> Segmented:
    """The best-scoring candidate, if it reaches ``min_score`` and its mask is not empty."""

    if masks is None or scores is None:
        return Segmented(False, reason="SAM3 returned no candidate")
    masks, scores = _array(masks), _array(scores).reshape(-1)
    if masks.size == 0 or scores.size == 0:
        return Segmented(False, reason="SAM3 returned no candidate")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or masks.shape[0] != scores.shape[0]:
        raise RuntimeError(
            f"unexpected SAM3 candidate shapes: masks={masks.shape}, scores={scores.shape}"
        )
    index = int(np.argmax(scores))
    score = float(scores[index])
    box: list[float] | None = None
    if boxes is not None:
        boxes = _array(boxes)
        if boxes.ndim >= 2 and index < boxes.shape[0]:
            box = [float(value) for value in boxes[index].reshape(-1)[:4]]
    if score < min_score:
        return Segmented(
            False,
            score=score,
            box=box,
            reason=f"top score {score:.3f} is below min_score {min_score:.3f}",
        )
    mask = np.asarray(masks[index]) > 0
    if mask.ndim != 2 or not mask.any():
        return Segmented(False, score=score, box=box, reason="SAM3 returned an empty mask")
    return Segmented(True, score=score, box=box, mask=mask)


def select_all(masks: Any, scores: Any, boxes: Any, min_score: float) -> list[Segmented]:
    """Copy every nonempty candidate above the threshold, without picking a winner.

    This is an additional read-only image-analysis interface. The existing
    ``select_top``/``segment`` robot perception behavior is unchanged. Candidate
    masks may overlap; their number is not an object-count or identity guarantee.
    """
    if masks is None or scores is None:
        return []
    masks, scores = _array(masks), _array(scores).reshape(-1)
    if masks.size == 0 and scores.size == 0:
        return []
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or masks.shape[0] != scores.shape[0]:
        raise RuntimeError(f"unaligned SAM3 candidates: masks={masks.shape}, scores={scores.shape}")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise RuntimeError("SAM3 candidate scores must be finite probabilities")
    if boxes is not None:
        boxes = _array(boxes)
        if boxes.shape != (len(scores), 4) or not np.isfinite(boxes).all():
            raise RuntimeError("SAM3 candidate boxes must be finite aligned Nx4 pixel coordinates")
    height, width = masks.shape[1:]
    selected = []
    for index, score in enumerate(scores):
        mask = np.array(masks[index] > 0, dtype=bool, copy=True)
        if score < min_score or not mask.any():
            continue
        if boxes is None:
            rows, cols = np.nonzero(mask)
            box = [float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1)]
        else:
            box = [float(v) for v in boxes[index]]
            box = [max(0.0, min(width, box[0])), max(0.0, min(height, box[1])),
                   max(0.0, min(width, box[2])), max(0.0, min(height, box[3]))]
            if box[0] >= box[2] or box[1] >= box[3]:
                raise RuntimeError("SAM3 candidate box is empty or inverted within image bounds")
        selected.append(Segmented(True, score=float(score), box=box, mask=mask))
    return selected


def _device_index(cuda_device: int | None) -> int:
    """Resolve a host GPU ordinal against the launcher visibility mask.

    A single visible GPU is authoritative (including UUID/MIG masks). For a
    multi-GPU ordinal mask, map the requested host ordinal to its local index.
    Never rewrite visibility after another component may have initialized CUDA.
    """
    if cuda_device is not None and cuda_device < 0:
        raise ValueError("cuda_device must be nonnegative")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return 0 if cuda_device is None else cuda_device
    devices = [part.strip() for part in visible.split(",") if part.strip()]
    if not devices or devices == ["-1"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES exposes no GPU for SAM3")
    if len(devices) == 1 or cuda_device is None:
        return 0
    if str(cuda_device) not in devices:
        raise ValueError(f"SAM3 GPU {cuda_device} is not in CUDA_VISIBLE_DEVICES={visible!r}")
    return devices.index(str(cuda_device))


class LocalSam3:
    """SAM3 loaded on one GPU of this process."""

    def __init__(self, checkpoint: str | None = None, cuda_device: int | None = None) -> None:
        checkpoint = checkpoint or os.environ.get("SAM3_CHECKPOINT_PATH")
        if not checkpoint:
            raise RuntimeError("SAM3_CHECKPOINT_PATH is not set")
        resolved = Path(checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {resolved}")
        device_index = _device_index(cuda_device)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "SAM3 in this process needs torch and the sam3 package "
                "(https://github.com/facebookresearch/sam3)"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 in this process needs a CUDA-capable GPU")
        self._device = torch.device("cuda", device_index)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with torch.cuda.device(self._device), warnings.catch_warnings():
            try:
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model
            except ImportError as exc:
                raise RuntimeError("SAM3 in this process needs the sam3 package") from exc
            warnings.simplefilter("ignore")
            model = build_sam3_image_model(
                device="cuda",
                checkpoint_path=str(resolved),
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
            # The upstream builder accepts 'cuda', so its current-device context
            # selects the card. The processor must also retain an indexed device.
            self._processor = Sam3Processor(model, device=self._device, confidence_threshold=0.0)
        self._torch = torch
        self._model = model
        # every candidate is kept: the caller's min_score decides
        self._lock = threading.Lock()
        self._digest: str | None = None
        self._state: dict[str, Any] | None = None

    def _autocast(self) -> Any:
        return self._torch.autocast("cuda", dtype=self._torch.bfloat16)

    def _state_for(self, pixels: np.ndarray) -> dict[str, Any]:
        """The image features, computed once for as many prompts as are put to one picture."""

        from PIL import Image

        digest = hashlib.sha256(repr(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        if digest != self._digest or self._state is None:
            with self._autocast():
                self._state = self._processor.set_image(Image.fromarray(pixels, mode="RGB"))
            self._digest = digest
        return self._state

    def segment(
        self,
        image: Any,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> Segmented:
        has_text = isinstance(text_prompt, str) and bool(text_prompt.strip())
        if has_text == (point is not None):
            raise ValueError("provide exactly one of text_prompt or point")
        if point is not None and (not isinstance(point, (list, tuple)) or len(point) != 2):
            raise ValueError("point must be [row, col]")
        try:
            min_score = float(min_score)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"min_score must be a number, got {min_score!r}") from exc
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        pixels = np.ascontiguousarray(np.asarray(image)[..., :3].astype(np.uint8))
        with self._lock, self._torch.cuda.device(self._device), warnings.catch_warnings():
            # the model runs inside the program's own process here: what it warns about (a
            # skipped post-processing kernel on a new GPU) is not the program's to read
            warnings.simplefilter("ignore")
            state = self._state_for(pixels)
            if has_text:
                assert text_prompt is not None
                with self._autocast():
                    out = self._processor.set_text_prompt(prompt=text_prompt.strip(), state=state)
                return select_top(out.get("masks"), out.get("scores"), out.get("boxes"), min_score)
            assert point is not None
            row, col = int(point[0]), int(point[1])
            height, width = int(state["original_height"]), int(state["original_width"])
            if row < 0 or col < 0 or row >= height or col >= width:
                raise ValueError(
                    f"point [row, col] {list(point)} is outside image shape [{height}, {width}]"
                )
            if getattr(self._model, "inst_interactive_predictor", None) is None:
                raise RuntimeError("SAM3 instance interactivity is not enabled")
            with self._autocast():
                masks, scores, _ = self._model.predict_inst(
                    state,
                    point_coords=np.asarray([[col, row]], dtype=np.float32),
                    point_labels=np.asarray([1], dtype=np.int64),
                    multimask_output=True,
                )
            return select_top(masks, scores, None, min_score)

    def segment_all(
        self, image: Any, *, text_prompt: str, min_score: float = 0.2,
    ) -> list[Segmented]:
        """Return all qualifying text-prompted masks from the shared processor.

        Uses the same checkpoint, image cache, lock and CUDA device as robot
        ``segment``. It does not modify presence scoring, perform overlap
        suppression, infer geometry or query the robot. Returned arrays are
        copies, so later calls cannot mutate a saved result.
        """
        if not isinstance(text_prompt, str) or not text_prompt.strip():
            raise ValueError("text_prompt must be a nonempty string")
        if isinstance(min_score, bool):
            raise ValueError("min_score must be a finite number between 0 and 1")
        try:
            min_score = float(min_score)
        except (TypeError, ValueError) as exc:
            raise ValueError("min_score must be a finite number between 0 and 1") from exc
        if not np.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("min_score must be a finite number between 0 and 1")
        pixels = np.asarray(image)
        if pixels.ndim != 3 or pixels.shape[2] < 3 or not pixels.shape[0] or not pixels.shape[1]:
            raise ValueError("image must be a nonempty HxWx3 RGB array")
        pixels = np.ascontiguousarray(pixels[..., :3].astype(np.uint8))
        with self._lock, self._torch.cuda.device(self._device), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state = self._state_for(pixels)
            with self._autocast():
                out = self._processor.set_text_prompt(prompt=text_prompt.strip(), state=state)
            return select_all(out.get("masks"), out.get("scores"), out.get("boxes"), min_score)


_SHARED: dict[tuple[str, str | None, int], LocalSam3] = {}
_SHARED_LOCK = threading.Lock()


def shared(checkpoint: str | None = None, cuda_device: int | None = None) -> LocalSam3:
    """The process's one :class:`LocalSam3` for a checkpoint and the cards it may see.

    A policy server plays one episode after another: the model is loaded for the first and
    serves the rest (a second copy would cost its load time and the card its memory).
    """

    url = os.environ.get("PYRUALEAN_SAM3_URL")
    if url:
        from .sam3_service import RemoteSam3
        return RemoteSam3(url)
    path = checkpoint or os.environ.get("SAM3_CHECKPOINT_PATH") or ""
    key = (str(Path(path).expanduser().resolve()) if path else "", os.environ.get("CUDA_VISIBLE_DEVICES"),
           _device_index(cuda_device))
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = LocalSam3(checkpoint, cuda_device)
        return _SHARED[key]


__all__ = ["LocalSam3", "Segmented", "select_top", "select_all", "shared"]
