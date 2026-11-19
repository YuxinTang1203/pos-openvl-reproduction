"""Optional automatic visual grounding at PyRUA's text-perception boundary.

Only this adapter instance wraps locate/segment. Ambiguous queries raise before
the original top-one geometry is returned, so uncaught exceptions stop the
current Python cell. This is explicit adaptation behavior, not native OpenVL
recovery and not a safety boundary against Python that catches the exception.
"""
import functools
import inspect
import math

from .openvl_visual import OpenVLVisualAdapter


class PerceptionChoiceRequired(RuntimeError):
    """The current visual query needs an explicit instance selection."""


class PerceptionRoutingError(RuntimeError):
    """Automatic grounding failed; original top-one perception was not run."""


def validate_ambiguity_thresholds(score_margin, duplicate_mask_iou):
    """Validate generic routing heuristics; scores are not calibrated odds."""
    for name, value in (("ambiguity_score_margin", score_margin),
                        ("duplicate_mask_iou", duplicate_mask_iou)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number between 0 and 1")
    if duplicate_mask_iou == 0:
        raise ValueError("duplicate_mask_iou must be greater than zero")


def assess_ambiguity(data, *, score_margin=.10, duplicate_mask_iou=.90):
    """Group competitive masks without deleting or renumbering source proposals.

    Highest-score-first representatives merge proposals whose mask IoU reaches
    ``duplicate_mask_iou``. A chain of overlapping proposals does not collapse
    distinct representatives. Neither SAM scores nor this grouping establish
    target identity, correctness or calibrated uncertainty.
    """
    import numpy as np
    from pycocotools import mask as mask_utils

    validate_ambiguity_thresholds(score_margin, duplicate_mask_iou)
    boxes, scores, masks = data.get("boxes") or [], data.get("scores") or [], data.get("masks") or []
    if not (len(boxes) == len(scores) == len(masks)):
        raise ValueError("Grounding boxes/scores/masks must be aligned for ambiguity assessment")
    size = data.get("size")
    decoded, empty = {}, []
    for i, score in enumerate(scores):
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Grounding scores must be finite numbers between 0 and 1")
        mask = np.asarray(mask_utils.decode({"size": size, "counts": masks[i]}), dtype=bool)
        if mask.shape != tuple(size):
            raise ValueError("Grounding mask shape is not the source image shape")
        if mask.any():
            decoded[i] = mask
        else:
            empty.append(i + 1)
    top = max((scores[i] for i in decoded), default=None)
    competitive = [i for i in decoded if top - scores[i] <= score_margin + 1e-12]
    competitive.sort(key=lambda i: (-scores[i], i))
    groups = []
    for index in competitive:
        for group in groups:
            representative = group["representative_number"] - 1
            intersection = np.count_nonzero(decoded[index] & decoded[representative])
            union = np.count_nonzero(decoded[index] | decoded[representative])
            iou = intersection / union
            if iou >= duplicate_mask_iou:
                group["candidate_numbers"].append(index + 1)
                break
        else:
            groups.append({"representative_number": index + 1, "candidate_numbers": [index + 1]})
    return {"choice_required": len(groups) > 1, "top_score": top,
            "score_margin": score_margin, "duplicate_mask_iou": duplicate_mask_iou,
            "competitive_candidate_numbers": [i + 1 for i in competitive],
            "hypothesis_groups": groups, "empty_candidate_numbers": empty,
            "source_candidate_count": len(boxes),
            "reason": ("no_valid_candidates" if not groups else
                       "distinct_competitive_hypotheses" if len(groups) > 1 else "one_competitive_hypothesis"),
            "interpretation": "Uncalibrated score-gap/mask-overlap heuristic; not a guarantee of target identity or correctness. All original candidates remain available."}


class OpenVLRoutedAdapter(OpenVLVisualAdapter):
    def __init__(self, kit, *, config, initial_images, segmenter=None,
                 ambiguity_score_margin=.10, duplicate_mask_iou=.90):
        validate_ambiguity_thresholds(ambiguity_score_margin, duplicate_mask_iou)
        super().__init__(kit, config=config, initial_images=initial_images, segmenter=segmenter)
        self.ambiguity_score_margin = float(ambiguity_score_margin)
        self.duplicate_mask_iou = float(duplicate_mask_iou)
        self.routing_enabled = True
        self.automatic_grounding_calls = 0
        self.autonomous_tool_calls = 0
        self.choice_boundaries = 0
        self.pending_routes = []
        self._original_perception = {}
        self._routing_depth = 0
        self.provenance.update(
            mode="automatically_routed_visual_workspace",
            routing="locate/segment text queries; distinct score-competitive mask hypotheses require explicit choice",
            ambiguity_heuristic={"version": "score-gap-mask-iou-v1",
                                 "score_margin": self.ambiguity_score_margin,
                                 "duplicate_mask_iou": self.duplicate_mask_iou,
                                 "grouping": "highest-score-first representative masks; original candidates retained",
                                 "calibrated_probability_guarantee": False},
            semantic_change="Ambiguous text perception raises before original top-one geometry; no automatic replay",
            selection_guard="Ordinary catchable Python exception, not a global action interlock",
        )
        for name in ("locate", "segment"):
            original = getattr(kit.robo, name, None)
            if not callable(original):
                continue
            self._original_perception[name] = original
            setattr(kit.robo, name, self._wrapper(name, original))
        self.persist()

    def set_routing_enabled(self, enabled):
        """Harness control for exact prefix replay; not exposed as a model tool."""
        if type(enabled) is not bool:
            raise ValueError("routing enabled must be boolean")
        self.routing_enabled = enabled
        self.event("routing_enabled", enabled=enabled)

    def enable_routing(self):
        self.set_routing_enabled(True)

    def disable_routing(self):
        self.set_routing_enabled(False)

    def _original_call(self, original, args, kwargs):
        self._routing_depth += 1
        try:
            return original(*args, **kwargs)
        finally:
            self._routing_depth -= 1

    def _wrapper(self, name, original):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def routed(*args, **kwargs):
            if not self.routing_enabled or self._routing_depth:
                return self._original_call(original, args, kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            # Point prompts select an instance by pixel already. Retain the
            # original validation when point/text are both supplied illegally.
            if values.get("point") is not None:
                return self._original_call(original, args, kwargs)
            query = values.get("text_prompt", values.get("prompt"))
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Automatic grounding requires a nonempty text prompt")
            camera = values.get("camera", self.kit.robo.CAMERAS[0])
            if camera not in self.kit.robo.CAMERAS:
                raise ValueError("Unknown perception camera")
            if values.get("step", -1) != -1:
                raise PerceptionRoutingError("Text routing requires the current observation; inspect historical images with their frame IDs")
            try:
                min_score = float(values.get("min_score", .2))
            except (TypeError, ValueError) as exc:
                raise ValueError("min_score must be finite and between 0 and 1") from exc
            if not math.isfinite(min_score) or not 0 <= min_score <= 1:
                raise ValueError("min_score must be finite and between 0 and 1")
            result = self._ground_for_perception(name, query, camera, min_score)
            payload = result.result
            if payload.get("error"):
                payload["route_status"] = "grounding_error"
                raise PerceptionRoutingError(str(payload["error"]))
            candidates = payload.get("candidates")
            bindings = payload.get("binding_ids")
            if not isinstance(candidates, list) or not bindings:
                payload["route_status"] = "grounding_error"
                raise PerceptionRoutingError("Grounding returned no usable candidate schema/binding")
            try:
                assessment = assess_ambiguity(self.groundings[bindings[0]]["data"],
                    score_margin=self.ambiguity_score_margin, duplicate_mask_iou=self.duplicate_mask_iou)
            except Exception as exc:
                payload["route_status"] = "grounding_error"
                payload["error"] = f"Ambiguity assessment failed: {type(exc).__name__}: {exc}"
                raise PerceptionRoutingError(payload["error"]) from exc
            payload["ambiguity_assessment"] = assessment
            if assessment["choice_required"]:
                payload["route_status"] = "choice_required"
                self.choice_boundaries += 1
                self.event("perception_choice_required", primitive=name, binding_id=bindings[0], candidate_count=len(candidates))
                geometry_hint = (
                    "After selecting a candidate, prefer robo.locate_candidate(binding_id, candidate_number) "
                    "for its complete ObjectSight geometry. Check found, center_xyz, views and note before acting. "
                    "visual_candidate plus robo.back_project remains an alternate single-surface-pixel path. "
                    if self._supports_locate_candidate() else
                    "Choose a candidate with visual_candidate(binding_id, candidate_number), then use "
                    "robo.back_project and check its surface and uncertainty. "
                )
                raise PerceptionChoiceRequired(
                    f"{name} returned {len(assessment['hypothesis_groups'])} distinct competitive mask hypotheses "
                    f"among {len(candidates)} candidate regions (score-gap heuristic, not a correctness guarantee). "
                    f"Inspect the attached numbered image for {bindings[0]!r}. "
                    + geometry_hint +
                    "Original top-one geometry was not returned. Earlier actions already happened; do not replay them."
                )
            # No replacement typed result or geometry is fabricated. The
            # original primitive still supplies its own found/error/uncertainty.
            payload["route_status"] = "original_perception"
            return self._original_call(original, args, kwargs)

        return routed

    def _ground_for_perception(self, primitive, query, camera, min_score):
        from .arms import ArmResult

        self.automatic_grounding_calls += 1
        client = self.tools["Visual_Grounding_Tool"].client
        previous_threshold = client.min_score
        try:
            client.min_score = min_score
            result = super().invoke("Visual_Grounding_Tool", {"query": query, "camera": camera})
        except Exception as exc:
            result = ArmResult({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            client.min_score = previous_threshold
        result.result["routing"] = {"trigger": "automatic_text_perception", "primitive": primitive,
                                    "query": query, "camera": camera, "min_score": min_score}
        self.pending_routes.append(result)
        self.event("automatic_grounding", primitive=primitive, camera=camera, min_score=min_score,
                   ok=not bool(result.result.get("error")))
        return result

    def invoke(self, name, args):
        self.autonomous_tool_calls += 1
        self.event("autonomous_visual_tool", name=name)
        return super().invoke(name, args)

    def before_tool(self, *, name=None, args=None):
        super().before_tool(name=name, args=args)
        self.pending_routes = []

    def after_tool(self, result):
        # Register actual Python image outputs first; automatic numbered images
        # already have immutable source aliases and must not be re-registered as
        # fresh camera captures after actions in the same cell.
        result = super().after_tool(result)
        pending, self.pending_routes = self.pending_routes, []
        if pending:
            for row in pending:
                # Unambiguous perception may be followed by motion in the same
                # cell, or policy code may catch a choice exception and move.
                # Recompute delivery-time freshness without relabeling capture.
                source = row.result.get("source") or {}
                if source.get("image_id") in self.frames:
                    row.result["source"] = self.frame_context(source["image_id"])
                row.result["image_contexts"] = [self.frame_context(ref)
                    for ref in row.result.get("images", []) if ref in self.frames]
                self.event("automatic_perception_result", payload=row.result)
            # Keep unambiguous primitive feedback unchanged. Evidence is
            # archived for all routes, but the model only needs a new decision
            # when candidate selection or a grounding failure interrupted it.
            visible = [row for row in pending
                       if row.result.get("route_status") in {"choice_required", "grounding_error"}]
            if visible:
                result.result["routed_perception"] = [row.result for row in visible]
            for row in visible:
                result.images.extend(row.images)
                if row.result.get("images"):
                    result.result.setdefault("images", []).extend(row.result["images"])
        return result

    def prompt(self):
        return super().prompt() + """
Automatic visual routing is enabled for robo.locate/segment TEXT queries. They
automatically obtain OpenVL candidate evidence. A configurable score-gap/mask-IoU
heuristic requires a choice only when multiple distinct masks score close to the
highest candidate. Duplicate mask proposals count as one hypothesis for this
check; all original proposals and numbers remain available. This is not a
probability or identity guarantee. Other calls use the original primitive,
return type and feedback (visual evidence stays archived). Ambiguous calls raise
PerceptionChoiceRequired BEFORE returning top-one geometry. Unless caught, that
exception stops the current cell, while earlier actions and variables persist.
Inspect routed_perception and the attached numbered image to select the intended
instance. On the supported native RGB-only RoboDojo backend, prefer
robo.locate_candidate(binding_id, candidate_number): it keeps the selected mask
and returns the original full ObjectSight geometry (centre, top, size, yaw, views,
note). Check found, center_xyz, views and note before an action. For a single
visible surface point, visual_candidate plus back_project remains an alternate
path. Unsupported backends must use their documented geometry API; locate_candidate
raises explicitly and never silently substitutes a text top-one selection.
Do not replay earlier
actions or re-run the ambiguous text query expecting a different winner. A failed
Grounding call raises PerceptionRoutingError without silently using top one.
Point-based segment calls keep their original behavior. SAM regions may include
false positives; a choice boundary does not prove multiple real objects exist.
This automatic routing is a robot adaptation, not autonomous visual-tool use.
""" + f"Current heuristic: score gap <= {self.ambiguity_score_margin}; duplicate mask IoU >= {self.duplicate_mask_iou}.\n"

    def summary(self):
        return {**super().summary(), "routing_enabled": self.routing_enabled,
                "automatic_grounding_calls": self.automatic_grounding_calls,
                "autonomous_tool_calls": self.autonomous_tool_calls,
                "choice_boundaries": self.choice_boundaries}
