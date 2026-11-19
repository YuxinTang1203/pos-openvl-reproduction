"""Public-feedback dependencies; no perception, evaluator, or robot replay.

Facts describe API reports, not semantic grasp/insertion success. Dependencies
are executor declarations, not host-verified causal relationships.
"""
from __future__ import annotations

import copy
from collections import deque

ARMS = ("left", "right", "single")
FACTS = tuple(f"{kind}:{arm}" for kind in ("holding", "arrival", "geometry") for arm in ARMS)
DEPENDENCY_SCHEMA = {
    "type": "array", "maxItems": 8,
    "description": "Optional dependencies for this cell. Declarations are not verified facts.",
    "items": {"type": "object", "additionalProperties": False,
              "properties": {"fact": {"type": "string", "enum": list(FACTS)},
                             "belief_ids": {"type": "array", "maxItems": 8,
                                            "uniqueItems": True, "items": {"type": "string"}}},
              "required": ["fact", "belief_ids"]},
}


class FailureDependencies:
    def __init__(self):
        self.facts = {}
        self.intents = deque(maxlen=24)
        self.events = deque(maxlen=24)
        self.warnings = []
        self.cell = None
        self.serial = 0
        self.pending_boundary = None
        self.last_rejected_request = {}

    def begin(self, cell, step, belief):
        self.cell = cell
        self.pending_boundary = None  # Cell boundary permits deliberate recovery.
        self.warnings = []
        step = step if isinstance(step, dict) else {}
        bindings = step.get("dependencies", [])
        world_ids = {r["id"] for kind in ("states", "relations")
                     for r in belief["world"][kind]}
        try:
            if not isinstance(bindings, list) or len(bindings) > 8:
                raise ValueError("dependencies must be a list of at most eight bindings")
            for binding in bindings:
                if not isinstance(binding, dict) or set(binding) != {"fact", "belief_ids"}:
                    raise ValueError("each dependency needs fact and belief_ids")
                ids = binding["belief_ids"]
                if binding["fact"] not in FACTS or not isinstance(ids, list) or len(ids) > 8:
                    raise ValueError("unknown fact or invalid belief_ids")
                if any(not isinstance(i, str) or i not in world_ids for i in ids):
                    raise ValueError("belief_ids must name existing physical states/relations")
                if len(set(ids)) != len(ids):
                    raise ValueError("duplicate belief_ids")
        except (ValueError, TypeError, KeyError) as exc:
            self.warnings.append(str(exc))
            bindings = []  # Bad bookkeeping never resubmits or rejects robot code.
        self.intents.append({"cell": cell, "goal": step.get("goal"),
                             "phase": step.get("phase"), "bindings": copy.deepcopy(bindings),
                             "dependency_origin": "executor declaration, not causal proof"})

    @staticmethod
    def signature(parameters):
        """Exact submitted request, not a test of kinematic equivalence."""
        import json
        return json.dumps(parameters, sort_keys=True, default=str, separators=(",", ":"))

    def before_action(self, method, parameters):
        arm = parameters.get("arm", "single")
        rejected = self.last_rejected_request.get(arm)
        return {"same_cell_after_failure": self.pending_boundary is not None,
                "failure_event_id": self.pending_boundary,
                "exact_rejected_request_repeated": bool(
                    rejected and method == "move_to" and
                    rejected["signature"] == self.signature(parameters)),
                "comparison_scope": "exact supplied parameters; omitted orientation and scene may change"}

    def observe(self, record, *, close, open_, cell, request_signature=None):
        """Consume one real public ledger record, in original execution order."""
        s, p = record.summary, record.kwargs
        arm = p.get("arm", "single")
        if arm not in (*ARMS, "both") or not record.stateful:
            return None
        source = {"call": record.index, "cell": cell, "tool": record.tool}
        updates, reason, question = {}, None, None
        if record.error or arm == "both":
            for affected_arm in ARMS[:2] if arm == "both" else (arm,):
                for kind in ("holding", "arrival", "geometry"):
                    updates[f"{kind}:{affected_arm}"] = "unknown"
            reason = "tool_error_unknown_effect" if record.error else "other_action_unknown_effect"
        elif record.tool == "grip":
            if p.get("gripper") == open_:
                updates[f"holding:{arm}"] = "released_by_command"
                updates[f"geometry:{arm}"] = "unknown"
                reason = "release"  # OPEN/holding=False is not an empty grasp.
            elif p.get("gripper") == close:
                holding = s.get("holding")
                updates[f"holding:{arm}"] = (
                    "reported_holding" if holding is True else
                    "contradicted" if holding is False else "unknown")
                updates[f"geometry:{arm}"] = "unknown"
                if holding is False:
                    reason = "empty_grasp"
                    question = "Where is the intended target relative to the gripper now?"
                elif holding is True:
                    reason = "holding_report"
        elif record.tool in {"move_pose", "move_to"}:
            steps = s.get("steps_used")
            # Explicit IK feedback only certifies non-execution with explicit zero steps.
            refused = bool(s.get("ik_error")) and type(steps) in (int, float) and steps == 0
            if refused:
                updates[f"arrival:{arm}"] = "contradicted"
                reason = "ik_refused_no_execution"
                question = "Which reported pose, orientation, path or arm change addresses this refusal?"
                self.last_rejected_request[arm] = {
                    "signature": request_signature or self.signature(p), "call": record.index}
            else:
                # Executed/unknown outcomes can change contact and retention. No collision diagnosis.
                updates[f"geometry:{arm}"] = "unknown"
                updates[f"arrival:{arm}"] = (
                    "reported_reached" if s.get("reached") is True else "unknown")
                if p.get("gripper") == open_:
                    updates[f"holding:{arm}"] = "released_by_command"
                elif f"holding:{arm}" in self.facts:
                    updates[f"holding:{arm}"] = "unknown"
                if s.get("blocked") or s.get("reached") is False or s.get("ik_error"):
                    reason = "partial_or_unknown_motion"
                    question = "Is the intended object still held, and what is its current relation to the target?"
                elif s.get("reached") is True:
                    reason = "arrival_report"
                self.last_rejected_request.pop(arm, None)
        else:
            # Unhandled native actions cannot preserve a certified retention/geometry estimate.
            updates[f"geometry:{arm}"] = "unknown"
            updates[f"holding:{arm}"] = "unknown"
            updates[f"arrival:{arm}"] = "unknown"
            reason = "other_action_unknown_effect"
        if not updates:
            return None
        for fact, status in updates.items():
            self.facts[fact] = {"status": status, "source": copy.deepcopy(source),
                                "physical_success_verified": False}
        if reason is None:
            reason = "unknown_grip_outcome"
        review_required = reason in {"empty_grasp", "ik_refused_no_execution", "partial_or_unknown_motion",
                                     "tool_error_unknown_effect"}
        invalidated = {k for k, v in updates.items()
                       if v in {"contradicted", "unknown", "released_by_command"}}
        affected = []
        for intent in self.intents:
            hits = [b for b in intent["bindings"] if b["fact"] in invalidated]
            if hits:
                affected.append({"cell": intent["cell"], "goal": intent["goal"],
                                 "facts": [b["fact"] for b in hits],
                                 "belief_ids": sorted({i for b in hits for i in b["belief_ids"]})})
        self.serial += 1
        event = {"id": f"dependency-{self.serial}", "reason": reason,
                 "review_required": review_required, "source": source, "arm": arm,
                 "public_feedback": copy.deepcopy(s), "updates": updates,
                 "affected_declared_dependencies": affected,
                 "observation_question": question,
                 "task_success_inferred": False, "identity_inferred": False}
        self.events.append(event)
        if review_required:
            self.pending_boundary = event["id"]
        return copy.deepcopy(event)

    def context(self):
        recent = [{k: copy.deepcopy(v) for k, v in event.items() if k != "public_feedback"}
                  for event in list(self.events)[-3:]]
        intent = self.intents[-1] if self.intents else None
        prerequisites = [{"fact": binding["fact"],
                          "report": copy.deepcopy(self.facts.get(binding["fact"])),
                          "satisfied_by_host": None}
                         for binding in intent["bindings"]] if intent else []
        return {"facts": copy.deepcopy(self.facts), "recent_events": recent,
                "current_intent": copy.deepcopy(intent), "declared_prerequisites": prerequisites,
                "metadata_warnings": list(self.warnings), "complete_history": False,
                "semantics": "Source-stamped API reports and declared dependency invalidations, not world truth. "
                             "Observation alone never clears a contradiction; a later action report may supersede it. "
                             "reported_holding does not identify the object; reported_reached is not task completion."}
