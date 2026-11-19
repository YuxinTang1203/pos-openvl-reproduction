"""Scoped, public-feedback execution guard; a small PoS/OpenVL prerequisite intervention.

Self-authored adaptation, reusing PyRUA's ExecutionFeedback boundary hooks.
ProgPrompt is a conceptual reference, not a source-code dependency or reproduction.
"""
from dataclasses import asdict, is_dataclass

from .action_feedback import ActionFeedback, ExecutionFeedback, RecoveryRequired
from .ablation_v17 import (
    _FusionControl, _PoSControl, effective_config,
    validate_options as validate_base,
)

KIND = "v19_prerequisites"
PROMPT = """
# Action prerequisites and local recovery
Declare object acquisition with step.phase='grasp' or
`with robo.action_scope('grasp'):`. Express dependencies in Python:

with robo.action_scope('grasp'):
    arrival = robo.move_to(target_xyz, arm='right')
    checked(arrival, require='reached')
    grasp = robo.set_gripper('right', robo.CLOSE)
    checked(grasp, require='holding')
    # Only now lift; verify the returned lift before further transport.

Do not close at a target after its approach failed. Do not lift/carry the intended
object after empty closing. End that sequence and decide recovery in a NEW Python
call. Previous effects and variables remain: never replay the whole sequence.
Explicit IK errors invalidate the requested motion in every scope. Only numeric
steps_used=0 supports no-execution. Read actual final pose, not requested pose.
Use action_scope('contact') for intentional pressing/insertion; reached=False
may be expected contact, so use task-appropriate feedback rather than blindly
checking arrival. holding=True is occupancy, not identity or placement success.
The failure_brief is historical public feedback, not a current-state oracle:
- IK rejection without motion: reconsider reachability/pose/arm; do not automatically
  discard unaffected object localization or repeat broad perception.
- Failed arrival: check final pose and the relevant approach/obstruction.
- Empty acquisition: verify the affected object's location and jaw alignment;
  acquire fresh observations when needed before another grasp.
Retain unrelated evidence only if its original validity still holds. The brief
does not refresh evidence, certify recovery or remove original task obligations.
The same prompt and scope syntax apply with the execution check on or off.
Always write explicit conditional code. Do not catch a failed check to continue
its dependent sequence. No automatic retry or sensing is supplied.
"""


class ScopedFeedback(ExecutionFeedback):
    """Latch same-cell motion after scoped failure, without automatic sensing.

The passive condition exposes identical scope syntax and feedback guidance.
No truth about objects, task success, or failure causes is manufactured here.
"""
    capture_failure_images = False

    def __init__(self, *, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.intent = "unspecified"
        self.latest_failure = None
        self.goal = None

    def begin_turn(self):
        super().begin_turn()
        self.intent = "unspecified"

    def observe(self, tool, arguments, result, index):
        ActionFeedback.observe(self, tool, arguments, result, index)
        data = asdict(result) if is_dataclass(result) else result
        if not isinstance(data, dict):
            return
        reason = None
        if data.get("ik_error"):
            reason = "ik_failure"
        elif self.intent in {"move", "grasp"} and data.get("reached") is False:
            reason = "arrival_not_reported"
        elif (self.intent == "grasp" and data.get("holding") is False
              and (data.get("commanded") == "close" or arguments.get("gripper") == 1)):
            reason = "empty_grasp"
        if reason is None:
            return
        packet = {
            "reason": reason, "tool": tool, "call_index": index,
            "intent": self.intent, "arguments": arguments, "feedback": data,
            "no_execution_reported": bool(data.get("ik_error"))
                and type(data.get("steps_used")) in (int, float)
                and data["steps_used"] == 0,
            "state": "No rollback or automatic retry; use a new cell to decide recovery.",
            "task_success_inferred": False, "object_identity_inferred": False,
        }
        packet["goal"] = self.goal
        self.latest_failure = packet
        self.events.append({"scoped_failure": packet, "gate_enabled": self.enabled})
        if self.enabled:
            self.blocked = packet
            raise RecoveryRequired(reason + "; redecide in a new python call; do not replay")

    def before(self, *, tool=None, arguments=None):
        if self.blocked:
            self.events.append({"prerequisite_interception": {
                "tool": tool, "arguments": arguments,
                "source_call_index": self.blocked["call_index"],
                "reason": self.blocked["reason"],
            }})
        super().before(tool=tool, arguments=arguments)

    def brief(self):
        if self.latest_failure is None:
            return None
        from copy import deepcopy
        p = self.latest_failure
        scopes = {
            "ik_failure": ("requested motion", "reachability / pose / arm"),
            "arrival_not_reported": ("arrival at requested pose", "actual final pose / approach"),
            "empty_grasp": ("acquisition of intended object", "object location / jaw alignment"),
        }
        invalid, review = scopes[p["reason"]]
        return {
            "source_call_index": p["call_index"], "goal": p["goal"],
            "reason": p["reason"], "intent": p["intent"],
            "arguments": deepcopy(p["arguments"]), "feedback": deepcopy(p["feedback"]),
            "invalidated_prerequisite": invalid, "review_focus": review,
            "no_execution_reported": p["no_execution_reported"],
            "historical_only": True, "recovery_verified": False,
            "evidence_validity_updated": False,
            "execution_check_enabled": self.enabled,
        }


def validate_options(config):
    if config.get("kind") != KIND or type(config.get("guard_enabled", True)) is not bool:
        raise ValueError("v19_prerequisites requires boolean guard_enabled")
    base = {k: v for k, v in config.items() if k != "guard_enabled"}
    base["kind"] = "v17_ablation"
    validate_base(base)


class _FeedbackMixin:
    def __init__(self, *args, guard_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_guard = ScopedFeedback(enabled=guard_enabled)
        # Keep action_scope's original type check and original primitive hooks.
        self.kit.robo._action_feedback = self.feedback_guard
        from .action_feedback import checked
        self.kit._globals["checked"] = checked
        self.provenance["v19_prerequisites"] = {
            "enabled": guard_enabled, "author_code_port": False,
            "reused_component": "PyRUA ExecutionFeedback primitive boundary hooks",
            "conceptual_reference": "ProgPrompt assertion/recovery",
            "new_model_calls": 0, "automatic_sensing_calls": 0,
            "automatic_retries": 0, "hidden_state_access": False,
            "dependency_scope": "same-cell conservative stop, not a causal graph",
            "failure_brief": "latest historical public failure; no evidence refresh",
        }

    def prompt(self):
        return super().prompt() + PROMPT

    def execute_python(self, args, execute):
        # Only explicit metadata sets intent. Never inherit a previous cell's grasp.
        raw = args.get("step") if isinstance(args, dict) else None
        phase = raw.get("phase") if isinstance(raw, dict) else None
        self.feedback_guard.goal = raw.get("goal") if isinstance(raw, dict) else None
        self.feedback_guard.intent = {"grasp": "grasp", "transport": "move"}.get(
            phase, "unspecified")
        return super().execute_python(args, execute)

    def after_tool(self, name, result, records):
        brief = self.feedback_guard.brief()
        if brief is not None:
            result.result["failure_brief"] = brief
        return super().after_tool(name, result, records)

    def context(self):
        context = super().context()
        brief = self.feedback_guard.brief()
        if brief is not None:
            context["failure_brief"] = brief
        return context


class _GuardedPoS(_FeedbackMixin, _PoSControl):
    pass


class _GuardedFusion(_FeedbackMixin, _FusionControl):
    pass


def build_harness(kit, config, task, initial_images=()):
    validate_options(config)
    base = {k: v for k, v in config.items() if k != "guard_enabled"}
    base["kind"] = "v17_ablation"
    cls = _GuardedPoS
    if config["profile"] != "pos":
        from .openvl_routed import OpenVLRoutedAdapter
        visual = config["openvl"]
        kit.evidence_adapter = OpenVLRoutedAdapter(
            kit, config=visual["config"], initial_images=initial_images,
            ambiguity_score_margin=visual.get("ambiguity_score_margin", .10),
            duplicate_mask_iou=visual.get("duplicate_mask_iou", .90))
        cls = _GuardedFusion
    harness = cls(kit, effective_config(base), task, initial_images,
                  static_rules=config.get("static_rules", False),
                  guard_enabled=config.get("guard_enabled", True))
    harness._log("v19_prerequisites_configured",
                 config=harness.provenance["v19_prerequisites"])
    return harness
