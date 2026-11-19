"""Scoped, public-feedback execution guard; an independent V17 intervention.

Self-authored adaptation, reusing PyRUA's ExecutionFeedback boundary hooks.
ProgPrompt is a conceptual reference, not a source-code dependency or reproduction.
"""
from dataclasses import asdict, is_dataclass

from .action_feedback import ActionFeedback, ExecutionFeedback, RecoveryRequired
from .ablation_v17 import (
    _FusionControl, _PoSControl, effective_config,
    validate_options as validate_base,
)

KIND = "v18_feedback_guard"
PROMPT = """
# Scoped execution feedback
Use the existing public robot API. Declare step.phase='grasp' only for object
acquisition, or use `with robo.action_scope('grasp'):` around acquisition.
In that scope, CLOSE holding=False invalidates the grasp prerequisite; do not
continue lifting/carrying that object. In a move/grasp scope, reached=False
invalidates arrival. An explicit IK error invalidates the requested motion in
every scope. Only explicit steps_used=0 supports a no-execution interpretation.
Use `with robo.action_scope('contact'):` for intentional pressing/insertion;
blocked/reached=False can be expected contact. Neither proves task completion.
Unscoped empty closing is ambiguous and is not classified as failed grasp.
holding=True reports occupancy, not object identity or successful placement.
If recovery_required is returned, previous actions/variables remain. Do not
replay the entire cell. Read current feedback, reobserve/relocalize the affected
object when necessary, and change the failed pose/path/orientation/arm before a
new attempt in a NEW python call. Never catch a stop to continue same-cell motion.
The execution-check intervention may be active or passive in this experiment;
write the same correct conditional code in either case. Existing Python if/
assert checks are encouraged. All original task obligations and audits remain.
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
        self.events.append({"scoped_failure": packet, "gate_enabled": self.enabled})
        if self.enabled:
            self.blocked = packet
            raise RecoveryRequired(reason + "; redecide in a new python call; do not replay")


def validate_options(config):
    if config.get("kind") != KIND or type(config.get("guard_enabled", True)) is not bool:
        raise ValueError("v18_feedback_guard requires boolean guard_enabled")
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
        self.provenance["v18_feedback_guard"] = {
            "enabled": guard_enabled, "author_code_port": False,
            "reused_component": "PyRUA ExecutionFeedback primitive boundary hooks",
            "conceptual_reference": "ProgPrompt assertion/recovery",
            "new_model_calls": 0, "automatic_sensing_calls": 0,
            "automatic_retries": 0, "hidden_state_access": False,
            "dependency_scope": "same-cell conservative stop, not a causal graph",
        }

    def prompt(self):
        return super().prompt() + PROMPT

    def execute_python(self, args, execute):
        # Only explicit metadata sets intent. Never inherit a previous cell's grasp.
        raw = args.get("step") if isinstance(args, dict) else None
        phase = raw.get("phase") if isinstance(raw, dict) else None
        self.feedback_guard.intent = {"grasp": "grasp", "transport": "move"}.get(
            phase, "unspecified")
        return super().execute_python(args, execute)


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
    harness._log("v18_feedback_guard_configured",
                 config=harness.provenance["v18_feedback_guard"])
    return harness
