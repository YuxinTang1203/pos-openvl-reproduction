"""Opt-in robot evidence, shared by harness adapters; no simulator-private state."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass


def checked(result, *, require="reached"):
    """Keep dependent steps in one Python cell; raise on an explicit failed check.

    Use require='holding' after a grasp. Contact motions (e.g. pressing) need
    their own evidence check and should not blindly require reached=True.
    """
    if require not in {"reached", "holding", "found"}:
        raise ValueError("require must be reached, holding or found")
    value = result.get(require) if isinstance(result, dict) else getattr(result, require, None)
    if value is not True:
        raise RuntimeError(f"Robot check {require}=True failed: {result!r}")
    return result


class ActionFeedback:
    """Observe public primitive returns even when a cell forgets to print them."""

    def __init__(self):
        self.events = []

    def observe(self, tool, arguments, result, index):
        data = asdict(result) if is_dataclass(result) else result
        if not isinstance(data, dict):
            return
        reason = None
        if data.get("reached") is False:
            reason = "target_not_reached"
        # An empty closed gripper can be intentional (pressing); report evidence,
        # never automatically stop the next action or assert task failure.
        if data.get("holding") is False and (
            data.get("commanded") == "close" or arguments.get("gripper") == 1
        ):
            reason = "closed_without_holding"
        if reason:
            keys = (
                "reached",
                "holding",
                "opening",
                "width_m",
                "dist_to_target_m",
                "final_dist_m",
                "final_xyz",
                "ik_error",
                "blocked",
                "note",
                "steps_used",
            )
            self.events.append(
                {
                    "call_index": index,
                    "tool": tool,
                    "arguments": arguments,
                    "observation": reason,
                    "feedback": {k: data[k] for k in keys if k in data},
                }
            )

    def since(self, start):
        events = self.events[start:]
        if not events:
            return None
        return {
            "events": events[-8:],
            "omitted_events": max(0, len(events) - 8),
            "guidance": "These are primitive observations, not task verdicts. Inspect full "
            "feedback before dependent actions. Use checked(result) for movements and "
            "checked(result, require='holding') for grasps inside a multi-step cell; "
            "contact tasks need separate verification. Do not equate reached with success.",
        }


RECOVERY_INSTRUCTIONS = """
Execution checks are enabled at each robot primitive boundary. Keep multi-action
Python cells, loops and helpers. Ordinary failed motion aborts dependent actions;
state and Python variables are NOT rolled back. Recovery happens in a NEW code call.
Use `with robo.action_scope('grasp'):` around acquisition: empty close aborts.
Use `with robo.action_scope('contact'):` ONLY for intentional pressing/insertion:
blocked/reached=False may be expected contact, but IK errors still abort.
Contact scope is not proof of success: inspect returned feedback and verify the
physical result before advancing the task. Unscoped empty close is ambiguous and
reported, not treated as failed grasp. On recovery_required inspect current images,
re-localize relevant objects, change the failed pose/grasp, and retry ONLY the
failed stage. Do not replay completed transfers. Never catch an execution stop to
continue motion in the same cell. VISTA history, when available, can compare the
pre-failure view with the current view; historical coordinates are not current.
"""


class RecoveryRequired(RuntimeError):
    """A primitive failed; further motion in this code call is blocked."""


class ExecutionFeedback(ActionFeedback):
    """Primitive-boundary guard using only public results, with explicit intent."""

    def __init__(self):
        super().__init__()
        self.blocked = None
        self.intent = "move"

    def begin_turn(self):
        self.blocked = None
        self.intent = "move"

    def before(self, *, tool=None, arguments=None):
        if self.blocked:
            raise RecoveryRequired(self.blocked["reason"] + "; recover in a new code call")

    def scope(self, intent):
        from contextlib import contextmanager

        if intent not in {"move", "grasp", "contact"}:
            raise ValueError("intent must be move, grasp or contact")

        @contextmanager
        def scoped():
            previous = self.intent
            self.intent = intent
            try:
                yield
            finally:
                self.intent = previous

        return scoped()

    def observe(self, tool, arguments, result, index):
        super().observe(tool, arguments, result, index)
        data = asdict(result) if is_dataclass(result) else result
        if not isinstance(data, dict):
            return
        reason = None
        if data.get("ik_error"):
            reason = "ik_failure"
        elif self.intent != "contact" and (data.get("reached") is False or data.get("blocked")):
            reason = "motion_failed"
        elif (
            self.intent == "grasp"
            and data.get("holding") is False
            and (data.get("commanded") == "close" or arguments.get("gripper") == 1)
        ):
            reason = "empty_grasp"
        if not reason:
            return
        recovery = {
            "ik_failure": (
                "Inspect current pose; change approach/orientation or waypoint. "
                "Do not close or insert after a rejected pose."
            ),
            "motion_failed": (
                "Inspect obstruction and final pose; re-localize target and choose "
                "a reachable approach before retrying this motion."
            ),
            "empty_grasp": (
                "Re-localize the object; check jaw opening and grasp alignment, "
                "then retry acquisition before lifting or transferring."
            ),
        }
        self.blocked = {
            "reason": reason,
            "tool": tool,
            "call_index": index,
            "intent": self.intent,
            "arguments": arguments,
            "feedback": data,
            "next_step": recovery[reason],
            "state": (
                "No rollback. Earlier actions and Python variables remain; "
                "remaining motion in this code call is blocked."
            ),
        }
        # Include the mandatory recovery packet in action_feedback so native
        # HarnessPro output compaction retains it as well.
        self.events.append({"recovery_required": self.blocked})
        self.before()
