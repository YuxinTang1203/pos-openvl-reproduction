"""Admit the existing final chain for a stale state rather than strand it.
No new budget, model role, task-specific branch or execution interlock.
"""
from .pos_v4 import EventSchedule


class FreshnessSchedule(EventSchedule):
    def decide(self, **kw):
        normal = super().decide(**kw)
        if (not normal["due"] and self.attempted
                and normal["trigger"] == "stale_state_checkpoint"
                and normal["reserve_s"] > 0
                and normal["reason"] == "auxiliary_or_final_chain_reserve"
                and normal["auxiliary_remaining_s"] >= kw["minimum"]
                and kw["remaining"] >= kw["minimum"]):
            # Reuse the original final-chain admission calculation. This does
            # not set the harness finish flag or relax its independent audit.
            fallback = super().decide(**{**kw, "finish": True})
            if fallback["due"]:
                fallback.update(trigger="stale_state_reserved_chain",
                                reason="reserved_chain_released_for_stale_state")
                return fallback
        return normal
