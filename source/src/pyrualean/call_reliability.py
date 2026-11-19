"""Bounded transport recovery and observed-latency allocation, not a robot policy."""
from collections import defaultdict, deque
from collections.abc import Mapping


class CallReliability:
    max_retries = 1

    def __init__(self):
        self.latencies = defaultdict(lambda: deque(maxlen=4))

    @staticmethod
    def retryable(code, payload):
        if code == 400:
            error = payload.get('error') if isinstance(payload, Mapping) else None
            return (isinstance(error, Mapping) and isinstance(error.get('message'), str)
                    and error['message'].strip().lower() == 'download multimodal file timed out')
        if code == 0:
            return str(payload).split(':', 1)[0] in {
                'ReadTimeout', 'ConnectTimeout', 'WriteTimeout', 'PoolTimeout',
                'TimeoutError', 'ConnectError', 'ReadError', 'WriteError',
                'RemoteProtocolError', 'ConnectionError', 'ConnectionResetError'}
        return code in (408, 409, 425, 429, 500, 502, 503, 504)

    def observe(self, role, latency, ok, timeout=False):
        if latency >= 0:
            self.latencies[role].append((float(latency), bool(ok), bool(timeout)))

    # A few observations cannot estimate a tail-latency percentile. Keep the
    # configured attempt timeout; record samples for diagnostics only.
    backoff_max_s = 3.0

    def target(self, role, nominal):
        return nominal

    def recovery_allowance(self, scheduled, available, round_limit, nominal, min_calls):
        # Reserve capacity, not spending. Actual elapsed time is still charged
        # once to the unchanged auxiliary/episode budgets. Never borrow the
        # scheduler's final-chain reserve or the planner's time.
        wanted = (min_calls + 1) * nominal + self.backoff_max_s + 1.0
        # One second for local payload setup between reservation and stage entry.
        hard = min(available, round_limit)
        return min(hard, max(scheduled, wanted))

    def allocate(self, role, left, followups, nominal):
        reserve = followups * nominal
        room = left - reserve
        if room + 1e-6 < nominal:
            return None
        retry_funded = room + 1e-6 >= 2 * nominal + self.backoff_max_s
        cap = 2 * nominal + self.backoff_max_s if retry_funded else nominal
        return {'stage_cap_s': cap, 'minimum_followup_s': reserve,
                'target_s': nominal, 'followup_target_s': nominal,
                'samples': len(self.latencies[role]), 'minimum_stage_s': nominal,
                'attempt_timeout_s': nominal, 'retry_capacity_reserved': retry_funded,
                'allocation_policy': 'fixed_attempt_full_audit_reserved_retry'}
