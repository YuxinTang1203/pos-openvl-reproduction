"""Opt-in cross-turn recovery using public OpenVL perception and robot returns."""
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass

from .action_feedback import ExecutionFeedback, RecoveryRequired

PROMPT = """
OpenVL cross-turn recovery is enabled. A failed primitive remains pending across
Python cells. Declare acquisition with robo.action_scope('grasp') and intentional
contact with robo.action_scope('contact'). Empty unscoped closure is ambiguous
and is only reported; blocked contact is not proof of task success. Ordinary movement and finish(success) are blocked until repaired.
After a failure, re-localize the affected object with locate/segment; choose a
new found, current evidence ID from evidence_find(). Then use:
with robo.recovery_attempt(reference, change='specific revised approach', prepare=False):
    robo.move_pose(...)  # exactly ONE retry of the failed tool, on the same arm
For an empty grasp retry grip(close) instead. Set prepare=True for a single
waypoint move, or opening an empty gripper; preparation never resolves failure.
Each attempt needs evidence captured AFTER the failure and current at entry.
A retry must change the failed arguments or follow successful preparation.
Only measured reached=True (motion) / holding=True (grasp) resolves the primitive
failure. This does NOT verify a button, placement, object identity or task success.
Missing evidence remains unknown. Failed contact with IK error still needs a
measured successful motion; expected blocked contact alone cannot resolve it.
Do not catch the guard and continue dependent actions. Re-observe between attempts.
"""


class OpenVLRecovery(ExecutionFeedback):
    persistent = True

    def __init__(self, adapter):
        super().__init__()
        self.adapter = adapter
        self.turn = 0
        self.failed_turn = -1
        self.attempt = None
        self.prepared = False
        self.old_refs = set()

    def begin_turn(self):
        self.turn += 1
        self.intent = 'move'

    def before(self, *, tool=None, arguments=None):
        if self.attempt and self.attempt['used']:
            raise RecoveryRequired('One primitive per recovery attempt; inspect its result first')
        if not self.blocked:
            return
        a = self.attempt
        if tool is None or self.turn == self.failed_turn or a is None or a['used']:
            raise RecoveryRequired('Pending failure: observe and use recovery_attempt in a new cell')
        args = arguments or {}
        original = self.blocked['arguments']
        if args.get('arm') != original.get('arm'):
            raise RecoveryRequired('Recovery must use the affected arm')
        if a['prepare']:
            opening_empty = (self.blocked['reason'] == 'empty_grasp' and tool == 'grip'
                             and args.get('gripper') == -1)
            if not (tool in {'move_to', 'move_pose'} and args.get('gripper', 0) == 0) and not opening_empty:
                raise RecoveryRequired('Preparation allows a waypoint without gripper change, or opening after empty grasp')
        else:
            if tool != self.blocked['tool']:
                raise RecoveryRequired('Retry the failed primitive; preparation is separate')
            if args == original and not self.prepared:
                raise RecoveryRequired('Change failed arguments or complete preparation first')
        a['used'] = True

    @contextmanager
    def recovery_attempt(self, reference, *, change, prepare=False):
        if not self.blocked or self.attempt or self.turn == self.failed_turn:
            raise RecoveryRequired('Recovery requires a pending failure from an earlier cell')
        if type(prepare) is not bool or not isinstance(change, str) or not change.strip():
            raise ValueError('Provide a concrete change and boolean prepare')
        bridge = self.adapter.bridge
        bridge.require_reference(reference)
        if reference in self.old_refs or not bridge.validity(reference)['usable']:
            raise RecoveryRequired('Capture new current target evidence after the failure')
        if bridge.raw(reference).get('found') is not True:
            raise RecoveryRequired('Recovery needs a found target; missing evidence is unknown')
        self.attempt = {'prepare': prepare, 'used': False, 'reference': reference, 'change': change}
        self.adapter.event('recovery_attempt', **self.attempt)
        try:
            with self.scope(self.blocked['intent']):
                yield
        finally:
            self.attempt = None

    def execution_error(self, tool, arguments, index, error):
        self.blocked = dict(reason='execution_error', tool=tool, arguments=arguments,
                            intent=self.intent, call_index=index, feedback={'error': error})
        self.failed_turn = self.turn
        self.old_refs = set(self.adapter.bridge.refs)
        self.prepared = False
        self.events.append({'recovery_required': self.blocked})
        self.adapter.event('recovery_pending', failure=self.blocked)

    def observe(self, tool, arguments, result, index):
        previous = self.blocked
        data = asdict(result) if is_dataclass(result) else result
        try:
            super().observe(tool, arguments, result, index)
        except RecoveryRequired:
            self.failed_turn = self.turn
            self.old_refs = set(self.adapter.bridge.refs)
            self.prepared = False
            self.adapter.event('recovery_pending', failure=self.blocked)
            raise
        if previous and self.attempt and isinstance(data, dict):
            if self.attempt['prepare']:
                if data.get('reached') is True or data.get('opening') == 1.0:
                    self.prepared = True
            else:
                key = 'holding' if (previous['reason'] == 'empty_grasp' or
                                    (previous['tool'] == 'grip' and
                                     previous['arguments'].get('gripper') == 1)) else 'reached'
                if data.get(key) is True:
                    self.blocked = None
                    self.prepared = False
                    self.events.append({'recovery_resolved': {'call_index': index, 'criterion': key}})
                    self.adapter.event('recovery_resolved', call_index=index, criterion=key)


def install(kit):
    kit.enable_action_feedback()
    guard = OpenVLRecovery(kit.evidence_adapter)
    kit.robo._action_feedback = guard
    kit.robo.recovery_attempt = guard.recovery_attempt
    kit.evidence_adapter.provenance['options']['cross_turn_recovery'] = True
    import json
    (kit.evidence_adapter.root / 'provenance.json').write_text(
        json.dumps(kit.evidence_adapter.provenance, indent=2) + '\n')
    return guard
