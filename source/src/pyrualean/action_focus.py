"""Advisory action-scoped gap indexing; never a world model or action gate.

SayCan inspires separating task usefulness from current feasibility. We have no
learned affordance/value function. Existing ProgressLite already adapts Inner
Monologue; this module adds no second reflection loop. Recipes are original,
pure-data examples inspired by Code as Policies, not transplanted robot skills.
"""
from __future__ import annotations

import copy
import json

FOCUS_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'next_action': {'type': 'string', 'minLength': 1, 'maxLength': 240},
        'needed_now': {'type': 'array', 'maxItems': 6, 'uniqueItems': True,
                       'items': {'type': 'string', 'maxLength': 160}},
        'deferred': {'type': 'array', 'maxItems': 6, 'uniqueItems': True,
                     'items': {'type': 'string', 'maxLength': 160}},
        'observation_question': {'type': ['string', 'null'], 'maxLength': 240},
        'answer_changes': {'type': ['string', 'null'], 'maxLength': 240},
    },
    'required': ['next_action', 'needed_now', 'deferred',
                 'observation_question', 'answer_changes'],
    'description': 'Optional advisory plan. Gap IDs are existing PoS epistemic IDs. '
        'Deferred gaps stay unresolved. This is not an execution prerequisite or a success claim.',
}

# These functions are displayed as optional recipes, never installed as tools or
# called by the host. No hidden state, numeric geometry, or robot action occurs.
FIELD_RECIPE = '''def read_public_fields(reply, names):
    if not isinstance(reply, dict):
        return {"values": {}, "missing": list(names)}
    return {"values": {k: reply[k] for k in names if k in reply and reply[k] is not None},
            "missing": [k for k in names if k not in reply or reply[k] is None]}'''

GRIP_RECIPE = '''def describe_grip_reply(reply, command, close_value, open_value):
    values = reply if isinstance(reply, dict) else {}
    event = "close" if command == close_value else "open" if command == open_value else "other"
    return {"command_event": event, "reported_holding": values.get("holding"),
            "reported_width": values.get("width"), "identity_verified": False}'''

FOCUS_PROMPT = '''
# Action-scoped gaps (advisory; existing PoS and feedback loop retained)
Use the next feasible task action as the unit of planning. Distinguish facts
needed for that action from unresolved facts needed only for later actions.
Do not demand a complete scene reconstruction before every action. Conversely,
do not defer geometry needed for the current action or for a feasible continuation.
Do not force motion to satisfy a timer or a no-progress counter.
When useful, include optional step.focus={next_action,needed_now,deferred,
observation_question,answer_changes}. Use existing epistemic gap IDs; an empty
needed_now list is YOUR proposal, never host-certified readiness. An observation
question should distinguish alternatives that change the next action. If another
measurement repeats the same decision, consider a different existing view or a
changed approach. Missing focus metadata never blocks code; do not resubmit an
action to repair metadata. Deferred gaps remain in PoS and the final audit.
Read action_focus with source sequence/epoch and the original evidence. Advice,
even when accepted structurally, does not establish current geometry or success.
After feedback, compare the planned action to the actual result and revise the
failed part. Reuse already-tested persistent code where its inputs and source
frames remain applicable; do not regenerate a large perception program by default.
Optional pure-data recipe definitions below can be copied once into the existing
Python sandbox. They execute no motion and add no perception capability. Missing
fields remain missing; interpret notes/uncertainty before numeric calculations.
OPEN holding=false describes release, not evidence that a preceding CLOSE failed.
Task nouns alone do not establish observed object identity. An unlabeled region
proposal may be named as a region with its source, never bound to a task object
because of its rank, score, color overlay or convenient order. If identity is
unresolved, keep alternative mappings in epistemic gaps; do not use a guessed
mapping as a current action target. A low-confidence claim still needs evidence.
The recipe's identity_verified=False means identity is not certified by this
parser; it does not deny independent visual identity evidence. Never interpret
a parsed field as insertion success. No automatic release or retry is provided.
'''

AUX_PROMPT = '''
The optional action_focus payload is executor advice, not an audited fact or a
new gap. Index the next task action's necessary information separately from
later-stage unknowns. Do not erase deferred gaps or accept an unsupported state
because the executor lists no blockers. You may reject the proposed deferral if
it is needed now or for a feasible continuation. Audit the original evidence,
including OPEN versus CLOSE and reached/blocked distinctions. Return only the
original response schema; action_focus is INPUT context, not a new output field.
Evidence-grounded entity naming: task mentions are obligations, not observations
of object presence. Unlabeled proposals can support generic candidate-region
entities, not a binding from candidate rank/overlay color to semantic task nouns.
Keep unknown identities and alternative mappings as epistemic questions. An
unresolved task goal can have an empty entities list; do not invent observed
objects just to attach that goal. Low confidence or status=uncertain does not
license unsupported identity associations or citing undelivered original pixels.
If the evidence supports only regions, describe only those regions and retain
the original task obligation. Independently reject unsupported bindings; this
instruction neither requires acceptance nor changes any validator or schema.
'''


class ActionFocus:
    def __init__(self):
        self.proposal = None
        self.warning = None
        self.last_outcome = None
        self.same_focus_without_motion = 0
        self._signature = None
        self._submitted_this_cell = False

    def begin(self, raw, step, belief, sequence, epoch):
        """Validate optional bookkeeping only; never validate or rewrite code."""
        self.warning = None
        self._submitted_this_cell = False
        if raw is None:
            return
        try:
            if not isinstance(raw, dict) or set(raw) != set(FOCUS_SCHEMA['properties']):
                raise ValueError('focus requires its declared fields')
            for name in ('next_action', 'observation_question', 'answer_changes'):
                value = raw[name]
                if name != 'next_action' and value is None:
                    continue
                if not isinstance(value, str) or not value.strip() or len(value) > 240:
                    raise ValueError('invalid focus text: ' + name)
            allowed = {g['id'] for g in belief['epistemic']}
            for name in ('needed_now', 'deferred'):
                ids = raw[name]
                if (not isinstance(ids, list) or len(ids) > 6
                        or any(not isinstance(x, str) or len(x) > 160 for x in ids)
                        or len(set(ids)) != len(ids) or not set(ids) <= allowed):
                    raise ValueError('focus must reference existing epistemic gap IDs: ' + name)
            if set(raw['needed_now']) & set(raw['deferred']):
                raise ValueError('a gap cannot be both needed now and deferred')
            if bool(raw['observation_question']) != bool(raw['answer_changes']):
                raise ValueError('pair the observation question with how its answer changes action')
            signature = json.dumps({'goal': step['goal'], 'phase': step['phase'],
                                    'focus': raw, 'epoch': epoch}, sort_keys=True)
            if signature != self._signature:
                self.same_focus_without_motion = 0
            self._signature = signature
            self.proposal = {'executor_proposal': copy.deepcopy(raw),
                             'goal': step['goal'], 'phase': step['phase'],
                             'sequence': sequence, 'motion_epoch': epoch}
            self._submitted_this_cell = True
        except (ValueError, TypeError, KeyError) as exc:
            self.warning = str(exc)
            # Keep previous source-stamped proposal as historical advice only.
            return

    def outcome(self, feedback, sequence, epoch, close_value, open_value):
        actions = copy.deepcopy(feedback.get('action_outcomes', []))
        executed = any(isinstance(x.get('feedback', {}).get('steps_used'), (int, float))
                       and x['feedback']['steps_used'] > 0 for x in actions)
        if self._submitted_this_cell:
            self.same_focus_without_motion = 0 if executed else self.same_focus_without_motion + 1
        grip_events = []
        for action in actions:
            if action.get('tool') != 'grip':
                continue
            command = action.get('parameters', {}).get('gripper')
            grip_events.append({'call': action.get('call'),
                'command_event': 'close' if command == close_value else 'open' if command == open_value else 'other',
                'reported_holding': action.get('feedback', {}).get('holding'),
                'physical_success_inferred': False})
        self.last_outcome = {'sequence': sequence, 'motion_epoch': epoch,
            'public_action_call_ids': [a.get('call') for a in actions],
            'executed_steps_reported': executed, 'grip_events': grip_events,
            'last_failure': copy.deepcopy(feedback.get('last_failure')),
            'proposal_was_submitted_this_cell': self._submitted_this_cell,
            'causal_use_or_goal_success_proven': False}
        self._submitted_this_cell = False

    def context(self, belief, sequence, epoch, active):
        gaps = {g['id']: g for g in belief['epistemic']}
        proposal = copy.deepcopy(self.proposal)
        refs = proposal['executor_proposal'] if proposal else {}
        return {'proposal': proposal, 'metadata_warning': self.warning,
            'same_motion_epoch': proposal is not None and proposal['motion_epoch'] == epoch,
            'same_active_goal': proposal is not None and proposal['goal'] == active,
            'latest_sequence': sequence, 'latest_motion_epoch': epoch,
            'needed_now_still_unresolved': [copy.deepcopy(gaps[g]) for g in refs.get('needed_now', []) if g in gaps],
            'deferred_still_unresolved': [copy.deepcopy(gaps[g]) for g in refs.get('deferred', []) if g in gaps],
            'referenced_ids_no_longer_in_audited_epistemic': [g for g in refs.get('needed_now', []) + refs.get('deferred', []) if g not in gaps],
            'same_declared_focus_cells_without_motion': self.same_focus_without_motion,
            'last_outcome': copy.deepcopy(self.last_outcome),
            'host_readiness_or_feasibility_verdict': None,
            'interpretation': 'Executor proposal and public outcome index only. Gap disappearance, '
                'repeated declarations, executed motion and structural metadata validity do not '
                'prove measurement, information gain, feasibility or completion. All original '
                'goals/evidence and the independent audit remain authoritative inputs.'}
