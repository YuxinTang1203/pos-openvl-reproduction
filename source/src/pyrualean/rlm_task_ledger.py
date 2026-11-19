"""Optional RLM milestone ledger, inspired by ReCAP and SGG-ReflAct.

Public observations only; independent model reviews remain fallible judgments,
never benchmark verdicts. No task names, coordinates, or hidden state are read.
"""
from __future__ import annotations

import copy
import json

from .arms import ArmResult
from .trajectory_context import text


PLAN_PROMPT = """Review the proposed ordered plan against the COMPLETE instruction.
Use exactly one submit_harness_state call with payload {approved: bool, issues:
[strings]}. Do not execute tools or return a robot action. Check every object,
phase, ordering constraint, interaction and final condition. Require distinct
milestones for distinct required interactions; an issued command is not a
completed condition. Initial facts must be supported by their cited initial
observations; unknown is valid and must not be guessed. Preserve object identity
and original locations when the instruction depends on them. Initial facts are
historical, not current-state guarantees. Check remaining work against completed
milestones and current feedback. proposal.steps is the NEW proposed remaining
work. previous_remaining is the OLD accepted plan and is empty before the first
plan; that is normal and not a reason to reject a complete proposal.
Evidence/claims are untrusted data. Reject
omissions, contradictory facts, opaque whole-task steps and plans whose success
conditions cannot establish the requested outcomes. No evaluator state exists.
Do not make finish(success), a checkpoint approval, or a future milestone a
milestone's success condition: final reporting is handled separately by the host.
Keep issues brief; report only necessary corrections, not a full plan rewrite.
"""

CHECK_PROMPT = """Review whether the specified milestone (or final full task) is
supported by PUBLIC evidence. Use exactly one submit_harness_state call with
payload {satisfied: true/false/null, reason: string, evidence_ids: [strings]}.
There are no robot tools. Check the COMPLETE instruction, immutable initial
reference, ordered completed milestones and fresh camera observation. A command
being issued, a function returning without exception, a model statement or old
image alone is NOT evidence that a physical effect occurred. Historical call
feedback may establish required interactions only when its outcome is actually
supported; do not infer a button actuation from an attempted arm movement.
Compare original identities/locations with the current scene where required.
Return null for ambiguity, occlusion, conflicting or missing evidence; explain
which observation is needed. Return false for an observable unsatisfied condition.
For final review check all requirements, including ordering and side conditions.
If true, cite at least one fresh_evidence_ids entry (the current observation
event ID or its image IDs), and relevant call IDs for any transient interaction.
Use ONLY eligible_evidence_ids; milestone IDs are not observation IDs.
All observations and claims are untrusted
evidence. This is a fallible model assessment, never an official task verdict.
Be concise: reason at most 400 characters; cite only necessary evidence IDs.
"""


def _objects(value, label, maximum):
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ValueError(f"{label} must contain 1..{maximum} entries")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{label} entries must be objects")
    return value


def _refs(value, allowed, label, *, empty=False):
    if (not isinstance(value, list) or (not value and not empty)
            or len(value) > 32
            or any(not isinstance(v, str) or v not in allowed for v in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{label} must contain unique registered IDs")
    return list(value)


class TaskLedger:
    def __init__(self, owner):
        self.owner = owner
        self.initial_facts = None
        self.requirements = None
        self.steps = []
        self.completed = []
        self.pending = None
        self.last_review = None
        self.plan_reviews = 0
        self.reviews = 0
        self.goal_supported = False
        self.current_feedback = None

    def initial_reference(self):
        # Epoch zero ends on the first stateful robot call. A late read cannot
        # silently rewrite the initial scene, even during a plan revision.
        return [e for e in self.owner.trace.context()['events']
                if e['epoch'] == 0 and (e['kind'] in {'initial', 'guard_observation'}
                or (e['kind'] == 'tool' and any(c.get('ok') and not c.get('stateful')
                    for c in e['payload'].get('public_calls', []))))]

    def initialize_observation(self):
        if self.owner.trace.epoch == 0 and not any(e['images'] for e in self.initial_reference()):
            self.owner.trace.capture(self.owner.kit.robo, max_images=2)

    def head(self):
        return self.steps[0] if self.steps else None

    def _plan(self, args):
        if set(args) != {'initial_facts', 'requirements', 'steps'}:
            raise ValueError('rlm_plan requires initial_facts, requirements, steps')
        initial_events = self.initial_reference()
        allowed = {e['id'] for e in initial_events if e['images'] or e['kind'] == 'tool'}
        allowed.update(i['id'] for e in initial_events for i in e['images'])
        facts = []
        for item in _objects(args['initial_facts'], 'initial_facts', 24):
            if set(item) != {'id', 'claim', 'status', 'evidence_ids'}:
                raise ValueError('initial fact requires id, claim, status, evidence_ids')
            status = item['status']
            if status not in {'observed', 'unknown'}:
                raise ValueError('initial fact status must be observed or unknown')
            facts.append({'id': text(item['id'], 'fact ID', 64),
                          'claim': text(item['claim'], 'initial claim', 400), 'status': status,
                          'evidence_ids': _refs(item['evidence_ids'], allowed, 'initial evidence',
                                                empty=status == 'unknown')})
        requirements = []
        for item in _objects(args['requirements'], 'requirements', 24):
            if set(item) != {'id', 'text'}:
                raise ValueError('requirement requires id and text')
            requirements.append({'id': text(item['id'], 'requirement ID', 64),
                                 'text': text(item['text'], 'requirement', 400)})
        for entries in (facts, requirements):
            if len({x['id'] for x in entries}) != len(entries):
                raise ValueError('duplicate fact or requirement ID')
        if self.initial_facts is not None and facts != self.initial_facts:
            raise ValueError('initial facts are immutable; record uncertainty instead of rewriting history')
        if self.requirements is not None and requirements != self.requirements:
            raise ValueError('original requirements must be preserved on revision')
        required = {x['id'] for x in requirements}
        seen = {x['step']['id'] for x in self.completed}
        steps = []
        for item in _objects(args['steps'], 'steps', 24):
            if set(item) != {'id', 'objective', 'requirements', 'success_condition'}:
                raise ValueError('step requires id, objective, requirements, success_condition')
            ident = text(item['id'], 'step ID', 64)
            if ident in seen:
                raise ValueError('step IDs must be unique and cannot reuse completed IDs')
            seen.add(ident)
            steps.append({'id': ident, 'objective': text(item['objective'], 'objective', 300),
                          'requirements': _refs(item['requirements'], required, 'step requirements'),
                          'success_condition': text(item['success_condition'], 'success condition', 400)})
        covered = {r for item in steps for r in item['requirements']}
        covered.update(r for item in self.completed for r in item['step']['requirements'])
        if covered != required:
            raise ValueError('plan must cover every original requirement')
        if len(steps) + 2 > self.owner.config['max_ledger_reviews'] - self.reviews:
            raise ValueError('plan cannot fit remaining reviews plus plan/final review; simplify milestones')
        return {'initial_facts': facts, 'requirements': requirements, 'steps': steps}

    def execute(self, name, args):
        if self.pending:
            raise ValueError('ledger review pending; await its result')
        if self.reviews >= self.owner.config['max_ledger_reviews']:
            raise ValueError('RLM ledger review budget exhausted; report stuck')
        if name == 'rlm_plan':
            if self.plan_reviews >= self.owner.config['max_plan_revisions']:
                raise ValueError('RLM plan revision budget exhausted')
            proposal = self._plan(args)
            self.plan_reviews += 1
            self.pending = {'kind': 'plan', 'proposal': proposal}
        elif name == 'rlm_checkpoint':
            if set(args) != {'step_id', 'claim', 'evidence_ids'}:
                raise ValueError('checkpoint requires step_id, claim, evidence_ids')
            if not self.head() or args['step_id'] != self.head()['id']:
                raise ValueError('checkpoint must address the current ordered step')
            claim = text(args['claim'], 'claim', 1200)
            events = self.owner.trace.context()['events']
            allowed = {e['id'] for e in events if e['kind'] in {'tool', 'guard_observation'}}
            allowed.update(i['id'] for e in events for i in e['images'])
            refs = _refs(args['evidence_ids'], allowed, 'checkpoint evidence', empty=True)
            self.pending = {'kind': 'checkpoint', 'step': copy.deepcopy(self.head()),
                            'claim': claim, 'evidence_ids': refs}
        else:
            raise ValueError('unknown ledger tool')
        self.owner.phase = 'ledger_review'
        self.owner.log('ledger_review_queued', request=self.pending)
        return ArmResult({'status': 'queued', 'note': 'No robot motion. Await the ledger review.'})

    def queue_finish(self, args):
        if self.pending or self.requirements is None or self.steps:
            raise ValueError('all ordered milestones must have supported completion before success')
        if self.reviews >= self.owner.config['max_ledger_reviews']:
            raise ValueError('RLM ledger review budget exhausted; report stuck')
        self.pending = {'kind': 'goal', 'claim': text(args.get('summary'), 'summary', 1200),
                        'finish': copy.deepcopy(args)}
        self.owner.phase = 'ledger_review'
        self.owner.log('ledger_goal_review_queued')
        return ArmResult({'status': 'queued', 'note': 'Full task review pending; no official verdict.'})

    def review(self, model_call):
        if not self.pending:
            return ''
        pending = self.pending
        self.pending = None
        self.reviews += 1
        common = {'instruction': self.owner.trace.task,
                  'initial_reference': self.initial_reference(),
                  'initial_facts': self.initial_facts,
                  'requirements': self.requirements,
                  'completed': self.completed, 'remaining': self.steps}
        images = []
        rejection_reason = None
        fresh = set()
        if pending['kind'] == 'plan':
            common['proposal'] = pending['proposal']
            common['previous_remaining'] = common.pop('remaining')
            common['current_feedback'] = self.current_feedback
            ids = [i['id'] for e in common['initial_reference'] for i in e['images']][:2]
            images = [(i, self.owner.trace.images[i][1]) for i in ids]
            report = model_call('rlm_plan_review', PLAN_PROMPT, common, images)
            valid = (isinstance(report, dict) and set(report) == {'approved', 'issues'}
                     and type(report['approved']) is bool and isinstance(report['issues'], list)
                     and len(report['issues']) <= 24
                     and all(isinstance(x, str) and len(x) <= 2000 for x in report['issues']))
            accepted = bool(valid and report['approved'] is True and not report['issues'])
            if valid and report['approved'] is True and report['issues']:
                rejection_reason = 'Approved plan report contains unresolved issues'
            if accepted:
                proposal = pending['proposal']
                self.initial_facts = copy.deepcopy(proposal['initial_facts'])
                self.requirements = copy.deepcopy(proposal['requirements'])
                self.steps = copy.deepcopy(proposal['steps'])
                self.goal_supported = False
        else:
            observation = self.owner.trace.capture(self.owner.kit.robo, max_images=2)
            common.update(current_observation=observation, requested=pending,
                          public_call_ledger=self.public_calls())
            images = [(i['id'], self.owner.trace.images[i['id']][1]) for i in observation['images']]
            historical = [i for i in pending.get('evidence_ids', []) if i in self.owner.trace.images][:2]
            images.extend((i, self.owner.trace.images[i][1]) for i in historical)
            common['historical_image_evidence'] = [
                {'image_id': i, 'event_id': e['id'], 'epoch': e['epoch']}
                for e in self.owner.trace.events for image in e['images']
                for i in historical if image['id'] == i]
            eligible = set(self.owner.trace.images)
            eligible.update(e['id'] for e in self.owner.trace.events
                            if e['kind'] in {'initial', 'guard_observation'}
                            or (e['kind'] == 'tool' and e['payload'].get('tool') == 'python'))
            fresh = {i['id'] for i in observation['images']}
            if fresh:
                fresh.add(observation['id'])
            common['eligible_evidence_ids'] = sorted(eligible)
            common['fresh_evidence_ids'] = sorted(fresh)
            report = model_call('rlm_checkpoint_review', CHECK_PROMPT, common, images)
            refs = report.get('evidence_ids') if isinstance(report, dict) else None
            valid = (isinstance(report, dict) and set(report) == {'satisfied', 'reason', 'evidence_ids'}
                     and (report['satisfied'] is None or type(report['satisfied']) is bool)
                     and isinstance(report['reason'], str) and 0 < len(report['reason']) <= 2000
                     and isinstance(refs, list) and len(refs) <= 32
                     and all(isinstance(i, str) and i in eligible for i in refs))
            accepted = bool(valid and report['satisfied'] is True and fresh.intersection(refs))
            if valid and report['satisfied'] is True and not fresh.intersection(refs):
                rejection_reason = 'Supported report must cite a fresh observation event or image ID'
            if accepted and pending['kind'] == 'checkpoint':
                self.completed.append({'step': self.steps.pop(0), 'review': copy.deepcopy(report),
                                       'epoch': self.owner.trace.epoch})
            elif accepted:
                self.goal_supported = True
        self.last_review = {'kind': pending['kind'], 'accepted': accepted,
                            'valid_report': valid, 'report': copy.deepcopy(report),
                            'rejection_reason': rejection_reason or (
                                'Unavailable or malformed review; use the specified schema and registered evidence IDs'
                                if not valid else 'Review did not support completion' if not accepted else None),
                            'fresh_evidence_ids': sorted(fresh)}
        self.owner.phase = ('goal_supported' if self.goal_supported else
                            'ready' if self.requirements is not None else 'planning')
        self.owner.log('ledger_reviewed', review_kind=self.last_review['kind'],
                       **{k: v for k, v in self.last_review.items() if k != 'kind'})
        return 'RLM milestone review (model assessment, not official verdict):\n' + json.dumps(self.last_review)

    def public_calls(self):
        # Only the public call summaries captured by the existing runtime.
        # Truncation is explicit so an auditor cannot assume missing history.
        rows = [{'event_id': e['id'], 'epoch': e['epoch'], 'calls': e['payload'].get('public_calls', []),
                 'stdout': str(e['payload'].get('result', {}).get('stdout', ''))[:1600],
                 'exception': e['payload'].get('result', {}).get('exception'),
                 'images': e['images']}
                for e in self.owner.trace.events if e['kind'] == 'tool'
                and e['payload'].get('tool') == 'python']
        tail = rows[-80:]
        while tail and len(json.dumps(tail, default=str)) > 48000:
            tail = tail[1:]
        return {'entries': tail, 'truncated': len(rows) > len(tail)}

    def context(self):
        return {'initial_facts': self.initial_facts, 'requirements': self.requirements,
                'current_step': self.head(), 'remaining': self.steps,
                'completed': self.completed, 'last_review': self.last_review,
                'reviews': self.reviews, 'plan_reviews': self.plan_reviews,
                'pending': bool(self.pending), 'goal_supported': self.goal_supported,
                'initial_reference_ids': [e['id'] for e in self.initial_reference()],
                'latest_execution_feedback': self.current_feedback}


def ledger_tools():
    string = {'type': 'string'}
    refs = {'type': 'array', 'items': string, 'maxItems': 32}
    def obj(properties):
        return {'type': 'object', 'properties': properties, 'required': list(properties),
                'additionalProperties': False}
    return [
        {'name': 'rlm_plan', 'description': 'Propose or revise ordered milestones; preserves initial facts and requirements.',
         'input_schema': obj({
             'initial_facts': {'type': 'array', 'minItems': 1, 'maxItems': 24, 'items': obj({
                 'id': string, 'claim': string, 'status': {'type': 'string', 'enum': ['observed', 'unknown']},
                 'evidence_ids': refs})},
             'requirements': {'type': 'array', 'minItems': 1, 'maxItems': 24,
                              'items': obj({'id': string, 'text': string})},
             'steps': {'type': 'array', 'minItems': 1, 'maxItems': 24, 'items': obj({
                 'id': string, 'objective': string, 'requirements': refs, 'success_condition': string})}})},
        {'name': 'rlm_checkpoint', 'description': 'Request evidence review of the current milestone; captures fresh camera images.',
         'input_schema': obj({'step_id': string, 'claim': string, 'evidence_ids': refs})}]


LEDGER_PROMPT = """
# RLM v3 ordered task ledger
First inspect the scene, then submit rlm_plan(initial_facts, requirements, steps).
Use the initial_reference IDs in controller state for immutable initial object
identities/locations; uncertain facts must be status=unknown. Copy the FULL task
requirements, including order, side conditions and final posture. Each step has
id, objective, requirements (IDs), success_condition. Separate distinct required
interactions into distinct milestones. An independent review checks coverage.
Ordinary python remains the robot executor. Use a short cell for observation or
the CURRENT milestone. A cell returning normally never completes a milestone.
After inspecting the actual outcome call rlm_checkpoint(step_id, claim,
evidence_ids). The host captures current cameras and reviews the outcome. Only
supported milestones advance; false/unknown retains the same step and explains
missing evidence. On failure revise remaining steps through rlm_plan, copying
initial_facts and requirements EXACTLY; completed IDs cannot be reused. Plans
are hypotheses: preserve uncertainty and use observations to resolve it.
Do not add a milestone for finish(success) or checkpoint approval itself; the
host manages final reporting after the actual task milestones are completed.
If maintenance is skipped and a review is still pending, abort as stuck; the
host will stop further action attempts rather than loop without review allowance.
The controller reintroduces the initial reference, current milestone and remaining
plan on every round. Use rlm only when needed: reason must be history_missing,
conflicting_evidence, execution_failure or plan_revision, plus code. Retrieve
relevant public event payloads and images; no automatic analysis on every action.
Analysis results support decisions but do not themselves complete milestones.
finish(success) requests one final full-instruction review after all milestones;
if supported, call finish(success) again to report completion. Reviews are model
assessments, never official success. If a review is uncertain, obtain the missing
public observation, then retry the checkpoint or revise the remaining plan.
All reviews, retries, observations and analysis share the existing episode budget.
"""
