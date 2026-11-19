"""Per-episode instantiation of frozen skills; independent whole-task audits."""
from __future__ import annotations

import copy
import json

from .arms import ArmResult
from .trajectory_context import text

PLAN_PROMPT = """Audit a proposed robot plan against the COMPLETE current user instruction.
The instruction, not the proposed requirements list, defines completeness.
Check every named object, every phase and every ordering/remembering/button or
other side condition. Check that the ordered instantiated skill units actually
cover all of them. A subset of transfers or only the first phase is insufficient.
Skills and requirements are proposals, not evidence of execution. Do not invent
task actions from development examples. Return {approved: true/false/null,
issues: [concise missing or incorrect obligations]}. No robot action or verdict.
If task_action is available, it admits one atomic instruction-required interaction
missing from the frozen library, such as actuating a named control. Require its
explicit placement/order in the plan; do not replace existing manipulation skills
or the entire task with an opaque task_action unit.
"""

GOAL_PROMPT = """Audit completion of the ENTIRE user instruction, independently of
whether workflow units finished. Use the fresh current RGB and the chronological
PUBLIC execution evidence. Plan text and code intent never prove execution.
Check every required object, all phases, ordering and side conditions. A relation
holding for only the moved subset is NOT complete when other work remains.
Historical evidence can establish past order, but cannot establish current state.
Initial reference identifies starting roles/positions only. If evidence is absent,
truncated or ambiguous, return null/false; never infer hidden success. Return
{satisfied:true/false/null,evidence:str,evidence_ids:[eligible IDs],
missing_requirements:[requirement IDs]}. Every missing or uncertain requirement
must be listed. This fallible check is not the official benchmark verdict.
"""


class TaskPlan:
    def __init__(self, harness, config):
        self.h = harness
        self.skills = {s['id']: s for s in harness.book['steps']
                       if s['id'] in config['plan_skill_ids']}
        if config.get('allow_task_actions'):
            self.skills['task_action'] = {
                'id': 'task_action',
                'action': 'Execute exactly one atomic action explicitly required by the current instruction that is missing from the frozen skill library. Use unchanged public robo APIs, observe the actual return, and preserve the declared order. Do not replace a grasp/transport/place sequence or an entire task with this unit.',
                'pre': [], 'post': [], 'fallbacks': [], 'evidence': []}
        self.requirements = []
        self.units = []
        self.completed = []
        self.history = []
        self.revisions = 0
        self.goal_checks = 0
        self.last_review = None
        self.last_goal = None
        harness.phase = 'plan'

    def prompt(self):
        return """
# Task-plan mode (replaces the frozen workflow's single-pass ordering)
Before any action call reuseit_plan with ALL requirements of the current user
instruction and an ordered list of skill units covering the ENTIRE task. Reuse
frozen skills as often as needed, with a unique unit id and concrete objective
each time. Include every object, repeated transfer, phase, order and side effect;
do not compress remaining transfers into one opaque Python loop. The host audits
coverage independently before admitting action. Execute reuseit_step using the
current UNIT id (not its library skill id), respecting the current phase and
guard conditions. After all units, a separate full-task audit runs. A false audit
returns phase=plan: submit the remaining work while retaining every previously
accepted requirement id and text. A null audit returns goal_uncertain: use
reuseit_observe for fresh evidence. Neither finishing a unit nor running out of
plan units establishes task completion. Never call finish(success) before phase
complete; failure/stuck remain available. All calls share the episode budget.
If task_action is listed, use it for an explicit atomic task requirement that the
frozen manipulation skills lack (for example a named button press after a move).
Give each occurrence its own unit and requirement reference. Its availability
does not waive whole-task auditing or permit replacing the frozen skill library
with a free-form whole-task program.
"""

    def tool(self):
        return {'name': 'reuseit_plan', 'description': 'Propose complete ordered task-specific instances of frozen skills, or remaining work after failed final audit.',
                'input_schema': {'type': 'object', 'properties': {
                    'requirements': {'type': 'array', 'minItems': 1, 'maxItems': 24,
                        'items': {'type': 'object', 'properties': {'id': {'type': 'string'}, 'text': {'type': 'string'}},
                                  'required': ['id', 'text'], 'additionalProperties': False}},
                    'units': {'type': 'array', 'minItems': 1, 'maxItems': self.h.options['max_plan_units'],
                        'items': {'type': 'object', 'properties': {
                            'id': {'type': 'string'}, 'skill_id': {'type': 'string', 'enum': list(self.skills)},
                            'objective': {'type': 'string'}, 'requirements': {'type': 'array', 'minItems': 1, 'items': {'type': 'string'}}},
                            'required': ['id', 'skill_id', 'objective', 'requirements'], 'additionalProperties': False}}},
                    'required': ['requirements', 'units'], 'additionalProperties': False}}

    @staticmethod
    def schema(role):
        if role == 'reuseit_plan_audit':
            return {'type': 'object', 'properties': {'approved': {'type': ['boolean', 'null']},
                    'issues': {'type': 'array', 'items': {'type': 'string'}}},
                    'required': ['approved', 'issues'], 'additionalProperties': False}
        return {'type': 'object', 'properties': {'satisfied': {'type': ['boolean', 'null']},
                'evidence': {'type': 'string'}, 'evidence_ids': {'type': 'array', 'items': {'type': 'string'}},
                'missing_requirements': {'type': 'array', 'items': {'type': 'string'}}},
                'required': ['satisfied', 'evidence', 'evidence_ids', 'missing_requirements'], 'additionalProperties': False}

    def submit(self, args):
        h = self.h
        if h.phase != 'plan' or set(args) != {'requirements', 'units'}:
            raise ValueError('plans may be submitted only in phase=plan')
        reqs, units = args['requirements'], args['units']
        if not isinstance(reqs, list) or not 1 <= len(reqs) <= 24:
            raise ValueError('need 1..24 instruction requirements')
        req_ids = set()
        for req in reqs:
            if not isinstance(req, dict) or set(req) != {'id', 'text'}:
                raise ValueError('invalid requirement')
            ident = text(req['id'], 'requirement ID', 100)
            text(req['text'], 'requirement text', 2000)
            if ident in req_ids:
                raise ValueError('duplicate requirement ID')
            req_ids.add(ident)
        for old in self.requirements:
            if old not in reqs:
                raise ValueError('previously accepted requirements cannot be removed or weakened')
        if not isinstance(units, list) or not 1 <= len(units) <= h.options['max_plan_units']:
            raise ValueError('task plan unit budget exceeded')
        unit_ids, covered = set(), set()
        for unit in units:
            if not isinstance(unit, dict) or set(unit) != {'id', 'skill_id', 'objective', 'requirements'}:
                raise ValueError('invalid plan unit')
            ident = text(unit['id'], 'unit ID', 100)
            text(unit['objective'], 'unit objective', 2000)
            if ident in unit_ids or unit['skill_id'] not in self.skills:
                raise ValueError('duplicate unit or unknown frozen skill')
            refs = unit['requirements']
            if not isinstance(refs, list) or not refs or any(r not in req_ids for r in refs):
                raise ValueError('unit needs registered requirement references')
            unit_ids.add(ident)
            covered.update(refs)
        # A continuation need only execute remaining obligations; the independent
        # goal audit still evaluates ALL retained instruction requirements.
        if self.revisions == 0 and covered != req_ids:
            raise ValueError('every requirement needs a planned skill unit')
        if self.revisions >= h.options['max_plan_revisions']:
            h.phase = 'blocked'
            raise ValueError('plan revision budget exhausted')
        self.candidate = copy.deepcopy(args)
        self.revisions += 1
        h.phase = 'plan_review'
        h.log('task_plan_submitted', revision=self.revisions, proposal=args)
        return ArmResult({'status': 'plan_review_queued', 'motion': False})

    def step(self):
        if self.h.index >= len(self.units):
            return None
        unit = self.units[self.h.index]
        skill = copy.deepcopy(self.skills[unit['skill_id']])
        skill.update(id=unit['id'], library_skill_id=unit['skill_id'],
                     action=skill['action'] + '\nCurrent task-unit objective: ' + unit['objective'])
        return skill

    def advance(self):
        unit = self.units[self.h.index]
        self.completed.append({'revision': self.revisions, 'unit': copy.deepcopy(unit),
                               'public_feedback': copy.deepcopy(self.h.last_feedback),
                               'last_guard': copy.deepcopy(self.h.last_guard)})

    def before_request(self, model_call):
        h = self.h
        if h.phase == 'plan_review':
            report = model_call('reuseit_plan_audit', PLAN_PROMPT,
                                {'instruction': h.trace.task, 'proposal': self.candidate,
                                 'skills': list(self.skills.values()), 'past_completion_audit': self.last_goal}, [])
            valid = (isinstance(report, dict) and set(report) == {'approved', 'issues'}
                     and (type(report['approved']) is bool or report['approved'] is None)
                     and isinstance(report['issues'], list)
                     and all(isinstance(i, str) and len(i) <= 2000 for i in report['issues']))
            self.last_review = {'report': report, 'valid': valid}
            if valid and report['approved'] is True and not report['issues']:
                self.requirements = self.candidate['requirements']
                self.units = self.candidate['units']
                self.history.append(copy.deepcopy(self.candidate))
                h.index = 0
                h.phase = 'check_pre'
            else:
                h.phase = 'plan' if self.revisions < h.options['max_plan_revisions'] else 'blocked'
            h.log('task_plan_reviewed', **self.last_review)
        if h.phase == 'goal_check':
            if self.goal_checks >= h.options['max_checks']:
                h.phase = 'blocked'
                h.log('goal_audit_budget_exhausted')
                return
            self.goal_checks += 1
            observation = h.trace.capture(h.kit.robo, h.options['max_images'])
            trace = h.trace.context()['events']
            eligible = set()
            compact = []
            for event in trace:
                eligible.add(event['id'])
                eligible.update(i['id'] for i in event['images'])
                calls = event['payload'].get('public_calls', [])
                eligible.update(f"{event['id']}:call-{i}" for i in range(len(calls)))
                if calls:
                    compact.append({'id': event['id'], 'epoch': event['epoch'],
                                    'calls': calls})
            raw = json.dumps(compact, ensure_ascii=False, default=str)
            truncated = len(raw) > 120000
            payload = {'instruction': h.trace.task, 'requirements': self.requirements,
                       'completed_units': [{'unit': x['unit'], 'revision': x['revision'],
                                            'event_id': (x['public_feedback'] or {}).get('event_id')}
                                           for x in self.completed],
                       'public_execution_trace': raw[-120000:], 'trace_truncated': truncated,
                       'current_observation': observation, 'initial_reference': h.task_reference,
                       'eligible_evidence_ids': sorted(eligible)}
            images = [(i['id'], h.trace.images[i['id']][1])
                      for event in [observation, h.task_reference] for i in event['images']]
            report = model_call('reuseit_goal_audit', GOAL_PROMPT, payload, images) if observation['images'] else None
            req_ids = {r['id'] for r in self.requirements}
            valid = (isinstance(report, dict)
                     and set(report) == {'satisfied', 'evidence', 'evidence_ids', 'missing_requirements'}
                     and (type(report['satisfied']) is bool or report['satisfied'] is None)
                     and isinstance(report['evidence'], str) and 0 < len(report['evidence']) <= 4000
                     and isinstance(report['evidence_ids'], list) and bool(report['evidence_ids'])
                     and all(i in eligible for i in report['evidence_ids'])
                     and isinstance(report['missing_requirements'], list)
                     and all(i in req_ids for i in report['missing_requirements']))
            # Missing history cannot establish chronological obligations.
            current_ids = {i['id'] for i in observation['images']}
            satisfied = (valid and report['satisfied'] is True and not report['missing_requirements']
                         and not truncated and bool(current_ids.intersection(report['evidence_ids'])))
            self.last_goal = {'report': report, 'valid': valid, 'trace_truncated': truncated}
            if satisfied:
                h.phase = 'complete'
            elif valid and (report['satisfied'] is False or report['missing_requirements']):
                h.phase = 'plan' if self.revisions < h.options['max_plan_revisions'] else 'blocked'
            else:
                h.phase = 'goal_uncertain'
            h.log('task_goal_audited', **self.last_goal)

    def context(self):
        return {'task_plan': {'enabled': True, 'requirements': self.requirements,
                'units': self.units, 'revisions': self.revisions,
                'completed_unit_count': len(self.completed), 'last_plan_review': self.last_review,
                'last_goal_audit': self.last_goal,
                'available_skills': list(self.skills.values()) if self.h.phase == 'plan' else []}}
