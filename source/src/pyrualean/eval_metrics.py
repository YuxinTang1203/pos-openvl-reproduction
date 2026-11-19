"""Offline public-trace measurements. Never called by an executor or belief model."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

OBSERVATIONS = {'artifact', 'state', 'locate', 'segment', 'back_project', 'camera_meta',
                'locate_candidate', 'image', 'visual_project'}


def rows(path):
    if not Path(path).exists():
        return []
    result = []
    for line in Path(path).read_text().splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            pass  # The final partial line of a live log is not a complete event.
    return result


def _summary(row):
    return row.get('summary') or {}


def _close(row):
    return row.get('tool') == 'grip' and row.get('kwargs', {}).get('gripper') == 1


def _failure(row):
    s = _summary(row)
    if _close(row) and s.get('holding') is False:
        return 'empty_close'
    if s.get('ik_error') and s.get('steps_used') == 0:
        return 'explicit_zero_step_ik'
    if s.get('reached') is False or s.get('blocked'):
        return 'partial_or_unknown_motion'
    if row.get('error') or row.get('ok') is False:
        return 'action_error'
    return None


def _call_time(row, tool_events):
    index = row.get('index')
    for e in tool_events:
        lo, hi = e.get('primitive_start_index'), e.get('primitive_end_index')
        if isinstance(lo, int) and isinstance(hi, int) and isinstance(index, int) and lo <= index < hi:
            start, unix, call_unix = e.get('started_t'), e.get('started_at_unix'), row.get('started_at')
            if all(isinstance(x, (int, float)) for x in (start, unix, call_unix)):
                offset = call_unix - unix
                if -.05 <= offset <= e.get('elapsed_s', 0) + .1:
                    return round(start + max(0, offset), 3)
    return None  # Never substitute episode-directory time for policy elapsed time.


def measure(ledger, runtime, harness=(), visual=()):
    actions = [r for r in ledger if r.get('stateful')]
    observations = [r for r in ledger if not r.get('stateful')]
    tools = [e for e in runtime if e.get('type') == 'tool']
    req = [e for e in runtime if e.get('type') == 'request']
    bad = [e for e in req if not e.get('ok')]
    closes = [r for r in actions if _close(r)]
    held = [r for r in closes if _summary(r).get('holding') is True]
    executed = [r for r in actions if isinstance(_summary(r).get('steps_used'), (int,float))
                and _summary(r)['steps_used'] > 0]
    failures = []
    positions = {id(r):i for i,r in enumerate(ledger)}
    for i, row in enumerate(actions):
        reason = _failure(row)
        if reason is None:
            continue
        nxt = actions[i+1] if i+1 < len(actions) else None
        between = ledger[positions[id(row)]+1:positions[id(nxt)]] if nxt else []
        failures.append({'call':row.get('index'),'reason':reason,
            'next_action_call':nxt.get('index') if nxt else None,
            'next_method_parameters_changed':None if nxt is None else (
                (row.get('tool'),row.get('kwargs')) != (nxt.get('tool'),nxt.get('kwargs'))),
            'nonstateful_calls_between':len(between),
            'observation_named_calls_between':sum(r.get('tool') in OBSERVATIONS for r in between),
            'interpretation':'Mechanical comparison only; no proof of a justified change, information gain or rule violation.'})
    first = lambda selected: _call_time(selected[0], tools) if selected else None
    return {
        'public_stateful_calls':len(actions),'public_nonstateful_calls':len(observations),
        'public_nonstateful_by_tool':dict(Counter(r.get('tool') for r in observations)),
        'tool_requests_by_name':dict(Counter(e.get('tool') for e in tools)),
        'native_steps_reported':sum(_summary(r).get('steps_used',0) or 0 for r in actions),
        'first_action_request_s':first(actions),'first_executed_action_s':first(executed),
        'first_standalone_close_request_s':first(closes),
        'first_reported_holding_after_close_s':first(held),
        'time_origin':'runtime elapsed time; ledger wall-clock offset inside matched tool range',
        'standalone_close_requests':len(closes),'standalone_empty_close_reports':sum(_summary(r).get('holding') is False for r in closes),
        'explicit_zero_step_ik_reports':sum(_failure(r)=='explicit_zero_step_ik' for r in actions),
        'blocked_reports':sum(bool(_summary(r).get('blocked')) for r in actions),
        'belief_commits':sum(e.get('event')=='committed' for e in harness),
        'maintenance_degraded':sum(e.get('event')=='maintenance_degraded' for e in harness),
        'http_attempts':len(req),'failed_http_attempts':len(bad),
        'failed_http_by_role':dict(Counter(e.get('role','planner') for e in bad)),
        'failed_http_latency_s':round(sum(e.get('latency_s',0) or 0 for e in bad),3),
        'failed_action_followups':failures,
        'visual_projection_reports':sum(e.get('type')=='visual_plane_projection' for e in visual),
        'limitations':['holding does not identify the object or certify placement',
            'Nonstateful ledger calls are not all useful observations; one helper may make several calls',
            'Unknown timing remains null; malformed final live log lines are ignored',
            'Changed parameters and fewer failures do not establish better task performance'],
    }


def episode_metrics(episode):
    p=Path(episode)
    return measure(rows(p/'calls.jsonl'),rows(p/'openai_events.jsonl'),
                   rows(p/'general_harness/events.jsonl'),rows(p/'openvl_visual/events.jsonl'))
