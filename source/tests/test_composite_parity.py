"""Composition must preserve the existing component behavior, not just files."""
import copy
from pathlib import Path

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.composite_harness import CompositeHarness
from pyrualean.progress_lite import ProgressLiteHarness
from tests.test_composite_harness import setup, args, events
from tests.test_pos_harness import Script, audit, belief


def pair(robo, tmp_path):
    kit = setup(robo, tmp_path/'composite', visual=False)
    base_kit = ArmToolkit(robo, arm='cells', workspace=tmp_path/'base/workspace',
                          images='on-demand', feedback='pure')
    baseline = ProgressLiteHarness(base_kit, {'kind':'pos_progress_lite'}, 'Put the block on the plate')
    return kit.general_harness, baseline


def test_policy_methods_are_the_original_modified_pos_methods():
    for name in ['maintenance_plan','maintenance_requirements','response_schema','_validate',
                 '_validate_step','execute_finish']:
        assert name not in CompositeHarness.__dict__, name
        assert getattr(CompositeHarness,name) is getattr(ProgressLiteHarness,name), name
    assert not hasattr(CompositeHarness,'planner_instruction')
    assert not hasattr(CompositeHarness,'invalid_state_report')


@pytest.mark.parametrize('spent,remaining', [(0,5400),(232,4453),(414,4172),
    (623.1386202406138,3702.49),(752.613,3378),(910.825,3000),(733,1100),(1000,500)])
@pytest.mark.parametrize('trigger', ['initial','error','stale','normal'])
def test_original_scheduler_decisions_identical_over_full_lifecycle(robo,tmp_path,spent,remaining,trigger):
    composite, baseline = pair(robo,tmp_path)
    for h in [composite,baseline]:
        h.sequence=15;h.motion_epoch=3
        h.error_event=trigger=='error'
        h.important_event=False
        h.schedule.attempted=trigger!='initial'
        h.schedule.last_sequence=7 if trigger=='stale' else 13
        h.schedule.last_elapsed=1227.931
    kw=dict(spent=spent,remaining=remaining,total=5400,auxiliary=1080,round_limit=480,minimum=240)
    actual=composite.maintenance_plan(**kw); original=baseline.maintenance_plan(**kw)
    if actual.get('trigger')=='stale_state_reserved_chain':
        assert not original['due'] and original['reason']=='auxiliary_or_final_chain_reserve'
        assert original['trigger']=='stale_state_checkpoint'
        assert actual['due'] and actual['reserve_s']==0
        assert actual['allowance_s']<=min(kw['auxiliary']-kw['spent'],kw['round_limit'],kw['remaining'])
    else:
        assert actual==original
    assert composite.maintenance_requirements()==baseline.maintenance_requirements()


def test_the_recorded_budget_override_no_longer_occurs(robo,tmp_path):
    h,_=pair(robo,tmp_path)
    h.sequence=15;h.motion_epoch=3;h.error_event=True
    h.schedule.attempted=True;h.schedule.last_sequence=13;h.schedule.last_elapsed=1227.931
    p=h.maintenance_plan(spent=623.1386202406138,remaining=3702.49,total=5400,
                         auxiliary=1080,round_limit=480,minimum=240)
    assert not p['due'] and p['reserve_s']==240
    assert p['allowance_s']==pytest.approx(216.86137975938618)


def test_original_schema_and_auxiliary_call_inputs_are_preserved(robo,tmp_path):
    h,b=pair(robo,tmp_path)
    for role in ['pos_belief','pos_repair','pos_sentinel']:
        assert h.response_schema(role)==b.response_schema(role)
    directive={'goal_id':'a-root','mode':'observe','next_step':'Inspect the placement outcome.',
               'decision_critical_missing':'Outcome unverified.','why_change':'No placement evidence.'}
    a=Script(copy.deepcopy(h.belief),{**audit(None),'directive':directive})
    c=Script(copy.deepcopy(b.belief),{**audit(None),'directive':directive})
    h.before_request(a);b.before_request(c)
    assert len(a.calls)==len(c.calls)
    for actual, original in zip(a.calls,c.calls):
        assert {k:v for k,v in actual.items() if k!='prompt'}=={k:v for k,v in original.items() if k!='prompt'}
        assert actual['prompt']==original['prompt']+h.response_contract(actual['role'])
    assert h.belief==b.belief and h.directive==b.directive
    assert len(a.calls)==2


def test_original_validator_does_not_rewrite_invalid_states(robo,tmp_path):
    h,b=pair(robo,tmp_path)
    candidate=belief('unknown-evidence')
    messages=[]
    for harness in [h,b]:
        with pytest.raises(ValueError)as exc:harness._validate(copy.deepcopy(candidate))
        messages.append(str(exc.value))
    assert messages[0]==messages[1]
    assert not events(h.kit,'state_freshness_downgraded')


def test_no_motion_wrappers_or_added_action_gate(robo,tmp_path):
    move=robo.move_to;stateful=robo._stateful
    kit=setup(robo,tmp_path,visual=False)
    assert robo.move_to==move and robo._stateful==stateful
    h=kit.general_harness
    h.directive={'goal_id':'a-root','mode':'observe','next_step':'Observe first.',
                 'decision_critical_missing':'Unverified outcome.','why_change':'Need evidence.'}
    # Advisory PoS guidance remains advisory; the bridge does not invent a gate.
    out=kit.execute_tool('python',args('robo.move_to([0,0,.6])'))
    assert 'observation_precondition_blocked' not in str(out.result)
    assert not events(kit,'observation_precondition_blocked')
    assert not events(kit,'batch_probe_stopped')


def test_handoff_preserves_full_pos_directive_and_completion_behavior(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False);h=kit.general_harness
    directive={'goal_id':'a-root','mode':'recover','next_step':'x'*350,
               'decision_critical_missing':'Specific uncertainty.','why_change':'Latest failure.'}
    h.before_request(Script(copy.deepcopy(h.belief),{**audit(None),'directive':directive}))
    out=kit.execute_tool('python',args('print(robo.state())'))
    assert out.result['task_progress']['controller_directive']==directive
    assert out.result['composite_decision']['directive']==directive
    assert not out.result['composite_decision']['belief_updated_for_latest_feedback']
    assert 'COMPOSITE_HANDOFF' in h.runtime_context()
    assert not h.completion_approved
    assert h.execute_finish.__func__ is ProgressLiteHarness.execute_finish
