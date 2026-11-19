import copy
from types import SimpleNamespace
import pytest
from pyrualean.arms import ArmResult
from pyrualean.composite_schedule import FreshnessSchedule
from pyrualean.pos_v4 import EventSchedule
from tests.test_composite_harness import setup,args,events
from tests.test_pos_harness import Script,audit


def directive():
    return {'goal_id':'a-root','mode':'act','next_step':'Attempt the first grasp.',
            'decision_critical_missing':None,'why_change':'Fresh evidence.'}


def test_motion_archives_directive_across_all_current_surfaces(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False);h=kit.general_harness
    original=directive();h.before_request(Script(copy.deepcopy(h.belief),{**audit(None),'directive':original}))
    assert h.decision_context()['directive']==original
    out=kit.execute_tool('python',args('robo.move_to([0,0,.6])'))
    assert h.directive is None
    assert out.result['task_progress']['controller_directive'] is None
    assert out.result['task_progress']['historical_controller_directive']==original
    assert out.result['composite_decision']['directive'] is None
    assert out.result['composite_decision']['historical_directive']==original
    assert h.last_audited_directive==original
    assert events(kit,'directive_archived')
    next_directive={**original,'next_step':'Inspect contact outcome.'}
    h.before_request(Script(copy.deepcopy(h.belief),{**audit(None),'directive':next_directive}))
    assert h.decision_context()['directive']==next_directive
    assert h.decision_context()['historical_directive'] is None


def test_read_only_feedback_does_not_invalidate_same_epoch_directive(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False);h=kit.general_harness
    d=directive();h.before_request(Script(copy.deepcopy(h.belief),{**audit(None),'directive':d}))
    out=kit.execute_tool('python',args('print(robo.state())'))
    assert out.result['task_progress']['controller_directive']==d
    assert h.decision_context()['directive']==d


def test_blocked_contact_is_reported_even_when_arrival_tolerance_passes(robo,tmp_path):
    h=setup(robo,tmp_path,visual=False).general_harness
    r=SimpleNamespace(index=50,tool='move_pose',stateful=True,kwargs={},error=None,
                      summary={'reached':True,'blocked':'something is in the way','steps_used':9})
    feedback=h._feedback(ArmResult({}),[r])
    assert feedback['last_failure']['type']=='contact_or_motion_shortfall'
    assert feedback['last_failure']['call']==50
    assert h.error_event and h.last_failure==feedback['last_failure']
    assert 'fresh images' in feedback['recovery_guidance']


@pytest.mark.parametrize('spent',[695.798,752.613,733.0])
def test_recorded_stranded_reserve_admits_existing_audit_chain(spent):
    s=FreshnessSchedule();s.attempted=True;s.last_sequence=3;s.last_elapsed=1000
    kw=dict(sequence=15,epoch=3,gap=None,important=False,error=False,finish=False,
            remaining=2500,total=5400,spent=spent,auxiliary=1080,round_limit=480,minimum=240)
    original=EventSchedule();original.__dict__.update(s.__dict__)
    assert not original.decide(**kw)['due']
    p=s.decide(**kw)
    assert p['due'] and p['trigger']=='stale_state_reserved_chain'
    assert p['reserve_s']==0 and 240<=p['allowance_s']<=1080-spent


@pytest.mark.parametrize('spent,remaining,seq,last_elapsed',[
    (900,2500,15,1000),(695.8,100,15,1000),(695.8,3000,4,2300),
])
def test_fallback_never_admits_underfunded_or_fresh_maintenance(spent,remaining,seq,last_elapsed):
    s=FreshnessSchedule();s.attempted=True;s.last_sequence=3;s.last_elapsed=last_elapsed
    kw=dict(sequence=seq,epoch=3,gap=None,important=False,error=False,finish=False,
            remaining=remaining,total=5400,spent=spent,auxiliary=1080,round_limit=480,minimum=240)
    assert not s.decide(**kw)['due']


def test_feedback_boundary_is_advice_and_never_intercepts_code(robo,backend,tmp_path):
    kit=setup(robo,tmp_path,visual=False)
    assert 'Printing a result is not a test' in kit.general_harness.prompt()
    assert 'return and inspect that image' in kit.general_harness.prompt()
    out=kit.execute_tool('python',args('robo.move_to([0,0,.6]); robo.move_to([.1,0,.6])'))
    assert sum(n=='move_to' for n,_ in backend.calls)==2
    assert not out.result.get('observation_precondition_blocked')


def test_disabled_reflact_has_no_residual_required_reflection_instructions(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False,reflection=False)
    text=kit.general_harness.prompt()
    assert "ReflAct's observed_state/goal_gap" not in text
    schema=next(t['input_schema'] for t in kit.get_tools_spec() if t['name']=='python')
    assert schema['required']==['code'] and 'observed_state' not in schema['properties']
