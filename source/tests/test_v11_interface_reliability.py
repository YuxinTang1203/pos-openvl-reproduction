import copy,json
import pytest
from pyrualean.composite_harness import CompositeHarness
from pyrualean.reflact_harness import validate_options
from tests.test_composite_harness import setup,args,events
from tests.test_openai_runtime import ScriptedTransport,_reply
from pyrualean.openai_runtime import run_openai_agent


def advisory(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False)
    kit.general_harness=CompositeHarness(kit,{'kind':'openvl_pos_reflact',
        'openvl':{'enabled':False},'reflact':{'enabled':True,'summary_length_policy':'advisory'}},'Keep original task unfinished until verified')
    return kit


def test_advisory_schema_does_not_publish_contradictory_hard_summary_cap(robo,tmp_path):
    kit=advisory(robo,tmp_path)
    schema=next(t['input_schema'] for t in kit.get_tools_spec() if t['name']=='python')
    assert schema['required']==['observed_state','goal_gap','code']
    assert 'maxLength' not in schema['properties']['observed_state']
    assert schema['properties']['code']['minLength']==1
    assert 'brevity target' in kit.general_harness.prompt()
    assert 'only observed_state and goal_gap' in kit.general_harness.prompt()


@pytest.mark.parametrize('field',['observed_state','goal_gap'])
def test_long_valid_summary_executes_once_and_is_recorded_without_truncation(robo,tmp_path,field):
    kit=advisory(robo,tmp_path);report=args('executions=globals().get("executions",0)+1')
    report[field]='Unverified physical evidence. '*40
    original=copy.deepcopy(report)
    result=kit.execute_tool('python',report)
    assert kit._globals['executions']==1 and kit._cell_index==1
    assert not result.result.get('error')
    recorded=events(kit,'reflection_recorded')
    assert len(recorded)==1 and recorded[0][field]==report[field].strip()
    warning=events(kit,'reflection_length_warning')
    assert warning[0]['code_blocked'] is False and warning[0]['text_truncated'] is False
    assert report==original


@pytest.mark.parametrize('bad',[None,'','   ',4])
def test_missing_or_invalid_code_still_rejects_before_execution(robo,backend,tmp_path,bad):
    kit=advisory(robo,tmp_path);a=args('robo.move_to([0,0,.6])');a['code']=bad
    out=kit.execute_tool('python',a)
    assert not out.result['reflact_protocol']['accepted']
    assert out.result['reflact_protocol']['code_executed'] is False
    assert 'code' in out.result['reflact_protocol']['required_arguments']
    assert kit._cell_index==0 and not any(n=='move_to' for n,_ in backend.calls)


def test_invalid_call_then_model_corrected_call_does_not_replay_actions(robo,tmp_path):
    kit=advisory(robo,tmp_path)
    transport=ScriptedTransport([
      _reply([('missing','python',{'observed_state':'Unverified','goal_gap':'Unfinished'})]),
      _reply([('valid','python',args('one=globals().get("one",0)+1'))]),
      _reply([('end','finish',{'status':'stuck','summary':'Synthetic interface check.'})])])
    out=run_openai_agent(kit.general_harness.prompt(),[],kit,model='offline',base_url='http://unused',
       api_key=None,max_requests=3,max_turns=3,budget_s=10,auxiliary_budget_s=0,
       transport=transport,out_dir=tmp_path/'runtime')
    assert out['python_cells']==1 and kit._globals['one']==1
    assert len(events(kit,'reflection_rejected'))==1
    assert len(events(kit,'reflection_recorded'))==1


def test_strict_mode_keeps_old_behavior_and_unknown_length_policy_is_invalid(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False)
    kit.execute_tool('python',args('must_not_run=1',observed_state='x'*321))
    assert kit._cell_index==0 and 'must_not_run' not in kit._globals
    with pytest.raises(ValueError):
      validate_options({'kind':'reflact','summary_length_policy':'ignore_all_validation'})
