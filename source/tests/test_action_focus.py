import copy

import pytest

from pyrualean.action_focus import ActionFocus, FIELD_RECIPE, GRIP_RECIPE
from pyrualean.composite_harness import CompositeHarness, validate_options
from tests.test_composite_harness import setup, events, args
from tests.test_pos_harness import Script, audit
from tests.test_openvl_routed import CONFIG


def belief():
    return {'epistemic': [{'id': 'e-grasp', 'target': 'grasp alignment unknown'},
                           {'id': 'e-place', 'target': 'placement geometry unknown'}]}


def proposal(**kw):
    return {'next_action': 'Choose a grasp after checking its local alignment',
            'needed_now': ['e-grasp'], 'deferred': ['e-place'],
            'observation_question': 'Which candidate matches the intended body?',
            'answer_changes': 'Use the matched candidate or change the view', **kw}


def enabled(robo, tmp_path, reflection=False):
    kit = setup(robo, tmp_path, reflection=reflection)
    kit.general_harness = CompositeHarness(kit, {'kind': 'openvl_pos_reflact',
        'openvl': {'config': str(CONFIG)}, 'reflact': {'enabled': reflection},
        'action_focus': {'enabled': True}}, 'Put the block on the plate')
    return kit


def test_scopes_gaps_without_rewriting_or_certifying_readiness():
    b=belief();original=copy.deepcopy(b);f=ActionFocus()
    f.begin(proposal(), {'goal':'a-root','phase':'observe'}, b, 4, 0)
    c=f.context(b,5,0,'a-root')
    assert [g['id'] for g in c['needed_now_still_unresolved']]==['e-grasp']
    assert [g['id'] for g in c['deferred_still_unresolved']]==['e-place']
    assert c['host_readiness_or_feasibility_verdict'] is None and b==original


@pytest.mark.parametrize('patch', [
    {'needed_now':['invented']}, {'needed_now':['e-place']},
    {'needed_now':['e-grasp','e-grasp']}, {'needed_now':'e-grasp'},
    {'needed_now':[{}]}, {'next_action':''}, {'next_action':'x'*241},
    {'observation_question':None}, {'answer_changes':None},
    {'extra':'undeclared'}, {'deferred':list(range(7))},
])
def test_bad_advice_is_not_adopted(patch):
    f=ActionFocus();f.begin(proposal(**patch),{'goal':'g','phase':'observe'},belief(),1,0)
    assert f.warning and f.proposal is None


def test_missing_optional_plan_is_no_readiness_claim():
    f=ActionFocus();f.begin(None,{'goal':'g','phase':'grasp'},belief(),1,0)
    c=f.context(belief(),1,0,'g')
    assert c['proposal'] is None and c['host_readiness_or_feasibility_verdict'] is None


def test_empty_needed_list_and_removed_gap_never_certify_measurement():
    f=ActionFocus();f.begin(proposal(needed_now=[],observation_question=None,answer_changes=None),
                          {'goal':'g','phase':'grasp'},belief(),1,0)
    b={'epistemic':[]};c=f.context(b,2,1,'g')
    assert c['host_readiness_or_feasibility_verdict'] is None
    assert c['referenced_ids_no_longer_in_audited_epistemic']==['e-place']
    assert not c['same_motion_epoch']


def test_prior_proposal_is_source_stamped_when_new_advice_is_invalid():
    f=ActionFocus();f.begin(proposal(),{'goal':'g','phase':'observe'},belief(),1,0)
    f.begin({'bad':'metadata'},{'goal':'g','phase':'observe'},belief(),5,2)
    c=f.context(belief(),5,2,'other')
    assert c['proposal']['sequence']==1 and not c['same_motion_epoch']
    assert not c['same_active_goal'] and c['metadata_warning']


def test_repetition_diagnostic_never_means_no_information_gain():
    f=ActionFocus()
    for seq in range(3):
        f.begin(proposal(),{'goal':'g','phase':'observe'},belief(),seq,0)
        f.outcome({'action_outcomes':[]},seq+1,0,1,-1)
    assert f.same_focus_without_motion==3
    assert f.last_outcome['causal_use_or_goal_success_proven'] is False
    f.begin(proposal(next_action='Change the observation'),{'goal':'g','phase':'observe'},belief(),4,0)
    assert f.same_focus_without_motion==0


def test_open_false_and_close_false_keep_distinct_event_semantics():
    f=ActionFocus()
    actions=[{'call':i,'tool':'grip','parameters':{'gripper':command},
              'feedback':{'holding':False,'steps_used':1}} for i,command in [(1,1),(2,-1)]]
    f.outcome({'action_outcomes':actions},3,2,1,-1)
    assert [r['command_event'] for r in f.last_outcome['grip_events']]==['close','open']
    assert all(r['physical_success_inferred'] is False for r in f.last_outcome['grip_events'])


@pytest.mark.parametrize('reply,names,missing', [
    (None,['xyz'],['xyz']), ({'xyz':None},['xyz'],['xyz']),
    ({'xyz':[0,0,0],'score':0},['xyz','score'],[]),
    ({'reached':False,'blocked':True},['reached','blocked','holding'],['holding']),
])
def test_reusable_field_recipe_preserves_unknown_and_false(reply,names,missing):
    scope={};exec(FIELD_RECIPE,scope)
    original=copy.deepcopy(reply);out=scope['read_public_fields'](reply,names)
    assert out['missing']==missing and reply==original
    if isinstance(reply,dict) and 'reached' in reply:assert out['values']['reached'] is False


@pytest.mark.parametrize('command,event', [(1,'close'),(-1,'open'),(0,'other')])
def test_reusable_grip_recipe_does_not_infer_identity_or_success(command,event):
    scope={};exec(GRIP_RECIPE,scope)
    out=scope['describe_grip_reply']({'holding':False},command,1,-1)
    assert out['command_event']==event and out['reported_holding'] is False
    assert out['reported_width'] is None and out['identity_verified'] is False


def test_enabled_metadata_optional_and_no_extra_tools(robo,tmp_path):
    kit=enabled(robo,tmp_path);tools=kit.get_tools_spec()
    schema=next(t['input_schema'] for t in tools if t['name']=='python')
    assert schema['required']==['code'] and 'step' not in schema['required']
    assert 'focus' in schema['properties']['step']['properties']
    assert 'focus' not in schema['properties']['step']['required']
    assert len(tools)==5 and kit.general_harness.reflection is None


def test_bad_focus_does_not_block_duplicate_or_modify_actions(robo,backend,tmp_path):
    kit=enabled(robo,tmp_path)
    out=kit.execute_tool('python',{'code':'robo.move_to([0,0,.6])',
         'step':{'goal':'task-root','phase':'grasp','expected':'Read feedback',
                 'focus':{'bad':'schema'}}})
    assert sum(k=='move_to' for k,_ in backend.calls)==1 and kit._cell_index==1
    assert kit.general_harness.action_focus.warning
    assert not out.result.get('general_harness_error')
    assert out.result['composite_decision']['action_focus']['host_readiness_or_feasibility_verdict'] is None


def test_current_and_later_gaps_reach_same_original_auxiliary_chain(robo,tmp_path):
    kit=enabled(robo,tmp_path);h=kit.general_harness
    # Schema-valid existing audited epistemic records, not asserted geometry.
    h.belief['epistemic']=[{'id':'e-grasp','target':'grasp pose unknown','entities':[]},
                         {'id':'e-place','target':'placement pose unknown','entities':[]}]
    h.belief['active_gap']={'type':'epistemic','id':'e-grasp'}
    kit.execute_tool('python',{'code':'print("read-only")',
        'step':{'goal':'task-root','phase':'observe','expected':'Read the decision question','focus':proposal()}})
    before=copy.deepcopy(h.belief);calls=[]
    d={'goal_id':'a-root','mode':'observe','next_step':'Inspect the local grasp candidate.',
       'decision_critical_missing':'grasp pose','why_change':'Do not repeat the whole scene.'}
    script=Script(copy.deepcopy(before),{**audit(None),'directive':d})
    def model(**kw):
        calls.append(copy.deepcopy(kw));return script(**kw)
    h.before_request(model)
    assert [x['role'] for x in calls]==['pos_belief','pos_sentinel']
    assert all(x['payload']['action_focus']['proposal']['executor_proposal']['deferred']==['e-place'] for x in calls)
    assert h.belief['epistemic']==before['epistemic']
    assert all('action_focus is INPUT context' in x['prompt'] for x in calls)


def test_no_focus_and_no_new_gap_required_to_execute(robo,backend,tmp_path):
    kit=enabled(robo,tmp_path)
    kit.execute_tool('python',{'code':'robo.move_to([0,0,.6])'})
    assert sum(k=='move_to' for k,_ in backend.calls)==1 and kit._cell_index==1
    assert kit.general_harness.action_focus.proposal is None


def test_invalid_reflection_does_not_submit_focus_or_run_code(robo,backend,tmp_path):
    kit=enabled(robo,tmp_path,reflection=True)
    kit.execute_tool('python',{'code':'robo.move_to([0,0,.6])','step':{'focus':proposal()}})
    assert not any(k=='move_to' for k,_ in backend.calls)
    assert kit.general_harness.action_focus.proposal is None


@pytest.mark.parametrize('value', [False,1,'yes',None,{'enabled':1},{'enabled':True,'extra':1}])
def test_bad_config_rejected_without_changing_legacy(value):
    with pytest.raises(ValueError):
        validate_options({'kind':'openvl_pos_reflact','openvl':{'enabled':False},'action_focus':value})


def test_default_disabled_preserves_original_prompt_and_schema(robo,tmp_path):
    kit=setup(robo,tmp_path,reflection=False);h=kit.general_harness
    assert h.action_focus is None and 'action_focus' not in h.decision_context()
    schema=next(t['input_schema']for t in kit.get_tools_spec()if t['name']=='python')
    assert 'focus' not in schema['properties']['step']['properties']
    assert '# Action-scoped gaps' not in h.prompt()
