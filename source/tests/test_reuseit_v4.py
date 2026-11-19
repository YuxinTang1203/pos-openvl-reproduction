import pytest
from tests.test_reuseit_rlm_ports import kit_for,guard,workflow_config


def test_readonly_diagnostic_rechecks_without_physical_retry(robo,backend,tmp_path,workflow_config):
    kit=kit_for(robo,tmp_path,workflow_config);h=kit.general_harness
    guard(h,False);assert h.phase=='fallback'
    steps=len(backend.steps)
    result=kit.execute_tool('reuseit_observe',{'queries':[{'query':'black bowl','camera':robo.CAMERAS[0]}]})
    assert 'error' not in result.result and result.result['readonly_diagnostics']
    assert h.phase=='check_pre' and h.retries==0 and len(backend.steps)==steps
    assert h.last_feedback['public_calls'] and all(not c['stateful'] for c in h.last_feedback['public_calls'])
    guard(h)
    assert h.phase=='execute' and h.retries==0 and kit._cell_index==0


@pytest.mark.parametrize('queries', [[{'query':'x','camera':'invalid-camera'}],[{'query':'x','code':'robo.move_to(...)'}],[{'query':'x'}]*4])
def test_diagnostics_validate_queries_before_any_call(robo,backend,tmp_path,workflow_config,queries):
    kit=kit_for(robo,tmp_path,workflow_config);h=kit.general_harness
    guard(h,False);before=len(backend.calls)
    result=kit.execute_tool('reuseit_observe',{'queries':queries})
    assert 'error' in result.result and len(backend.calls)==before and h.phase=='fallback'


def test_atomic_task_specific_action_covers_missing_skill_and_keeps_goal_audit(robo,backend,tmp_path,workflow_config):
    kit=kit_for(robo,tmp_path,{**workflow_config,'task_plan':True,'plan_skill_ids':['observe'],'allow_task_actions':True})
    h=kit.general_harness
    assert 'task_action' in h.task_plan.skills
    proposal={'requirements':[{'id':'press','text':'Press the explicitly named control after placement'}],
              'units':[{'id':'press_after_place','skill_id':'task_action','objective':'Press the named control once after placement','requirements':['press']}]}
    kit.execute_tool('reuseit_plan',proposal)
    def model(role,prompt,payload,images):
        if role=='reuseit_plan_audit':return {'approved':True,'issues':[]}
        assert role=='reuseit_goal_audit'
        return {'satisfied':False,'evidence':'Synthetic fixture has not actually pressed a control',
                'evidence_ids':[payload['current_observation']['images'][0]['id']],'missing_requirements':['press']}
    h.before_request(model);assert h.phase=='execute'
    kit.execute_tool('reuseit_step',{'step_id':'press_after_place','mode':'action','code':"print('synthetic protocol only')"})
    h.before_request(model)
    assert h.phase=='plan' and not backend.solved()
    assert 'error' in kit.execute_tool('finish',{'status':'success','summary':'skip audit'}).result


def test_task_specific_actions_require_audited_plan_mode(robo,tmp_path,workflow_config):
    with pytest.raises(ValueError,match='task_plan'):
        kit_for(robo,tmp_path,{**workflow_config,'allow_task_actions':True})


def test_diagnostic_uses_locate_when_backend_provides_it(robo,backend,tmp_path,workflow_config,monkeypatch):
    seen=[]
    def locate(query,camera):
        seen.append((query,camera))
        return robo.segment(query,camera=camera)
    monkeypatch.setattr(robo,'locate',locate,raising=False)
    kit=kit_for(robo,tmp_path,workflow_config);h=kit.general_harness
    guard(h,False);steps=len(backend.steps)
    result=kit.execute_tool('reuseit_observe',{'queries':[{'query':'black bowl'}]})
    assert seen==[('black bowl',robo.CAMERAS[0])]
    assert result.result['readonly_diagnostics'][0]['method']=='locate'
    assert isinstance(result.result['readonly_diagnostics'][0]['result'],dict)
    assert len(backend.steps)==steps and h.retries==0
