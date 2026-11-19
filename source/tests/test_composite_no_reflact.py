import copy
from tests.test_composite_harness import setup,events
from tests.test_pos_harness import Script,audit
from tests.test_openai_runtime import ScriptedTransport,_reply
from pyrualean.openai_runtime import run_openai_agent


def test_disabled_schema_and_provenance_have_no_reflection_requirement(robo,tmp_path):
    kit=setup(robo,tmp_path,reflection=False)
    h=kit.general_harness
    assert h.reflection is None and h.provenance['reflact'] is None
    assert h.visual_enabled and kit.evidence_adapter is not None
    schema=next(t['input_schema'] for t in kit.get_tools_spec() if t['name']=='python')
    assert schema['required']==['code']
    assert not {'observed_state','goal_gap'} & set(schema['properties'])
    assert 'ReflAct summaries are disabled' in h.prompt()
    assert "ReflAct's observed_state/goal_gap" not in h.prompt()


def test_disabled_executes_code_once_and_preserves_feedback(robo,backend,tmp_path):
    kit=setup(robo,tmp_path,visual=False,reflection=False)
    out=kit.execute_tool('python',{'code':'robo.move_to([0,0,.6])'})
    assert sum(k=='move_to' for k,_ in backend.calls)==1
    assert kit._cell_index==1 and kit.general_harness.motion_epoch>0
    assert out.result['task_progress'] and out.result['composite_decision']
    assert events(kit,'decision_bound')
    assert not events(kit,'reflection_recorded') and not events(kit,'reflection_rejected')


def test_disabled_visual_results_still_enter_pos_with_sources(robo,tmp_path):
    kit=setup(robo,tmp_path,reflection=False)
    out=kit.execute_tool('python',{'code':"robo.segment('blocks')"})
    assert events(kit,'visual_evidence_linked')
    assert any(e.get('openvl_source') for e in kit.general_harness.evidence.values())
    assert not out.result.get('general_harness_error')


def test_disabled_preserves_audited_pos_commit_and_archives_after_motion(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False,reflection=False);h=kit.general_harness
    d={'goal_id':'a-root','mode':'act','next_step':'Inspect before placement.',
       'decision_critical_missing':None,'why_change':'Current evidence.'}
    h.before_request(Script(copy.deepcopy(h.belief),{**audit(None),'directive':d}))
    assert h.revision==1 and h.decision_context()['directive']==d
    kit.execute_tool('python',{'code':'robo.move_to([0,0,.6])'})
    assert h.decision_context()['directive'] is None
    assert h.decision_context()['historical_directive']==d
    assert events(kit,'committed') and events(kit,'directive_archived')


def test_disabled_runtime_code_only_calls_and_no_reflection_events(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False,reflection=False)
    transport=ScriptedTransport([
        _reply([('read','python',{'code':'print(robo.state())'})]),
        _reply([('end','finish',{'status':'stuck','summary':'Read-only interface check.'})])])
    r=run_openai_agent(kit.general_harness.prompt(),[],kit,model='offline',
        base_url='http://unused/v1',api_key=None,transport=transport,
        budget_s=60,max_turns=3,max_requests=3,auxiliary_budget_s=0,out_dir=tmp_path/'runtime')
    assert not r['errors'] and r['python_cells']==1
    assert kit.finish['status']=='stuck'
    assert not events(kit,'reflection_recorded') and not events(kit,'reflection_rejected')


def test_disabled_keeps_success_audit_requirement(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False,reflection=False)
    out=kit.execute_tool('finish',{'status':'success','summary':'No evidence.'})
    assert out.result['completion_review_pending'] and kit.finish is None
