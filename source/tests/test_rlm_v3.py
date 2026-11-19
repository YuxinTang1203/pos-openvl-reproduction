"""Synthetic public-evidence protocol checks, not robot success claims."""
import copy
import json

import pytest

from tests.test_reuseit_rlm_ports import kit_for, run, call, report, end
from pyrualean.migrations import validate_migration


def make(robo, tmp_path, **options):
    kit = kit_for(robo, tmp_path, {'kind': 'rlm', 'task_ledger': True, **options})
    kit.general_harness.before_request(lambda *a: pytest.fail('unexpected model call'))
    return kit


def proposal(kit, count=2):
    h = kit.general_harness
    reference = h.ledger.initial_reference()
    iid = next(i['id'] for e in reference for i in e['images'])
    return {'initial_facts': [{'id': 'start', 'claim': 'Synthetic initial scene',
                              'status': 'observed', 'evidence_ids': [iid]}],
            'requirements': [{'id': 'r1', 'text': 'Complete each phase in order'}],
            'steps': [{'id': f's{i}', 'objective': f'Synthetic phase {i}',
                       'requirements': ['r1'], 'success_condition': f'Observed phase {i} outcome'}
                      for i in range(count)]}


def reviewer(satisfied=True, approved=True, stale=False):
    def model(role, prompt, payload, images):
        assert images and 'instruction' in payload
        if role == 'rlm_plan_review':
            assert 'previous_remaining' in payload and 'remaining' not in payload
            return {'approved': approved, 'issues': [] if approved else ['Missing a required interaction']}
        assert role == 'rlm_checkpoint_review'
        ids = ([payload['initial_reference'][1]['images'][0]['id']] if stale else
               [payload['current_observation']['images'][0]['id']])
        return {'satisfied': satisfied, 'reason': 'Synthetic controlled observation', 'evidence_ids': ids}
    return model


def approve(kit, count=2):
    p = proposal(kit, count)
    r = kit.execute_tool('rlm_plan', p)
    assert 'error' not in r.result
    kit.general_harness.before_request(reviewer())
    return p


def checkpoint(kit):
    step = kit.general_harness.ledger.head()
    return kit.execute_tool('rlm_checkpoint', {'step_id': step['id'], 'claim': 'Observed outcome', 'evidence_ids': []})


def test_v3_options_are_explicit_and_old_mode_unchanged(robo, tmp_path):
    for bad in ({'task_ledger': 1}, {'max_ledger_reviews': 3}, {'task_ledger': True, 'max_plan_revisions': 0}):
        with pytest.raises(ValueError):
            validate_migration({'kind': 'rlm', **bad})
    kit = kit_for(robo, tmp_path, {'kind': 'rlm'})
    assert kit.general_harness.ledger is None
    assert [s['name'] for s in kit.get_tools_spec()] == ['python', 'finish', 'rlm']
    assert kit.general_harness.provenance['implementation'] == 'pyrualean-rlm-v2-isolated-worker'
    kit.general_harness.close()


def test_ordered_milestones_gate_finish_and_final_audit(robo, backend, tmp_path):
    kit = make(robo, tmp_path);h = kit.general_harness
    p = approve(kit)
    assert 'error' in kit.execute_tool('finish', {'status': 'success', 'summary': 'skip'}).result
    # Successful code execution does not complete the current milestone.
    kit.execute_tool('python', {'code': 'answer=46'})
    assert h.ledger.head()['id'] == 's0'
    assert 'error' in kit.execute_tool('rlm_checkpoint', {'step_id': 's1', 'claim': 'skip', 'evidence_ids': []}).result
    checkpoint(kit);h.before_request(reviewer())
    assert h.ledger.head()['id'] == 's1'
    assert h.context()['task_ledger']['initial_facts'] == p['initial_facts']
    checkpoint(kit);h.before_request(reviewer())
    assert not h.ledger.steps and not kit.finish
    kit.execute_tool('finish', {'status': 'success', 'summary': 'all supported'})
    assert h.ledger.pending['kind'] == 'goal' and not kit.finish
    h.before_request(reviewer())
    kit.execute_tool('finish', {'status': 'success', 'summary': 'model claim only'})
    assert kit.finish and not backend.solved()
    h.close()


@pytest.mark.parametrize('satisfied,stale', [(False, False), (None, False), (True, True), (1, False)])
def test_unsupported_or_stale_review_never_advances(robo, tmp_path, satisfied, stale):
    kit = make(robo, tmp_path);h=kit.general_harness
    approve(kit, 1);checkpoint(kit);h.before_request(reviewer(satisfied=satisfied, stale=stale))
    assert h.ledger.head()['id'] == 's0' and not h.ledger.completed
    assert h.ledger.last_review['accepted'] is False
    h.close()


def test_rejected_plan_and_skipped_maintenance_do_not_enable_finish(robo, tmp_path):
    kit=make(robo,tmp_path);h=kit.general_harness
    kit.execute_tool('rlm_plan',proposal(kit));h.before_request(reviewer(approved=False))
    assert not h.ledger.requirements and h.phase=='planning'
    assert 'error' in kit.execute_tool('finish',{'status':'success','summary':'skip'}).result
    kit.execute_tool('rlm_plan',proposal(kit));h.before_request(lambda *a:None)
    assert not h.ledger.requirements and not h.ledger.last_review['valid_report']
    h.close()


def test_replan_preserves_original_reference_and_requirements(robo, tmp_path):
    kit=make(robo,tmp_path);h=kit.general_harness;p=approve(kit)
    checkpoint(kit);h.before_request(reviewer())
    for field in ['initial_facts', 'requirements']:
        bad=copy.deepcopy(p)
        bad[field][0]['claim' if field=='initial_facts' else 'text']='substituted'
        bad['steps']=bad['steps'][1:]
        assert 'error' in kit.execute_tool('rlm_plan',bad).result
    p['steps']=p['steps'][1:]
    p['steps'][0]['objective']='Recover remaining phase'
    kit.execute_tool('rlm_plan',p);h.before_request(reviewer())
    assert h.ledger.head()['objective']=='Recover remaining phase' and len(h.ledger.completed)==1
    assert len(h.context()['task_ledger']['completed'])==1
    h.close()


def test_initial_reference_excludes_late_images_and_queued_self_references(robo,tmp_path):
    kit=make(robo,tmp_path);h=kit.general_harness
    h.trace.append('tool', {'tool':'rlm', 'arguments':{'code':'fake observation'},'public_calls':[]})
    h.trace.epoch+=1
    later=h.trace.capture(robo)
    p=proposal(kit);p['initial_facts'][0]['evidence_ids']=[later['images'][0]['id']]
    assert 'error' in kit.execute_tool('rlm_plan',p).result
    assert all(e['epoch']==0 for e in h.ledger.initial_reference())
    assert not any(e['payload'].get('tool')=='rlm' for e in h.ledger.initial_reference())
    h.close()


def test_review_and_revision_budgets_and_abort(robo,tmp_path):
    kit=make(robo,tmp_path,max_ledger_reviews=3,max_plan_revisions=1);h=kit.general_harness
    approve(kit,1)
    checkpoint(kit);h.before_request(reviewer(satisfied=False))
    checkpoint(kit);h.before_request(reviewer(satisfied=False))
    assert 'error' in checkpoint(kit).result
    kit.execute_tool('finish',{'status':'stuck','summary':'review budget exhausted'})
    assert kit.finish['status']=='stuck'
    h.close()


def test_plan_reserves_final_review_and_readonly_child_has_ledger(robo,tmp_path):
    kit=make(robo,tmp_path,max_ledger_reviews=3);h=kit.general_harness
    assert 'error' in kit.execute_tool('rlm_plan',proposal(kit,2)).result
    approve(kit,1)
    kit.execute_tool('rlm',{'code':"print(context['task_ledger']['current_step']['id'])",'reason':'plan_revision'})
    h.before_request(lambda *a:pytest.fail('unexpected model call'))
    assert h.last_analysis=={'stdout':'s0\n'}
    h.close()


def test_final_review_can_reject_and_actions_invalidate_supported_goal(robo,tmp_path):
    kit=make(robo,tmp_path);h=kit.general_harness;approve(kit,1)
    checkpoint(kit);h.before_request(reviewer())
    args={'status':'success','summary':'claim'}
    kit.execute_tool('finish',args);h.before_request(reviewer(satisfied=False))
    assert not h.ledger.goal_supported and not kit.finish
    kit.execute_tool('finish',args);h.before_request(reviewer())
    assert h.ledger.goal_supported
    kit.execute_tool('python',{'code':'answer=1'})
    assert not h.ledger.goal_supported
    h.close()


def test_demand_driven_analysis_reason_and_duplicate_result(robo,tmp_path):
    kit=make(robo,tmp_path);h=kit.general_harness
    assert 'error' in kit.execute_tool('rlm',{'code':'print(46)'}).result
    assert 'error' in kit.execute_tool('rlm',{'code':'print(46)','reason':'every_action'}).result
    args={'code':'print(46)','reason':'history_missing'}
    kit.execute_tool('rlm',args);h.before_request(lambda *a:pytest.fail('unexpected recursion'))
    assert h.last_analysis=={'stdout':'46\n'}
    assert 'error' in kit.execute_tool('rlm',args).result
    assert h.analysis_calls==1
    h.trace.capture(robo)
    assert 'error' not in kit.execute_tool('rlm',args).result
    h.before_request(lambda *a:pytest.fail('unexpected recursion'))
    assert h.analysis_calls==2
    h.close()


def test_plan_reviewer_distinguishes_candidate_from_empty_previous_plan(robo,tmp_path):
    kit=make(robo,tmp_path);h=kit.general_harness;p=proposal(kit,1)
    kit.execute_tool('rlm_plan',p)
    def review(role,prompt,payload,images):
        assert payload['previous_remaining']==[] and payload['proposal']['steps']==p['steps']
        assert 'NEW proposed remaining' in prompt
        return {'approved':True,'issues':[]}
    h.before_request(review)
    kit.execute_tool('python',{'code':"print('synthetic-observation')"})
    ledger=h.ledger.public_calls()
    assert ledger['entries'][-1]['stdout'].strip()=='synthetic-observation'
    assert ledger['entries'][-1]['calls']==[]
    h.close()


def test_current_observation_event_and_initial_reference_are_registered_evidence(robo,tmp_path):
    kit=make(robo,tmp_path);h=kit.general_harness;approve(kit,1);checkpoint(kit)
    def model(role,prompt,payload,images):
        current=payload['current_observation']['id']
        initial=payload['initial_reference'][0]['id']
        assert current in payload['fresh_evidence_ids'] and initial in payload['eligible_evidence_ids']
        return {'satisfied':True,'reason':'Synthetic initial and current observation','evidence_ids':[current,initial]}
    h.before_request(model)
    assert not h.ledger.steps and h.ledger.last_review['accepted']
    h.close()


def test_unknown_evidence_id_and_missing_fresh_citation_have_actionable_feedback(robo,tmp_path):
    kit=make(robo,tmp_path);h=kit.general_harness;approve(kit,1)
    checkpoint(kit)
    h.before_request(lambda *a:{'satisfied':True,'reason':'invented proof','evidence_ids':['not-registered']})
    assert not h.ledger.last_review['accepted'] and 'registered evidence' in h.ledger.last_review['rejection_reason']
    checkpoint(kit)
    old=h.ledger.initial_reference()[0]['id']
    h.before_request(lambda *a:{'satisfied':True,'reason':'old proof only','evidence_ids':[old]})
    assert not h.ledger.last_review['accepted'] and 'fresh observation' in h.ledger.last_review['rejection_reason']
    assert h.ledger.last_review['fresh_evidence_ids']
    h.close()


def test_unprocessed_review_aborts_instead_of_spinning_main_budget(robo,tmp_path):
    kit=make(robo,tmp_path);h=kit.general_harness;p=proposal(kit,1)
    transport,result=run(kit,tmp_path,[call('rlm_plan',p),call('python',{'code':'must_not_run=1'}),end()],
                         auxiliary_budget_s=0)
    assert kit.finish['status']=='stuck' and h.closed
    assert len(transport.requests)==2 and result['auxiliary_requests']==0
    assert 'must_not_run' not in kit._globals and not h.ledger.completed
    assert 'ledger_review_unavailable' in (h.directory/'events.jsonl').read_text()
    kit=make(robo,tmp_path/'analysis');h=kit.general_harness
    transport,result=run(kit,tmp_path/'analysis',[
        call('rlm',{'code':'print(46)','reason':'history_missing'}),
        call('python',{'code':'must_not_run=1'}),end()],auxiliary_budget_s=0)
    assert kit.finish['status']=='stuck' and h.closed and len(transport.requests)==2
    assert 'must_not_run' not in kit._globals and not h.last_analysis


def test_child_protocol_one_repair_shared_budget_and_explicit_tool_prompt(robo,tmp_path):
    kit=make(robo,tmp_path);h=kit.general_harness
    requests=[]
    answers=iter([None,{'action':'final','answer':'46'}])
    def model(role,prompt,payload,images):
        assert role=='rlm_query' and 'EXACTLY ONCE' in prompt
        requests.append(payload)
        return next(answers)
    assert h._query(model,'sum',[17,29],[],1)=='46'
    assert h.subcalls==2 and 'No valid report' in requests[1]['repl_feedback']['error']
    with pytest.raises(ValueError,match='unavailable'):
        h._query(lambda *a:None,'sum',[17,29],[],1)
    assert h.subcalls==4
    h.close()


def test_shared_runtime_ledger_survives_trimming_without_automatic_analysis(robo,backend,tmp_path):
    kit=make(robo,tmp_path,window_rounds=1);h=kit.general_harness;p=proposal(kit,1)
    from pyrualean.openai_runtime import run_openai_agent
    from pyrualean.migrations import migration_prompt
    requests=[]
    def transport(url,body,timeout):
        requests.append(body)
        if body['tools'][0]['function']['name']=='submit_harness_state':
            message=body['messages'][1]['content']
            if isinstance(message,list):message=''.join(x.get('text','') for x in message if x['type']=='text')
            payload=json.loads(message.split('\nAttached image labels')[0])
            if 'proposal' in payload:return 200,report({'approved':True,'issues':[]})
            return 200,report({'satisfied':True,'reason':'Synthetic current observation',
                              'evidence_ids':[payload['current_observation']['images'][0]['id']]})
        if not h.ledger.requirements:return 200,call('rlm_plan',p)
        if h.ledger.head():return 200,call('rlm_checkpoint',{'step_id':'s0','claim':'observed','evidence_ids':[]})
        return 200,end('success')
    result=run_openai_agent(migration_prompt('synthetic task',kit),[],kit,model='same-model',
        base_url='http://unused/v1',api_key=None,transport=transport,max_requests=12,budget_s=60,
        max_turns=12,out_dir=tmp_path/'out',retries=0,tool_choice='auto')
    assert not result['errors'] and result['auxiliary_requests']==3
    assert h.analysis_calls==0 and h.subcalls==0 and result['compactions']>0
    assert kit.finish['status']=='success' and not backend.solved()
    assert h.context()['task_ledger']['completed'][0]['step']['id']=='s0'
    assert all(b['model']=='same-model' for b in requests)
