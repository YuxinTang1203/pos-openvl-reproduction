import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_reuseit_rlm_ports import kit_for, workflow_config
from pyrualean.analysis_repl import AnalysisREPL


def planned(robo, tmp_path, config):
    return kit_for(robo, tmp_path, {**config, 'task_plan': True, 'plan_skill_ids': ['observe']})


def plan(n=2):
    return {'requirements': [{'id': 'all', 'text': 'Complete all ordered phases for all named objects'}],
            'units': [{'id': f'unit{i}', 'skill_id': 'observe', 'objective': f'Execute ordered phase {i}',
                       'requirements': ['all']} for i in range(n)]}


def model(goal=False, approved=True, stale=False):
    def call(role, prompt, payload, images):
        if role == 'reuseit_plan_audit':
            assert payload['instruction']
            return {'approved': approved, 'issues': [] if approved else ['Required last phase omitted']}
        if role == 'reuseit_goal_audit':
            assert payload['instruction'] and payload['completed_units']
            ids = [payload['initial_reference']['images'][0]['id'] if stale else payload['current_observation']['images'][0]['id']]
            return {'satisfied': goal, 'evidence': 'Synthetic controlled fixture, not a physical claim',
                    'evidence_ids': ids, 'missing_requirements': [] if goal else ['all']}
        return {'satisfied': True, 'evidence': 'Synthetic current observation',
                'evidence_ids': payload['eligible_evidence_ids']}
    return call


def execute_current(kit):
    h = kit.general_harness
    return kit.execute_tool('reuseit_step', {'step_id': h._step()['id'], 'mode': 'action', 'code': 'x=1'})


def test_repeated_skills_and_independent_goal_block_partial_completion(robo, backend, tmp_path, workflow_config):
    kit = planned(robo, tmp_path, workflow_config); h = kit.general_harness
    assert h.phase == 'plan'
    assert 'error' in kit.execute_tool('reuseit_step', {'step_id': 'observe', 'mode': 'action', 'code': 'x=1'}).result
    kit.execute_tool('reuseit_plan', plan())
    h.before_request(model())
    assert h.phase == 'execute' and h._step()['id'] == 'unit0'
    execute_current(kit); h.before_request(model())
    assert h.phase == 'execute' and h._step()['id'] == 'unit1'
    execute_current(kit); h.before_request(model())
    assert h.phase == 'plan' and len(h.task_plan.completed) == 2
    assert 'error' in kit.execute_tool('finish', {'status': 'success', 'summary': 'partial'}).result
    continuation = plan(1); continuation['units'][0]['id'] = 'remaining'
    kit.execute_tool('reuseit_plan', continuation); h.before_request(model())
    execute_current(kit); h.before_request(model(goal=True))
    assert h.phase == 'complete' and len(h.task_plan.completed) == 3
    kit.execute_tool('finish', {'status': 'success', 'summary': 'model claim'})
    assert kit.finish and not backend.solved()


def test_plan_audit_rejects_omitted_order_before_motion(robo, tmp_path, workflow_config):
    kit = planned(robo, tmp_path, workflow_config);h = kit.general_harness
    kit.execute_tool('reuseit_plan', plan());h.before_request(model(approved=False))
    assert h.phase == 'plan' and kit._cell_index == 0
    assert h.task_plan.last_review['report']['issues']


@pytest.mark.parametrize('mutation', ['unknown_skill', 'duplicate_unit', 'uncovered_requirement'])
def test_invalid_plan_never_admits_execution(robo, tmp_path, workflow_config, mutation):
    kit = planned(robo, tmp_path, workflow_config); p = plan()
    if mutation == 'unknown_skill': p['units'][0]['skill_id'] = 'unregistered'
    if mutation == 'duplicate_unit': p['units'][1]['id'] = p['units'][0]['id']
    if mutation == 'uncovered_requirement': p['requirements'].append({'id': 'missing', 'text': 'another phase'})
    assert 'error' in kit.execute_tool('reuseit_plan', p).result
    assert kit.general_harness.phase == 'plan' and kit._cell_index == 0


def test_continuation_cannot_weaken_original_requirements(robo, tmp_path, workflow_config):
    kit = planned(robo, tmp_path, workflow_config);h=kit.general_harness
    kit.execute_tool('reuseit_plan', plan(1));h.before_request(model())
    execute_current(kit);h.before_request(model())
    p=plan(1);p['requirements'][0]['text']='Do only one object'
    assert 'error' in kit.execute_tool('reuseit_plan',p).result


def test_goal_audit_requires_current_visual_evidence(robo, tmp_path, workflow_config):
    kit = planned(robo, tmp_path, workflow_config);h=kit.general_harness
    kit.execute_tool('reuseit_plan',plan(1));h.before_request(model())
    execute_current(kit);h.before_request(model(goal=True,stale=True))
    assert h.phase == 'goal_uncertain'
    assert 'error' in kit.execute_tool('finish',{'status':'success','summary':'stale'}).result
    kit.execute_tool('reuseit_observe',{});h.before_request(model(goal=True))
    assert h.phase == 'complete'


def test_analysis_worker_is_lightweight_persistent_and_scrubs_env(monkeypatch):
    monkeypatch.setenv('RLM_TEST_SECRET','not-for-child')
    kernel=AnalysisREPL(timeout_s=2)
    try:
        assert kernel.run('answer=sum(context); print(answer)', [17,29]+[0]*1000000,None)=={'stdout':'46\n'}
        env=Path(f'/proc/{kernel.process.pid}/environ').read_bytes()
        assert b'RLM_TEST_SECRET' not in env
        assert kernel.run('print(answer)',[],None)=={'stdout':'46\n'}
        result=kernel.run("print(llm_query('sum', context))",[20,26],lambda q,c,ids:str(sum(c)))
        assert result=={'stdout':'46\n'}
    finally:kernel.close()


def test_analysis_does_not_reimport_heavy_application_entrypoint(tmp_path):
    import pyrualean.analysis_repl as module
    package_root=str(Path(module.__file__).parent.parent)
    script=tmp_path/'app.py'
    script.write_text('''import torch
import json
if __name__ == '__mp_main__':
    raise RuntimeError('application must never be re-imported by analysis worker')
from pyrualean.analysis_repl import AnalysisREPL
if __name__ == '__main__':
    k=AnalysisREPL(timeout_s=3)
    try:
        print(json.dumps(k.run('print(sum(context))',[17,29]+[0]*1000000,None)))
    finally:
        k.close()
''')
    env={**os.environ,'PYTHONPATH':package_root,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    result=subprocess.run([sys.executable,str(script)],env=env,capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout.strip())=={'stdout':'46\n'}


def test_runtime_stops_before_provider_when_worker_preflight_fails(robo,tmp_path,monkeypatch):
    from tests.test_reuseit_rlm_ports import run,end
    kit=kit_for(robo,tmp_path,{'kind':'rlm'})
    monkeypatch.setattr(kit.general_harness.kernel,'run',lambda *a,**kw:{'error':'EOFError: ','namespace_reset':True})
    transport,result=run(kit,tmp_path,[end()])
    assert not transport.requests and result['errors']
    assert 'preflight failed' in result['errors'][0]


def test_runtime_stops_on_worker_transport_failure_after_preflight(robo,tmp_path,monkeypatch):
    from tests.test_reuseit_rlm_ports import run,call,end
    kit=kit_for(robo,tmp_path,{'kind':'rlm'})
    responses=iter([{'stdout':'rlm_worker_ready\n'},
                    {'error':'ConnectionResetError: test','namespace_reset':True}])
    monkeypatch.setattr(kit.general_harness.kernel,'run',lambda *a,**kw:next(responses))
    transport,result=run(kit,tmp_path,[call('rlm',{'code':'print(context)'}),end()])
    assert len(transport.requests)==1 and result['errors']
    assert 'infrastructure failure' in result['errors'][0]


def test_task_plan_shared_runtime_covers_plan_guard_and_final_audit(robo,backend,tmp_path,workflow_config):
    from pyrualean.openai_runtime import run_openai_agent
    from pyrualean.migrations import migration_prompt
    from tests.test_reuseit_rlm_ports import call,report,end
    kit=planned(robo,tmp_path,workflow_config);h=kit.general_harness;requests=[]
    def transport(url,body,timeout):
        requests.append(body)
        names=[t['function']['name'] for t in body['tools']]
        if names==['submit_harness_state']:
            content=body['messages'][1]['content']
            if isinstance(content,list):content=''.join(x.get('text','') for x in content if x.get('type')=='text')
            payload=json.loads(content.split('\nAttached image labels')[0])
            if 'proposal' in payload: reply=report({'approved':True,'issues':[]})
            elif 'completed_units' in payload:
                reply=report({'satisfied':True,'evidence':'synthetic current fixture',
                    'evidence_ids':[payload['current_observation']['images'][0]['id']], 'missing_requirements':[]})
            else: reply=report({'satisfied':True,'evidence':'synthetic current fixture','evidence_ids':payload['eligible_evidence_ids']})
        elif h.phase=='plan':reply=call('reuseit_plan',plan(1))
        elif h.phase=='execute':reply=call('reuseit_step',{'step_id':'unit0','mode':'action','code':'answer=46'})
        else:
            assert h.phase=='complete'
            reply=end('success')
        return 200,reply
    result=run_openai_agent(migration_prompt('Synthetic task only',kit),[],kit,model='same-model',
        base_url='http://unused/v1',api_key=None,transport=transport,max_requests=12,budget_s=60,max_turns=12,
        out_dir=tmp_path/'out',retries=0,tool_choice='auto')
    assert result['errors']==[] and result['auxiliary_requests']==4
    assert len(requests)==7 and kit._globals['answer']==46
    assert kit.finish['status']=='success' and not backend.solved()
