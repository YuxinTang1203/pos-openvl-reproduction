"""Real toolkit/runtime composition with scripted models and synthetic SAM masks."""
import copy
import json
from pathlib import Path

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.composite_harness import CompositeHarness, validate_options
from pyrualean.migrations import configure_migration, cli_migration
from pyrualean.openai_runtime import run_openai_agent
from pyrualean.openvl_routed import OpenVLRoutedAdapter
from tests.test_openai_runtime import ScriptedTransport, _reply
from tests.test_openvl_routed import Candidates, CONFIG
from tests.test_pos_harness import Script, belief, audit


def setup(robo, tmp_path, *, visual=True, reflection=True):
    kit = ArmToolkit(robo, arm='cells', workspace=tmp_path/'workspace', images='on-demand', feedback='pure')
    config={'kind':'openvl_pos_reflact','openvl':{'enabled':visual,'config':str(CONFIG)},'reflact':{'enabled':reflection}}
    if visual:
        kit.evidence_adapter=OpenVLRoutedAdapter(kit,config=CONFIG,initial_images={},segmenter=Candidates())
    kit.general_harness=CompositeHarness(kit,config,'Put the block on the plate')
    return kit


def args(code='pass', **kw):
    return {'observed_state':'Outcome remains unverified.','goal_gap':'The original placement remains unfinished.','code':code,**kw}


def events(kit, kind):
    return [json.loads(l) for l in (kit.general_harness.directory/'events.jsonl').read_text().splitlines()
            if json.loads(l)['event']==kind]


def test_schema_retains_optional_lite_and_reflection_precedes_execution(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False)
    schema=next(t['input_schema'] for t in kit.get_tools_spec() if t['name']=='python')
    assert schema['required']==['observed_state','goal_gap','code']
    assert 'step' in schema['properties'] and 'step' not in schema['required']
    out=kit.execute_tool('python',args('marker=11',step={'goal':'invented-goal','phase':'invalid'}))
    assert kit._globals['marker']==11 and kit._cell_index==1
    assert out.result['task_progress']['metadata_warnings']
    assert len(events(kit,'reflection_recorded'))==1
    assert events(kit,'progress_step')[0]['step']['goal'] in ('task-root','a-root')
    assert not out.result.get('general_harness_error')


def test_missing_reflection_executes_zero_actions_and_consumes_zero_cells(robo,backend,tmp_path):
    kit=setup(robo,tmp_path,visual=False)
    out=kit.execute_tool('python',{'code':'robo.move_to([0,0,.6])'})
    assert out.result['reflact_protocol']['accepted'] is False
    assert kit._cell_index==0 and not any(k=='move_to' for k,_ in backend.calls)
    assert not events(kit,'progress_step')


def test_ambiguous_visual_evidence_reaches_pos_and_never_replays_actions(robo,backend,tmp_path):
    kit=setup(robo,tmp_path)
    out=kit.execute_tool('python',args("robo.move_to([0,0,.6])\np=robo.segment('blocks')\nrobo.move_to([.1,0,.6])"))
    assert 'PerceptionChoiceRequired' in out.result['exception']
    assert sum(k=='move_to' for k,_ in backend.calls)==1
    h=kit.general_harness
    observation=json.loads(h.evidence['obs-1']['text'])
    visual=observation['openvl_evidence']['delivered_visual_results']['routed_perception'][0]
    assert visual['route_status']=='choice_required' and len(visual['candidates'])==2
    images=[e for e in h.evidence.values() if e['type']=='image']
    assert images and all(not e['current_eligible'] for e in images)  # annotations alone
    assert all(e.get('openvl_source') for e in images)
    assert not out.result.get('general_harness_error')
    assert len(events(kit,'reflection_recorded'))==1
    assert len(events(kit,'visual_evidence_linked'))==1


def test_recalled_camera_image_is_historical_after_motion(robo,tmp_path):
    kit=setup(robo,tmp_path)
    first=kit.execute_tool('python',args("robo.show()"))
    assert first.images
    source=first.result['visual_images'][0]['image_id']
    kit.execute_tool('python',args('robo.move_to([0,0,.6])'))
    result=kit.execute_tool('Recall_Images',{'image_ids':[source]})
    h=kit.general_harness
    entry=next(e for e in h.evidence.values() if e.get('sequence')==h.sequence and e['type']=='image')
    assert not entry['current_eligible'] and not entry['openvl_source']['current_camera_pixels']
    assert not result.result.get('general_harness_error')


def test_finish_requires_original_pos_audit_but_failure_remains_available(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False)
    out=kit.execute_tool('finish',{'status':'success','summary':'executor says done'})
    assert out.result['completion_review_pending'] and kit.finish is None
    kit.execute_tool('finish',{'status':'stuck','summary':'no supporting evidence'})
    assert kit.finish['status']=='stuck'


def test_scripted_runtime_rejects_bad_call_then_executes_once_with_shared_schema(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False)
    transport=ScriptedTransport([
        _reply([('bad','python',{'code':'bad=1'})]),
        _reply([('good','python',args('good=7',step='malformed optional metadata'))]),
        _reply([('end','finish',{'status':'stuck','summary':'offline'})]),
    ])
    result=run_openai_agent(kit.general_harness.prompt(),[],kit,model='offline',base_url='http://unused/v1',
        api_key=None,max_turns=3,budget_s=10,out_dir=tmp_path/'episode',transport=transport,auxiliary_budget_s=0)
    assert result['requests']==3 and result['auxiliary_requests']==0
    assert result['python_cells']==1 and kit._globals['good']==7 and 'bad' not in kit._globals
    assert not result['errors']
    assert len(events(kit,'reflection_recorded'))==1


def test_pos_sentinel_receives_delivered_visual_metadata_and_preserves_original_goals(robo,tmp_path):
    kit=setup(robo,tmp_path)
    kit.execute_tool('python',args("robo.segment('blocks')"))
    h=kit.general_harness
    candidate=copy.deepcopy(h.belief)
    review={**audit(None),'directive':{'goal_id':'a-root','mode':'observe',
        'next_step':'Select the intended block from the numbered candidates.',
        'decision_critical_missing':'Which candidate is the target?','why_change':'Ambiguity remains.'}}
    model=Script(candidate,review)
    context=h.before_request(model)
    assert h.last_status=='committed' and 'TASK_PROGRESS_DIRECTIVE' in context
    assert len(model.calls)==2
    assert all('routed_perception' in json.dumps(c['payload']) for c in model.calls)
    assert h.belief['achievement'][0]['id']=='a-root'
    assert h.directive['goal_id']=='a-root'


def test_ablation_removes_reflection_without_changing_lite_contract(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False,reflection=False)
    spec=next(t for t in kit.get_tools_spec() if t['name']=='python')
    assert spec['input_schema']['required']==['code']
    kit.execute_tool('python',{'code':'marker=1'})
    assert kit._globals['marker']==1 and not events(kit,'reflection_recorded')


@pytest.mark.parametrize('update',[
    {'pos':{'kind':'pos'}},{'openvl':{'enabled':'yes'}},{'openvl':{'config':''}},
    {'reflact':{'enabled':True,'unexpected':1}},{'pos':{'unknown':1}},
])
def test_config_rejects_implicit_versions_or_unknown_options(update):
    with pytest.raises(ValueError):
        validate_options({'kind':'openvl_pos_reflact','openvl':{'config':str(CONFIG)},**update})


def test_registered_configuration_builds_both_layers(robo,tmp_path,monkeypatch):
    import pyrualean.openvl_routed as routed
    original=routed.OpenVLRoutedAdapter
    monkeypatch.setattr(routed,'OpenVLRoutedAdapter',lambda *a,**kw:original(*a,**kw,segmenter=Candidates()))
    kit=ArmToolkit(robo,arm='cells',workspace=tmp_path/'workspace',images='on-demand',feedback='pure')
    record=configure_migration(kit,{'kind':'openvl_pos_reflact','openvl':{'config':str(CONFIG)}},
        task='Place block',initial_images={})
    assert kit.evidence_adapter and isinstance(kit.general_harness,CompositeHarness)
    assert record['adaptation']['pos']['implementation']=='pyrualean-pos-progress-im-lite-v1'
    with pytest.raises(ValueError,match='clean toolkit'):
        configure_migration(kit,{'kind':'baseline'},task='Place block',initial_images={})


def test_runtime_maintains_visual_evidence_before_actual_reflection_decision(robo,tmp_path):
    kit=setup(robo,tmp_path)
    kit.execute_tool('python',args("robo.segment('blocks')"))
    h=kit.general_harness
    candidate=copy.deepcopy(h.belief)
    reviewer={**audit(None),'directive':{'goal_id':'a-root','mode':'observe','next_step':'Inspect a current original view.',
        'decision_critical_missing':'Outcome is unknown.','why_change':'Placement is unverified.'}}
    requests=[]
    roots=0
    def transport(url,body,timeout):
        nonlocal roots
        requests.append(copy.deepcopy(body))
        if body['tools'][0]['function']['name']=='submit_harness_state':
            schema=body['tools'][0]['function']['parameters']['properties']['payload']
            report=reviewer if 'accepted' in schema['properties'] else candidate
            return 200,_reply([('audit','submit_harness_state',{'payload':report})])
        roots+=1
        return 200,(_reply([('read','python',args('print(robo.state()); robo.show()'))])if roots==1
                    else _reply([('end','finish',{'status':'stuck','summary':'offline'})]))
    result=run_openai_agent(h.prompt(),[],kit,model='offline',base_url='http://unused/v1',api_key=None,
        max_turns=4,budget_s=5400,out_dir=tmp_path/'runtime',transport=transport)
    assert result['auxiliary_requests']==2 and result['requests']==4
    assert h.revision==1 and len(events(kit,'reflection_recorded'))==2
    assert not result['errors']
    assert 'routed_perception' in json.dumps(requests[0]['messages'])
    assert any('TASK_PROGRESS_DIRECTIVE' in str(m.get('content',''))for m in requests[2]['messages'])
    assert result['options']['auxiliary_budget_s']==1080


def test_lite_unverified_completion_is_downgraded_without_action_replay(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False)
    kit.execute_tool('python',args('visits=1'))
    result=kit.execute_tool('python',args('visits+=1',step={'previous_outcome':'observed_complete'}))
    assert kit._globals['visits']==2 and kit._cell_index==2
    warnings=result.result['task_progress']['metadata_warnings']
    assert any(w['kind']=='completion_not_verified'for w in warnings)
    assert not kit.general_harness.completion_approved


def test_cli_resolves_nested_visual_config_relative_to_json(tmp_path):
    from types import SimpleNamespace
    file=tmp_path/'combination.json'
    file.write_text(json.dumps({'kind':'openvl_pos_reflact','openvl':{'config':'visual.yml'}}))
    result=cli_migration(SimpleNamespace(general_harness_config=file,runtime='openai'))
    assert result['openvl']['config']==str(tmp_path/'visual.yml')


