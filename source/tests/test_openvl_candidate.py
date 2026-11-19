"""Candidate: existing robot API, native Zoom and on-demand linked evidence."""
import json
import yaml
import pytest
from pyrualean.arms import ArmToolkit
from pyrualean.migrations import configure_migration, validate_migration
from pyrualean.openvl_adapter import adapt_prompt
from tests.test_openvl_native import ROOT


def candidate(robo,tmp_path,images='on-demand'):
    kit=ArmToolkit(robo,arm='cells',workspace=tmp_path/'workspace',images=images,feedback='pure')
    cfg=yaml.safe_load((ROOT/'manipulation_candidate.yml').read_text())
    configure_migration(kit,cfg['migration'],task='Generic manipulation',initial_images={},model=cfg['model'])
    kit.evidence_adapter.bridge.validate_geometry=False
    return kit


def test_candidate_preserves_python_api_and_does_not_advertise_duplicate_tools(robo,tmp_path):
    kit=candidate(robo,tmp_path)
    assert {s['name'] for s in kit.get_tools_spec()}=={'python','finish','Zoom_In_Tool'}
    prompt=adapt_prompt('FULL ORIGINAL API DOCUMENTATION',kit)
    assert 'FULL ORIGINAL API DOCUMENTATION' in prompt
    assert 'Robot_Perception_Tool' not in prompt and 'Robot_Evidence_Tool' not in prompt
    assert 'evidence_context' in prompt and 'robo.locate' in prompt
    assert not getattr(kit.evidence_adapter,'completion_reviewer',None)
    assert not kit.robo._action_feedback and not kit.stage_verifier
    kit.execute_tool('python',{'code':"marker=17; s=robo.segment('bowl'); print(s.found)"})
    assert kit.evidence_adapter.bridge.refs
    out=kit.execute_tool('python',{'code':'print(marker)'})
    assert '17' in out.result['stdout']
    directory=out.result.get('openvl_environment_update','')
    assert '- trace_' not in directory
    assert kit.evidence_adapter.trace.find(kind='cell_start')


def test_action_bundle_is_historical_readonly_and_includes_errors(robo,backend,tmp_path):
    kit=candidate(robo,tmp_path);a=kit.evidence_adapter
    kit.execute_tool('python',{'code':"robo.segment('bowl'); robo.move_to([0,0,0.6]); robo.segment('bowl')"})
    first=a.trace.find(kind='action')[0]['reference']
    bundle=a.trace.context_for_action(first)
    assert bundle['perception_before'] and bundle['perception_after']
    assert bundle['images_after'] and 'images_before' in bundle['missing']
    assert all(not p['current_target'] for p in bundle['perception_before']+bundle['perception_after'])
    assert bundle['states']['state_before']['relation']=='cached_before_action_not_refreshed'
    backend.error_next={'error':'fixture failure','code':'bad_target'}
    kit.execute_tool('python',{'code':"try:\n robo.move_to([0,0,0.5])\nexcept Exception:\n pass"})
    last=a.trace.find(kind='action')[0]['reference'];calls=len(backend.calls)
    failure=kit._globals['evidence_context'](last)
    assert len(backend.calls)==calls
    assert failure['exceptions'] and failure['images_before']
    assert failure['scope']['task_effect']=='not_verified' and not failure['scope']['current_target']
    bundle['action']['tool']='mutated'
    assert a.trace.read(first)['tool']=='move_to'
    # An image linked through the bundle is still a real native Zoom input.
    image_ref=failure['images_before'][0]['image_ref']
    out=kit.execute_tool('Zoom_In_Tool',{'image_id':image_ref,'coords':[0,0,500,500]})
    assert out.images and 'error' not in out.result


def test_context_never_associates_perception_after_later_action(robo,tmp_path):
    kit=candidate(robo,tmp_path,images='none');a=kit.evidence_adapter
    kit.execute_tool('python',{'code':"robo.move_to([0,0,0.6]); robo.move_to([0,0,0.55]); robo.segment('bowl')"})
    newest,oldest=a.trace.find(kind='action')
    early=a.trace.context_for_action(oldest['reference'])
    late=a.trace.context_for_action(newest['reference'])
    assert not early['perception_after'] and late['perception_after']
    assert 'images_after' in late['missing']
    with pytest.raises(ValueError):a.trace.context_for_action('missing')
    with pytest.raises(ValueError):a.trace.context_for_action(newest['reference'],limit=True)


def test_directory_filters_audit_only_updates_without_fallback_leak(robo,tmp_path):
    kit=candidate(robo,tmp_path);a=kit.evidence_adapter
    assert a.compact_inventory('Directory:\n- trace_action: historical')==''
    assert a.compact_inventory('Unrecognized native format')=='Unrecognized native format'
    cfg=yaml.safe_load((ROOT/'manipulation_candidate.yml').read_text())
    validate_migration(cfg['migration'])
    assert cfg['migration']['feedback_mode']=='ondemand'
    assert cfg['migration']['completion_review']=='off' and not cfg['migration']['local_coding']
    with pytest.raises(ValueError):validate_migration(dict(cfg['migration'],expose_robot_tools='false'))


def test_candidate_and_baseline_match_execution_and_sampling(tmp_path):
    from pyrualean import LiberoRobot
    from tests.conftest import FakeLiberoBackend
    configs=[yaml.safe_load((ROOT/name).read_text()) for name in ['manipulation_candidate.yml','candidate_baseline.yml']]
    assert {k:v for k,v in configs[0].items() if k not in {'migration','result_dir'}} == {k:v for k,v in configs[1].items() if k not in {'migration','result_dir'}}
    results=[]
    for mode in ('baseline','candidate'):
        backend=FakeLiberoBackend();robot=LiberoRobot(backend)
        kit=candidate(robot,tmp_path/mode) if mode=='candidate' else ArmToolkit(robot,arm='cells',workspace=tmp_path/mode/'workspace',images='on-demand',feedback='pure')
        result=kit.execute_tool('python',{'code':"a=robo.move_to([0,0,2]); b=robo.move_to([0,0,0.6]); print(a.reached,b.reached); retained=42"})
        assert kit._globals['retained']==42
        results.append((result.result['stdout'],backend.eef,[name for name,_ in backend.calls if name=='move_to']))
    assert results[0]==results[1]


def test_image_directory_keeps_visible_refs_and_archived_images_queryable(robo,tmp_path):
    kit=candidate(robo,tmp_path);a=kit.evidence_adapter
    out=kit.execute_tool('python',{'code':'robo.move_to([0,0,0.6])'})
    action=a.trace.find(kind='action')[0]['reference'];bundle=a.trace.context_for_action(action)
    archived=bundle['images_after'][0]['image_ref']
    assert archived in a.memory.state.images.registry
    assert f'- {archived}:' not in out.result.get('openvl_environment_update','')
    zoom=kit.execute_tool('Zoom_In_Tool',{'image_id':archived,'coords':[0,0,300,300]})
    assert zoom.images
    assert any(ref in a._directory_visible_images for ref in zoom.result['openvl_image_refs'])
