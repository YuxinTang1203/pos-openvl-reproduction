"""Ablation presentation preserves native observations and freshness barriers."""
import json
import pytest
from pyrualean.openvl_feedback import zoom_arguments, present_feedback
from tests.test_openvl_native import setup

@pytest.mark.parametrize('box', [[1,0,0,1], [0,0,1001,5], [0,0,float('nan'),5], [0,0,5], [False,0,5,5]])
def test_invalid_zoom_rejected(box):
    with pytest.raises(ValueError):
        zoom_arguments({'coords':box})

@pytest.mark.parametrize('margin', [-1, float('inf'), True])
def test_invalid_margin_rejected(margin):
    with pytest.raises(ValueError):
        zoom_arguments({'coords':[0,0,200,200], 'padding':margin})

@pytest.mark.parametrize('mode', ['ondemand','auto'])
def test_real_crop_defaults_and_explicit_adjustment(robo, tmp_path, mode):
    kit, _ = setup(robo, tmp_path, feedback_mode=mode)
    a = kit.evidence_adapter
    assert a.provenance["feedback_mode"] == mode
    for extra in ({}, {'padding':0}):
        out = kit.execute_tool('Zoom_In_Tool', {'image_id':'head.png', 'coords':[0,0,400,400], **extra})
        assert 'error' not in out.result, out.result
        assert out.images
        assert 'effect_status' not in out.result  # Zoom makes no robot-effect claim
    assert 'effective_padding' in json.dumps(out.result)
    schema = next(s for s in a.specs() if s['name']=='Zoom_In_Tool')['input_schema']
    assert schema['properties']['padding']['minimum'] == 50
    bad = kit.execute_tool('Zoom_In_Tool', {'image_id':'head.png', 'coords':[0,0,1001,2]})
    assert bad.result['error']['code'] == 'invalid_zoom_arguments'

@pytest.mark.parametrize('mode', ['ondemand','auto'])
def test_references_comparison_and_original_feedback(robo, tmp_path, mode):
    kit, _ = setup(robo, tmp_path, feedback_mode=mode)
    a = kit.evidence_adapter
    assert a.provenance["feedback_mode"] == mode
    a.bridge.validate_geometry = False
    def perceive():
        return kit.execute_tool('Robot_Perception_Tool', {'operation':'segment','query':'bowl','camera':'agentview'})
    first = perceive().result['evidence_refs'][-1]['reference']
    kit.execute_tool('python', {'code':'robo.move_to([0,0,0.6])'})
    second = perceive().result['evidence_refs'][-1]
    assert second['previous_ref'] == first
    comparison = a.compare_evidence(second['reference'])
    assert comparison['before']['current_target'] is False
    assert comparison['after']['usable']
    assert a.compare_evidence(first)['after']['measurement'] is None
    out = kit.execute_tool('python', {'code':"print('original measurements'); raise ValueError('original error')"})
    assert 'original measurements' in out.result['stdout']
    assert 'original error' in json.dumps(out.result)
    # New native memory IDs remain discoverable even in compact feedback.
    if 'openvl_environment_update' in out.result:
        assert 'not a freshness certificate' in out.result['openvl_environment_update']
    assert 'evidence_update' not in out.result
    assert (a.root/'full_feedback.jsonl').exists()

def test_presentation_does_not_mutate_or_upgrade_measurements():
    original = {'stdout':'reached=True', 'error':'found=False', 'data':{'found':False},
                'evidence_update':{'history':'large'},
                'evidence_temporal_context':{'output_scope':{'actions':[{'tool':'move_to'}]}}}
    out = present_feedback(original, [], automatic=True)
    assert out['data'] == original['data'] and out['stdout'] == original['stdout']
    assert out['error'] == original['error']
    assert out['effect_status']['task_effect'] == 'unverified_by_this_adapter'
    assert 'evidence_update' in original


def test_explicit_cross_query_comparison_does_not_claim_identity(robo, tmp_path):
    kit, _ = setup(robo, tmp_path, feedback_mode='ondemand')
    a = kit.evidence_adapter
    refs = []
    for query in ['bowl', 'cup']:
        out = kit.execute_tool('Robot_Perception_Tool', {
            'operation':'segment', 'query':query, 'camera':'agentview'})
        refs.append(out.result['evidence_refs'][-1]['reference'])
    compared = a.compare_evidence(refs[1], previous_ref=refs[0])
    assert compared['association'] == {
        'same_query':False, 'same_camera':True,
        'object_identity':'unverified', 'causality':'unverified'}
    assert a.compare_evidence(refs[1])['before'] is None


def test_invalid_previous_ref_is_not_silently_replaced(robo, tmp_path):
    kit, _ = setup(robo, tmp_path, feedback_mode='ondemand')
    out = kit.execute_tool('Robot_Perception_Tool', {
        'operation':'segment', 'query':'bowl', 'camera':'agentview'})
    ref = out.result['evidence_refs'][-1]['reference']
    with pytest.raises((ValueError, TypeError)):
        kit.evidence_adapter.compare_evidence(ref, previous_ref='')


def test_archive_failure_preserves_executed_cell(robo, tmp_path, monkeypatch):
    from pathlib import Path
    kit, _ = setup(robo, tmp_path, feedback_mode='ondemand')
    original = Path.open
    def fail_archive(path, *args, **kwargs):
        if path.name == 'full_feedback.jsonl':
            raise OSError('simulated archive failure')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail_archive)
    out = kit.execute_tool('python', {'code':"completed = 1; print('completed once')"})
    assert kit._globals['completed'] == 1
    assert 'completed once' in out.result['stdout']
    assert 'simulated archive failure' in out.result['feedback_archive_error']


def test_generic_verification_without_case_specific_strategy(robo, tmp_path):
    kit, _ = setup(robo, tmp_path, feedback_mode='auto')
    prompt = kit.evidence_adapter.prompt()
    assert 'Before claiming task completion' in prompt
    assert 'Clearly mark anything that cannot be confirmed as unconfirmed.' in prompt
    feedback = present_feedback({'evidence_temporal_context': {
        'output_scope': {'actions': [{'tool': 'move_to'}]}}}, [], automatic=True)
    text = prompt + json.dumps(feedback)
    for phrase in ['button activation', 'coverage or a stable stack',
                   'Before advancing a required task phase', 'do not silently skip']:
        assert phrase not in text
