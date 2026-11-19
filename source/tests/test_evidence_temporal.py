import copy

import pytest

from pyrualean.evidence_temporal import annotate_replay_history
from tests.test_openvl_trace import local


def test_replay_labels_order_without_inventing_capture_time():
    records = [dict(cell=8, message_index=17, code='x=0.8011', feedback={}),
               dict(cell=9, message_index=19, code='move()', feedback={})]
    images = [dict(message_index=16, path='old.png')]
    original = copy.deepcopy(records)
    data = annotate_replay_history(records, images, cutoff_message=19)
    assert data['records'][0]['temporal_context']['later_cells'] == [9]
    assert data['images'][0]['temporal_context']['later_cells'] == [8, 9]
    assert data['images'][0]['temporal_context']['capture_step'] is None
    assert not data['records'][1]['temporal_context']['current_target']
    assert records == original and 'temporal_context' not in images[0]
    with pytest.raises(ValueError):
        annotate_replay_history(records, images, cutoff_message=17)


def test_direct_stdout_gets_staleness_update_without_helper_read(robo, backend, tmp_path):
    kit = local(robo, tmp_path)
    first = kit.execute_tool('python', {'code': "p=robo.segment('bowl', camera='agentview'); print(p)"})
    before = copy.deepcopy(first.result)
    ref = kit.evidence_adapter.bridge.refs[-1]
    assert first.result['evidence_temporal_context']['perception'][-1]['validity']['usable']
    out = kit.execute_tool('python', {'code': 'robo.move_to([0,0,0.6]); print(p)'})
    context = out.result['evidence_temporal_context']
    row = next(r for r in context['perception'] if r['reference'] == ref)
    assert not row['validity']['usable']
    assert row['source']['step'] < context['assessment_step']
    assert context['output_scope']['actions']
    assert 'stdout' in context['output_scope']['constraint']
    assert first.result == before  # Past messages remain immutable.
    assert kit.evidence_adapter.bridge.last_read is None
    assert sum(n == 'segment' for n, _ in backend.calls) == 1


def test_trace_text_and_zoom_keep_historical_phase(robo, tmp_path):
    kit = local(robo, tmp_path)
    kit.execute_tool('python', {'code': 'robo.move_to([0,0,0.6])'})
    trace = kit.evidence_adapter.trace
    visual = trace.read(trace.find(kind='visual')[0]['reference'])
    ref = visual['image_ref']
    old_step = visual['step']
    kit.execute_tool('python', {'code': 'robo.move_to([0,0,0.7])'})
    context = trace.image_context(ref)
    assert any(s['source_step'] == old_step for s in context['sources'])
    assert context['assessment_step'] > old_step
    assert not context['current_target']
    out = kit.execute_tool('Zoom_In_Tool', {'image_id': ref, 'coords': [0, 0, 1000, 1000]})
    assert 'error' not in out.result
    assert out.result['source_image_temporal_context'][0]['sources'] == context['sources']
    for img in out.result['evidence_temporal_context']['images']:
        assert not img['current_target']
        assert img['derived_from'] or img['image_ref'] == ref
    end = trace.read(trace.find(kind='cell_end')[0]['reference'])
    assert not end['temporal_context']['current_target']


def test_inventory_delta_preserves_references_and_temporal_updates(robo, tmp_path):
    kit=local(robo,tmp_path)
    adapter=kit.evidence_adapter
    content='Environment Update:\n- trace_action: dict (keys: arguments, result)\n- robot_a: dict (keys: found)'
    compact=adapter.compact_inventory(content)
    assert 'trace_action' in compact and 'robot_a' in compact
    assert adapter.compact_inventory(content)==''
    assert 'robot_b' in adapter.compact_inventory(content+'\n- robot_b: dict (keys: found)')
    kit.execute_tool('python',{'code':"robo.segment('bowl',camera='agentview')"})
    out=kit.execute_tool('python',{'code':'robo.move_to([0,0,0.6]); print("keep raw feedback")'})
    assert 'keep raw feedback' in out.result['stdout']
    assert out.result['evidence_temporal_context']['perception']
    assert not out.result['evidence_temporal_context']['perception'][-1]['validity']['usable']
