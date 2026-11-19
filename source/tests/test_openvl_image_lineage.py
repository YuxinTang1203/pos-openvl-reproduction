"""Image history stays readable after many observations and nested crops."""
import json

from tests.test_openvl_trace import local


def test_repeated_perception_and_nested_zoom_keep_tool_json_intact(robo, tmp_path):
    kit = local(robo, tmp_path)
    adapter = kit.evidence_adapter
    kit.execute_tool('python', {'code': "for i in range(35):\n robo.segment('bowl', camera='agentview')"})
    original = adapter.bridge.values[adapter.bridge.refs[-1]]['source']['image_ref']
    ref = original
    for _ in range(8):
        result = kit.execute_tool('Zoom_In_Tool', {
            'image_id': ref, 'coords': [200, 200, 800, 800], 'padding': 50})
        assert 'error' not in result.result
        # Real rendering is the failure boundary: the old recursive metadata
        # exceeded 20k and left the planner with truncated, invalid JSON.
        rendered = json.loads(result.content_blocks[0]['text'])
        ref = result.result['data']['image']
        assert rendered['data']['image'] == ref
        assert rendered['artifact_refs'] and result.images
        context = adapter.trace.image_context(ref)
        assert original in context['root_image_refs']
        assert context['lineage_complete'] and not context['current_target']
        assert all(isinstance(parent, str) for parent in context['derived_from'])
        assert any(s['camera'] == 'agentview' and s['source_step'] == 0 for s in context['sources'])
    group = next(s for s in context['sources'] if s['phase'] == 'perception_source')
    assert group['associations'] == 35
    assert group['source_call_range'][0] < group['source_call_range'][1]


def test_zoom_lineage_keeps_old_steps_after_motion_and_resolves_aliases(robo, tmp_path):
    kit = local(robo, tmp_path)
    adapter = kit.evidence_adapter
    kit.execute_tool('python', {'code': "robo.segment('bowl', camera='agentview')"})
    source = adapter.bridge.values[adapter.bridge.refs[-1]]['source']['image_ref']
    cropped = kit.execute_tool('Zoom_In_Tool', {'image_id': source, 'coords': [100, 100, 600, 600]})
    ref = cropped.result['data']['image']
    kit.execute_tool('python', {'code': 'robo.move_to([0,0,0.6])'})
    context = adapter.trace.image_context(ref)
    assert any(s['source_step'] == 0 and 'historical' in ' '.join(s['validity_statuses'])
               for s in context['sources'])
    assert context['assessment_step'] > 0 and not context['current_target']
    # Legacy assignments are accepted, but replies no longer recursively embed them.
    adapter.trace.derived_images['legacy_crop'] = [adapter.trace.image_context('head.png')]
    legacy = adapter.trace.image_context('legacy_crop')
    assert legacy['derived_from'] == [adapter.trace._image_reference('head.png')]
    assert all(isinstance(parent, str) for parent in legacy['derived_from'])


def test_source_overflow_and_lineage_cycles_are_explicit(robo, tmp_path):
    kit = local(robo, tmp_path)
    trace = kit.evidence_adapter.trace
    ref = trace._image_reference('head.png')
    for step in range(20):
        trace.rows.append((f'visual_{step}', dict(kind='visual', image_ref=ref, camera='agentview',
            step=step, call_index=step, cell=step, relation='after_action_same_step')))
    context = trace.image_context(ref)
    assert len(context['sources']) == 8 and context['omitted_sources'] == 12
    assert {s['source_step'] for s in context['sources']} >= {0, 19}
    trace.record_derivation('a', ['b'])
    trace.record_derivation('b', ['a'])
    context = trace.image_context('a')
    assert context['cycle_detected'] and not context['lineage_complete']
    assert not context['current_target']
