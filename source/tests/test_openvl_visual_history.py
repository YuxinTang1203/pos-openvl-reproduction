"""Passive history must never add sensing, rewrite feedback, or become policy."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyrualean import LiberoRobot
from pyrualean.arms import ArmToolkit
from pyrualean.openvl_routed import OpenVLRoutedAdapter
from tests.conftest import FakeLiberoBackend
from tests.test_openvl_visual import adapter, AllObjects, CONFIG
from tests.test_openvl_locate_candidate import configured


def record(history, row):
    return history.context(row['reference'])['record']


def test_same_calls_results_and_persistent_python_without_extra_sensing(tmp_path):
    outputs, calls, artifacts = [], [], []
    for enabled in (False, True):
        backend = FakeLiberoBackend()
        count = []
        original = backend.artifact
        backend.artifact = lambda *a, **k: (count.append((a, k)), original(*a, **k))[1]
        robo = LiberoRobot(backend)
        if enabled:
            kit, a = adapter(robo, tmp_path / 'visual')
            a.tools['Visual_Grounding_Tool'].client.segmenter = SimpleNamespace(
                segment_all=lambda *a, **k: pytest.fail('History must not call SAM'))
        else:
            kit = ArmToolkit(robo, arm='cells', workspace=tmp_path/'baseline/workspace',
                             images='on-demand', feedback='pure')
        out = kit.execute_tool('python', {'code':
            "s=robo.state(); p=robo.back_project(100,100,camera='agentview'); "
            "a=robo.move_to([0,0,.6]); b=robo.move_to([0,0,.55]); marker=17; "
            "print(a.reached,b.reached)"})
        outputs.append({k: v for k, v in out.result.items() if k != 'elapsed_s'})
        calls.append(copy.deepcopy(backend.calls)); artifacts.append(count)
        assert kit._globals['marker'] == 17
        if enabled:
            assert 'visual_history' not in out.result and 'robot_trace' not in out.result
            assert not a.frames and not a.summary()['trace_errors']
            assert a.history.find(kind='perception', tool='back_project')
            assert kit._globals['visual_find'].__self__ is a.history
            assert robo.visual_context.__self__ is a.history
    assert outputs[0] == outputs[1]
    assert calls[0] == calls[1] and artifacts[0] == artifacts[1]


def test_cached_before_returned_after_and_unknown_without_envelope(robo, backend, tmp_path, monkeypatch):
    kit, a = adapter(robo, tmp_path)
    kit.execute_tool('python', {'code': 'robo.state(); robo.move_to([0,0,.6]); robo.move_to([0,0,.55])'})
    first, second = [record(a.history, r) for r in reversed(a.history.find(kind='action'))]
    before = a.history.context(first['state_before'])['record']
    after = a.history.context(first['state_after'])['record']
    next_before = a.history.context(second['state_before'])['record']
    assert before['relation'] == 'cached_before_action_not_refreshed'
    assert after['relation'] == 'returned_action_observation'
    assert next_before['value'] == after['value']
    assert first['target_association'] == 'unknown'
    cached = copy.deepcopy(robo._last_state)
    original = backend.execute_tool
    monkeypatch.setattr(backend, 'execute_tool', lambda n, k:
                        SimpleNamespace(result={'public': 7}) if n == 'no_envelope' else original(n, k))
    assert robo._stateful('no_envelope', {}, lambda x: x) == {'public': 7}
    action = record(a.history, a.history.find(kind='action', tool='no_envelope')[0])
    unknown = a.history.context(action['state_after'])['record']
    assert unknown['availability'] == 'unknown' and unknown['value'] is None
    assert unknown['relation'] == 'unknown_no_returned_observation'
    assert robo._last_state == cached


def test_original_primitive_exception_and_cell_failure_remain_linked(robo, backend, tmp_path):
    kit, a = adapter(robo, tmp_path)
    backend.error_next = {'error': 'original refusal', 'code': 'bad_target'}
    out = kit.execute_tool('python', {'code':
        "try:\n    robo.move_to([0,0,.6])\nexcept Exception as e:\n    caught=type(e).__name__\n"
        "robo.move_to([0,0,.55]); raise ValueError('original cell failure')"})
    assert kit._globals['caught'] == 'ToolError'
    assert 'ValueError: original cell failure' in out.result['exception']
    actions = [record(a.history, r) for r in a.history.find(kind='action')]
    assert len(actions) == 2 and not actions[-1]['record']['ok']
    errors = [record(a.history, r) for r in a.history.find(kind='exception')]
    assert {r['scope'] for r in errors} == {'primitive', 'cell'}
    assert any('original refusal' in r['error'] and r['links'] for r in errors)
    assert sum(n == 'move_to' for n, _ in backend.calls) == 2
    assert not a.summary()['trace_errors']


def test_frames_grounding_and_helper_results_are_linked_without_refresh(robo, tmp_path):
    kit, a = adapter(robo, tmp_path)
    result = kit.execute_tool('Visual_Grounding_Tool', {'query': 'blocks'})
    binding = result.result['binding_ids'][0]
    frame = result.result['source']['image_id']
    context = a.history.context(binding)
    assert context['record']['source_image_id'] == frame
    assert any(r['kind'] == 'frame' for r in context['related'])
    original_context = a.history.context(frame)['record']
    kit.execute_tool('python', {'code': f"pixel=robo.visual_candidate({binding!r},2)\nrobo.move_to([0,0,.6])"})
    contexts = a.history.context(binding)['related']
    assert any(r['kind'] == 'visual_candidate_read' for r in contexts)
    assert a.history.context(frame)['record'] == original_context
    assert not a.frame_context(frame)['current_camera_pixels']
    assert not a.history.context(frame)['current_target']
    with pytest.raises(ValueError, match='current original'):
        robo.visual_candidate(binding, 2)
    context['record']['source_image_id'] = 'mutated'
    assert a.history.context(binding)['record']['source_image_id'] == frame
    native_ref = a.history.context(binding)['reference']
    assert native_ref in a.memory.state.bindings.bindings
    manifest = json.loads((a.root/'manifest.json').read_text())
    assert manifest['history']['aliases'][binding] == native_ref


@pytest.mark.parametrize('failure', ['binding', 'begin', 'perception', 'action', 'finish', 'disk', 'postprocessing'])
def test_recording_failures_never_change_execution(robo, backend, tmp_path, monkeypatch, failure):
    kit, a = adapter(robo, tmp_path)
    def failed(*args, **kwargs):
        raise OSError('optional history unavailable')
    if failure == 'binding':
        monkeypatch.setattr(a.memory, 'store_binding', failed)
    elif failure == 'disk':
        original_write = Path.write_text
        monkeypatch.setattr(Path, 'write_text', lambda p, *args, **kwargs:
                            failed() if p.is_relative_to(a.root) else original_write(p, *args, **kwargs))
    elif failure == 'postprocessing':
        monkeypatch.setattr(a, '_after_visual_tool', failed)
    else:
        name = {'begin': 'begin', 'perception': '_perception', 'action': '_action', 'finish': '_finish'}[failure]
        monkeypatch.setattr(a.history, name, failed)
    out = kit.execute_tool('python', {'code':
        "p=robo.back_project(100,100,camera='agentview'); r=robo.move_to([0,0,.6]); marker=23; print(r.reached)"})
    assert not out.result.get('exception') and not out.result.get('error')
    assert kit._globals['marker'] == 23 and out.result['stdout'].strip() == 'True'
    assert 'evidence_error' not in out.result
    assert sum(n == 'move_to' for n, _ in backend.calls) == 1
    assert a.summary()['trace_errors']
    backend.error_next = {'error': 'original refusal', 'code': 'bad_target'}
    out = kit.execute_tool('python', {'code': 'robo.move_to([0,0,.55])'})
    assert 'ToolError' in out.result['exception'] and 'original refusal' in out.result['exception']
    assert 'optional history unavailable' not in out.result['exception']


def test_routed_uses_same_passive_history_and_records_choice_exception(robo, tmp_path):
    kit = ArmToolkit(robo, arm='cells', workspace=tmp_path/'workspace', images='on-demand', feedback='pure')
    class Competitive(AllObjects):
        def segment_all(self, *args, **kwargs):
            rows = super().segment_all(*args, **kwargs)
            return [type(r)(r.found, .9, r.box, r.mask) for r in rows]
    a = OpenVLRoutedAdapter(kit, config=CONFIG, initial_images={}, segmenter=Competitive())
    kit.evidence_adapter = a
    out = kit.execute_tool('python', {'code': "robo.segment('blocks',camera='agentview'); robo.move_to([0,0,.6])"})
    assert 'PerceptionChoiceRequired' in out.result['exception']
    assert not a.history.find(kind='action')
    assert a.history.find(kind='perception_choice_required')
    assert a.history.find(kind='exception')
    binding = out.result['routed_perception'][0]['binding_ids'][0]
    assert a.history.context(binding)['record']['kind'] == 'grounding'


@pytest.mark.parametrize('limit', [0, 51, True, 1.5])
def test_bounded_queries_reject_bad_limits(robo, tmp_path, limit):
    _, a = adapter(robo, tmp_path)
    with pytest.raises(ValueError, match='limit'):
        a.history.find(limit=limit)
    with pytest.raises(ValueError, match='limit'):
        a.history.context('missing', limit=limit)


def test_query_unknown_reference_and_native_memory_copies(robo, tmp_path):
    kit, a = adapter(robo, tmp_path)
    kit.execute_tool('python', {'code': 'robo.move_to([0,0,.6])'})
    ref = a.history.find(kind='action')[0]['reference']
    a.history.context(ref)['record']['result']['reached'] = 'changed'
    a.memory.copy_bindings_for_execution()[ref]['result']['reached'] = 'changed'
    assert a.history.context(ref)['record']['result']['reached'] is True
    with pytest.raises(ValueError, match='Unknown'):
        a.history.context('invented')
    assert len(a.history.find(limit=1)) == 1


def test_selected_geometry_keeps_binding_frame_and_ledger_index(configured):
    _, _, _, robo, kit, a, binding = configured
    out = kit.execute_tool('python', {'code': f"selected=robo.locate_candidate({binding!r},2)"})
    assert not out.result.get('exception')
    row = record(a.history, a.history.find(kind='selected_mask_geometry')[0])
    assert row['result']['binding_id'] == binding
    assert row['result']['candidate_number'] == 2
    assert row['result']['geometry_payload']['center_xyz'] == list(kit._globals['selected'].center_xyz)
    call = robo._ledger.records[row['call_index']]
    assert call.tool == 'locate_mask' and not call.stateful
    assert binding in row['links'] and row['result']['source_image_id'] in row['links']
    assert a.history.context(binding)['record']['source_image_id'] == row['result']['source_image_id']


def test_event_and_image_archive_failure_preserve_native_pixels_and_python_result(robo, tmp_path, monkeypatch):
    kit, a = adapter(robo, tmp_path)
    original_open = Path.open
    def failed_archive(path, mode='r', *args, **kwargs):
        if path.is_relative_to(a.root) and any(letter in mode for letter in 'wa+'):
            raise OSError('visual archive unavailable')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', failed_archive)
    ground = kit.execute_tool('Visual_Grounding_Tool', {'query': 'blocks'})
    assert not ground.result.get('error') and ground.images
    binding, source = ground.result['binding_ids'][0], ground.result['source']['image_id']
    pixel = robo.visual_candidate(binding, 1)
    assert pixel['candidate_number'] == 1
    recalled = kit.execute_tool('Recall_Images', {'image_ids': [source]})
    assert recalled.images and not recalled.result.get('error')
    out = kit.execute_tool('python', {'code': "robo.move_to([0,0,.6]); print('original'); robo.show('agentview')"})
    assert out.result['stdout'].strip() == 'original' and out.images
    assert not out.result.get('error') and not out.result.get('exception')
    assert 'evidence_error' not in out.result and a.summary()['trace_errors']


def test_python_frame_registration_failure_preserves_original_exception_and_images(robo, tmp_path, monkeypatch):
    kit, a = adapter(robo, tmp_path)
    monkeypatch.setattr(a, 'register_frame', lambda *args, **kwargs:
                        (_ for _ in ()).throw(OSError('optional frame registration unavailable')))
    out = kit.execute_tool('python', {'code':
        "robo.move_to([0,0,.6]); robo.show('agentview'); print('before error'); raise ValueError('original')"})
    assert out.result['stdout'].strip() == 'before error' and out.images
    assert 'ValueError: original' in out.result['exception']
    assert 'optional frame' not in out.result['exception'] and 'evidence_error' not in out.result
    assert a.summary()['trace_errors']


def test_action_context_finds_later_same_cell_evidence_without_target_claim(robo, tmp_path):
    kit, a = adapter(robo, tmp_path)
    out = kit.execute_tool('python', {'code':
        "robo.move_to([0,0,.6]); robo.back_project(100,100,camera='agentview'); robo.show('agentview')"})
    assert not out.result.get('exception')
    action = a.history.find(kind='action')[0]
    context = a.history.context(action['reference'])
    observations = context['same_cell_observations']
    assert {'frame', 'perception'} <= {r['kind'] for r in observations}
    assert all(r['relation'] == 'same_cell_temporal_only' for r in observations)
    assert context['record']['target_association'] == 'unknown'
    assert len(a.history.context(action['reference'], limit=1)['same_cell_observations']) == 1
