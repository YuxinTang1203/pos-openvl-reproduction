"""Exercise visual tools through the real PyRUA message/image delivery path."""
import base64
import copy
import json
from pathlib import Path

import yaml

from pyrualean.arms import ArmToolkit
from pyrualean.openai_runtime import run_openai_agent
from pyrualean.openvl_visual import OpenVLVisualAdapter
from tests.test_openai_runtime import _reply
from tests.test_openvl_grounding import FakeSam, data_url, scene


def visual_kit(robo, tmp_path):
    config = yaml.safe_load((Path(__file__).resolve().parents[1] /
        'integrations/migrations/openvl/capabilities_visual.yml').read_text())
    # Native configuration validation expects a service address even though the
    # test injects the CPU fixture backend and never opens an HTTP connection.
    config['services']['tools']['sam3_segment']['server_url'] = 'http://127.0.0.1:1'
    path = tmp_path / 'visual.yml'
    path.write_text(yaml.safe_dump(config))
    kit = ArmToolkit(robo, arm='cells', workspace=tmp_path/'workspace', images='on-demand', feedback='pure')
    png = base64.b64decode(data_url(scene()[0]).split(',', 1)[1])
    adapter = OpenVLVisualAdapter(kit, config=path,
        initial_images={f'{robo.CAMERAS[0]}.png': png}, segmenter=FakeSam())
    kit.evidence_adapter = adapter
    return kit, adapter


class VisualRoundTrip:
    def __init__(self):
        self.requests = []
        self.outputs = []

    def __call__(self, url, body, timeout_s):
        self.requests.append(copy.deepcopy(body))
        turn = len(self.requests)
        if turn == 1:
            return 200, _reply([('ground', 'Visual_Grounding_Tool', {'query': 'blocks', 'image_id': 'frame_0001'})])
        result = json.loads(next(m['content'] for m in reversed(body['messages']) if m['role'] == 'tool'))
        self.outputs.append(result)
        if turn == 2:
            return 200, _reply([('zoom', 'Zoom_In_Tool', {'binding_id': result['binding_ids'][0], 'candidate_number': 1})])
        if turn == 3:
            return 200, _reply([('recall', 'Recall_Images', {'image_ids': ['frame_0001', result['data']['image']]})])
        return 200, _reply([('done', 'finish', {'status': 'stuck', 'summary': 'Offline visual tool protocol test'})])


def test_three_visual_tools_deliver_images_and_reusable_ids_to_real_main_loop(robo, backend, tmp_path):
    kit, adapter = visual_kit(robo, tmp_path)
    transport = VisualRoundTrip()
    generation = run_openai_agent(adapter.prompt(), [], kit, model='fixture',
        base_url='http://unused/v1', api_key=None, max_turns=10, budget_s=30,
        out_dir=tmp_path/'episode', transport=transport)
    assert generation['status'] == 'completed'
    assert len(transport.requests) == 4 and len(transport.outputs) == 3
    assert generation['tool_calls'] == {'Visual_Grounding_Tool': 1, 'Zoom_In_Tool': 1,
                                      'Recall_Images': 1, 'finish': 1}
    for output in transport.outputs:
        assert 'error' not in output
        assert output['images'] and all(isinstance(ref, str) and ref in adapter.frames for ref in output['images'])
        assert all(not c['current_camera_pixels'] for c in output['image_contexts'] if c['parent_image_id'])
        for artifact in output.get('artifact_refs', []):
            if artifact['kind'] == 'image':
                assert artifact['id'] in adapter.frames
    ground, zoom, recall = transport.outputs
    assert ground['data']['rendered_image'] in adapter.frames
    assert zoom['data']['image'] in adapter.frames
    assert all('tool_generated_image_' not in text for output in transport.outputs for text in output.get('evidence', []))
    assert zoom['source']['image_id'] == 'frame_0001'  # Crop raw pixels, not the numbered overlay.
    assert all('world_xyz' not in output['data'] for output in (ground, zoom))
    assert all(sum(p.get('type') == 'image_url' for m in request['messages']
                   if isinstance(m.get('content'), list) for p in m['content']) > 0
               for request in transport.requests[1:])
    assert not any(name in {'move_to', 'back_project', 'segment'} for name, _ in backend.calls)


def test_python_display_capture_is_unknown_but_show_is_captured_at_cell_end(robo, tmp_path):
    kit, adapter = visual_kit(robo, tmp_path)
    output = kit.execute_tool('python', {'code':
        "old = robo.image('agentview')\nrobo.move_to([0,0,0.6])\ndisplay(old)"})
    assert 'exception' not in output.result
    context = output.result['visual_images'][0]
    assert context['source_step'] is None and not context['current_camera_pixels']
    assert context['delivery_step'] > 0
    output = kit.execute_tool('python', {'code': "robo.show('agentview')\nrobo.move_to([0,0,0.7])"})
    assert 'exception' not in output.result
    context = output.result['visual_images'][0]
    assert context['source_step'] == robo._backend.env_steps()
    assert context['current_camera_pixels']


def test_grounding_data_copy_and_historical_candidate_zoom_never_refresh_world_geometry(robo, tmp_path):
    kit, adapter = visual_kit(robo, tmp_path)
    ground = kit.execute_tool('Visual_Grounding_Tool', {'query': 'blocks', 'image_id': 'frame_0001'})
    assert 'error' not in ground.result
    binding = ground.result['binding_ids'][0]
    measurement = adapter.read_data(binding)
    measurement['data']['pred_boxes'][0][0] = -12345
    assert adapter.read_data(binding)['data']['pred_boxes'][0][0] >= 0
    kit.execute_tool('python', {'code': 'robo.move_to([0,0,0.6])'})
    crop = kit.execute_tool('Zoom_In_Tool', {'binding_id': binding, 'candidate_number': 2})
    assert 'error' not in crop.result
    assert crop.result['source']['source_step'] == 0
    assert crop.result['source']['actions_since_capture'] > 0
    assert not crop.result['source']['current_camera_pixels']
    assert all(c['source_step'] == 0 and not c['current_camera_pixels'] for c in crop.result['image_contexts'])
