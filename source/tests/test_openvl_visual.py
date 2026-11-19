"""Native visual tools preserve PyRUA execution while adding multi-object evidence."""
import json
from pathlib import Path

import numpy as np
import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.openvl_adapter import adapt_prompt
from pyrualean.openvl_visual import OpenVLVisualAdapter
from pyrualean.hosts.sam3_local import Segmented
from pyrualean.migrations import validate_migration

CONFIG = Path(__file__).resolve().parents[1] / 'integrations/migrations/openvl/capabilities_visual.yml'


class AllObjects:
    def segment_all(self, image, *, text_prompt, min_score):
        h, w = np.asarray(image).shape[:2]
        rows = []
        for x, score in ((w // 4, .9), (w // 2, .7)):
            mask = np.zeros((h, w), dtype=bool)
            mask[h // 4:h // 2, x:x + w // 8] = True
            rows.append(Segmented(True, score, [x, h // 4, x + w // 8, h // 2], mask))
        return rows


def adapter(robo, tmp_path):
    kit = ArmToolkit(robo, arm='cells', workspace=tmp_path / 'workspace', images='on-demand', feedback='pure')
    kit.evidence_adapter = OpenVLVisualAdapter(kit, config=CONFIG, initial_images={}, segmenter=AllObjects())
    return kit, kit.evidence_adapter


def test_grounding_all_candidates_native_memory_and_zoom_without_coordinate_guess(robo, tmp_path):
    kit, a = adapter(robo, tmp_path)
    out = kit.execute_tool('Visual_Grounding_Tool', {'query': 'blocks'})
    assert 'error' not in out.result, out.result
    assert out.result['data']['num_masks'] == 2
    assert len(out.images) >= 1
    binding = out.result['binding_ids'][0]
    assert len(a.read_data(binding)['data']['pred_boxes']) == 2
    zoom = kit.execute_tool('Zoom_In_Tool', {'binding_id': binding, 'candidate_number': 1})
    assert zoom.images and 'error' not in zoom.result, zoom.result
    assert zoom.result['source']['image_id'] == out.result['source']['image_id']
    assert zoom.result['image_contexts'][0]['parent_image_id'] == out.result['source']['image_id']
    recalled = kit.execute_tool('Recall_Images', {'image_ids': [out.result['source']['image_id'], zoom.result['images'][0]]})
    assert len(recalled.images) == 2
    json.loads(recalled.content_blocks[0]['text'])


def test_original_api_and_motion_feedback_preserved_no_automatic_capture(robo, backend, tmp_path):
    kit, a = adapter(robo, tmp_path)
    assert {t['name'] for t in kit.get_tools_spec()} == {'python', 'finish', 'Visual_Grounding_Tool', 'Zoom_In_Tool', 'Recall_Images'}
    prompt = adapt_prompt('FULL API DOCUMENTATION', kit)
    assert 'FULL API DOCUMENTATION' in prompt
    assert 'Coding_Agent' not in prompt and 'evidence_context' not in prompt
    assert robo._evidence_adapter is a.history and robo._action_feedback is None
    assert a.provenance['robot_action_hook'] == 'passive_history_only'
    out = kit.execute_tool('python', {'code': 'a=17; r=robo.move_to([0, 0, .6]); print(r.reached)'})
    assert kit._globals['a'] == 17 and not a.frames
    assert set(out.result) <= {'stdout', 'stderr', 'value', 'exception', 'cell', 'error', 'elapsed_s'}
    validate_migration({'kind': 'openvl_visual', 'config': str(CONFIG)})


def test_historical_grounding_stays_historical_and_dedup_does_not_refresh_old_frames(robo, tmp_path):
    kit, a = adapter(robo, tmp_path)
    first = kit.execute_tool('Visual_Grounding_Tool', {'query': 'blocks'})
    source = first.result['source']['image_id']
    assert a.frame_context(source)['current_camera_pixels']
    kit.execute_tool('python', {'code': 'robo.move_to([0, 0, .6])'})
    later = kit.execute_tool('Visual_Grounding_Tool', {'query': 'blocks'})
    new = later.result['source']['image_id']
    assert new != source
    assert a.frame_context(new)['current_camera_pixels']
    assert not a.frame_context(source)['current_camera_pixels']
    zoom = kit.execute_tool('Zoom_In_Tool', {'binding_id': first.result['binding_ids'][0], 'candidate_number': 1})
    assert zoom.result['source']['actions_since_capture'] == 1
    assert not zoom.result['image_contexts'][0]['current_camera_pixels']


def test_zoom_pixel_coords_and_explicit_errors(robo, tmp_path):
    kit, a = adapter(robo, tmp_path)
    out = kit.execute_tool('Visual_Grounding_Tool', {'query': 'blocks'})
    source = out.result['source']['image_id']
    h, w = out.result['source']['size_hw']
    zoom = kit.execute_tool('Zoom_In_Tool', {'image_id': source, 'coords': [0, 0, w/2, h/2], 'coord_space': 'pixels'})
    assert zoom.images and 'error' not in zoom.result
    for args in ({'image_id': 'bad', 'coords': [0, 0, 500, 500]},
                 {'binding_id': out.result['binding_ids'][0], 'candidate_number': 99},
                 {'image_id': source, 'coords': [[0, 0, 500, 500]]}):
        assert 'error' in kit.execute_tool('Zoom_In_Tool', args).result


def test_visible_returned_image_ids_can_be_reused(robo, tmp_path):
    kit, a = adapter(robo, tmp_path)
    out = kit.execute_tool('Visual_Grounding_Tool', {'query': 'blocks'})
    rendered = out.result['data']['rendered_image']
    recalled = kit.execute_tool('Recall_Images', {'image_ids': [rendered]})
    assert recalled.images and 'error' not in recalled.result
    z = kit.execute_tool('Zoom_In_Tool', {'image_id': rendered, 'coords': [0, 0, 500, 500]})
    assert z.images and 'error' not in z.result


def test_images_none_registers_memory_without_delivering_pixels(robo, tmp_path):
    kit, a = adapter(robo, tmp_path)
    kit.images = 'none'
    out = kit.execute_tool('Visual_Grounding_Tool', {'query': 'blocks'})
    assert not out.images and out.result['data']['rendered_image'] in a.frames
    crop = kit.execute_tool('Zoom_In_Tool', {'binding_id': out.result['binding_ids'][0], 'candidate_number': 1})
    assert not crop.images and crop.result['data']['image'] in a.frames
    recall = kit.execute_tool('Recall_Images', {'image_ids': [crop.result['data']['image']]})
    assert not recall.images and not recall.result['pixels_delivered']
