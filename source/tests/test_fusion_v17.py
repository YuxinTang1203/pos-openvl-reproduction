"""Regression for the observed field-access failure and isolated prompt ablation."""
import copy
import json

import pytest

from pyrualean.minimal_fusion import MinimalFusionHarness, validate_options
from pyrualean.openvl_visual import ProjectionResult
from tests.test_minimal_fusion import setup
from tests.test_openvl_locate_candidate import configured


def test_observed_getattr_pattern_through_public_tool(configured):
    scene, sam, native, robo, kit, adapter, binding = configured
    frame = adapter.groundings[binding]['source_image_id']
    before = scene.step_count, len(native.state.records)
    # This is the access pattern that discarded valid projections in V16 cell005.
    out = kit.execute_tool('python', {'code': (
        f"p = robo.visual_project({frame!r}, 200, 380, .81)\n"
        "st = getattr(p, 'status', None)\n"
        "xyz = getattr(p, 'xyz_m', None)\n"
        "assert st == 'ok' and xyz is not None\n"
        "assert xyz == p['xyz_m']\nprint(st, xyz)"
    )})
    assert not out.result.get('error'), out.result
    assert kit._globals['st'] == 'ok'
    assert (scene.step_count, len(native.state.records)) == before
    assert kit._globals['p'].source['current_camera_pixels']
    assert kit._globals['p'].plane_sigma_m is None


def test_unknown_projection_stays_unknown_and_serializes_identically():
    raw = {'status': 'unknown', 'xyz_m': None, 'reason': 'ray_parallel_to_plane'}
    p = ProjectionResult(raw)
    assert p.status == 'unknown' and p.xyz_m is None
    assert json.dumps(p) == json.dumps(raw)
    assert copy.deepcopy(p).reason == raw['reason']
    with pytest.raises(AttributeError):
        p.missing_field
    with pytest.raises(KeyError):
        p['missing_field']


def test_next_action_ablation_keeps_failure_instrumentation_off(tmp_path):
    _, robo, kit = setup(tmp_path, mode='off')
    baseline = kit.general_harness.prompt()
    config = {'kind': 'minimal_fusion', 'openvl': {'enabled': False},
              'dependency_mode': 'off', 'observation_scope': 'next_action'}
    before = robo.move_to
    harness = MinimalFusionHarness(kit, config, 'Insert all three tubes')
    assert harness.prompt().startswith(baseline)
    assert '# Observation for the next action' in harness.prompt()
    assert '# Failure dependencies' not in harness.prompt()
    assert robo.move_to == before
    assert harness.provenance['observation_scope'] == 'next_action'
    assert harness.provenance['additional_model_requests'] == 0


def test_invalid_scope_rejected():
    with pytest.raises(ValueError, match='observation_scope'):
        validate_options({'kind': 'minimal_fusion', 'observation_scope': 'disable_freshness'})
