import pytest

from tests.test_reuseit_rlm_ports import kit_for, guard, workflow_config


def action(code, mode='action'):
    return {'step_id': 'observe', 'mode': mode, 'code': code}


def test_code_error_requires_observation_and_precheck_before_repair(robo, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, workflow_config)
    h = kit.general_harness
    guard(h)
    result = kit.execute_tool('reuseit_step', action("raise TypeError('bad helper argument')"))
    assert h.phase == 'check_pre' and h.retries == 0 and h.code_repairs == 1
    assert result.images and h.repair_observation['images']
    assert 'error' in kit.execute_tool('reuseit_step', action('x=1', 'repair')).result
    guard(h)
    assert h.phase == 'repair'
    assert 'error' in kit.execute_tool('reuseit_step', action('x=1')).result
    kit.execute_tool('reuseit_step', action('x=46', 'repair'))
    assert h.phase == 'check_post' and h.retries == 0 and kit._globals['x'] == 46
    assert 'error' in kit.execute_tool('finish', {'status': 'success', 'summary': 'skip'}).result
    guard(h)
    assert h.phase == 'complete' and h.code_repairs == 0


def test_fallback_code_error_does_not_consume_physical_retry(robo, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, {**workflow_config, 'max_retries': 1})
    h = kit.general_harness
    guard(h, False)
    kit.execute_tool('reuseit_step', action("raise ValueError('shape mismatch')", 'fallback'))
    assert h.retries == 0 and h.code_repairs == 1 and h.phase == 'check_pre'
    guard(h)
    kit.execute_tool('reuseit_step', action('fixed=True', 'repair'))
    assert h.retries == 1 and h.phase == 'check_pre'
    guard(h, False)
    assert h.phase == 'blocked'


def test_partial_motion_exception_is_observed_without_automatic_replay(robo, backend, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, workflow_config)
    h = kit.general_harness
    guard(h)
    kit.execute_tool('reuseit_step', action("robo.move_to([0.05, -0.10, 0.6])\nraise TypeError('after motion')"))
    stateful = [e for e in h.last_feedback['public_calls'] if e['stateful']]
    assert stateful and h.trace.epoch > 0
    assert h.repair_observation['epoch'] == h.trace.epoch
    assert h.repair_observation['time_monotonic'] >= h.trace.events[-2]['time_monotonic']
    steps = len(backend.steps)
    # Precondition failure must revoke the requested code repair.
    guard(h, False)
    assert h.phase == 'fallback' and h.pending_repair is None
    assert 'error' in kit.execute_tool('reuseit_step', action('robo.set_gripper(1)', 'repair')).result
    assert len(backend.steps) == steps


def test_repeated_code_errors_are_bounded(robo, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, {**workflow_config, 'max_code_repairs': 1})
    h = kit.general_harness
    guard(h)
    kit.execute_tool('reuseit_step', action("raise ValueError('first')"))
    guard(h)
    kit.execute_tool('reuseit_step', action("raise ValueError('second')", 'repair'))
    assert h.phase == 'blocked' and h.code_repairs == 1 and h.retries == 0
    assert 'error' in kit.execute_tool('reuseit_step', action('x=1', 'repair')).result


def test_missing_fresh_camera_blocks_code_repair(robo, tmp_path, workflow_config, monkeypatch):
    kit = kit_for(robo, tmp_path, workflow_config)
    h = kit.general_harness
    guard(h)
    monkeypatch.setattr(robo, 'image', lambda *a, **kw: (_ for _ in ()).throw(ValueError('camera unavailable')))
    kit.execute_tool('reuseit_step', action("raise TypeError('bad helper')"))
    assert h.phase == 'repair_wait' and not h.repair_observation['images']
    assert 'error' in kit.execute_tool('reuseit_step', action('x=1', 'repair')).result
    kit.execute_tool('reuseit_observe', {})
    assert h.phase == 'repair_wait'


@pytest.mark.parametrize('limit', [-1, True, 257])
def test_code_repair_budget_is_validated(robo, tmp_path, workflow_config, limit):
    with pytest.raises(ValueError, match='max_code_repairs'):
        kit_for(robo, tmp_path, {**workflow_config, 'max_code_repairs': limit})


def test_runtime_recovers_code_error_with_shared_guard_calls(robo, tmp_path, workflow_config):
    from tests.test_reuseit_rlm_ports import run, call, report, end
    kit = kit_for(robo, tmp_path, workflow_config)
    def yes(event):
        return report({'satisfied': True, 'evidence': 'Synthetic controlled observation',
                       'evidence_ids': [f'event-{event:06d}-image-0']})
    transport, result = run(kit, tmp_path, [
        yes(2), call('reuseit_step', action("raise TypeError('bad argument')")),
        yes(6), call('reuseit_step', action('answer=46', 'repair')),
        yes(9), end('success')], tool_choice='auto')
    assert result['errors'] == [] and result['auxiliary_requests'] == 3
    assert kit.general_harness.phase == 'complete' and kit._globals['answer'] == 46
    assert kit.general_harness.index == 1


def test_non_programming_failure_retains_physical_fallback_limit(robo, tmp_path, workflow_config):
    kit = kit_for(robo, tmp_path, {**workflow_config, 'max_retries': 1})
    h = kit.general_harness
    guard(h)
    kit.execute_tool('reuseit_step', action("raise RuntimeError('primitive unavailable')"))
    assert h.phase == 'fallback' and h.code_repairs == 0
    kit.execute_tool('reuseit_step', action("raise RuntimeError('still unavailable')", 'fallback'))
    assert h.phase == 'blocked' and h.retries == 1 and h.code_repairs == 0
