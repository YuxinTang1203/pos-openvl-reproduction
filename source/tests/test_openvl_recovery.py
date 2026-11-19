from types import SimpleNamespace

import pytest

from pyrualean.action_feedback import RecoveryRequired
from pyrualean.openvl_recovery import OpenVLRecovery, install
from tests.test_independent_migrations import configure


class Bridge:
    refs = ['old']
    def require_reference(self, ref):
        if ref not in self.refs:
            raise ValueError(ref)
    def validity(self, ref):
        return {'usable': ref != 'stale'}
    def raw(self, ref):
        return {'found': ref != 'missing'}


def guard():
    bridge = Bridge()
    g = OpenVLRecovery(SimpleNamespace(bridge=bridge, event=lambda *a, **kw: None))
    g.begin_turn()
    with pytest.raises(RecoveryRequired):
        g.observe('move_to', {'xyz': [0, 0, 2]}, {'reached': False}, 1)
    bridge.refs = ['old', 'fresh', 'stale', 'missing']
    return g


def test_persistent_failure_and_evidence_gate():
    g = guard()
    with pytest.raises(RecoveryRequired):
        with g.recovery_attempt('fresh', change='new approach'):
            pass
    g.begin_turn()
    with pytest.raises(RecoveryRequired):
        g.before(tool='move_to', arguments={'xyz': [0, 0, .5]})
    for ref in ['old', 'stale', 'missing']:
        with pytest.raises(RecoveryRequired):
            with g.recovery_attempt(ref, change='new approach'):
                pass
    assert g.blocked


def test_preparation_does_not_resolve_and_one_retry_requires_measurement():
    g = guard()
    g.begin_turn()
    with g.recovery_attempt('fresh', change='waypoint', prepare=True):
        g.before(tool='move_to', arguments={'xyz': [0, 0, .4]})
        g.observe('move_to', {'xyz': [0, 0, .4]}, {'reached': True}, 2)
    assert g.blocked
    with g.recovery_attempt('fresh', change='retry'):
        g.before(tool='move_to', arguments={'xyz': [0, 0, 2]})
        g.observe('move_to', {'xyz': [0, 0, 2]}, {}, 3)
    assert g.blocked  # missing result is not success
    with g.recovery_attempt('fresh', change='revised height'):
        g.before(tool='move_to', arguments={'xyz': [0, 0, .5]})
        g.observe('move_to', {'xyz': [0, 0, .5]}, {'reached': True}, 4)
        with pytest.raises(RecoveryRequired):
            g.before(tool='grip', arguments={'gripper': -1})
    assert g.blocked is None


def test_release_wrong_arm_and_identical_retry_are_rejected():
    g = guard()
    g.begin_turn()
    for tool, args in [('grip', {'gripper': -1}), ('move_to', {'arm': 'right'}),
                       ('move_to', {'xyz': [0, 0, 2]})]:
        with pytest.raises(RecoveryRequired):
            with g.recovery_attempt('fresh', change='retry'):
                g.before(tool=tool, arguments=args)
    assert g.blocked


def test_openvl_python_boundary_persists_and_rejects_finish(robo, backend, tmp_path):
    kit, _ = configure(robo, tmp_path, 'openvl_evidence')
    install(kit)
    first = kit.execute_tool('python', {'code': 'robo.move_to([0, 0, 2])'})
    assert first.result['recovery_required']
    count = len(backend.calls)
    second = kit.execute_tool('python', {'code': 'robo.move_to([0, 0, .5])'})
    assert second.result['recovery_required']
    assert not any(n == 'move_to' for n, _ in backend.calls[count:])
    assert 'error' in kit.execute_tool('finish', {'status': 'success', 'summary': 'done'}).result


def test_real_openvl_binding_recovery_and_stale_pre_failure_rejected(robo, tmp_path):
    from tests.test_openvl_adapter import perceive
    kit, _ = configure(robo, tmp_path, 'openvl_evidence')
    g = install(kit)
    old, _ = perceive(kit)
    kit.execute_tool('python', {'code': 'robo.move_to([0, 0, 2])'})
    kit.execute_tool('python', {'code': 'print("new turn")'})
    with pytest.raises(RecoveryRequired):
        with g.recovery_attempt(old, change='lower target'):
            pass
    fresh, _ = perceive(kit)
    result = kit.execute_tool('python', {'code':
        f"with robo.recovery_attempt({fresh!r}, change='lower target'):\n    robo.move_to([0, 0, .5])"})
    assert 'exception' not in result.result, result.result
    assert g.blocked is None
    assert 'recovery_required' not in result.result


def test_backend_exception_remains_pending(robo, backend, tmp_path, monkeypatch):
    kit, _ = configure(robo, tmp_path, 'openvl_evidence')
    g = install(kit)
    execute = backend.execute_tool
    def fail(name, args):
        if name == 'move_to':
            return {'error': 'IK infeasible'}
        return execute(name, args)
    monkeypatch.setattr(backend, 'execute_tool', fail)
    kit.execute_tool('python', {'code': 'robo.move_to([0, 0, .5])'})
    assert g.blocked['reason'] == 'execution_error'
    g.begin_turn()
    with pytest.raises(RecoveryRequired):
        g.before(tool='grip', arguments={'gripper': -1})


def test_grasp_and_contact_do_not_confuse_arrival_with_holding():
    g = guard()
    g.begin_turn()
    g.blocked = None
    with g.scope('contact'):
        g.observe('move_to', {}, {'blocked': True, 'reached': False}, 2)
    assert g.blocked is None
    with g.scope('grasp'), pytest.raises(RecoveryRequired):
        g.observe('grip', {'gripper': 1}, {'holding': False}, 3)
    g.adapter.bridge.refs = [*g.adapter.bridge.refs, 'new']
    g.begin_turn()
    with g.recovery_attempt('new', change='reopen', prepare=True):
        g.before(tool='grip', arguments={'gripper': -1})
        g.observe('grip', {'gripper': -1}, {'opening': 1.0, 'holding': False}, 4)
    assert g.blocked
    with g.recovery_attempt('new', change='close after preparation'):
        g.before(tool='grip', arguments={'gripper': 1})
        g.observe('grip', {'gripper': 1}, {'holding': True}, 5)
    assert g.blocked is None


def test_native_migration_opt_in_and_prompt(robo, tmp_path):
    from pyrualean.arms import ArmToolkit
    from pyrualean.migrations import configure_migration, validate_migration
    from pyrualean.openvl_adapter import adapt_prompt
    from tests.test_openvl_native import ROOT
    config = {'kind': 'openvl', 'config': str(ROOT / 'capabilities_local.yml'), 'recovery': True}
    kit = ArmToolkit(robo, arm='cells', workspace=tmp_path / 'work', images='none', feedback='pure')
    record = configure_migration(kit, config, task='test', initial_images={})
    assert record['execution_recovery'] is True
    assert isinstance(kit.robo._action_feedback, OpenVLRecovery)
    assert 'robo.recovery_attempt' in adapt_prompt('task', kit)
    with pytest.raises(ValueError):
        validate_migration({**config, 'recovery': 'true'})
