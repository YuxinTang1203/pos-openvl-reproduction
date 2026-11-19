import pytest

from pyrualean.action_feedback import ExecutionFeedback, RecoveryRequired
from pyrualean.arms import ArmToolkit


def test_abort_caught_failure_blocks_dependencies_and_next_cell_recovers(robo, backend, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, images="none", feedback="pure")
    kit.enable_execution_recovery()
    before = len(backend.calls)
    result = kit.execute_tool(
        "python",
        {
            "code": """counter = 17
try:
    robo.move_to([0, 0, 2])
except Exception:
    pass
robo.move_to([0, 0, 0.5])
counter = 99"""
        },
    )
    assert result.result["recovery_required"]["reason"] == "motion_failed"
    assert sum(name == "move_to" for name, _ in backend.calls[before:]) == 1
    result = kit.execute_tool("python", {"code": "robo.move_to([0, 0, 0.5])\nprint(counter)"})
    assert "recovery_required" not in result.result
    assert "exception" not in result.result
    assert result.result["stdout"] == "17\n"


def test_contact_grasp_and_ik_have_distinct_recovery():
    guard = ExecutionFeedback()
    with guard.scope("contact"):
        guard.observe("move_pose", {}, {"reached": False, "blocked": True}, 0)
        guard.observe("grip", {"gripper": 1}, {"holding": False}, 1)
        assert guard.blocked is None
        with pytest.raises(RecoveryRequired):
            guard.observe("move_pose", {}, {"ik_error": "arm into table"}, 2)
    assert guard.blocked["reason"] == "ik_failure"
    guard.begin_turn()
    with guard.scope("grasp"), pytest.raises(RecoveryRequired):
        guard.observe("grip", {"gripper": 1}, {"holding": False}, 3)
    assert guard.blocked["reason"] == "empty_grasp"
    assert guard.intent == "move"
    guard.begin_turn()
    guard.observe("grip", {"gripper": 1}, {"holding": False}, 4)
    assert guard.blocked is None


def test_failure_attaches_current_observation(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, images="on-demand", feedback="pure")
    kit.enable_execution_recovery()
    result = kit.execute_tool("python", {"code": "robo.move_to([0, 0, 2])"})
    assert result.result["recovery_required"]
    assert result.images


def test_program_mode_checks_each_primitive(robo, backend, tmp_path):
    kit = ArmToolkit(robo, arm="program", workspace=tmp_path, images="none", feedback="pure")
    kit.enable_execution_recovery()
    before = len(backend.calls)
    result = kit.execute_tool(
        "run_program",
        {
            "code": """def run(robo):
    with robo.action_scope('move'):
        robo.move_to([0, 0, 2])
        robo.move_to([0, 0, 0.5])
"""
        },
    )
    assert result.result["recovery_required"]["reason"] == "motion_failed"
    assert sum(name == "move_to" for name, _ in backend.calls[before:]) == 1


@pytest.mark.parametrize("recovery", [False, True])
def test_harness_feedback_with_optional_dependency_guard(robo, backend, tmp_path, recovery):
    """Separate the independent harness from an explicitly enabled execution guard."""
    from pyrualean.migrations import migration_prompt
    from pyrualean.openai_runtime import run_openai_agent
    from tests.test_independent_migrations import configure
    from tests.test_openai_runtime import ScriptedTransport, _reply

    kit, _ = configure(robo, tmp_path, "harnesspro")
    if recovery:
        kit.enable_execution_recovery()
    transport = ScriptedTransport([
        _reply([("failed", "python", {"code": """print('x' * 18000)
counter = 17
try:
    robo.move_to([0, 0, 2])
except Exception:
    pass
robo.move_to([0, 0, 0.5])
counter = 99
"""})]),
        _reply([("recover", "python", {"code": "robo.move_to([0, 0, 0.6]); print(counter)"})]),
        _reply([("done", "finish", {"status": "stuck", "summary": "diagnostic only"})]),
    ])
    result = run_openai_agent(
        migration_prompt("Test", kit), [], kit, model="test", base_url="http://unused/v1",
        api_key=None, max_turns=100, max_requests=3, budget_s=30,
        out_dir=tmp_path / "episode", transport=transport)
    assert result["errors"] == []
    assert sum(name == "move_to" for name, _ in backend.calls) == (2 if recovery else 3)
    assert kit._globals["counter"] == (17 if recovery else 99)
    first = next(m["content"] for m in transport.requests[1]["body"]["messages"]
                 if m.get("tool_call_id") == "failed")
    assert "Tool output compacted" in first
    assert '"reached": false' in first
    assert ("recovery_required" in first) == recovery
    second = next(m["content"] for m in transport.requests[2]["body"]["messages"]
                  if m.get("tool_call_id") == "recover")
    assert "recovery_required" not in second
    assert '"reached": true' in second
