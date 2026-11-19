"""Optional robot helpers reflect installed instance capabilities, not class names."""

import pytest

from pyrualean import LiberoRobot, TaskCard, api_reference, render_prompt
from pyrualean.arms import ArmToolkit
from pyrualean.prompt import enabled_helper_reference

GUARDIAN = ("guardian_checkpoint", "guardian_retry", "guardian_plan")
STAGE = ("stage_begin", "stage_review", "stage_record", "stage_retry")
OPTIONAL = (*GUARDIAN, *STAGE, "action_scope")


def toolkit(robo, tmp_path, *, arm="cells"):
    return ArmToolkit(robo, arm=arm, workspace=tmp_path / "ws", images="on-demand")


class Verifier:
    def verify_subtask(self, **kwargs):
        return True, "success"

    verify_plan = verify_subtask


def test_baseline_reference_has_native_api_without_disabled_helpers(robo):
    reference = api_reference(robo)
    assert reference == api_reference(type(robo))
    assert enabled_helper_reference(robo) == ""
    for name in OPTIONAL:
        assert f"robo.{name}(" not in reference
    for name in ("move_to", "state", "image", "show", "artifact", "set_gripper"):
        assert f"robo.{name}(" in reference
    assert "robo.task" in reference and "robo.done" in reference
    assert "class Move(" in reference and "ToolError" in reference


def test_guardian_reference_depends_on_instance_not_shared_class(robo, backend, tmp_path):
    other = LiberoRobot(backend)
    kit = toolkit(robo, tmp_path)
    kit.enable_guardian(client=Verifier(), cameras=robo.CAMERAS[:1])
    enabled = api_reference(robo)
    for name in GUARDIAN:
        assert f"robo.{name}(" in enabled
        assert f"robo.{name}(" not in api_reference(other)
        assert f"robo.{name}(" not in api_reference(type(robo))
    for name in (*STAGE, "action_scope"):
        assert f"robo.{name}(" not in enabled
    assert "robo.guardian_checkpoint(" in render_prompt(robot=robo).system
    assert "robo.guardian_checkpoint(" not in render_prompt(robot=other).system


def test_action_scope_requires_execution_recovery_not_feedback_collector(robo, tmp_path):
    kit = toolkit(robo, tmp_path)
    kit.enable_action_feedback()
    assert "robo.action_scope(" not in api_reference(robo)
    kit.enable_execution_recovery()
    assert "robo.action_scope(" in api_reference(robo)
    assert all(f"robo.{name}(" not in api_reference(robo) for name in STAGE)
    kit.enable_stage_verification()
    reference = api_reference(robo, primitives="no-vla")
    assert all(f"robo.{name}(" in reference for name in (*STAGE, "action_scope"))
    assert "robo.pi0_pick(" not in reference
    assert "robo.move_to(" in reference


@pytest.mark.parametrize("arm", ["cells", "program"])
def test_tools_reflect_late_enable_without_accumulating_helper_docs(robo, tmp_path, arm):
    kit = toolkit(robo, tmp_path, arm=arm)
    original = kit.get_tools_spec()
    assert "Enabled robot helpers" not in original[0]["description"]
    assert all(name not in original[0]["description"] for name in OPTIONAL)
    kit.enable_stage_verification()
    enabled = kit.get_tools_spec()
    assert enabled == kit.get_tools_spec()
    assert enabled[0]["description"].startswith(original[0]["description"])
    assert enabled[0]["description"].count("Enabled robot helpers:") == 1
    assert all(f"robo.{name}(" in enabled[0]["description"] for name in (*STAGE, "action_scope"))
    assert all(f"robo.{name}(" not in enabled[0]["description"] for name in GUARDIAN)
    assert "Enabled robot helpers" not in original[0]["description"]
    assert enabled[0]["input_schema"] == original[0]["input_schema"]
    assert enabled[1:] == original[1:]


def test_render_prompt_rejects_mismatched_instance_and_class(robo):
    class OtherRobot(LiberoRobot):
        pass

    with pytest.raises(ValueError, match="instance of robot_cls"):
        render_prompt(robot=robo, robot_cls=OtherRobot)


def test_episode_build_prompt_uses_enabled_robot_instance(robo, tmp_path):
    from pyrualean.play import build_prompt

    card = TaskCard(suite="test", task=0, seed=0, task_language="Move the object", object_names=[])
    options = dict(budget_s=60, max_programs=None, robot=robo)
    baseline = build_prompt("cells", card, **options)
    assert all(f"robo.{name}(" not in baseline for name in OPTIONAL)
    kit = toolkit(robo, tmp_path)
    kit.enable_guardian(client=Verifier(), cameras=robo.CAMERAS[:1])
    enabled = build_prompt("cells", card, **options)
    assert all(f"robo.{name}(" in enabled for name in GUARDIAN)
    assert "robo.move_to(" in baseline and "robo.move_to(" in enabled
    preview = build_prompt("cells", card, budget_s=60, max_programs=None)
    assert all(f"robo.{name}(" not in preview for name in OPTIONAL)
