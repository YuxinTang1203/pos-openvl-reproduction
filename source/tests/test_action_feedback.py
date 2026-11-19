from pyrualean.arms import ArmToolkit


def test_failure_feedback_survives_silent_cell_and_checked_stops_dependencies(
    robo, backend, tmp_path
):
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path, images="none", feedback="pure")
    toolkit.enable_action_feedback()
    result = toolkit.execute_tool("python", {"code": "robo.move_to([0, 0, 2]); print('attempted')"})
    assert result.result["action_feedback"]["events"][0]["observation"] == "target_not_reached"
    before = len(backend.calls)
    result = toolkit.execute_tool(
        "python", {"code": "checked(robo.move_to([0, 0, 2]))\nrobo.move_to([0, 0, 0.5])"}
    )
    assert "Robot check" in str(result.result)
    assert sum(name == "move_to" for name, _ in backend.calls[before:]) == 1
    result = toolkit.execute_tool(
        "python",
        {
            "code": "counter = 0\nfor p in ([0, 0, 0.6], [0.1, 0, 0.6]):\n    checked(robo.move_to(p))\n    counter += 1"
        },
    )
    assert "exception" not in result.result
    assert toolkit.execute_tool("python", {"code": "print(counter)"}).result["stdout"] == "2\n"


def test_plain_baseline_feedback_is_unchanged(robo, tmp_path):
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path, images="none", feedback="pure")
    result = toolkit.execute_tool("python", {"code": "robo.move_to([0, 0, 2]); print('attempted')"})
    assert "action_feedback" not in result.result
    assert "checked" not in toolkit._globals
