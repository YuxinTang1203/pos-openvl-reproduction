"""Exercise the opt-in prompt setting through the real model request path."""
import copy

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.execution_style import BATCH_ACTIONS_GUIDANCE, execution_prompt
from pyrualean.openai_runtime import run_openai_agent


@pytest.mark.parametrize("enabled", [False, True])
def test_batch_option_reaches_request_without_changing_robot_tool_interface(robo, tmp_path, enabled):
    requests = []

    def transport(url, body, timeout_s):
        requests.append(copy.deepcopy(body))
        return 200, {"choices": [{"message": {"role": "assistant", "content": "Done."},
                                  "finish_reason": "stop"}]}

    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace")
    result = run_openai_agent(
        "TASK AND API", [], kit, model="test", base_url="http://test/v1", api_key=None,
        max_turns=1, budget_s=10, out_dir=tmp_path / "out", nudges=0,
        transport=transport, batch_actions=enabled,
    )
    content = requests[0]["messages"][0]["content"]
    text = content if isinstance(content, str) else content[0]["text"]
    expected = "TASK AND API" + ("\n\n" + BATCH_ACTIONS_GUIDANCE if enabled else "")
    assert text == expected
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == ["python", "finish"]
    assert "batch_actions" not in requests[0]  # Host option, not an unsupported provider field.
    assert result["options"]["batch_actions"] is enabled


def test_program_arm_rejects_cell_specific_guidance():
    with pytest.raises(ValueError, match="cells"):
        execution_prompt("TASK", batch_actions=True, arm="program")
