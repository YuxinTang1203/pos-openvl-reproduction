"""Native-code mode keeps the official observation/action loop intact."""

import threading

import numpy as np
import pytest

from pyrualean.hosts.robodojo_native_code import NativeCodeToolkit, validate_action
from pyrualean.hosts.xpolicylab_bridge import Bridge, EpisodeOver


def observation(step=0):
    return {
        "instruction": "move an arm",
        "state": {
            "left_arm_joint_state": np.zeros(6),
            "right_arm_joint_state": np.zeros(6),
            "left_ee_joint_state": np.ones(1),
            "right_ee_joint_state": np.ones(1),
        },
        "vision": {"cam_head": {"color": np.zeros((4, 4, 3), dtype=np.uint8)}},
        "additional_info": {"pyrualean": {"step": step, "step_limit": 5}},
    }


def hold():
    return {
        "left_arm_joint_state": [0.1] + [0.0] * 5,
        "right_arm_joint_state": [0.0] * 6,
        "left_ee_joint_state": [1.0],
        "right_ee_joint_state": [1.0],
    }


def test_native_code_submits_joint_action_and_reads_next_observation(tmp_path):
    bridge = Bridge(poll_s=0.1)
    bridge.update_obs(observation())
    toolkit = NativeCodeToolkit(bridge, tmp_path, max_turns=2)
    assert {spec["name"] for spec in toolkit.get_tools_spec()} == {"python", "finish"}
    assert "robo" not in toolkit._globals
    received = []

    def client():
        received.extend(bridge.get_action())
        bridge.update_obs(observation(1))

    thread = threading.Thread(target=client)
    thread.start()
    result = toolkit.execute_tool("python", {"code": f"a = {hold()!r}\nprint(step(a)['status']['step'])"})
    thread.join(1)
    assert received == [hold()]
    assert result.result["stdout"].strip() == "1"
    assert result.result["native_actions"] == 1
    assert not result.result.get("exception")


def test_native_code_rejects_invalid_actions_before_bridge(tmp_path):
    bridge = Bridge(poll_s=0.01)
    bridge.update_obs(observation())
    toolkit = NativeCodeToolkit(bridge, tmp_path, max_turns=1)
    bad = hold()
    bad["left_arm_joint_state"][0] = 0.5
    with pytest.raises(ValueError, match="exceeds"):
        validate_action(bad, bridge.get_obs()["state"])
    result = toolkit.execute_tool("python", {"code": "step({'left_arm_joint_state': [0]*6})"})
    assert "exactly" in result.result["exception"]
    assert toolkit._actions == []


def test_native_code_uses_previous_command_when_measured_joint_lags():
    state = observation()["state"]
    first = hold()
    second = hold()
    second["left_arm_joint_state"][0] = 0.2
    assert validate_action(second, state, first)["left_arm_joint_state"][0] == 0.2
    with pytest.raises(ValueError, match="exceeds"):
        validate_action(second, state)


def test_bridge_finish_interrupts_native_step():
    bridge = Bridge(poll_s=0.01)
    bridge.update_obs(observation())
    failures = []

    def submit():
        try:
            bridge.step(hold(), timeout_s=2)
        except EpisodeOver as exc:
            failures.append(str(exc))

    thread = threading.Thread(target=submit)
    thread.start()
    assert bridge.get_action() == [hold()]
    bridge.finish()
    thread.join(1)
    assert failures
