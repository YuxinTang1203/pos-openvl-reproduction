"""Exercise the real upstream loop/memory with scripted providers and robot backends."""

from __future__ import annotations

import copy
import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("openvlharness")

from pyrualean.arms import ArmToolkit
from pyrualean.openvl_runtime import run_openvl_agent


def reply(name=None, args=None, text="<answer>Done</answer>"):
    message = {"role": "assistant", "content": text if name is None else "Checking the scene."}
    if name:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ]
    return {
        "choices": [{"message": message}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 10},
        },
    }


class Provider:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def __call__(self, url, body, timeout):
        self.requests.append(copy.deepcopy(body))
        return 200, next(self.replies)


def run(robo, tmp_path, replies, **options):
    from PIL import Image

    path = tmp_path / "initial.png"
    Image.new("RGB", (64, 64), "red").save(path)
    toolkit = options.pop("toolkit", None) or ArmToolkit(
        robo, arm="cells", workspace=tmp_path / "ws", images="on-demand", feedback="pure"
    )
    transport = options.pop("transport", None) or Provider(replies)
    result = run_openvl_agent(
        "Use the robot tools to perform the task.",
        [path],
        toolkit,
        model=options.pop("model", "qwen3-vl-8b-instruct"),
        base_url="http://provider.test/v1",
        api_key="test",
        max_turns=None,
        budget_s=options.pop("budget_s", 30),
        out_dir=tmp_path / "out",
        transport=transport,
        **options,
    )
    return result, toolkit, transport


def test_native_loop_keeps_robot_observations_and_native_zoom(robo, tmp_path):
    from openvlharness.agent.agent import MainAgent

    result, toolkit, provider = run(
        robo,
        tmp_path,
        [
            reply("python", {"code": "counter = 7\nrobo.show('agentview')"}),
            reply(
                "Zoom_In_Tool",
                {"image_id": "tool_generated_image_1", "coords": [100, 100, 500, 500]},
            ),
            reply("python", {"code": "print(counter + 1)"}),
            reply("finish", {"status": "stuck", "summary": "Integration probe complete"}),
        ],
    )
    assert result["status"] == "completed" and result["stop_reason"] == "finish"
    assert result["requests"] == 4  # host stop adds no model request
    assert result["input_tokens"] == 400 and result["output_tokens"] == 80
    assert result["cached_input_tokens"] == 40
    assert len(toolkit.turns) == 3  # zoom is a perception tool, not a robot action
    second = json.dumps(provider.requests[1])
    third = json.dumps(provider.requests[2])
    assert "tool_generated_image_1" in second and "data:image/png;base64," in second
    assert "tool_generated_image_2" in third  # the upstream zoom stored its crop
    assert "8" in json.dumps(provider.requests[3])
    assert "agent/agent.py" in result["source"]["files"]["MainAgent"]["path"]
    assert MainAgent.__module__ == "openvlharness.agent.agent"
    trajectory = json.loads((tmp_path / "out/openvlharness_trajectory.json").read_text())
    turns = trajectory["extra_records"]["trajectory"]["chat_history"]
    assert turns[0]["memory_outputs"][0]["image_refs"] == ["tool_generated_image_1"]
    assert turns[1]["memory_outputs"][0]["canonical_result"]["image"] == "tool_generated_image_2"
    assert "data:image/" not in json.dumps(trajectory)
    assert list((tmp_path / "out/images").glob("*"))


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_changed_inventory_preserves_images_and_old_zoom_references(robo, tmp_path, protocol):
    runs = {}
    for policy in ("always", "on_change"):
        directory = tmp_path / policy
        directory.mkdir()
        replies = [
            reply("python", {"code": "counter = 7\nrobo.show('agentview')"}),
            reply("python", {"code": "counter += 1\nprint(counter)"}),
            reply(
                "Zoom_In_Tool",
                {"image_id": "tool_generated_image_1", "coords": [100, 100, 500, 500]},
            ),
            reply("python", {"code": "print(counter)"}),
            reply(),
        ]
        if protocol == "responses":
            converted = []
            for response in replies:
                message = response["choices"][0]["message"]
                if message.get("tool_calls"):
                    call = message["tool_calls"][0]
                    output = {"type": "function_call", "call_id": call["id"], **call["function"]}
                else:
                    output = {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": message["content"]}],
                    }
                converted.append({"output": [output]})
            replies = converted
        result, _, provider = run(
            robo,
            directory,
            replies,
            model="gateway-vision-model",
            protocol=protocol,
            memory_emit_policy=policy,
        )
        assert result["errors"] == []
        assert result["source"]["memory_emit_policy"] == policy
        trajectory = json.loads((directory / "out/openvlharness_trajectory.json").read_text())
        turns = trajectory["extra_records"]["trajectory"]["chat_history"]
        assert turns[1]["memory_outputs"][0]["canonical_result"]["stdout"] == "8\n"
        assert (
            turns[2]["memory_outputs"][0]["canonical_result"]["image"] == "tool_generated_image_2"
        )
        runs[policy] = result, provider, turns

    old, old_provider, old_turns = runs["always"]
    new, new_provider, new_turns = runs["on_change"]
    assert new["requests"] == old["requests"] == 5
    assert sum(bool(t["memory_turn_message_indices"]) for t in new_turns) < sum(
        bool(t["memory_turn_message_indices"]) for t in old_turns
    )
    assert new_turns[0]["memory_turn_message_indices"]  # newly observed image
    assert new_turns[2]["memory_turn_message_indices"]  # newly generated crop
    assert not new_turns[1]["memory_turn_message_indices"]  # state unchanged
    assert (
        new["request_metrics"][-1]["context_string_chars"]
        < old["request_metrics"][-1]["context_string_chars"]
    )

    def images(value):
        if isinstance(value, str):
            return [value] if value.startswith("data:image/") else []
        if isinstance(value, dict):
            return images(list(value.values()))
        if isinstance(value, list):
            return [image for item in value for image in images(item)]
        return []

    for before, after in zip(old_provider.requests, new_provider.requests, strict=True):
        assert images(before) == images(after)


def test_one_cell_runs_multiple_motions_and_reuses_python_state(robo, backend, tmp_path):
    result, toolkit, provider = run(
        robo,
        tmp_path,
        [
            reply(
                "python",
                {
                    "code": (
                        "def move_pair():\n"
                        "    for position in ([0.1, 0.1, 0.5], [0.1, 0.2, 0.5]):\n"
                        "        moved = robo.move_to(position)\n"
                        "        assert moved.reached\n"
                        "move_pair()\ncounter = 2\nprint(counter)"
                    )
                },
            ),
            reply("python", {"code": "move_pair()\ncounter += 2\nprint(counter)"}),
            reply(),
        ],
    )
    assert result["errors"] == [] and len(toolkit.turns) == 2
    assert sum(name == "move_to" for name, _ in backend.calls) == 4
    feedback = [
        message["content"]
        for message in provider.requests[-1]["messages"]
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    assert any('"stdout": "4\\n"' in text for text in feedback)
    assert result["source"]["memory_emit_policy"] == "on_change"


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_permanent_http_errors_do_not_repeat_identical_requests(robo, tmp_path, code):
    result, toolkit, _ = run(
        robo,
        tmp_path,
        [],
        transport=lambda *args: (code, {"error": "invalid model or request"}),
        retries=2,
    )
    assert result["status"] == "error" and result["requests"] == 1
    assert toolkit.turns == []
    assert result["request_metrics"][0]["status_code"] == code
    assert result["failed_request_wait_s"] == result["request_wait_s"]


def test_transport_timeout_is_timed_and_retried(robo, tmp_path):
    import httpx

    attempts = []

    def transport(*args):
        attempts.append(True)
        if len(attempts) == 1:
            time.sleep(0.01)
            raise httpx.ReadTimeout("timed out")
        return 200, reply()

    result, _, _ = run(robo, tmp_path, [], transport=transport, retries=1)
    assert result["requests"] == 2 and len(result["request_latency_s"]) == 2
    assert result["failed_request_wait_s"] >= 0.01
    assert result["request_wait_s"] == sum(result["request_latency_s"])
    assert result["request_metrics"][0]["ok"] is False
    assert result["request_metrics"][1]["ok"] is True


def test_gateway_image_timeout_remains_retryable(robo, tmp_path):
    attempts = []

    def transport(*args):
        attempts.append(True)
        if len(attempts) == 1:
            return 400, {"error": {"message": "Download multimodal file timed out"}}
        return 200, reply()

    result, _, _ = run(robo, tmp_path, [], transport=transport, retries=1)
    assert result["requests"] == 2 and result["errors"] == []


def test_invalid_arguments_do_not_execute_robot(robo, tmp_path):
    result, toolkit, provider = run(
        robo,
        tmp_path,
        [
            reply("python", {"code": 123}),
            reply("finish", {"status": "stuck", "summary": "stop"}),
        ],
    )
    assert len(toolkit.turns) == 1
    assert "invalid_tool_arguments" in json.dumps(provider.requests[1])
    assert result["errors"] == []


def test_robot_exception_reaches_next_model_turn(robo, tmp_path):
    result, _, provider = run(
        robo,
        tmp_path,
        [
            reply("python", {"code": "raise ValueError('robot feedback test')"}),
            reply(),
        ],
    )
    assert "robot feedback test" in json.dumps(provider.requests[1])
    assert result["errors"] == []


def test_finish_cannot_claim_benchmark_success(robo, backend, tmp_path):
    result, toolkit, _ = run(
        robo,
        tmp_path,
        [
            reply("finish", {"status": "success", "summary": "claimed success"}),
        ],
    )
    assert toolkit.finish["status"] == "success" and not backend.solved()
    assert result["stop_reason"] == "finish"


def test_native_multiple_call_rejection(robo, tmp_path):
    response = reply("python", {"code": "print(1)"})
    response["choices"][0]["message"]["tool_calls"].append(
        reply("finish", {"status": "stuck", "summary": "stop"})["choices"][0]["message"][
            "tool_calls"
        ][0]
    )
    result, toolkit, provider = run(robo, tmp_path, [response, reply()])
    assert toolkit.turns == []
    assert "multiple_tool_calls_not_allowed" in json.dumps(provider.requests[1])
    assert result["errors"] == []


def test_call_budget_stops_robot_and_loop(robo, tmp_path):
    toolkit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", max_turns=1)
    result, _, _ = run(
        robo,
        tmp_path,
        [
            reply("python", {"code": "print(1)"}),
            reply("python", {"code": "print(2)"}),
        ],
        toolkit=toolkit,
    )
    assert result["stop_reason"] == "call_budget"
    assert toolkit.turn_budget_exhausted
    assert result["requests"] <= 2


def test_timeout_discards_late_model_action(robo, tmp_path):
    def late(url, body, timeout):
        time.sleep(0.08)
        return 200, reply("python", {"code": "raise AssertionError('must not execute')"})

    stops = []
    result, toolkit, _ = run(
        robo, tmp_path, [], transport=late, budget_s=0.03, on_timeout=lambda: stops.append(True)
    )
    assert result["status"] == "timeout" and toolkit.turns == []
    assert stops == [True]


def test_external_episode_end_no_request(robo, tmp_path):
    result, toolkit, _ = run(robo, tmp_path, [], should_stop=lambda: True)
    assert result["requests"] == 0 and toolkit.turns == []
    assert result["stop_reason"] == "environment_ended"


def test_timeout_interrupts_robot_cell_before_next_motion(robo, backend, tmp_path):
    result, toolkit, _ = run(
        robo,
        tmp_path,
        # Leave room for upstream agent/schema initialization on a loaded host;
        # the timeout must occur inside the cell, not before the first request.
        [reply("python", {"code": "time.sleep(2.2)\nrobo.move_to([0.1, 0.1, 0.5])"})],
        budget_s=2.0,
        on_timeout=lambda: robo._stop_episode("timeout"),
    )
    assert result["status"] == "timeout"
    assert backend.frames == 0
    assert len(toolkit.turns) == 1


def test_cli_forwards_openvl_model_protocol_and_budget(tmp_path, monkeypatch):
    from pyrualean import cli

    captured = []

    def host(argv, **kwargs):
        captured.extend(argv)
        out = tmp_path / "episode"
        (out / "result.json").write_text(json.dumps({"status": "completed"}))
        return 0

    monkeypatch.setattr(cli, "run_host", host)
    cli.main(
        [
            "play",
            "--arm",
            "cells",
            "--runtime",
            "openvlharness",
            "--model",
            "qwen3.7-plus",
            "--openvl-protocol",
            "chat_completions",
            "--openvl-max-iterations",
            "23",
            "--openvl-memory-emit-policy",
            "always",
            "--out",
            str(tmp_path / "episode"),
            "--task",
            "0",
        ]
    )
    assert captured[captured.index("--runtime") + 1] == "openvlharness"
    assert captured[captured.index("--model") + 1] == "qwen3.7-plus"
    assert captured[captured.index("--openvl-protocol") + 1] == "chat_completions"
    assert captured[captured.index("--openvl-max-iterations") + 1] == "23"
    assert captured[captured.index("--openvl-memory-emit-policy") + 1] == "always"


def test_model_error_retains_native_partial_trajectory(robo, tmp_path):
    provider = Provider([reply("python", {"code": "x = 1"})])

    def fail(url, body, timeout):
        if provider.requests:
            return 503, {"error": "provider unavailable"}
        return provider(url, body, timeout)

    result, _, _ = run(robo, tmp_path, [], transport=fail, retries=0)
    assert result["status"] == "error" and result["errors"]
    trajectory = json.loads((tmp_path / "out/openvlharness_trajectory.json").read_text())
    assert len(trajectory["extra_records"]["trajectory"]["chat_history"]) == 1


def test_native_responses_profile(robo, tmp_path):
    response = {
        "output": [
            {
                "type": "function_call",
                "call_id": "finish_id",
                "name": "finish",
                "arguments": json.dumps({"status": "stuck", "summary": "stop"}),
            }
        ],
        "usage": {"input_tokens": 120, "output_tokens": 30},
    }
    result, _, provider = run(robo, tmp_path, [response], model="gpt-6-astra")
    assert result["stop_reason"] == "finish" and result["requests"] == 1
    assert result["input_tokens"] == 120
    assert "input" in provider.requests[0] and "messages" not in provider.requests[0]


def test_custom_gateway_model_uses_actual_id_and_native_chat_policy(robo, tmp_path):
    result, _, provider = run(
        robo,
        tmp_path,
        [
            reply("python", {"code": "robo.show('agentview')"}),
            reply("finish", {"status": "stuck", "summary": "stop"}),
        ],
        model="qwen3.7-plus",
        protocol="chat_completions",
        reasoning_effort="high",
    )
    assert result["source"]["model_binding"] == "custom_gateway"
    assert result["source"]["model_alias"] == "kimi-k3"
    assert result["model"] == "qwen3.7-plus"
    assert all(r["model"] == "qwen3.7-plus" for r in provider.requests)
    assert all(r["reasoning_effort"] == "high" for r in provider.requests)
    assert result["source"]["native_model_profile"]["context_policy"] == "provider_enforced"
    assert "tool_generated_image_1" in json.dumps(provider.requests[1])


def test_unknown_gateway_model_defaults_to_chat(robo, tmp_path):
    result, _, provider = run(robo, tmp_path, [reply()], model="my-vision-model")
    assert provider.requests[0]["model"] == "my-vision-model"
    assert result["source"]["protocol"] == "chat_completions"


def test_custom_responses_gateway_preserves_model_id(robo, tmp_path):
    response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done"}],
            }
        ]
    }
    result, _, provider = run(
        robo, tmp_path, [response], model="gateway-vision-model", protocol="responses"
    )
    assert provider.requests[0]["model"] == "gateway-vision-model"
    assert "input" in provider.requests[0]
    assert result["source"]["model_binding"] == "custom_gateway"


def test_iteration_ceiling_is_not_provider_failure(robo, tmp_path):
    result, _, _ = run(
        robo, tmp_path, [reply("python", {"code": "print(1)"}), reply(text="")], max_iterations=1
    )
    assert result["max_turns_hit"] and result["errors"] == []
    assert result["stop_reason"] == "iteration_limit"


def test_play_episode_dispatches_native_runtime(robo, backend, tmp_path, monkeypatch):
    from PIL import Image

    from pyrualean import play
    from pyrualean.robots import get_adapter

    png = tmp_path / "initial.png"
    Image.new("RGB", (64, 64), "red").save(png)
    card = SimpleNamespace(
        robot="libero",
        max_env_steps=600,
        task_language="move the bowl",
        images={"agentview": str(png)},
        render=lambda: "move the bowl",
    )
    host = SimpleNamespace(
        boot=lambda *a, **kw: (backend, []), dump_card=lambda *a: card, make_backend=lambda x: x
    )
    adapter = SimpleNamespace(**vars(get_adapter("libero")), host_module=lambda: host)
    monkeypatch.setattr(play, "get_adapter", lambda name: adapter)
    monkeypatch.setattr(play, "resolve_rpent_root", lambda explicit=None: tmp_path)
    monkeypatch.chdir(tmp_path)
    cell = SimpleNamespace(
        suite="test", task=0, seed=0, max_episode_steps=600, cuda_device=0, tag="test"
    )
    provider = Provider([reply("finish", {"status": "success", "summary": "claim only"})])
    result = play.play_episode(
        cell,
        tmp_path / "episode",
        arm="cells",
        model="qwen3-vl-8b-instruct",
        runtime="openvlharness",
        base_url="http://provider.test/v1",
        budget_s=30,
        openvl_options={"transport": provider},
    )
    assert result["status"] == "completed", result
    assert result["environment_success"] is False
    assert result["generation"]["runtime"] == "openvlharness"
    assert result["openvlharness_returncode"] == 0
    assert (tmp_path / "episode/openvlharness_source.json").is_file()


def test_native_bindings_reused_in_python_and_stale_geometry_rejected(robo, backend, tmp_path):
    result, toolkit, provider = run(robo, tmp_path, [
        reply('Robot_Perception_Tool', {'operation': 'segment', 'query': 'bowl', 'camera': 'agentview'}),
        reply('python', {'code': "a=evidence(); a['score']=-9\nprint(evidence()['score'])\npoint=evidence()['world_xyz']\nchecked(robo.move_to([0.05, -0.1, 0.6]))\nprint(evidence())"}),
        reply('python', {'code': "print(evidence(allow_historical=True)['world_xyz'])\nrobo.segment('bowl', camera='agentview')\nprint(evidence()['source']['step'])"}),
        reply('finish', {'status': 'stuck', 'summary': 'memory bridge tested'})])
    assert result['status'] == 'completed'
    assert 'Historical or untracked geometry' in json.dumps(provider.requests[2])
    assert '0.8' in json.dumps(provider.requests[2])  # mutation did not alter original binding
    assert sum(name == 'segment' for name, _ in backend.calls) == 2
    assert len(toolkit.openvl_evidence.refs) == 2
    events = [json.loads(l) for l in (tmp_path/'out/openvlharness_events.jsonl').read_text().splitlines()]
    assert any(e.get('type') == 'robot_evidence_stored' for e in events)
    assert result['source']['structured_evidence'] is True
