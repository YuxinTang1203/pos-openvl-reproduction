"""Exercise released hooks through real serialized HTTP, without a model service."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.migrations import configure_migration, migration_prompt
from pyrualean.openai_runtime import run_openai_agent
from tests.test_independent_migrations import BASE, PROJECTS
from tests.test_openai_runtime import _reply


@pytest.mark.parametrize("compress", [False, True])
def test_native_hook_effects_reach_http_receiver(robo, tmp_path, monkeypatch, compress):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    policy = json.loads((BASE / "integrations/migrations/harnesspro_policy.json").read_text())
    policy["session_context"] = "ROBOT_HOOK_SMOKE_TEST"
    policy["profiles"]["standard"].update(
        economy_after_tools=1, strict_after_tools=2, finalize_after_tools=3,
        strict_reminder_every=1, reasoning_effort="low")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace",
                     images="on-demand", feedback="pure", max_turns=100)
    configure_migration(kit, {"kind": "harnesspro", "root": str(PROJECTS / "HarnessPro"),
                             "policy": str(policy_path)}, task="Move an object", initial_images={})
    selection = json.loads(kit.harnesspro_hook.selection_path.read_text())
    assert [selection[k] for k in ("economy_after_tools", "strict_after_tools",
                                   "finalize_after_tools")] == [1, 2, 3]
    assert selection["tool_output_max_chars"] == policy["tool_output"]["max_chars"]

    # Two calls in one response also exercise tool-call/reply pairing.
    output_size = 18000 if compress else 20
    replies = [
        _reply([("a", "python", {"code": f"robo.move_to([0,0,2]); print('x'*{output_size})"}),
                ("b", "python", {"code": "print('SECOND_CELL')"})]),
        _reply([("c", "python", {"code": "third_executed = True"})]),
        _reply([("d", "python", {"code": "denied_executed = True"})]),
        _reply([("e", "finish", {"status": "stuck", "summary": "smoke"})]),
    ]
    received = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            index = len(received) - 1
            response = json.dumps(replies[index] if index < len(replies) else {}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = run_openai_agent(
            migration_prompt("Test", kit), [], kit, model="test",
            base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key=None,
            max_turns=100, max_requests=4, budget_s=30, retries=0,
            extra_body={"reasoning_effort": kit.harnesspro_hook.selection.profile.reasoning_effort},
            out_dir=tmp_path / "episode")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result["status"] == "completed"
    assert len(received) == 4
    for path, body in received:
        assert path == "/v1/chat/completions"
        assert body["reasoning_effort"] == "low"
        assert "ROBOT_HOOK_SMOKE_TEST" in json.dumps(body["messages"])
        pending = set()
        for message in body["messages"]:
            if message["role"] == "tool":
                pending.remove(message["tool_call_id"])
            else:
                assert not pending
                pending.update(call["id"] for call in message.get("tool_calls", []))
        assert not pending

    feedback = {m["tool_call_id"]: m["content"] for m in received[1][1]["messages"]
                if m["role"] == "tool"}
    assert "HarnessPro economy phase" in feedback["a"]
    assert "HarnessPro convergence phase" in feedback["b"]
    assert '"reached": false' in feedback["a"]
    assert ("Tool output compacted" in feedback["a"]) == compress
    if compress:
        assert "x" * output_size not in feedback["a"]
        assert len(feedback["a"]) < output_size
        assert "Protected robot feedback" in feedback["a"]
    denied = next(m["content"] for m in received[3][1]["messages"]
                  if m.get("tool_call_id") == "d")
    assert "budget exhausted" in denied
    assert kit._globals["third_executed"] is True
    assert "denied_executed" not in kit._globals
    assert kit.finish["status"] == "stuck"
    assert kit.harnesspro_hook.counters["python_cells"] == 3
    assert kit.harnesspro_hook.counters["denied_cells"] == 1
