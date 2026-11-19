"""Protect complete action feedback once and account for the text actually sent."""

import json

from pyrualean.arms import MAX_TEXT_CHARS, ArmResult
from pyrualean.harnesspro_runtime import NativeHarnessProHook


def hook_with_response(tmp_path, response):
    hook = object.__new__(NativeHarnessProHook)
    hook.audit_path = tmp_path / "audit.jsonl"
    inputs = []

    def call(event, name, payload):
        inputs.append(payload["tool_response"])
        return response

    hook._call = call
    return hook, inputs


def test_protected_data_never_enters_the_compressed_copy(tmp_path):
    hook, inputs = hook_with_response(tmp_path, {
        "decision": "block", "reason": "[Tool output compacted. Full raw result: raw.json]\nshort",
    })
    protected = {"primitive_feedback": [{"feedback": {"reached": False, "ik_error": "blocked"}}]}
    result = {"stdout": "x" * 18000, "robot_execution": protected, "exception": "keep me"}
    text = hook.after("python", result, 2)
    assert inputs == [{"stdout": "x" * 18000}]
    assert text.count('"robot_execution"') == 1
    assert text.count('"exception"') == 1
    assert "Full unprotected output:" in text
    delivered = json.loads(text.split("Protected robot feedback (not a task verdict):\n")[1])
    assert delivered == {"robot_execution": protected, "exception": "keep me"}
    audit = json.loads(hook.audit_path.read_text())
    assert audit["compacted"] is True
    assert audit["final_chars"] == len(text) < audit["original_chars"]
    assert audit["saved_chars"] == audit["original_chars"] - len(text)


def test_no_net_savings_delivers_original_without_a_compression_claim(tmp_path):
    hook, _ = hook_with_response(tmp_path, {"decision": "block", "reason": "overhead" * 500})
    result = {"stdout": "short", "robot_execution": {"task_outcome": "unknown"}}
    text = hook.after("python", result, 3)
    assert text == json.dumps(result, indent=2, ensure_ascii=False)
    audit = json.loads(hook.audit_path.read_text())
    assert audit["hook_requested_compaction"] is True
    assert audit["compacted"] is False and audit["saved_chars"] == 0
    assert audit["candidate_chars"] > audit["final_chars"]


def test_large_protected_feedback_is_complete_without_general_text_truncation(tmp_path):
    hook, inputs = hook_with_response(tmp_path, {})
    result = {"stdout": "ok", "robot_execution": {"primitive_feedback": [
        {"index": i, "reached": False, "note": "observed failure"} for i in range(600)
    ]}}
    text = hook.after("python", result, 4)
    assert len(text) > MAX_TEXT_CHARS
    arm_result = ArmResult(result, rendered_text=text)
    delivered = arm_result.content_blocks[0]["text"]
    assert json.loads(delivered) == result
    assert delivered.count('"robot_execution"') == 1
    assert inputs == [{"stdout": "ok"}]
    audit = json.loads(hook.audit_path.read_text())
    assert audit["compacted"] is False
    assert audit["final_chars"] == len(delivered)
