"""Historical inspection is explicit; stale current targets never return coordinates."""

import pytest

from tests.test_openvl_adapter import kit_for, perceive


def test_history_and_current_after_motion_do_not_interrupt_cell(robo, backend, tmp_path):
    kit = kit_for(robo, tmp_path, validate_geometry=False)
    ref, _ = perceive(kit)
    fresh = robo.evidence_current(ref)
    assert fresh["usable"] and fresh["measurement"] is not None
    kit._globals["ref"] = ref
    kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6])"})
    before = len(backend.calls)
    out = kit.execute_tool(
        "python",
        {
            "code": "old = evidence_history(ref)\n"
            "current = evidence_current(ref)\n"
            "if not current['usable']:\n"
            "    print(current['next_step']['operation'])\n"
            "    continued = True\n"
        },
    )
    assert "exception" not in out.result
    assert "relocalize" in out.result["stdout"] and kit._globals["continued"]
    assert kit._globals["old"]["current_target"] is False
    assert kit._globals["current"]["measurement"] is None
    assert not any(name in {"move_to", "segment"} for name, _ in backend.calls[before:])
    with pytest.raises(ValueError, match="Historical geometry"):
        robo.evidence(ref)


def test_relocalization_creates_a_new_current_reference(robo, tmp_path):
    kit = kit_for(robo, tmp_path, validate_geometry=False)
    old, _ = perceive(kit)
    kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6])"})
    stale = robo.evidence_current(old)
    assert stale["next_step"]["query"] == "bowl"
    assert stale["next_step"]["camera"] == "agentview"
    new, _ = perceive(kit)
    assert new != old and robo.evidence_current(new)["usable"]
    assert robo.evidence_current(old)["measurement"] is None
    history = robo.evidence_history(old)
    history["measurement"]["found"] = "modified"
    assert robo.evidence_history(old)["measurement"]["found"] != "modified"


@pytest.mark.parametrize("operation", ["history", "current"])
def test_native_capability_exposes_intent_and_requires_id(robo, tmp_path, operation):
    kit = kit_for(robo, tmp_path, validate_geometry=False)
    ref, _ = perceive(kit)
    kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6])"})
    out = kit.execute_tool("Robot_Evidence_Tool", {"operation": operation, "reference": ref})
    assert "error" not in out.result
    data = out.result["data"]
    if operation == "current":
        assert not data["usable"] and data["measurement"] is None
    else:
        assert data["purpose"] == "history" and not data["current_target"]
    missing = kit.execute_tool("Robot_Evidence_Tool", {"operation": operation})
    assert "error" in missing.result
