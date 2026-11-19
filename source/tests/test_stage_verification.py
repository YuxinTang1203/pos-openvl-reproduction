import json

import pytest

from pyrualean.arms import ArmToolkit


def toolkit(robo, tmp_path):
    kit = ArmToolkit(
        robo, arm="cells", workspace=tmp_path / "ws", images="on-demand", feedback="pure"
    )
    kit.enable_stage_verification()
    return kit


def assess(kit, status="observed"):
    s = kit.stage_verifier.current
    return [
        {
            "criterion": i,
            "status": status,
            "observation": "Observed the object in the compared views",
            "evidence": list(s["reviewed"].values()),
        }
        for i in range(len(s["criteria"]))
    ]


def test_stage_boundaries_preserve_multiaction_python_and_require_next_turn(
    robo, backend, tmp_path
):
    kit = toolkit(robo, tmp_path)
    denied = kit.execute_tool("python", {"code": "robo.move_to([0, 0, 0.6])"})
    assert "Stage review required" in denied.result["exception"]
    result = kit.execute_tool(
        "python",
        {
            "code": """
stage_begin('move into view', ['End effector is above the table'])
counter = 0
for p in ([0, 0, 0.6], [0.1, 0, 0.6]):
    robo.move_to(p)
    counter += 1
stage_review()
"""
        },
    )
    assert len(result.images) == 2
    assert kit._globals["counter"] == 2
    with pytest.raises(ValueError, match="NEXT"):
        kit.stage_verifier.record(assess(kit))
    denied = kit.execute_tool("python", {"code": "robo.move_to([0, 0, 0.7])"})
    assert "Stage review required" in denied.result["exception"]
    accepted = kit.execute_tool("python", {"code": f"stage_record({assess(kit)!r})"})
    assert "exception" not in accepted.result
    denied = kit.execute_tool("finish", {"status": "success", "summary": "done"})
    assert "error" in denied.result and kit.finish is None
    kit.execute_tool(
        "python",
        {
            "code": (
                "stage_begin('overall task', ['All requirements visible'], kind='final'); "
                "stage_review()"
            )
        },
    )
    kit.execute_tool("python", {"code": f"stage_record({assess(kit)!r})"})
    assert kit.execute_tool("finish", {"status": "success", "summary": "observed"}).result["ok"]


@pytest.mark.parametrize("kind", ["button", "placement", "insertion"])
def test_missing_contact_or_release_evidence_cannot_be_confirmed(robo, tmp_path, kind):
    kit = toolkit(robo, tmp_path)
    kit.execute_tool(
        "python",
        {"code": f"stage_begin('effect', ['Visible change'], kind='{kind}'); stage_review()"},
    )
    result = kit.execute_tool("python", {"code": f"stage_record({assess(kit)!r})"})
    assert "Evidence missing" in result.result["exception"]
    kit.execute_tool("python", {"code": f"stage_record({assess(kit, 'uncertain')!r})"})
    assert kit.stage_verifier.current["status"] == "needs_recovery"
    with pytest.raises(ValueError, match="Resolve"):
        kit.stage_verifier.begin("skip to other goal", ["done"])
    kit.stage_verifier.retry(change="Use wrist camera to see contact travel")
    with pytest.raises(ValueError, match="new concrete"):
        kit.stage_verifier.retry(change="Use wrist camera to see contact travel")


def test_contact_and_release_frames_are_separate_from_after(robo, tmp_path):
    kit = toolkit(robo, tmp_path)
    result = kit.execute_tool(
        "python",
        {
            "code": """
stage_begin('button press', ['Button travels and returns'], kind='button')
with robo.action_scope('contact'):
    robo.move_to([0, 0, 0.5])
robo.move_to([0, 0, 0.7])
stage_review()
"""
        },
    )
    assert len(result.images) == 3
    sv = kit.stage_verifier
    refs = sv.current["reviewed"]
    assert set(refs) == {"before", "contact", "after"}
    assert sv.snapshots[refs["contact"]]["revision"] < sv.snapshots[refs["after"]]["revision"]
    # Public primitive feedback, not an oracle, supplies held->released transition.
    kit.execute_tool("python", {"code": f"stage_record({assess(kit)!r})"})
    sv.begin("place object", ["Object rests on support"], kind="placement")
    sv.observe("grip", {"arm": "left", "gripper": robo.CLOSE}, {"holding": True})
    sv.observe("grip", {"arm": "left", "gripper": robo.OPEN}, {"holding": False})
    sv.review()
    assert "release" in sv.current["reviewed"]


def test_bad_references_and_stale_review_cannot_be_recorded(robo, tmp_path):
    kit = toolkit(robo, tmp_path)
    kit.execute_tool("python", {"code": "stage_begin('move', ['Visible']); stage_review()"})
    kit.stage_verifier.begin_turn()
    wrong = assess(kit)
    wrong[0]["evidence"] = ["invented_view"]
    with pytest.raises(ValueError, match="exact evidence"):
        kit.stage_verifier.record(wrong)
    kit.stage_verifier.motion_revision += 1
    with pytest.raises(ValueError, match="stale"):
        kit.stage_verifier.record(assess(kit))


def test_missing_camera_fails_closed(robo, backend, tmp_path, monkeypatch):
    kit = toolkit(robo, tmp_path)

    def unavailable(*a, **kw):
        raise FileNotFoundError("no camera")

    monkeypatch.setattr(backend, "artifact", unavailable)
    result = kit.execute_tool(
        "python", {"code": "stage_begin('move', ['Visible']); stage_review()"}
    )
    assert "Missing before/after" in result.result["exception"]
    assert not kit.stage_verifier.can_finish()
    events = [
        json.loads(line)
        for line in (kit.stage_verifier.root / "events.jsonl").read_text().splitlines()
    ]
    assert any(e["event"] == "capture_error" for e in events)


def test_stage_images_roundtrip_through_native_openvl_memory(robo, tmp_path):
    from openvlharness.memory.store import MemoryStore

    kit = toolkit(robo, tmp_path)
    memory = MemoryStore(emit_policy="on_change")
    memory.start([])
    kit.stage_verifier.memory = memory
    result = kit.execute_tool(
        "python", {"code": "stage_begin('observe', ['Visible scene']); stage_review()"}
    )
    assert "exception" not in result.result
    assert len(result.images) == 2
    assert all(s["native"] == "openvlharness" for s in kit.stage_verifier.snapshots.values())
    events = [
        json.loads(line)
        for line in (kit.stage_verifier.root / "events.jsonl").read_text().splitlines()
    ]
    assert (
        sum(e["event"] == "evidence_retrieved" and e["native"] == "openvlharness" for e in events)
        == 2
    )


def test_interrupted_primitive_is_not_logged_as_success(robo, tmp_path):
    kit = toolkit(robo, tmp_path)
    result = kit.execute_tool(
        "python", {"code": "stage_begin('approach', ['Target reached']); robo.move_to([0, 0, 2])"}
    )
    assert "RecoveryRequired" in result.result["exception"]
    failed = [r for r in robo._ledger.records if r.error and "RecoveryRequired" in r.error]
    assert failed and all(not r.ok for r in failed)
    assert kit.turns[-1]["ok"] is False


def test_program_arm_uses_same_public_stage_api(robo, tmp_path):
    kit = ArmToolkit(robo, arm="program", workspace=tmp_path / "ws", images="on-demand")
    kit.enable_stage_verification()
    result = kit.execute_tool(
        "run_program",
        {
            "code": """def run(robo):
    robo.stage_begin('approach', ['Visible end pose'])
    robo.move_to([0, 0, 0.6])
    robo.move_to([0.1, 0, 0.6])
    robo.stage_review()
"""
        },
    )
    assert "exception" not in result.result
    assert len(result.images) == 2


def test_two_tool_calls_from_same_model_reply_cannot_self_confirm(robo, tmp_path):
    kit = toolkit(robo, tmp_path)
    kit.model_turn = 1
    kit.execute_tool("python", {"code": "stage_begin('observe', ['Visible']); stage_review()"})
    response = kit.execute_tool("python", {"code": f"stage_record({assess(kit)!r})"})
    assert "NEXT" in response.result["exception"]
    kit.model_turn = 2
    response = kit.execute_tool("python", {"code": f"stage_record({assess(kit)!r})"})
    assert "exception" not in response.result


@pytest.mark.parametrize(
    "options",
    [
        {"arm": "program", "max_programs": 1},
        {"arm": "program", "max_programs": 3, "max_turns": 1},
        {"arm": "cells", "max_turns": 1},
        {"arm": "cells", "max_turns": 0},
    ],
)
def test_stage_rejects_insufficient_code_budget_without_installing_hooks(robo, tmp_path, options):
    kit = ArmToolkit(robo, workspace=tmp_path / "ws", images="on-demand", **options)
    original_globals = set(kit._globals)
    with pytest.raises(ValueError, match="at least two remaining code calls"):
        kit.enable_stage_verification()
    assert kit.stage_verifier is None and robo._stage_verifier is None
    assert robo._action_feedback is None
    assert set(kit._globals) == original_globals
    assert not (tmp_path / "stage_verification").exists()


def test_stage_budget_validation_counts_already_used_calls(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", max_turns=2)
    kit.execute_tool("python", {"code": "counter = 1"})
    kit.enable_action_feedback()
    previous_feedback = robo._action_feedback
    with pytest.raises(ValueError, match="at least two remaining code calls"):
        kit.enable_stage_verification()
    assert robo._action_feedback is previous_feedback
    assert kit._globals["counter"] == 1
    assert "action_scope" not in kit._globals
    assert kit.stage_verifier is None


def test_stage_image_validation_does_not_install_recovery(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "ws", images="none")
    with pytest.raises(ValueError, match="requires camera images"):
        kit.enable_stage_verification()
    assert kit.stage_verifier is None and robo._stage_verifier is None
    assert robo._action_feedback is None
    assert "checked" not in kit._globals and "action_scope" not in kit._globals


@pytest.mark.parametrize("arm", ["cells", "program"])
def test_two_code_calls_can_complete_final_stage_with_a_new_model_epoch(robo, tmp_path, arm):
    kit = ArmToolkit(robo, arm=arm, workspace=tmp_path / "ws", images="on-demand",
                     max_turns=2, max_programs=2 if arm == "program" else None)
    kit.enable_stage_verification()
    kit.model_turn = 1
    code = "robo.stage_begin('final', ['Visible target'], kind='final')\nrobo.stage_review()"
    tool = "python" if arm == "cells" else "run_program"
    if arm == "program":
        code = "def run(robo):\n" + "\n".join("    " + line for line in code.splitlines())
    first = kit.execute_tool(tool, {"code": code})
    assert "exception" not in first.result and len(first.images) == 2
    with pytest.raises(ValueError, match="NEXT"):
        kit.stage_verifier.record(assess(kit))
    kit.model_turn = 2
    code = f"robo.stage_record({assess(kit)!r})"
    if arm == "program":
        code = "def run(robo):\n    " + code
    second = kit.execute_tool(tool, {"code": code})
    assert "exception" not in second.result
    assert kit.stage_verifier.can_finish()
    assert kit.execute_tool("finish", {"status": "success", "summary": "observed"}).result["ok"]


def test_stage_cannot_replace_an_existing_pending_verifier(robo, tmp_path):
    kit = toolkit(robo, tmp_path)
    verifier = kit.stage_verifier
    verifier.begin("pending stage", ["Visible target"])
    with pytest.raises(ValueError, match="already enabled"):
        kit.enable_stage_verification()
    assert kit.stage_verifier is verifier and robo._stage_verifier is verifier
    assert verifier.current["goal"] == "pending stage"
