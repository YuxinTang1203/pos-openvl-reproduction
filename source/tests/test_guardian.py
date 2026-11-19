import json
from pathlib import Path

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.guardian import GuardianRejected


class Verifier:
    def __init__(self, answer=True, category="success"):
        self.answer, self.category = answer, category
        self.calls = []

    def verify_subtask(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer, self.category

    verify_plan = verify_subtask


def setup(robo, tmp_path, client=None, arm="cells", cameras=None):
    kit = ArmToolkit(robo, arm=arm, workspace=tmp_path / "workspace", images="none")
    guard = kit.enable_guardian(client=client or Verifier(), cameras=cameras or robo.CAMERAS[:1])
    return kit, guard


def test_three_views(robo, tmp_path, monkeypatch):
    monkeypatch.setattr(robo, "CAMERAS", ("head", "left", "right"))
    monkeypatch.setattr(robo, "_show_artifacts", lambda cameras: [c + ".png" for c in cameras])
    kit, guard = setup(robo, tmp_path, cameras=robo.CAMERAS)
    result = kit.execute_tool("python", {"code": """
x = 5
with robo.guardian_checkpoint('grasp bowl', 'bowl held above table', kind='grasp'):
    robo.move_to([0, 0, 0.5])
x += 1
"""})
    assert "exception" not in result.result
    assert kit._globals["x"] == 6
    call = guard.client.calls[0]
    paths = call["img_paths_list"]
    assert [Path(p).name for p in paths] == [
        f"0001_{phase}_{i}.png" for phase in ("before", "after") for i in range(3)]
    assert all(Path(p).is_file() for p in paths)
    assert "bowl held above table" in call["subtask_instruction"]
    assert guard.last["verdict"] == "pass"
    assert not guard.final_supported


@pytest.mark.parametrize("answer,category,verdict", [
    (False, "no progress", "fail"), (None, None, "uncertain"),
    ("False", "success", "uncertain"), (True, "no progress", "uncertain"),
])
def test_caught_failure_still_blocks_motion(robo, backend, tmp_path, answer, category, verdict):
    kit, guard = setup(robo, tmp_path, Verifier(answer, category))
    result = kit.execute_tool("python", {"code": """
try:
    with robo.guardian_checkpoint('grasp', 'bowl held'):
        robo.move_to([0, 0, 0.5])
except Exception:
    pass
robo.move_to([0, 0, 0.6])
"""})
    assert "GuardianRejected" in result.result["exception"]
    assert guard.last["verdict"] == verdict
    assert sum(n == "move_to" for n, _ in backend.calls) == 1
    result = kit.execute_tool("python", {"code": "robo.move_to([0, 0, 0.6])"})
    assert "GuardianRejected" in result.result["exception"]
    with pytest.raises(ValueError):
        guard.retry(change=" ")
    guard.retry(change="Observe wrist view and correct grasp offset")
    result = kit.execute_tool("python", {"code": "robo.move_to([0, 0, 0.6])"})
    assert "GuardianRejected" in result.result["exception"]
    assert sum(n == "move_to" for n, _ in backend.calls) == 1
    guard.client.answer, guard.client.category = True, "success"
    result = kit.execute_tool("python", {"code": """
with robo.guardian_checkpoint('revised grasp', 'bowl held above table', kind='grasp'):
    robo.move_to([0, 0, 0.6])
"""})
    assert not result.result.get("exception")
    assert sum(n == "move_to" for n, _ in backend.calls) == 2
    assert len(guard.client.calls) == 2


def test_final_gate(robo, tmp_path):
    kit, guard = setup(robo, tmp_path)
    assert "error" in kit.execute_tool("finish", {"status": "success"}).result
    with robo.guardian_checkpoint("check task", "bowl on plate", kind="final"):
        pass
    assert guard.final_supported
    assert "entire task" in guard.client.calls[0]["subtask_instruction"]
    with robo.guardian_checkpoint("move arm away", "arm clear of bowl"):
        robo.move_to([0, 0, 0.6])
    assert not guard.final_supported
    assert "error" in kit.execute_tool("finish", {"status": "success"}).result
    with robo.guardian_checkpoint("check task", "bowl on plate", kind="final"):
        pass
    assert kit.execute_tool("finish", {"status": "success"}).result["ok"]
    assert not robo._backend.solved()


def test_program_and_plan(robo, tmp_path):
    kit, guard = setup(robo, tmp_path, arm="program")
    robo.guardian_plan("grasp bowl, carry to plate, release")
    assert len(guard.client.calls[0]["img_paths_list"]) == 1
    result = kit.execute_tool("run_program", {"code": """
def run(robo):
    with robo.guardian_checkpoint('grasp bowl', 'bowl held'):
        robo.move_to([0, 0, 0.5])
"""})
    assert not result.result.get("exception")
    assert len(guard.client.calls) == 2


def test_capture_failure_prevents_action(robo, backend, tmp_path, monkeypatch):
    kit, guard = setup(robo, tmp_path)
    monkeypatch.setattr(robo, "artifact", lambda *a, **kw: b"bad png")
    result = kit.execute_tool("python", {"code": """
with robo.guardian_checkpoint('grasp', 'held'):
    robo.move_to([0, 0, 0.5])
"""})
    assert result.result["exception"]
    assert guard.blocked
    assert not guard.client.calls
    assert not any(n == "move_to" for n, _ in backend.calls)


def test_model_and_action_exception(robo, tmp_path):
    class Broken(Verifier):
        def verify_subtask(self, **kw):
            raise RuntimeError("model unavailable")
    kit, guard = setup(robo, tmp_path, Broken())
    with pytest.raises(GuardianRejected), robo.guardian_checkpoint("grasp", "held"):
        pass
    assert guard.last["verdict"] == "uncertain"
    guard.retry(change="repair model service")
    with pytest.raises(ValueError), robo.guardian_checkpoint("grasp", "held"):
        raise ValueError("action failed")
    assert guard.blocked
    events = [json.loads(line) for line in (guard.root / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "checkpoint_interrupted"


def test_disabled_baseline(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path, images="none")
    assert kit.execute_tool("finish", {"status": "success"}).result["ok"]
    assert kit.guardian_verifier is None


def test_unscoped_motion_is_rejected_before_backend_even_after_pass(robo, backend, tmp_path):
    kit, guard = setup(robo, tmp_path)
    original_calls = list(backend.calls)
    with pytest.raises(GuardianRejected, match="active guardian_checkpoint"):
        robo.move_to([0, 0, 0.5])
    assert backend.calls == original_calls
    # Observations do not require a checkpoint, even after rejection.
    robo.state()
    guard.plan("grasp bowl")
    original_calls = list(backend.calls)
    with pytest.raises(GuardianRejected, match="active guardian_checkpoint"):
        robo.move_to([0, 0, 0.5])
    assert backend.calls == original_calls
    with guard.checkpoint("grasp bowl", "bowl held", kind="grasp"):
        robo.move_to([0, 0, 0.5])
    original_calls = list(backend.calls)
    with pytest.raises(GuardianRejected, match="active guardian_checkpoint"):
        robo.move_to([0, 0, 0.6])
    assert backend.calls == original_calls


@pytest.mark.parametrize("variant", ["thinking", "vanilla"])
def test_official_loader_pairs_variant(tmp_path, monkeypatch, variant):
    from types import SimpleNamespace

    from pyrualean.guardian import LocalGuardian

    source = tmp_path / "examples" / "guardian.py"
    source.parent.mkdir()
    source.touch()
    seen = {}

    class Model:
        execution_instruction = "original prompt"

        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("pyrualean.guardian.importlib.import_module", lambda name: SimpleNamespace(
        __file__=str(source), Guardian=Model))
    loaded = LocalGuardian(
        repo=tmp_path, model_path=tmp_path / f"guardian-{variant}", variant=variant)
    assert seen["thinking"] is (variant == "thinking")
    assert "left wrist" in loaded.model.execution_instruction
    other = "vanilla" if variant == "thinking" else "thinking"
    with pytest.raises(ValueError, match="disagree"):
        LocalGuardian(repo=tmp_path, model_path=tmp_path / f"guardian-{other}", variant=variant)
