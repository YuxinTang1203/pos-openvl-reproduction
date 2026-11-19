"""Exercise the shipped standalone pair, including failure continuation."""

from pathlib import Path

import pytest
import yaml

from pyrualean.arms import ArmToolkit
from pyrualean.migrations import configure_migration

PAIR = Path(__file__).resolve().parents[1] / "integrations/migrations/openvl_pair"


def config(kind):
    return yaml.safe_load((PAIR / f"{kind}.yml").read_text())


def test_pair_has_identical_execution_settings():
    baseline, evidence = config("baseline"), config("openvl_evidence")
    assert baseline.pop("result_dir") != evidence.pop("result_dir")
    assert baseline.pop("migration") == {"kind": "baseline"}
    assert evidence.pop("migration") == {
        "kind": "openvl_evidence", "validate_geometry": True, "max_failure_images": 2,
    }
    assert baseline == evidence
    assert baseline["runtime"] == "openai"
    assert "openvl_evidence" not in baseline


@pytest.mark.parametrize("kind", ["baseline", "openvl_evidence"])
def test_pair_does_not_install_failure_guard(robo, backend, tmp_path, kind):
    cfg = config(kind)
    kit = ArmToolkit(
        robo, arm="cells", workspace=tmp_path / "workspace",
        images=cfg["images"], feedback=cfg["feedback"], max_turns=cfg["max_turns"],
    )
    record = configure_migration(kit, cfg["migration"], task="Move bowl", initial_images={})
    out = kit.execute_tool("python", {"code":
        "first = robo.move_to([0,0,2])\n"
        "second = robo.move_to([0,0,0.6])\n"
        "persistent_marker = 42\n"
        "print(first.reached, second.reached)"})
    assert "exception" not in out.result
    assert "False True" in out.result["stdout"]
    assert len([name for name, _ in backend.calls if name == "move_to"]) == 2
    later = kit.execute_tool("python", {"code": "print(persistent_marker)"})
    assert "42" in later.result["stdout"]
    assert "recovery_required" not in out.result
    assert robo._action_feedback is None and kit.stage_verifier is None
    assert kit.native_vista is None and kit.harnesspro_hook is None
    assert not {"checked", "action_scope", "stage_begin", "evidence_target"} & kit._globals.keys()
    assert record["execution_recovery"] is False
    assert record["stage_verification"] is False
    if kind == "openvl_evidence":
        assert out.result["evidence_update"]["recovery_evidence"]
        native = record["native"]
        assert native["options"]["temporal_feedback"] is False
        assert set(native["native_components"]) == {"MemoryStore", "BaseCapability"}
    else:
        assert kit.evidence_adapter is None
        assert "evidence_update" not in out.result
