"""Native capability integration, with real local execution and scripted services."""

import base64
import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

pytest.importorskip("openvlharness")

from pyrualean.arms import ArmToolkit
from pyrualean.migrations import configure_migration, validate_migration
from pyrualean.openai_runtime import run_openai_agent
from pyrualean.openvl_adapter import adapt_prompt
from tests.test_openai_runtime import ScriptedTransport, _reply

ROOT = Path(__file__).resolve().parents[1] / "integrations/migrations/openvl"


def settings():
    config = yaml.safe_load((ROOT / "capabilities_all.yml").read_text())
    config["services"]["client"]["base_url"] = "http://localhost:8101"
    for value in config["services"]["tools"].values():
        if "server_url" in value:
            value["server_url"] = "http://localhost:8102"
    for value in config["services"]["models"].values():
        value.update(base="http://localhost:8103/v1", key="test-secret-not-for-provenance")
    return config


def setup(robo, tmp_path, config=None, feedback_mode="full"):
    from unittest.mock import patch

    from openvlharness.agent.agent import MainAgent

    stream = io.BytesIO()
    Image.new("RGB", (80, 60), "red").save(stream, format="PNG")
    kit = ArmToolkit(
        robo,
        arm="cells",
        workspace=tmp_path / "workspace",
        images="on-demand",
        feedback="pure",
        max_turns=100,
    )
    with patch.object(MainAgent, "generate", side_effect=AssertionError("wrong main loop")):
        record = configure_migration(
            kit,
            {"kind": "openvl", "config": config or settings(), "feedback_mode": feedback_mode},
            task="Read label and move",
            initial_images={"head.png": stream.getvalue()},
            model="gpt-6-astra",
        )
    return kit, record


class Service:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def execute(self, name, args, *, tool_call_id):
        self.calls.append(copy.deepcopy(args))
        return {
            "ok": self.error is None,
            "result": copy.deepcopy(self.result),
            "error": self.error,
            "meta": {},
            "tool_call_id": tool_call_id,
            "tool_name": name,
        }


def inject(kit, name, value):
    service = Service(value)
    kit.evidence_adapter.tools[name].client = service
    return service


def test_all_nine_registered_without_changing_executor(robo, tmp_path):
    kit, record = setup(robo, tmp_path)
    assert len(kit.evidence_adapter.native_names) == 9
    assert len(kit.get_tools_spec()) == 13  # nine native, two robot evidence, python/finish
    assert not record["execution_recovery"] and not record["stage_verification"]
    assert robo._action_feedback is None and kit.stage_verifier is None
    assert kit.native_vista is None and kit.harnesspro_hook is None
    assert "test-secret" not in (tmp_path / "openvl_evidence/provenance.json").read_text()
    assert "initial:head.png" in adapt_prompt("through ONE tool,", kit)


def test_crop_ocr_python_actions_roundtrip(robo, tmp_path, monkeypatch):
    from openvlharness.agent.agent import MainAgent

    monkeypatch.setattr(MainAgent, "generate", lambda *a, **k: pytest.fail("wrong main loop"))
    kit, _ = setup(robo, tmp_path)
    ocr = inject(kit, "OCR_Tool", {"text": "slot 3", "model_id": "fixture"})
    cropped = kit.execute_tool("Zoom_In_Tool", {"image_id": "head.png", "coords": [0, 0, 400, 400]})
    assert "error" not in cropped.result, cropped.result
    assert cropped.images
    ref = cropped.result["data"]["image"]
    assert ref and "openvl_environment_update" in cropped.result
    provider = ScriptedTransport(
        [
            _reply([("ocr", "OCR_Tool", {"image_id": ref})]),
            _reply(
                [
                    (
                        "code",
                        "python",
                        {
                            "code": "label = openvl_data('ocr_result')['text']\n"
                            "robo.move_to([0,0,2])\nrobo.move_to([0,0,0.6])\nprint(label)"
                        },
                    )
                ]
            ),
            _reply([("again", "python", {"code": "print(label); robo.show()"})]),
            _reply([("end", "finish", {"status": "stuck", "summary": "offline test"})]),
        ]
    )
    result = run_openai_agent(
        adapt_prompt("Test", kit),
        [],
        kit,
        model="gpt-6-astra",
        base_url="http://unused/v1",
        api_key=None,
        max_turns=100,
        budget_s=60,
        out_dir=tmp_path / "episode",
        transport=provider,
    )
    assert result["status"] == "completed"
    assert kit._globals["label"] == "slot 3"
    assert "image_base64" in ocr.calls[0]
    assert ref in json.dumps(provider.requests)
    assert "openvl_environment_update" in json.dumps(provider.requests)
    manifest = json.loads((tmp_path / "openvl_evidence/native/manifest.json").read_text())
    assert ref in manifest["images"] and "ocr_result" in manifest["bindings"]


def test_depth_native_coder_and_python_share_copied_arrays(robo, tmp_path):
    kit, _ = setup(robo, tmp_path)
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    inject(
        kit,
        "Depth_Estimation_Tool",
        {
            "depth_meters_b64": base64.b64encode(array.tobytes()).decode(),
            "depth_shape": [2, 2],
            "size": [2, 2],
        },
    )
    out = kit.execute_tool("Depth_Estimation_Tool", {"image_id": "head.png"})
    assert "error" not in out.result, out.result
    ref = out.result["data"]["binding"]
    coder = inject(
        kit,
        "Python_Coding_Agent_Tool",
        {"generated_code": f"print({ref}['depth_meters'].mean())\n{ref}['depth_meters'][:] = 99"},
    )
    out = kit.execute_tool("Python_Coding_Agent_Tool", {"objective": "Mean depth"})
    assert "error" not in out.result and not out.result["data"]["execution_error"], out.result
    assert "2.5" in out.result["data"]["printed_output"]
    assert "binding_descriptors" in coder.calls[0]
    assert "depth_meters_b64" not in json.dumps(coder.calls)
    np.testing.assert_array_equal(kit._globals["openvl_data"](ref)["depth_meters"], array)
    assert list((tmp_path / "openvl_evidence/native").glob("*.npz"))


def test_search_web_and_image_reference_reuse(robo, tmp_path):
    kit, _ = setup(robo, tmp_path)
    inject(kit, "Text_Search_Tool", "Manual https://example.com/manual explains slots.")
    out = kit.execute_tool("Text_Search_Tool", {"query": "slot manual"})
    assert "error" not in out.result, out.result
    memory = kit.evidence_adapter.memory
    assert memory.state.urls.webpage_registry
    ref = next(iter(memory.state.urls.webpage_registry))
    visit = inject(kit, "Webpage_Visit_Tool", "Slot 3 is marked red.")
    out = kit.execute_tool("Webpage_Visit_Tool", {"urls": [f"[{ref}]"], "goal": "Read slot"})
    assert "error" not in out.result, out.result
    assert "https://example.com/manual" in json.dumps(visit.calls)
    url = memory.resolve_image("head.png")[0]
    search = inject(
        kit,
        "Image_Search_Tool",
        {
            "text": "Match __MULTIMODAL_IMG_REF_PLACEHOLDER__ https://example.com/match",
            "kept_image_data_urls": [url],
        },
    )
    out = kit.execute_tool("Image_Search_Tool", {"image_id": "head.png"})
    assert "error" not in out.result and out.images, out.result
    assert "image_base64" in search.calls[0]


def test_native_grounding_and_camera_arrays(robo, tmp_path):
    kit, _ = setup(robo, tmp_path)
    inject(
        kit,
        "Visual_Grounding_Tool",
        {
            "pred_boxes": [[1, 2, 20, 30]],
            "pred_scores": [0.9],
            "pred_masks": [],
            "num_masks": 1,
            "orig_img_w": 80,
            "orig_img_h": 60,
        },
    )
    out = kit.execute_tool("Visual_Grounding_Tool", {"image_id": "head.png", "query": "red block"})
    assert "error" not in out.result, out.result
    inject(
        kit,
        "Camera_Trajectory_Tool",
        {
            "intrinsics": [np.eye(3).tolist()] * 2,
            "extrinsics": [np.eye(4)[:3].tolist()] * 2,
            "num_images": 2,
            "reference_image_id": "head.png",
            "image_sizes": [[80, 60]] * 2,
            "coordinate_system": "reference camera",
            "cameras": [],
        },
    )
    out = kit.execute_tool("Camera_Trajectory_Tool", {"image_ids": ["head.png", "head.png"]})
    assert "error" not in out.result, out.result
    value = kit._globals["openvl_data"]("camera_trajectory_result")
    assert isinstance(value["intrinsics"], np.ndarray)


def test_errors_do_not_advertise_success_or_destroy_memory(robo, tmp_path):
    kit, _ = setup(robo, tmp_path)
    tool = kit.evidence_adapter.tools["OCR_Tool"]
    tool.client = Service(error={"code": "offline", "message": "not deployed"})
    out = kit.execute_tool("OCR_Tool", {"image_id": "head.png"})
    assert out.result["error"]["code"] == "offline"
    assert "openvl_environment_update" not in out.result
    out = kit.execute_tool("Zoom_In_Tool", {"image_id": "missing_id", "coords": [0, 0, 400, 400]})
    assert "error" in out.result
    assert kit.evidence_adapter.memory.resolve_image("head.png")[0]


def test_native_data_cannot_bypass_robot_freshness(robo, tmp_path):
    kit, _ = setup(robo, tmp_path)
    kit.execute_tool(
        "Robot_Perception_Tool", {"operation": "segment", "query": "bowl", "camera": "agentview"}
    )
    ref = kit.evidence_adapter.bridge.refs[-1]
    assert kit._globals["openvl_data"](ref)
    kit.execute_tool("python", {"code": "robo.move_to([0,0,0.6])"})
    with pytest.raises(ValueError, match="Historical"):
        kit._globals["openvl_data"](ref)


def test_local_profile_and_per_episode_isolation(robo, tmp_path):
    kit, _ = setup(robo, tmp_path, str(ROOT / "capabilities_local.yml"))
    assert kit.evidence_adapter.native_names == {"Zoom_In_Tool"}
    out = kit.execute_tool("Zoom_In_Tool", {"image_id": "head.png", "coords": [0, 0, 500, 500]})
    assert out.images
    kit.evidence_adapter.remaining = lambda: -1
    stopped = kit.execute_tool("Zoom_In_Tool", {"image_id": "head.png", "coords": [0, 0, 500, 500]})
    assert "episode ended" in stopped.result["error"]
    from pyrualean import LiberoRobot
    from tests.conftest import FakeLiberoBackend

    other, _ = setup(
        LiberoRobot(FakeLiberoBackend()), tmp_path / "second", str(ROOT / "capabilities_local.yml")
    )
    assert len(other.evidence_adapter.memory.state.images.registry) == 1
    assert not other.evidence_adapter.memory.state.bindings.bindings


def test_mismatched_coding_and_extra_interventions_rejected(robo, tmp_path):
    cfg = settings()
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace")
    with pytest.raises(ValueError, match="must match"):
        configure_migration(
            kit,
            {"kind": "openvl", "config": cfg},
            task="test",
            initial_images={},
            model="different-model",
        )
    for extra in ("execution_recovery", "stage_verification", "temporal_feedback"):
        with pytest.raises(ValueError):
            validate_migration({"kind": "openvl", "config": cfg, extra: True})
