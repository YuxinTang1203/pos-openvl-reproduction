"""Public-input boundaries, independent geometry, and real runtime integration."""
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.openvl_geometry import calibrated_view
from pyrualean.phase_geometry import triangulate_point
from pyrualean.phase_tools import PROMPT, ObservationPhases, install
from tests.test_openai_runtime import ScriptedTransport, _reply


def metadata(x):
    pose = np.eye(4)
    pose[0, 3] = x
    return {"shape": [480, 640, 3], "intrinsic_matrix": [[600, 0, 320], [0, 610, 240], [0, 0, 1]],
            "extrinsic_matrix": pose.tolist()}


def pixels(point, x):
    # Independent projection: public cameras look down -z, with +y up.
    px, py, pz = np.asarray(point) - [x, 0, 0]
    return [240 + 610 * py / pz, 320 - 600 * px / pz]


@pytest.mark.parametrize("point", [[.1, .05, -1.3], [0, -.1, -.7], [-.1, .08, -2]])
def test_geometry_recovers_points_without_a_table_or_depth(point):
    views = [calibrated_view(metadata(x)) for x in (-.2, .2)]
    result = triangulate_point(views, [pixels(point, x) for x in (-.2, .2)])
    assert result["status"] == "ok"
    np.testing.assert_allclose(result["xyz_m"], point, atol=1e-10)
    assert all(0 < x < .03 for x in result["std_xyz_m"])


def test_degenerate_and_inconsistent_rays_produce_no_target():
    a, b = [calibrated_view(metadata(x)) for x in (-.2, .2)]
    same = triangulate_point([a, a], [[240, 320], [240, 320]])
    assert same["status"] == "unknown" and same["xyz_m"] is None
    inconsistent = triangulate_point([a, b], [[200, 440], [280, 200]])
    assert inconsistent["reason"] == "inconsistent_correspondence"
    assert inconsistent["xyz_m"] is None
    assert triangulate_point([a], [[240, 320]])["xyz_m"] is None


def test_pixel_uncertainty_matches_monte_carlo():
    views = [calibrated_view(metadata(x)) for x in (-.2, .2)]
    px = np.array([pixels([.1, .05, -1.3], x) for x in (-.2, .2)])
    nominal = triangulate_point(views, px)
    rng = np.random.default_rng(12)
    points = [triangulate_point(views, px + rng.normal(0, .5, (2, 2)), pixel_sigma_px=.5)["xyz_m"]
              for _ in range(80)]
    observed = np.std(points, axis=0)
    predicted = np.array(nominal["std_xyz_m"]) / 2
    assert np.all((observed / predicted > .6) & (observed / predicted < 1.5))


@pytest.mark.parametrize("px", [[[float('nan'), 3], [1, 2]], [[-1, 3], [1, 2]], [[1], [2]]])
def test_bad_pixel_arguments_rejected(px):
    with pytest.raises(ValueError):
        triangulate_point([calibrated_view(metadata(x)) for x in (-.2, .2)], px)


@dataclass
class PublicState:
    step: int
    truncated: bool = False


class PublicRobot:
    CAMERAS = ("cam_head", "cam_right_wrist")

    def __init__(self):
        self.step = 0
        self._ledger = SimpleNamespace(records=[])
        self.calls = []
        self.reached = True
        self.data = np.zeros((480, 640, 3), dtype=np.uint8)

    def __getattr__(self, key):
        raise AssertionError(f"Unexpected/non-public input: {key}")

    def state(self):
        return PublicState(self.step)

    def image(self, camera):
        self.calls.append(("image", camera))
        return self.data.copy()

    def camera_meta(self, camera):
        return metadata(-.2 if camera == "cam_head" else .2)

    def move_to(self, position, **kwargs):
        assert kwargs["gripper"] == 0
        self.calls.append(("move_to", position))
        self.step += 1
        self._ledger.records.append(SimpleNamespace(stateful=True))
        return {"reached": self.reached, "blocked": not self.reached, "truncated": False}

    def locate(self, query, camera):
        self.calls.append(("locate", query))
        return {"found": False, "center_xyz": None, "note": "occluded"}


@pytest.fixture
def phase(tmp_path, monkeypatch):
    import pyrualean.phase_tools as module
    monkeypatch.setattr(module, "supported", lambda robo: True)
    robo = PublicRobot()
    shown = []
    kit = SimpleNamespace(robo=robo, workspace=tmp_path / "workspace", _displayed=shown,
                          _display=shown.append, _globals={})
    return install(kit), robo, kit


def test_snapshot_and_batched_measurement_use_only_public_inputs(phase):
    helper, robo, kit = phase
    result = helper.observe_scene(cameras=list(robo.CAMERAS))
    assert len(kit._displayed) == 2
    assert result["frames"]["cam_right_wrist"]["display_label"] == "display_1"
    matches = {"visible_corner": {c: pixels([.1, .05, -1.3], x)
                                 for c, x in zip(robo.CAMERAS, (-.2, .2))}}
    measured = helper.measure_points(result["snapshot_id"], matches)
    np.testing.assert_allclose(measured["points"]["visible_corner"]["xyz_m"], [.1, .05, -1.3])
    assert all(call[0] == "image" for call in robo.calls)
    assert not robo._ledger.records
    result["frames"].clear()  # Returned values must not mutate private provenance.
    assert helper.measure_points(measured["snapshot_id"], matches)["sources"]


def test_returned_calibration_cannot_mutate_snapshot_provenance(phase):
    helper, robo, _ = phase
    observed = helper.observe_scene(cameras=list(robo.CAMERAS))
    observed["frames"]["cam_head"]["calibration"]["intrinsic_matrix"][0][0] = 999
    matches = {"corner": {c: pixels([.1, .05, -1.3], x)
                          for c, x in zip(robo.CAMERAS, (-.2, .2))}}
    measured = helper.measure_points(observed["snapshot_id"], matches)
    measured["sources"]["cam_head"]["calibration"]["intrinsic_matrix"][0][0] = 123
    again = helper.measure_points(observed["snapshot_id"], matches)
    assert again["sources"]["cam_head"]["calibration"]["intrinsic_matrix"][0][0] == 600
    np.testing.assert_allclose(again["points"]["corner"]["xyz_m"], [.1, .05, -1.3])


@pytest.mark.parametrize("change", ["move", "pixels", "calibration"])
def test_stale_observation_never_becomes_a_current_target(phase, monkeypatch, change):
    helper, robo, _ = phase
    result = helper.observe_scene(cameras=list(robo.CAMERAS))
    if change == "move":
        robo.move_to([0, 0, 0], gripper=0)
    elif change == "pixels":
        robo.data[1, 1, 0] = 1
    else:
        monkeypatch.setattr(helper, "camera_meta", lambda c: metadata(.7))
    with pytest.raises(ValueError, match="Stale|changed"):
        helper.measure_points(result["snapshot_id"], {"point": {c: [240, 320] for c in robo.CAMERAS}})


def test_capture_race_rejected_before_delivering_images(phase, monkeypatch):
    helper, robo, kit = phase
    original = helper.camera_meta
    def changes(camera):
        robo.step += 1
        return original(camera)
    monkeypatch.setattr(helper, "camera_meta", changes)
    with pytest.raises(ValueError, match="Scene changed"):
        helper.observe_scene()
    assert not kit._displayed and not helper.snapshots


@pytest.mark.parametrize("reached", [True, False])
def test_move_once_then_observe_never_retries_or_grips(phase, reached):
    helper, robo, _ = phase
    robo.reached = reached
    result = helper.move_and_observe([0, 0, 1], locate_query="block")
    assert [c[0] for c in robo.calls] == (["move_to", "image", "locate"] if reached else ["move_to", "image"])
    assert result["status"] == ("observed" if reached else "motion_incomplete")
    assert not result.get("success")
    if reached:
        assert result["perception"]["found"] is False


def test_prompt_nested_snapshot_example_executes_after_one_motion(phase):
    import re

    helper, robo, kit = phase
    # Execute the model-visible example itself, with coordinates supplied by
    # the caller. A later measurement uses the same nested observation ID.
    example = re.search(
        r"  Use (r = move_and_observe\(.*?\);)\n  (print\(r\); snapshot_id = .*?)\.\n",
        PROMPT,
    )
    assert example is not None
    namespace = {**kit._globals, "x": .1, "y": .2, "z": .9}
    exec(example[1] + "\n" + example[2], namespace)
    result = namespace["r"]
    assert namespace["snapshot_id"] == result["observation"]["snapshot_id"]
    assert "snapshot_id" not in result and "images" not in result
    assert sum(c[0] == "move_to" for c in robo.calls) == 1
    matches = {"corner": {c: pixels([.1, .05, -1.3], offset)
                          for c, offset in zip(robo.CAMERAS, (-.2, .2))}}
    measured = helper.measure_points(namespace["snapshot_id"], matches)
    assert measured["points"]["corner"]["status"] == "ok"


def test_bad_observation_options_do_not_move_and_original_errors_propagate(phase, monkeypatch):
    helper, robo, _ = phase
    with pytest.raises(ValueError):
        helper.move_and_observe([0, 0, 1], cameras=["hidden_depth"])
    assert not robo.calls
    def failure(*args, **kwargs):
        raise RuntimeError("existing recovery guard")
    monkeypatch.setattr(robo, "move_to", failure)
    with pytest.raises(RuntimeError, match="existing recovery guard"):
        helper.move_and_observe([0, 0, 1])
    assert not robo.calls


def test_actual_rgb_toolkit_delivers_original_images_and_audits_option(tmp_path):
    from tests.test_openvl_locate_candidate import colour
    from pyrualean.hosts.robodojo_native import NativeToolkit, NativeBackend
    from pyrualean.robodojo_xpl_rgb import RobodojoOfficialColourRobot
    from pyrualean.openai_runtime import run_openai_agent
    scene = colour.Scene([colour.CUBE])
    toolkit = NativeToolkit(scene, colour.Segmenter(scene), tmp_path / "native", colour_only=True)
    robo = RobodojoOfficialColourRobot(NativeBackend(toolkit), primitives="no-vla")
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace", images="on-demand", feedback="pure")
    transport = ScriptedTransport([
        _reply([("one", "python", {"code": "print(observe_scene(cameras=['cam_head','cam_right_wrist']))"})]),
        _reply(text="stop"),
    ])
    result = run_openai_agent("TASK", [], kit, model="test", base_url="http://test/v1", api_key=None,
                             max_turns=2, budget_s=15, out_dir=tmp_path / "out", nudges=0,
                             transport=transport, phase_tools=True)
    assert result["options"]["phase_tools"] is True and result["requests"] == 2
    assert "phase_tools" not in transport.requests[0]["body"]
    user_content = transport.requests[1]["body"]["messages"][-1]["content"]
    assert sum(p["type"] == "image_url" for p in user_content) == 2
    assert not scene.step_count
    event = json.loads((tmp_path / "observation_phases/events.jsonl").read_text().splitlines()[0])
    assert event["type"] == "observation" and len(event["frames"]) == 2


def test_non_rgb_backend_cannot_enable_tools(robo, tmp_path):
    kit = ArmToolkit(robo, arm="cells", workspace=tmp_path / "workspace")
    with pytest.raises(ValueError, match="RGB-only"):
        ObservationPhases(kit)
