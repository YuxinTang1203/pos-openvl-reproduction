"""Independent pinhole geometry tests; no simulator, model, or live task state."""

from dataclasses import replace

import numpy as np
import pytest

from pyrualean.openvl_geometry import (
    LineObservation,
    axis_from_lines,
    calibrated_view,
    project_to_plane,
)


class Camera:
    """Independent CV +z camera; its API supplies world rays to the estimator."""

    shape = (800, 1000)

    def __init__(self, position, yaw=0.0, focal=650.0):
        self.pos = np.asarray(position, dtype=float)
        c, s = np.cos(yaw), np.sin(yaw)
        self.rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        self.focal = focal

    def project(self, xyz):
        x, y, z = self.rotation.T @ (np.asarray(xyz) - self.pos)
        return self.focal * y / z + 400, self.focal * x / z + 500

    def rays(self, row, col):
        return self.rotation @ np.array([
            (col - 500) / self.focal, (row - 400) / self.focal, 1.0,
        ])


def cameras():
    return [
        Camera([-.25, 0, 0], .04, 650),
        Camera([.25, .05, 0], -.04, 720),
        Camera([0, -.3, -.1], .02, 600),
    ]


def observations(direction=None, views=None):
    point = np.array([.1, -.08, 1.3])
    direction = np.array([.3, .8, -.2]) if direction is None else np.asarray(direction)
    views = cameras() if views is None else views
    result = []
    for i, view in enumerate(views):
        # Different physical endpoints across views must still recover the same axis.
        endpoints = [view.project(point + t * direction) for t in (-.16 + .02*i, .18 + .03*i)]
        result.append(LineObservation(view, f"camera{i}", "straight_item", "snapshot17",
                                      tuple(endpoints)))
    return result


@pytest.mark.parametrize("direction", [[.3, .8, -.2], [.1, -.3, .4], [-.6, .2, .1]])
@pytest.mark.parametrize("view_count", [2, 3])
def test_exact_direction_from_noncorresponding_visible_portions(direction, view_count):
    result = axis_from_lines(observations(direction, cameras()[:view_count]))
    assert result["status"] == "ok"
    unit = np.asarray(direction) / np.linalg.norm(direction)
    assert abs(np.array(result["axis_unsigned"]) @ unit) == pytest.approx(1)
    assert 0 < result["angle_sigma_deg"] < 15
    assert not {"xyz_m", "length_m", "endpoint", "depth", "holding"} & result.keys()


def test_real_view_coordinate_convention_against_independent_projection():
    from pyrualean.hosts.robodojo_sight import View

    point, direction = np.array([.12, .08, -1.4]), np.array([.2, .3, .1])
    obs = []
    for i, origin in enumerate([[-.25, 0, 0], [.25, .05, 0]]):
        pose = np.eye(4)
        pose[:3, 3] = origin
        intrinsic = np.array([[600, 0, 320], [0, 610, 240], [0, 0, 1]])
        view = View(intrinsic, pose)
        endpoints = []
        for t in (-.2 + .03*i, .2 + .02*i):
            x, y, z = point + t * direction - origin
            endpoints.append((240 + 610*y/z, 320 - 600*x/z))
        obs.append(LineObservation(view, str(i), "line", "scene", tuple(endpoints)))
    result = axis_from_lines(obs)
    assert result["status"] == "ok"
    assert abs(np.asarray(result["axis_unsigned"]) @ direction/np.linalg.norm(direction)) == pytest.approx(1)


def test_local_pixel_covariance_agrees_with_independent_monte_carlo():
    rng = np.random.default_rng(19)
    obs = observations()
    nominal = axis_from_lines(obs)
    direction = np.asarray(nominal["axis_unsigned"])
    angular_errors = []
    for _ in range(120):
        noisy = [replace(o, endpoints_rc=np.asarray(o.endpoints_rc) + rng.normal(0, 1, (2, 2)))
                 for o in obs]
        measured = axis_from_lines(noisy, max_plane_residual_deg=10)
        assert measured["status"] == "ok"
        cosine = np.clip(abs(np.asarray(measured["axis_unsigned"]) @ direction), 0, 1)
        angular_errors.append(np.arccos(cosine))
    rms = np.degrees(np.sqrt(np.mean(np.square(angular_errors))))
    assert .6 < rms / nominal["angle_sigma_deg"] < 1.5


def test_nearly_parallel_interpretation_planes_rejected():
    views = [Camera([-.25, 0, 0]), Camera([.25, 0, 0])]
    # Both image lines lie in the same plane through the camera baseline.
    obs = []
    for i, view in enumerate(views):
        endpoints = [view.project([x, 0, 1.3]) for x in (-.2, .2)]
        obs.append(LineObservation(view, str(i), "line", "scene", tuple(endpoints)))
    result = axis_from_lines(obs)
    assert result["reason"] == "degenerate_interpretation_planes"
    assert result["axis_unsigned"] is None


def test_short_and_uncertain_lines_reject_instead_of_claiming_an_axis():
    obs = observations()
    short = replace(obs[0], endpoints_rc=((400, 500), (401, 500)))
    assert axis_from_lines([short, obs[1]])["reason"] == "image_line_too_short"
    uncertain = [replace(o, pixel_sigma_px=100) for o in obs]
    result = axis_from_lines(uncertain)
    assert result["reason"] == "line_axis_too_uncertain"
    assert result["axis_unsigned"] is None


def test_three_inconsistent_lines_rejected():
    obs = observations()
    obs[2] = replace(obs[2], endpoints_rc=((300, 300), (550, 500)))
    assert axis_from_lines(obs)["reason"] == "inconsistent_lines_possible_mismatch"


def test_wrong_two_line_association_can_be_consistent_and_is_not_certified():
    obs = observations()[:2]
    other = observations(direction=[-.2, .6, .5])
    result = axis_from_lines([obs[0], other[1]])
    assert result["status"] == "ok"
    assert "Two mismatched image lines" in result["limitations"][-1]


@pytest.mark.parametrize("change", [
    {"frame_id": "old"}, {"object_id": "another"}, {"camera_id": "camera0"},
    {"pixel_sigma_px": 0}, {"pixel_sigma_px": float("nan")},
    {"endpoints_rc": ((-1, 10), (20, 10))}, {"endpoints_rc": ((1, 2, 3), (4, 5))},
])
def test_invalid_or_unrelated_evidence_is_unknown(change):
    obs = observations()
    result = axis_from_lines([obs[0], replace(obs[1], **change)])
    assert result["status"] == "unknown" and result["axis_unsigned"] is None


def test_missing_views_and_invalid_threshold():
    assert axis_from_lines(observations()[:1])["reason"] == "at_least_two_lines_required"
    with pytest.raises(ValueError, match="finite"):
        axis_from_lines(observations(), max_angle_sigma_deg=float("nan"))


def test_invalid_calibration_is_unknown():
    obs = observations()
    obs[0].view.rays = lambda row, col: np.array([float("nan"), 0, 0])
    assert axis_from_lines(obs)["reason"] == "invalid_line_calibration"


def metadata():
    return {"shape": [480, 640, 3],
            "intrinsic_matrix": [[600, 0, 320], [0, 610, 240], [0, 0, 1]],
            "extrinsic_matrix": np.eye(4).tolist()}


def test_plane_projection_off_centre_checks_both_signs_and_forward_depth():
    view = calibrated_view(metadata())
    # Independent OpenGL pinhole projection; centre-pixel tests would miss signs.
    x, y, z = .15, .08, -1.4
    row, col = 240 + 610*y/z, 320 - 600*x/z
    result = project_to_plane(view, row, col, z)
    assert result["status"] == "ok"
    np.testing.assert_allclose(result["xyz_m"], [x, y, z], atol=1e-12)
    assert result["forward_depth_m"] == pytest.approx(1.4)
    assert result["plane_sigma_m"] is None
    assert "plane uncertainty unknown" in result["uncertainty_scope"]
    assert result["std_xyz_m"][0] == pytest.approx(1.4/600)
    assert result["std_xyz_m"][1] == pytest.approx(1.4/610)
    assert result["std_xyz_m"][2] == 0  # Conditional on supplied plane, NOT known surface height.


def test_plane_noise_propagates_separately_from_pixel_noise():
    view = calibrated_view(metadata())
    result = project_to_plane(view, 220, 380, -1.4, pixel_sigma_px=0, plane_sigma_m=.01)
    ray = np.array([.1, 20/610, -1])
    expected = np.outer(ray/ray[2], ray/ray[2]) * .01**2
    np.testing.assert_allclose(result["covariance_m2"], expected, atol=1e-12)
    assert result["std_xyz_m"][2] == pytest.approx(.01)


def test_oblique_camera_projection_with_translation():
    info = metadata()
    angle = .6
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    origin = np.array([.2, -.1, 1.1])
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, origin
    info["extrinsic_matrix"] = pose.tolist()
    local = np.array([.1, -.04, -.7])
    expected = rotation @ local + origin
    result = project_to_plane(calibrated_view(info),
                              240 + 610*local[1]/local[2],
                              320 - 600*local[0]/local[2], expected[2])
    np.testing.assert_allclose(result["xyz_m"], expected, atol=1e-12)


def test_plane_behind_or_parallel_is_unknown():
    view = calibrated_view(metadata())
    assert project_to_plane(view, 240, 320, .5)["reason"] == "intersection_not_forward"
    assert project_to_plane(view, 240, 320, 0)["status"] == "unknown"
    info = metadata()
    info["extrinsic_matrix"] = [[1, 0, 0, 0], [0, 0, -1, 0],
                                [0, 1, 0, 1], [0, 0, 0, 1]]
    result = project_to_plane(calibrated_view(info), 240, 320, .5)
    assert result["reason"] == "ray_parallel_or_nearly_parallel_to_plane"


@pytest.mark.parametrize("key,value", [
    ("intrinsic_matrix", [[0, 0, 320], [0, 610, 240], [0, 0, 1]]),
    ("intrinsic_matrix", [[600, 1, 320], [0, 610, 240], [0, 0, 1]]),
    ("intrinsic_matrix", [[600, 0, float("nan")], [0, 610, 240], [0, 0, 1]]),
    ("extrinsic_matrix", np.diag([2, 1, 1, 1]).tolist()),
    ("shape", [480.5, 640, 3]),
])
def test_invalid_pinhole_calibration_rejected(key, value):
    with pytest.raises(ValueError):
        calibrated_view({**metadata(), key: value})


@pytest.mark.parametrize("row,col,z,sigma", [
    (-1, 10, -.5, 1), (10, 640, -.5, 1), (10, 10, float("nan"), 1), (10, 10, -.5, -1),
])
def test_invalid_projection_inputs_rejected(row, col, z, sigma):
    with pytest.raises(ValueError):
        project_to_plane(calibrated_view(metadata()), row, col, z, pixel_sigma_px=sigma)
