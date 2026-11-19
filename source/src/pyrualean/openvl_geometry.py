# Copyright 2026 PyRUA-Lean Contributors.
# Licensed under the Apache License, Version 2.0.

"""Pure calibrated-ray geometry for explicitly selected visual evidence.

This module consumes calibrated image lines, not robot state, masks, task names,
depth, or object poses. A caller must separately validate that its evidence is
current, synchronized, and depicts the same straight physical centreline. The
endpoints of each image line can delimit different visible portions; they do
not need to be the same physical endpoints across cameras.

Views use the ``robodojo_sight.View`` interface: ``shape`` and ``rays(row,col)``.
That View's extrinsic is camera-to-world, optical axis -z and +y up; pixel order
is row,col in the original image. Axis fitting makes no position/depth claim.
Plane projection constructs a point conditional on a caller-supplied height;
it does not identify the surface visible at that pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

LIMITATIONS = (
    "Assumes all observations depict the same straight physical centreline.",
    "Geometric consistency does not verify object correspondence or line correctness.",
    "Pixel-noise uncertainty excludes calibration, synchronization and annotation bias.",
    "Direction is up to sign; no position, extent, endpoint or depth is measured.",
    "No claim about rigid grasp, contact, support, or task success.",
    "Two mismatched image lines can give a perfectly consistent direction.",
)


def calibrated_view(metadata: dict):
    """Validate public rectilinear pinhole metadata in the documented convention."""
    from .hosts.robodojo_sight import View

    k = np.asarray(metadata["intrinsic_matrix"], dtype=float)
    pose = np.asarray(metadata["extrinsic_matrix"], dtype=float)
    shape = np.asarray(metadata["shape"][:2], dtype=float)
    if (k.shape != (3, 3) or pose.shape != (4, 4) or shape.shape != (2,)
            or not np.isfinite(k).all() or not np.isfinite(pose).all()
            or not np.isfinite(shape).all() or (shape <= 0).any()
            or not np.array_equal(shape, np.floor(shape))):
        raise ValueError("Camera metadata must have finite K(3x3), E(4x4) and positive image shape")
    if (k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
            or abs(k[0, 1]) > 1e-10 or abs(k[1, 0]) > 1e-10
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6)):
        raise ValueError("Expected rectilinear zero-skew K and rigid camera-to-world E")
    return View(k, pose, tuple(int(v) for v in shape))


def project_to_plane(view, row: float, col: float, plane_z: float, *,
                     pixel_sigma_px: float = 1.0, plane_sigma_m: float | None = None) -> dict:
    """Intersect a forward pixel ray with an explicitly supplied world z plane.

    This is a conditional geometric construction, not a measurement of the
    surface visible at the pixel. The plane height is never inferred here.
    """
    values = np.asarray([row, col, plane_z, pixel_sigma_px], dtype=float)
    if (not np.isfinite(values).all() or pixel_sigma_px < 0
            or not 0 <= row < view.shape[0] or not 0 <= col < view.shape[1]
            or (plane_sigma_m is not None
                and (not np.isfinite(plane_sigma_m) or plane_sigma_m < 0))):
        raise ValueError("Projection requires finite in-image pixels/plane and nonnegative noise")
    ray = np.asarray(view.rays(np.asarray(row), np.asarray(col)), dtype=float)
    origin = np.asarray(view.pos, dtype=float)
    result = {
        "status": "unknown", "reason": None, "xyz_m": None,
        "conditional_on_plane_z_m": float(plane_z), "pixel_rc": [float(row), float(col)],
        "plane_sigma_m": plane_sigma_m, "pixel_sigma_px": float(pixel_sigma_px),
        "interpretation": "Forward ray intersection with the supplied plane; not a measured surface, grasp or contact.",
        "uncertainty_scope": (
            "Pixel noise conditional on plane_z; plane uncertainty unknown; calibration error excluded."
            if plane_sigma_m is None else
            "Independent pixel and supplied plane-height noise; calibration error excluded."
        ),
    }
    if (ray.shape != (3,) or origin.shape != (3,) or not np.isfinite(ray).all()
            or not np.isfinite(origin).all() or np.linalg.norm(ray) == 0):
        raise ValueError("Camera must provide finite world ray and origin")
    if abs(ray[2]) / np.linalg.norm(ray) < 1e-6:
        return {**result, "reason": "ray_parallel_or_nearly_parallel_to_plane"}
    parameter = (plane_z - origin[2]) / ray[2]
    if parameter <= 0:
        return {**result, "reason": "intersection_not_forward"}
    point = origin + parameter * ray
    _, _, forward_depth = view.project(point)
    if not np.isfinite(point).all() or not np.isfinite(forward_depth) or forward_depth <= 0:
        return {**result, "reason": "intersection_not_forward"}
    # Local finite-pixel sensitivity, with plane height held fixed.
    derivatives = []
    delta = 1e-3
    for dr, dc in ((delta, 0), (0, delta)):
        points = []
        for sign in (1, -1):
            d = view.rays(np.asarray(row + sign*dr), np.asarray(col + sign*dc))
            if abs(d[2]) / np.linalg.norm(d) < 1e-6:
                return {**result, "reason": "pixel_uncertainty_crosses_parallel_ray"}
            points.append(origin + (plane_z - origin[2]) / d[2] * d)
        derivatives.append((points[0] - points[1]) / (2*delta))
    jacobian = np.asarray(derivatives).T
    covariance = jacobian @ jacobian.T * pixel_sigma_px**2
    if plane_sigma_m is not None:
        derivative_z = ray / ray[2]
        covariance += np.outer(derivative_z, derivative_z) * plane_sigma_m**2
    if not np.isfinite(covariance).all():
        return {**result, "reason": "nonfinite_uncertainty"}
    return {**result, "status": "ok", "xyz_m": point.tolist(),
            "forward_depth_m": float(forward_depth),
            "ray_distance_m": float(np.linalg.norm(point - origin)),
            "covariance_m2": covariance.tolist(),
            "std_xyz_m": np.sqrt(np.maximum(np.diag(covariance), 0)).tolist()}


@dataclass(frozen=True)
class LineObservation:
    """One asserted straight centreline, with independent endpoint pixel noise.

    ``frame_id`` denotes a shared scene instant, not an individual camera image
    ID. Artifact/binding provenance belongs to the caller. ``object_id`` is an
    explicit correspondence assertion; this function cannot verify its truth.
    Silhouette edges or visible-surface midpoints need not be the projection of
    an object's physical centreline, especially under occlusion or perspective.
    """

    view: Any
    camera_id: str
    object_id: str
    frame_id: str
    endpoints_rc: tuple[tuple[float, float], tuple[float, float]]
    pixel_sigma_px: float = 1.0


def _unknown(reason: str, diagnostics: dict | None = None) -> dict:
    return {
        "status": "unknown",
        "reason": reason,
        "axis_unsigned": None,
        "angle_sigma_deg": None,
        "diagnostics": diagnostics or {},
        "limitations": list(LIMITATIONS),
    }


def _fit(views: list, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = []
    for view, endpoints in zip(views, pixels):
        rays = [view.rays(np.asarray(p[0]), np.asarray(p[1])) for p in endpoints]
        normal = np.cross(rays[0], rays[1])
        norm = np.linalg.norm(normal)
        if normal.shape != (3,) or not np.isfinite(normal).all() or norm <= 1e-14:
            raise ValueError("Image line has no finite interpretation plane")
        normals.append(normal / norm)
    normals = np.asarray(normals)
    _, _, vt = np.linalg.svd(normals, full_matrices=True)
    axis = vt[-1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return axis, normals


def axis_from_lines(
    observations: list[LineObservation],
    *,
    min_plane_angle_deg: float = 2.0,
    min_line_length_px: float = 5.0,
    max_plane_residual_deg: float = 2.0,
    max_angle_sigma_deg: float = 15.0,
) -> dict:
    """Intersect interpretation planes to estimate an unsigned world direction.

    Pixel uncertainty is propagated by finite differences through the fit,
    assuming independent Gaussian endpoint errors and exact calibration. The
    angular sigma is the local RMS angular error, not a confidence bound on
    systematic errors. Thresholds reject weak geometry; they do not prove that
    a supplied apparent centreline is an appropriate physical measurement.
    """
    obs = list(observations)
    thresholds = (
        min_plane_angle_deg, min_line_length_px,
        max_plane_residual_deg, max_angle_sigma_deg,
    )
    if not np.isfinite(thresholds).all() or min(thresholds) <= 0 or min_plane_angle_deg >= 90:
        raise ValueError("Thresholds must be finite and positive; plane angle must be below 90")
    if len(obs) < 2:
        return _unknown("at_least_two_lines_required")
    if any(not o.camera_id or not o.object_id or not o.frame_id for o in obs):
        return _unknown("explicit_camera_object_and_frame_ids_required")
    if (
        len({o.camera_id for o in obs}) != len(obs)
        or len({o.object_id for o in obs}) != 1
        or len({o.frame_id for o in obs}) != 1
    ):
        return _unknown("line_identity_camera_or_frame_mismatch")
    try:
        pixels = np.asarray([o.endpoints_rc for o in obs], dtype=float)
        sigma = np.asarray([o.pixel_sigma_px for o in obs], dtype=float)
    except (ValueError, TypeError):
        return _unknown("invalid_line_pixels_or_noise")
    if (
        pixels.shape != (len(obs), 2, 2)
        or sigma.shape != (len(obs),)
        or not np.isfinite(pixels).all()
        or not np.isfinite(sigma).all()
        or (sigma <= 0).any()
    ):
        return _unknown("invalid_line_pixels_or_noise")
    lengths = np.linalg.norm(pixels[:, 1] - pixels[:, 0], axis=1)
    if (lengths < min_line_length_px).any():
        return _unknown("image_line_too_short", {"line_lengths_px": lengths.tolist()})
    views = [o.view for o in obs]
    for view, endpoints in zip(views, pixels):
        if (
            (endpoints < 0).any()
            or (endpoints[:, 0] >= view.shape[0]).any()
            or (endpoints[:, 1] >= view.shape[1]).any()
        ):
            return _unknown("pixel_outside_original_image")
    try:
        axis, normals = _fit(views, pixels)
    except (ValueError, np.linalg.LinAlgError):
        return _unknown("invalid_line_calibration")
    pairs = [
        {
            "camera_ids": [obs[i].camera_id, obs[j].camera_id],
            "acute_plane_angle_deg": float(
                np.degrees(np.arccos(np.clip(abs(normals[i] @ normals[j]), 0, 1)))
            ),
        }
        for i, j in combinations(range(len(obs)), 2)
    ]
    residuals = np.degrees(np.arcsin(np.clip(abs(normals @ axis), 0, 1)))
    diagnostics = {
        "line_lengths_px": lengths.tolist(),
        "pixel_sigma_px": sigma.tolist(),
        "plane_pairs": pairs,
        "plane_residual_angles_deg": residuals.tolist(),
        "interpretation_plane_normals": normals.tolist(),
    }
    if max(p["acute_plane_angle_deg"] for p in pairs) < min_plane_angle_deg:
        return _unknown("degenerate_interpretation_planes", diagnostics)
    if (residuals > max_plane_residual_deg).any():
        return _unknown("inconsistent_lines_possible_mismatch", diagnostics)
    gradients = []
    delta = 1e-3
    try:
        for index in np.ndindex(pixels.shape):
            plus, minus = pixels.copy(), pixels.copy()
            plus[index] += delta
            minus[index] -= delta
            ap, _ = _fit(views, plus)
            am, _ = _fit(views, minus)
            # The direction is unsigned; eliminate arbitrary SVD sign flips.
            ap *= 1 if ap @ axis >= 0 else -1
            am *= 1 if am @ axis >= 0 else -1
            gradients.append((ap - am) / (2 * delta) * sigma[index[0]])
    except (ValueError, np.linalg.LinAlgError):
        return _unknown("invalid_pixel_uncertainty", diagnostics)
    jacobian = np.asarray(gradients).T
    covariance = jacobian @ jacobian.T
    angle_sigma = float(np.degrees(np.sqrt(max(np.trace(covariance), 0))))
    diagnostics["angle_sigma_deg"] = angle_sigma
    if not np.isfinite(covariance).all() or angle_sigma > max_angle_sigma_deg:
        return _unknown("line_axis_too_uncertain", diagnostics)
    return {
        "status": "ok",
        "reason": None,
        "frame_id": obs[0].frame_id,
        "object_id": obs[0].object_id,
        "axis_unsigned": axis.tolist(),
        "axis_covariance": covariance.tolist(),
        "angle_sigma_deg": angle_sigma,
        "diagnostics": diagnostics,
        "uncertainty_model": "local independent endpoint pixel noise; exact calibration assumed",
        "limitations": list(LIMITATIONS),
    }
