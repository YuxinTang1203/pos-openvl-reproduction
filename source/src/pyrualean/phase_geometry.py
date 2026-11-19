"""Triangulation from explicitly matched RGB pixels and public calibration only."""
from itertools import combinations

import numpy as np


LIMITATIONS = [
    "The caller must select the same physical point in each synchronized view.",
    "Mask centroids and different silhouette corners are not corresponding points.",
    "Consistent rays do not prove correspondence, identity, support, grasp or task success.",
    "Uncertainty covers independent pixel noise, not calibration or matching bias.",
]


def triangulate_point(views, pixels_rc, *, pixel_sigma_px=1.0,
                      min_ray_angle_deg=2.0, max_reprojection_px=4.0, max_std_m=.03):
    """Estimate one point, without assuming its height or a support surface.

    Views implement the existing calibrated ``View`` ray/project interface.
    Bad geometry returns unknown with no position; malformed arguments raise.
    """
    thresholds = [pixel_sigma_px, min_ray_angle_deg, max_reprojection_px, max_std_m]
    if not np.isfinite(thresholds).all() or min(thresholds) <= 0 or min_ray_angle_deg >= 90:
        raise ValueError("Geometry thresholds must be finite and positive; ray angle < 90")
    unknown = {"status": "unknown", "reason": None, "xyz_m": None,
               "limitations": list(LIMITATIONS)}
    if not 2 <= len(views) <= 3:
        return {**unknown, "reason": "two_or_three_views_required"}
    pixels = np.asarray(pixels_rc, dtype=float)
    if pixels.shape != (len(views), 2) or not np.isfinite(pixels).all():
        raise ValueError("Provide one finite original-image row,col pair per view")
    for v, (r, c) in zip(views, pixels):
        if not (0 <= r < v.shape[0] and 0 <= c < v.shape[1]):
            raise ValueError("Pixels must be inside the original image")
    origins = np.asarray([v.pos for v in views], dtype=float)

    def fit(p):
        directions = np.asarray([v.rays(np.asarray(r), np.asarray(c))
                                 for v, (r, c) in zip(views, p)])
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        projections = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
        matrix = projections.sum(axis=0)
        rhs = np.einsum("nij,nj->i", projections, origins)
        point = np.linalg.solve(matrix, rhs)
        return point, directions, float(np.linalg.cond(matrix))

    try:
        point, directions, condition = fit(pixels)
    except np.linalg.LinAlgError:
        return {**unknown, "reason": "degenerate_rays"}
    angle = max(np.degrees(np.arcsin(np.clip(np.linalg.norm(np.cross(a, b)), 0, 1)))
                for a, b in combinations(directions, 2))
    diagnostic = {"ray_angle_deg": float(angle), "condition_number": condition}
    if not np.isfinite(point).all() or condition > 1e10 or angle < min_ray_angle_deg:
        return {**unknown, "reason": "insufficient_parallax", "diagnostics": diagnostic}
    reprojection = []
    for v, origin, direction, pixel in zip(views, origins, directions, pixels):
        row, col, depth = v.project(point)
        if depth <= 0 or np.dot(point - origin, direction) <= 0:
            return {**unknown, "reason": "point_behind_camera", "diagnostics": diagnostic}
        reprojection.append(float(np.linalg.norm(np.asarray([row, col]) - pixel)))
    diagnostic["reprojection_error_px"] = reprojection
    if not np.isfinite(reprojection).all() or max(reprojection) > max_reprojection_px:
        return {**unknown, "reason": "inconsistent_correspondence", "diagnostics": diagnostic}
    # Local sensitivity to each input coordinate; no hidden geometry is consulted.
    jacobian = []
    delta = 1e-3
    for i in range(len(views)):
        for axis in (0, 1):
            plus, minus = pixels.copy(), pixels.copy()
            plus[i, axis] += delta
            minus[i, axis] -= delta
            jacobian.append((fit(plus)[0] - fit(minus)[0]) / (2 * delta))
    jacobian = np.asarray(jacobian).T
    covariance = jacobian @ jacobian.T * pixel_sigma_px**2
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    diagnostic["std_xyz_m"] = std.tolist()
    if not np.isfinite(std).all() or max(std) > max_std_m:
        return {**unknown, "reason": "position_too_uncertain", "diagnostics": diagnostic}
    return {**unknown, "status": "ok", "xyz_m": point.tolist(),
            "std_xyz_m": std.tolist(), "covariance_m2": covariance.tolist(),
            "pixel_sigma_px": float(pixel_sigma_px), "diagnostics": diagnostic}
