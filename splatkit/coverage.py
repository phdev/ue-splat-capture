"""T2 (part 2) -- frustum coverage of the target AABB + camera-in-geometry check.

Operates on the EXPORTED transforms.json (poses/intrinsics/AABB in world meters)
and the scene geometry (scene.json, UE cm -> converted to meters here), so it
validates the same data a downstream trainer would consume.
"""
from __future__ import annotations

import numpy as np

from . import convert
from .convert import c2w_to_w2c


def _sample_aabb(aabb_min, aabb_max, n=6) -> np.ndarray:
    aabb_min = np.asarray(aabb_min, float)
    aabb_max = np.asarray(aabb_max, float)
    axes = [np.linspace(aabb_min[i], aabb_max[i], n) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return grid


def frustum_coverage(doc: dict, n_grid: int = 6, near: float = 0.01,
                     far: float = 1e3) -> dict:
    intr = {k: doc[k] for k in ("w", "h", "fl_x", "fl_y", "cx", "cy")}
    W, H = intr["w"], intr["h"]
    pts = _sample_aabb(doc["aabb_min"], doc["aabb_max"], n=n_grid)
    covered = np.zeros(pts.shape[0], bool)
    counts = np.zeros(pts.shape[0], int)
    for fr in doc["frames"]:
        M = np.asarray(fr["transform_matrix"], float)
        R_w2c, t_w2c = c2w_to_w2c(M)
        Xc = pts @ R_w2c.T + t_w2c
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = intr["fl_x"] * (Xc[:, 0] / z) + intr["cx"]
            v = intr["fl_y"] * (Xc[:, 1] / z) + intr["cy"]
        seen = (z > near) & (z < far) & (u >= 0) & (u <= W) & (v >= 0) & (v <= H)
        covered |= seen
        counts += seen.astype(int)
    return {"fraction": float(covered.mean()),
            "min_views_per_point": int(counts.min()),
            "mean_views_per_point": float(counts.mean()),
            "n_samples": int(pts.shape[0])}


# --------------------------------------------------------------------------- #
# Camera-inside-geometry (in world meters)
# --------------------------------------------------------------------------- #
def _solids_in_meters(scene: dict):
    """Yield solid primitives converted to world meters: spheres and boxes.
    Fiducials (also spheres) are skipped -- they are markers, not occluders we
    must keep cameras out of."""
    solids = []
    for p in scene["primitives"]:
        if "fiducial_id" in p:
            continue
        if p["type"] == "sphere":
            c = convert.ue_point_to_world(p["center"])
            r = float(p["radius"]) * convert.CM_TO_M
            solids.append(("sphere", c, r))
        elif p["type"] == "box":
            corners = convert._aabb_corners(p["min"], p["max"])
            cw = convert.ue_point_to_world(corners)
            solids.append(("box", cw.min(axis=0), cw.max(axis=0)))
        # plane handled separately (ground)
    return solids


def cameras_inside_geometry(doc: dict, scene: dict, ground_z: float = 0.0,
                            margin: float = 1e-3) -> dict:
    solids = _solids_in_meters(scene)
    offenders = []
    for fr in doc["frames"]:
        C = np.asarray(fr["transform_matrix"], float)[:3, 3]
        inside = False
        reason = None
        if C[2] < ground_z - margin:
            inside, reason = True, "below ground"
        for s in solids:
            if inside:
                break
            if s[0] == "sphere":
                if np.linalg.norm(C - s[1]) < s[2] - margin:
                    inside, reason = True, "inside sphere"
            elif s[0] == "box":
                if np.all(C > s[1] + margin) and np.all(C < s[2] - margin):
                    inside, reason = True, "inside box"
        if inside:
            offenders.append({"file_path": fr.get("file_path"), "reason": reason})
    return {"n_inside": len(offenders), "offenders": offenders[:10]}
