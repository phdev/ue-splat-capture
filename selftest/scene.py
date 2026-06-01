"""Canonical self-test scene (UE convention: cm, Z-up) + UE-native projection.

Everything here is deterministic. Coordinates are in centimeters. The scene is
intentionally diffuse and shadow-free so a Gaussian splat can reconstruct it
well (the recon gate is about the *pipeline* being correct, not about stress-
testing the renderer), while still carrying enough texture (checker ground,
distinct colored objects, bright fiducials) to constrain geometry.
"""
from __future__ import annotations

import numpy as np

from splatkit import convert, geom

TARGET_CENTER_CM = np.array([0.0, 0.0, 35.0])
ORBIT_RADIUS_CM = 360.0
RING_ELEVATIONS_DEG = (22.0, 48.0)
AZIMUTH_STEP_DEG = 15.0          # 24 azimuths per ring -> 48 cameras
HELDOUT_EVERY = 4                # hold out every 4th camera (interleaved)
IMG_W = IMG_H = 128
HFOV_DEG = 55.0


def intrinsics() -> dict:
    return convert.intrinsics_from_hfov(IMG_W, IMG_H, HFOV_DEG)


# --------------------------------------------------------------------------- #
# Scene geometry
# --------------------------------------------------------------------------- #
def _mat(albedo, emissive=False, checker=None):
    return {"albedo": list(map(float, albedo)), "emissive": bool(emissive),
            "checker": checker}


FIDUCIALS = [
    {"id": "F0", "loc_cm": [0.0, 0.0, 8.0], "radius": 7.0, "color": [1.0, 1.0, 1.0]},
    {"id": "F1", "loc_cm": [120.0, 120.0, 15.0], "radius": 7.0, "color": [1.0, 0.0, 1.0]},
    {"id": "F2", "loc_cm": [-120.0, 120.0, 90.0], "radius": 7.0, "color": [0.0, 1.0, 1.0]},
    {"id": "F3", "loc_cm": [120.0, -120.0, 60.0], "radius": 7.0, "color": [1.0, 0.5, 0.0]},
    {"id": "F4", "loc_cm": [-120.0, -120.0, 30.0], "radius": 7.0, "color": [0.2, 1.0, 0.2]},
    {"id": "F5", "loc_cm": [0.0, -140.0, 110.0], "radius": 7.0, "color": [1.0, 1.0, 0.0]},
    {"id": "F6", "loc_cm": [140.0, 0.0, 100.0], "radius": 7.0, "color": [0.5, 0.5, 1.0]},
    {"id": "F7", "loc_cm": [-30.0, 60.0, 150.0], "radius": 7.0, "color": [1.0, 0.3, 0.3]},
]

AABB_MIN_CM = [-150.0, -150.0, 0.0]
AABB_MAX_CM = [150.0, 150.0, 160.0]


def build_scene() -> dict:
    prims = []
    # ground plane (checkerboard albedo) at z=0, normal +Z
    prims.append({"type": "plane", "point": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0],
                  "mat": _mat([0.8, 0.8, 0.82],
                              checker={"color2": [0.22, 0.25, 0.30], "scale_cm": 40.0})})
    # solid spheres
    prims.append({"type": "sphere", "center": [70.0, 40.0, 45.0], "radius": 45.0,
                  "mat": _mat([0.85, 0.25, 0.20])})
    prims.append({"type": "sphere", "center": [-80.0, -30.0, 35.0], "radius": 35.0,
                  "mat": _mat([0.20, 0.45, 0.85])})
    prims.append({"type": "sphere", "center": [10.0, 90.0, 28.0], "radius": 28.0,
                  "mat": _mat([0.95, 0.80, 0.20])})
    # boxes
    prims.append({"type": "box", "min": [-130.0, 40.0, 0.0], "max": [-70.0, 100.0, 60.0],
                  "mat": _mat([0.30, 0.70, 0.40])})
    prims.append({"type": "box", "min": [60.0, -110.0, 0.0], "max": [120.0, -50.0, 40.0],
                  "mat": _mat([0.70, 0.50, 0.85])})
    # fiducials -> emissive spheres
    for f in FIDUCIALS:
        prims.append({"type": "sphere", "center": list(f["loc_cm"]), "radius": f["radius"],
                      "mat": _mat(f["color"], emissive=True), "fiducial_id": f["id"]})

    return {
        "intrinsics": intrinsics(),
        "background": [0.12, 0.14, 0.18],
        "ambient": 0.28,
        "lights": [
            {"dir": list(geom.normalize([0.4, 0.3, 1.0])), "intensity": 0.62},
            {"dir": list(geom.normalize([-0.5, -0.2, 0.6])), "intensity": 0.22},
        ],
        "primitives": prims,
        "fiducials": FIDUCIALS,
        "aabb_min_cm": AABB_MIN_CM,
        "aabb_max_cm": AABB_MAX_CM,
        "target_center_cm": TARGET_CENTER_CM.tolist(),
        "orbit_radius_cm": ORBIT_RADIUS_CM,
    }


# --------------------------------------------------------------------------- #
# Camera rig (orbit hemisphere). Deterministic ordering.
# --------------------------------------------------------------------------- #
def cameras() -> list[dict]:
    cams = []
    idx = 0
    n_az = int(round(360.0 / AZIMUTH_STEP_DEG))
    for elev in RING_ELEVATIONS_DEG:
        for k in range(n_az):
            az = k * AZIMUTH_STEP_DEG
            ar, er = np.radians(az), np.radians(elev)
            loc = TARGET_CENTER_CM + ORBIT_RADIUS_CM * np.array([
                np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])
            split = "heldout" if (idx % HELDOUT_EVERY == 1) else "train"
            cams.append({
                "index": idx, "ring_elev_deg": float(elev), "azimuth_deg": float(az),
                "loc_cm": loc.astype(float).tolist(), "split": split,
            })
            idx += 1
    return cams


def camera_basis(loc_cm) -> np.ndarray:
    """UE basis (cols forward,right,up) for a camera looking at the scene center."""
    return geom.look_at_basis_ue(loc_cm, TARGET_CENTER_CM)


# --------------------------------------------------------------------------- #
# Independent UE-native forward projection (NOT shared with splatkit.convert)
# --------------------------------------------------------------------------- #
def project_ue_native(loc_cm, basis_ue, intr, points_cm):
    """Project UE-cm world point(s) to pixels using the UE-native camera model.

    Returns (uv (...,2), depth (...,)). +u is camera-right, +v is downward.
    """
    P = np.asarray(points_cm, float).reshape(-1, 3)
    loc = np.asarray(loc_cm, float)
    f, r, u = basis_ue[:, 0], basis_ue[:, 1], basis_ue[:, 2]
    rel = P - loc
    d = rel @ f
    a = rel @ r
    b = rel @ u
    safe = np.where(np.abs(d) < 1e-9, 1e-9, d)
    px = intr["cx"] + intr["fl_x"] * (a / safe)
    py = intr["cy"] - intr["fl_y"] * (b / safe)
    uv = np.stack([px, py], axis=1)
    return uv.reshape(np.asarray(points_cm, float).shape[:-1] + (2,)), d
