"""THE coordinate-convention math: Unreal -> Nerfstudio/instant-ngp transforms.json.

Source (Unreal Engine):
    left-handed, Z-up, X-forward, Y-right, units = centimeters.
    A camera's orientation is given by a 3x3 basis whose COLUMNS are the
    world-space directions of its local axes (+X forward, +Y right, +Z up).

Target (transforms.json):
    right-handed, Z-up, units = meters.
    Camera local axes follow the OpenCV convention: +X right, +Y down,
    +Z forward (into the scene). ``transform_matrix`` is camera-to-world.

The single source of confusion in UE<->everything-else pipelines is handedness.
We change handedness by negating exactly one world axis (Y), encoded by the
fixed diagonal map D = diag(1, -1, 1). The two facts that make this provably
correct (see tests/test_convert.py):

  1. The resulting camera-to-world rotation is a PROPER rotation (det = +1).
     A common silent bug -- treating UE's left-handed data as if it were already
     right-handed -- yields det = -1 (an improper/mirrored "rotation"). We assert
     det≈+1, so that bug FAILS rather than silently mirroring.

  2. Reprojection is invariant to a *global* mirror (flipping the whole world AND
     all cameras), so the determinant check above is what guards global
     handedness, while the reprojection gate (T1) guards *asymmetric* flips
     (flipping points but not the camera basis, or vice-versa), which produce a
     genuine geometric mismatch.

All functions here are pure numpy and have no Unreal dependency.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from . import geom

# Fixed handedness-changing world map: UE (LH, cm) -> target (RH, m).
# Negate Y (one axis) to flip handedness; scale cm -> m.
CM_TO_M = 0.01
WORLD_FLIP = np.diag([1.0, -1.0, 1.0])          # directional part (handedness flip)
WORLD_MAP3 = WORLD_FLIP * CM_TO_M               # full linear map incl. scale

CAMERA_MODEL = "OPENCV"
WORLD_UP = "+Z"


# --------------------------------------------------------------------------- #
# Intrinsics
# --------------------------------------------------------------------------- #
def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> dict:
    """Pinhole intrinsics from a horizontal field of view (square pixels).

    UE's CineCamera/`FOVAngle` is a horizontal FOV. We derive fx from it and use
    square pixels (fy == fx) with a centered principal point. The self-test
    renderer back-projects pixels with these very intrinsics, so renderer and
    geometry never disagree about FOV/aspect.
    """
    if not (width > 0 and height > 0):
        raise ValueError("width/height must be positive")
    if not (0.0 < hfov_deg < 180.0):
        raise ValueError("hfov_deg must be in (0,180)")
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return {"w": int(width), "h": int(height),
            "fl_x": float(fx), "fl_y": float(fy),
            "cx": float(cx), "cy": float(cy)}


def intrinsics_matrix(intr: dict) -> np.ndarray:
    K = np.eye(3)
    K[0, 0] = intr["fl_x"]
    K[1, 1] = intr["fl_y"]
    K[0, 2] = intr["cx"]
    K[1, 2] = intr["cy"]
    return K


# --------------------------------------------------------------------------- #
# World points
# --------------------------------------------------------------------------- #
def ue_point_to_world(p_cm) -> np.ndarray:
    """UE point(s) in cm -> target world point(s) in meters (handedness flipped).

    Accepts shape (3,) or (...,3).
    """
    p = np.asarray(p_cm, dtype=np.float64)
    return p @ WORLD_MAP3.T


def world_point_to_ue(p_m) -> np.ndarray:
    """Inverse of :func:`ue_point_to_world` (meters -> cm). Used for round-trip."""
    p = np.asarray(p_m, dtype=np.float64)
    inv = np.linalg.inv(WORLD_MAP3)
    return p @ inv.T


# --------------------------------------------------------------------------- #
# Camera extrinsics
# --------------------------------------------------------------------------- #
def ue_camera_to_c2w(location_cm, basis_ue: np.ndarray) -> np.ndarray:
    """UE camera (location cm + basis) -> 4x4 OpenCV camera-to-world (meters).

    ``basis_ue`` columns are the world-space directions of the camera's UE local
    axes: column 0 = forward(+X), 1 = right(+Y), 2 = up(+Z).
    """
    location_cm = np.asarray(location_cm, float).reshape(3)
    basis_ue = np.asarray(basis_ue, float).reshape(3, 3)
    fwd_ue, right_ue, up_ue = basis_ue[:, 0], basis_ue[:, 1], basis_ue[:, 2]

    # Apply the handedness flip to the basis directions.
    fwd = geom.normalize(WORLD_FLIP @ fwd_ue)
    right = geom.normalize(WORLD_FLIP @ right_ue)
    up = geom.normalize(WORLD_FLIP @ up_ue)

    # OpenCV camera axes expressed in the (flipped) world:
    #   +X = right, +Y = down (= -up), +Z = forward
    R_c2w = np.stack([right, -up, fwd], axis=1)  # columns
    t_c2w = ue_point_to_world(location_cm)

    M = np.eye(4)
    M[:3, :3] = R_c2w
    M[:3, 3] = t_c2w
    return M


def c2w_to_w2c(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """camera-to-world (4x4) -> (R_w2c 3x3, t_w2c 3) for OpenCV projection."""
    M = np.asarray(M, float)
    R = M[:3, :3]
    t = M[:3, 3]
    R_w2c = R.T
    t_w2c = -R_w2c @ t
    return R_w2c, t_w2c


# --------------------------------------------------------------------------- #
# Deliberately-broken converters -- referenced ONLY by tests, to prove the
# gates have teeth. Wrong handedness must FAIL a test, never silently mirror.
# --------------------------------------------------------------------------- #
def _bad_no_handedness_flip(location_cm, basis_ue: np.ndarray) -> np.ndarray:
    """BUG: treat UE data as already right-handed (no Y flip anywhere).

    Geometrically self-consistent (a global mirror), so it would *pass*
    reprojection -- but it yields an improper rotation (det = -1). The
    determinant gate in T0 catches it.
    """
    location_cm = np.asarray(location_cm, float).reshape(3)
    basis_ue = np.asarray(basis_ue, float).reshape(3, 3)
    fwd, right, up = basis_ue[:, 0], basis_ue[:, 1], basis_ue[:, 2]
    R_c2w = np.stack([right, -up, fwd], axis=1)
    M = np.eye(4)
    M[:3, :3] = R_c2w
    M[:3, 3] = location_cm * CM_TO_M  # scaled but NOT flipped
    return M


def _bad_asymmetric_flip(location_cm, basis_ue: np.ndarray) -> np.ndarray:
    """BUG: flip the world translation but NOT the camera basis.

    A genuine inconsistency -> off-axis points reproject to the wrong place.
    The reprojection gate (T1) catches it.
    """
    location_cm = np.asarray(location_cm, float).reshape(3)
    basis_ue = np.asarray(basis_ue, float).reshape(3, 3)
    fwd, right, up = basis_ue[:, 0], basis_ue[:, 1], basis_ue[:, 2]  # NOT flipped
    R_c2w = np.stack([right, -up, fwd], axis=1)
    M = np.eye(4)
    M[:3, :3] = R_c2w
    M[:3, 3] = ue_point_to_world(location_cm)  # flipped
    return M


# --------------------------------------------------------------------------- #
# transforms.json assembly
# --------------------------------------------------------------------------- #
def build_transforms(intr: dict, frames: Iterable[dict],
                     aabb_min_ue=None, aabb_max_ue=None,
                     extra: dict | None = None) -> dict:
    """Assemble a Nerfstudio/instant-ngp transforms.json dict.

    ``frames`` is an iterable of dicts each with at least:
        file_path : str (relative path to the image)
        location_cm : (3,) UE camera location
        basis_ue : (3,3) UE camera basis (columns forward,right,up)
        split : "train" | "heldout"   (optional; defaults "train")
    plus any extra keys (e.g. fiducial GT pixels) carried through verbatim.
    """
    out_frames = []
    for fr in frames:
        M = ue_camera_to_c2w(fr["location_cm"], fr["basis_ue"])
        if not geom.is_proper_rotation(M):
            raise ValueError(
                f"converted pose for {fr.get('file_path')} is not a proper "
                f"right-handed rotation (det={geom.det3(M):.4f}); handedness bug")
        entry = {
            "file_path": fr["file_path"],
            "transform_matrix": M.tolist(),
        }
        if "split" in fr:
            entry["split"] = fr["split"]
        # carry through optional verification aids
        for k in ("fiducials_px", "fiducials_vis", "location_cm", "basis_ue"):
            if k in fr and k not in entry:
                v = fr[k]
                entry[k] = v.tolist() if isinstance(v, np.ndarray) else v
        out_frames.append(entry)

    doc = {
        "camera_model": CAMERA_MODEL,
        "world_up": WORLD_UP,
        "units": "meters",
        "convention_note": (
            "UE (left-handed, Z-up, cm) converted via diag(1,-1,1)*0.01. "
            "Camera axes are OpenCV: +X right, +Y down, +Z forward. "
            "transform_matrix is camera-to-world."),
        "w": intr["w"], "h": intr["h"],
        "fl_x": intr["fl_x"], "fl_y": intr["fl_y"],
        "cx": intr["cx"], "cy": intr["cy"],
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "frames": out_frames,
    }
    if aabb_min_ue is not None and aabb_max_ue is not None:
        # Convert the UE AABB corners and re-derive an axis-aligned box in world.
        corners_ue = _aabb_corners(aabb_min_ue, aabb_max_ue)
        corners_w = ue_point_to_world(corners_ue)
        doc["aabb_min"] = corners_w.min(axis=0).tolist()
        doc["aabb_max"] = corners_w.max(axis=0).tolist()
    if extra:
        doc.update(extra)
    return doc


def _aabb_corners(mn, mx) -> np.ndarray:
    mn = np.asarray(mn, float)
    mx = np.asarray(mx, float)
    out = []
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                out.append([mx[0] if i else mn[0],
                            mx[1] if j else mn[1],
                            mx[2] if k else mn[2]])
    return np.array(out, float)
