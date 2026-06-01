"""T1 -- fiducial reprojection verifier (the handedness/axis gate).

Projects known 3D fiducials (whose UE world coordinates are known exactly)
through the EXPORTED intrinsics + extrinsics using the OpenCV pinhole model,
and compares to ground-truth pixel locations that were produced by an entirely
independent UE-native projector (``selftest.scene.project_ue_native``). Two
independently-written projections agreeing to sub-pixel accuracy is a strong
proof that the coordinate conversion is correct; an asymmetric handedness bug
makes off-axis fiducials diverge by many pixels.

The success metric requires < 1.0 px MEAN error for *every* exported pose.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import convert
from .convert import c2w_to_w2c, intrinsics_matrix


def project_opencv(M: np.ndarray, intr: dict, points_world: np.ndarray):
    """Project world points (meters) to pixels via OpenCV pinhole.

    M: 4x4 camera-to-world. points_world: (N,3). Returns (uv (N,2), depth (N,)).
    """
    Pw = np.asarray(points_world, float).reshape(-1, 3)
    R_w2c, t_w2c = c2w_to_w2c(M)
    Xc = Pw @ R_w2c.T + t_w2c          # (N,3) camera-space (OpenCV: +Z fwd)
    z = Xc[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = intr["fl_x"] * (Xc[:, 0] / safe) + intr["cx"]
    v = intr["fl_y"] * (Xc[:, 1] / safe) + intr["cy"]
    return np.stack([u, v], axis=1), z


def per_pose_reprojection(c2w_list, intr, fiducials_world, gt_px, vis):
    """Mean reprojection error (px) per pose, over visible fiducials.

    fiducials_world: (F,3) meters. gt_px: (N,F,2). vis: (N,F) bool.
    Returns (per_pose_mean (N,), global_mean, global_max_pose_mean, n_pairs).
    """
    fiducials_world = np.asarray(fiducials_world, float)
    gt_px = np.asarray(gt_px, float)
    vis = np.asarray(vis, bool)
    per_pose = []
    all_err = []
    for i, M in enumerate(c2w_list):
        uv, _ = project_opencv(M, intr, fiducials_world)
        err = np.linalg.norm(uv - gt_px[i], axis=1)   # (F,)
        m = vis[i]
        if m.any():
            e = err[m]
            per_pose.append(float(e.mean()))
            all_err.append(e)
    per_pose = np.array(per_pose) if per_pose else np.array([0.0])
    all_err = np.concatenate(all_err) if all_err else np.array([0.0])
    return per_pose, float(all_err.mean()), float(per_pose.max()), int(all_err.size)


def _c2w_from_frames(doc: dict, converter=None) -> list[np.ndarray]:
    """camera-to-world per frame; from stored matrix, or re-converted via
    ``converter(location_cm, basis_ue)`` when one is supplied (for negative
    tests)."""
    out = []
    for fr in doc["frames"]:
        if converter is None:
            out.append(np.asarray(fr["transform_matrix"], float))
        else:
            out.append(converter(np.asarray(fr["location_cm"], float),
                                 np.asarray(fr["basis_ue"], float)))
    return out


def evaluate_doc(doc: dict, fiducials_ue: np.ndarray, converter=None):
    """Compute reprojection stats for a transforms doc + scene fiducials (UE cm).

    The frames must carry ``fiducials_px`` (GT, from the independent UE-native
    projector) and ``fiducials_vis``.
    """
    intr = {k: doc[k] for k in ("w", "h", "fl_x", "fl_y", "cx", "cy")}
    fiducials_world = convert.ue_point_to_world(np.asarray(fiducials_ue, float))
    gt_px = np.array([fr["fiducials_px"] for fr in doc["frames"]], float)
    vis = np.array([fr["fiducials_vis"] for fr in doc["frames"]], bool)
    c2w = _c2w_from_frames(doc, converter=converter)
    return per_pose_reprojection(c2w, intr, fiducials_world, gt_px, vis)


def run(transforms_path: str, scene_path: str, threshold_px: float = 1.0) -> dict:
    """Tier-T1 entry: returns a results dict (also written by the CLI)."""
    from . import results as R
    doc = json.loads(Path(transforms_path).read_text())
    scene = json.loads(Path(scene_path).read_text())
    fiducials_ue = np.array([f["loc_cm"] for f in scene["fiducials"]], float)
    per_pose, gmean, gmax, npairs = evaluate_doc(doc, fiducials_ue)
    checks = [
        R.check("max_pose_mean_reproj_px", gmax, threshold_px, "<",
                note=f"{len(per_pose)} poses, {npairs} fiducial obs"),
        R.check("global_mean_reproj_px", gmean, threshold_px, "<"),
    ]
    return {"checks": checks, "per_pose_mean_px": per_pose.tolist(),
            "n_poses": len(per_pose), "n_observations": npairs}


if __name__ == "__main__":
    import argparse
    from . import results as R
    ap = argparse.ArgumentParser(description="T1 reprojection verifier")
    ap.add_argument("--transforms", default="fixtures/selftest/transforms.json")
    ap.add_argument("--scene", default="fixtures/selftest/scene.json")
    ap.add_argument("--threshold", type=float, default=1.0)
    args = ap.parse_args()
    res = run(args.transforms, args.scene, args.threshold)
    doc = R.write_tier("t1", res["checks"], n_poses=res["n_poses"],
                       n_observations=res["n_observations"])
    R.print_tier(doc)
    raise SystemExit(0 if doc["pass"] else 1)
