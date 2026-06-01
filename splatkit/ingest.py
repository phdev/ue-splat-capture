"""Ingest a neutral `ue_poses.json` (written by the Unreal side) into the
Nerfstudio/instant-ngp `transforms.json` + `scene.json` the verifier consumes.

The two-interpreter split is deliberate: the Unreal side (UnrealEditor-Cmd's
own Python) cannot import this venv's packages, so it writes a plain JSON of raw
UE-convention poses + intrinsics + fiducials, and THIS module (pure Python, uv
venv) does the coordinate conversion via splatkit.convert.

`ue_poses.json` schema:
    {
      "intrinsics": {"w","h","fl_x","fl_y","cx","cy"},
      "background": [r,g,b],
      "aabb_min_cm":[...], "aabb_max_cm":[...],
      "fiducials": [{"id","loc_cm":[x,y,z],"radius","color":[r,g,b]}],
      "primitives": [...optional, for T2 camera-in-geometry...],
      "frames": [
        {"file_path","split","location_cm":[x,y,z],
         "basis_ue":[[fwd],[right],[up]] as 3x3 columns}
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from . import convert


def _detect_background(out_dir: Path, frames, n_sample=24, k=3):
    """Median corner colour across frames -> the true backdrop colour, so the
    trainer composites/inits against what UE actually rendered."""
    cols = []
    for fr in frames[:n_sample]:
        p = out_dir / fr["file_path"]
        if not p.exists():
            continue
        a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0
        h, w = a.shape[:2]
        cols += [a[:k, :k], a[:k, -k:], a[-k:, :k], a[-k:, -k:]]
    if not cols:
        return None
    return np.concatenate([c.reshape(-1, 3) for c in cols]).mean(0).tolist()


def _project_ue_native(loc_cm, basis_ue, intr, points_cm):
    """Independent UE-native projection (separate impl from convert.OpenCV path,
    so T1 stays a genuine cross-check)."""
    P = np.asarray(points_cm, float).reshape(-1, 3)
    loc = np.asarray(loc_cm, float)
    f, r, u = basis_ue[:, 0], basis_ue[:, 1], basis_ue[:, 2]
    rel = P - loc
    d = rel @ f
    safe = np.where(np.abs(d) < 1e-9, 1e-9, d)
    px = intr["cx"] + intr["fl_x"] * ((rel @ r) / safe)
    py = intr["cy"] - intr["fl_y"] * ((rel @ u) / safe)
    return np.stack([px, py], axis=1), d


def ingest(ue_poses_path: str, out_dir: str, copy_images: bool = True) -> dict:
    src = Path(ue_poses_path)
    up = json.loads(src.read_text())
    out = Path(out_dir)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "images" / "heldout_gt").mkdir(parents=True, exist_ok=True)

    intr = up["intrinsics"]
    W, H = intr["w"], intr["h"]
    fids = up.get("fiducials", [])
    fid_centers = np.array([f["loc_cm"] for f in fids], float) if fids else np.zeros((0, 3))

    frames = []
    for fr in up["frames"]:
        loc = np.asarray(fr["location_cm"], float)
        basis = np.asarray(fr["basis_ue"], float)
        if fid_centers.shape[0]:
            uv, depth = _project_ue_native(loc, basis, intr, fid_centers)
            vis = [bool(depth[k] > 1.0 and 0 <= uv[k, 0] <= W and 0 <= uv[k, 1] <= H)
                   for k in range(len(fids))]
        else:
            uv, vis = np.zeros((0, 2)), []

        # normalize file_path into the dataset layout
        sub = "heldout_gt" if fr.get("split") == "heldout" else "train"
        name = Path(fr["file_path"]).name
        rel = f"images/{sub}/{name}"
        if copy_images and Path(fr["file_path"]).exists():
            src_img = Image.open(fr["file_path"]).convert("RGB")
            if src_img.size != (W, H):     # box-downsample supersampled renders
                src_img = src_img.resize((W, H), Image.BOX)
            src_img.save(out / rel)
        frames.append({
            "file_path": rel, "split": fr.get("split", "train"),
            "location_cm": loc, "basis_ue": basis,
            "fiducials_px": uv, "fiducials_vis": vis,
        })

    bg = _detect_background(out, frames) or up.get("background", [0, 0, 0])
    doc = convert.build_transforms(
        intr, frames,
        aabb_min_ue=up.get("aabb_min_cm"), aabb_max_ue=up.get("aabb_max_cm"),
        extra={"n_fiducials": len(fids),
               "fiducials_world_m": convert.ue_point_to_world(fid_centers).tolist()
               if fid_centers.shape[0] else [],
               "background": bg,
               "source": "ue_capture (UnrealEditor-Cmd) -> splatkit.ingest"})
    (out / "transforms.json").write_text(json.dumps(doc, indent=1))

    scene = {
        "intrinsics": intr,
        "background": up.get("background", [0, 0, 0]),
        "fiducials": fids,
        "primitives": up.get("primitives", []),
        "aabb_min_cm": up.get("aabb_min_cm"), "aabb_max_cm": up.get("aabb_max_cm"),
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    return {"n_frames": len(frames), "out_dir": str(out)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest ue_poses.json -> transforms.json")
    ap.add_argument("--ue-poses", required=True)
    ap.add_argument("--out", default="fixtures/selftest")
    args = ap.parse_args()
    print(json.dumps(ingest(args.ue_poses, args.out), indent=2))
