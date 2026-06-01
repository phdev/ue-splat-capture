"""Generate the committed self-test fixtures (deterministic).

Renders the canonical scene from the orbit rig, computes per-frame fiducial
ground-truth pixels (via the independent UE-native projector) + visibility, and
writes:
    fixtures/selftest/images/{train,heldout_gt}/cam_###.png
    fixtures/selftest/transforms.json   (Nerfstudio/instant-ngp, OpenCV)
    fixtures/selftest/scene.json        (geometry, fiducials, AABB, cameras)

This stands in for `make capture` when Unreal is not used to produce fixtures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from splatkit import convert
from . import raytracer, scene as S

REPO_ROOT = Path(__file__).resolve().parent.parent
FIX_DIR = REPO_ROOT / "fixtures" / "selftest"


def _fiducial_prim_indices(scene_dict) -> dict[str, int]:
    out = {}
    for i, p in enumerate(scene_dict["primitives"]):
        if "fiducial_id" in p:
            out[p["fiducial_id"]] = i
    return out


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def generate(ss: int = 2) -> dict:
    scene_dict = S.build_scene()
    intr = scene_dict["intrinsics"]
    W, H = intr["w"], intr["h"]
    fid_centers = np.array([f["loc_cm"] for f in S.FIDUCIALS], float)
    fid_prim = _fiducial_prim_indices(scene_dict)

    (FIX_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (FIX_DIR / "images" / "heldout_gt").mkdir(parents=True, exist_ok=True)

    frames = []
    cams = S.cameras()
    for cam in cams:
        loc = np.asarray(cam["loc_cm"], float)
        basis = S.camera_basis(loc)
        img = raytracer.render(scene_dict, loc, basis, intr, ss=ss)

        sub = "heldout_gt" if cam["split"] == "heldout" else "train"
        rel = f"images/{sub}/cam_{cam['index']:03d}.png"
        Image.fromarray(_to_u8(img)).save(FIX_DIR / rel)

        uv, depth = S.project_ue_native(loc, basis, intr, fid_centers)
        vis = []
        for k, f in enumerate(S.FIDUCIALS):
            in_front = depth[k] > 1.0
            in_img = (0 <= uv[k, 0] <= W) and (0 <= uv[k, 1] <= H)
            unocc = raytracer.fiducial_visible(scene_dict, loc, f["loc_cm"],
                                               fid_prim[f["id"]])
            vis.append(bool(in_front and in_img and unocc))

        frames.append({
            "file_path": rel,
            "location_cm": loc,
            "basis_ue": basis,
            "split": cam["split"],
            "fiducials_px": uv,
            "fiducials_vis": vis,
        })

    doc = convert.build_transforms(
        intr, frames,
        aabb_min_ue=scene_dict["aabb_min_cm"], aabb_max_ue=scene_dict["aabb_max_cm"],
        extra={
            "n_fiducials": len(S.FIDUCIALS),
            "fiducials_world_m": convert.ue_point_to_world(fid_centers).tolist(),
            "render_ss": ss,
            "source": "selftest.make_fixtures (synthetic UE stand-in)",
        })

    (FIX_DIR / "transforms.json").write_text(json.dumps(doc, indent=1))

    # scene.json carries the geometry/fiducials/aabb/cameras for T1/T2.
    scene_out = dict(scene_dict)
    scene_out["cameras"] = cams
    (FIX_DIR / "scene.json").write_text(json.dumps(scene_out, indent=1))

    n_train = sum(f["split"] == "train" for f in frames)
    n_held = len(frames) - n_train
    return {"n_frames": len(frames), "n_train": n_train, "n_heldout": n_held,
            "resolution": [W, H], "n_fiducials": len(S.FIDUCIALS)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate self-test fixtures")
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    args = ap.parse_args()
    info = generate(ss=args.ss)
    print("fixtures written to", FIX_DIR)
    print(json.dumps(info, indent=2))
