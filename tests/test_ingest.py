"""End-to-end UE-data path: synthesize ue_poses.json from the canonical scene,
ingest it, and verify the resulting transforms.json validates and reprojects
fiducials to sub-pixel error -- exercising exactly what `make capture` feeds in.
"""
import json
from pathlib import Path

import numpy as np

from splatkit import ingest, reproject, schema
from selftest import scene as S

REPO = Path(__file__).resolve().parent.parent


def _ue_poses():
    sc = S.build_scene()
    frames = []
    for c in S.cameras():
        loc = np.asarray(c["loc_cm"], float)
        basis = S.camera_basis(loc)
        sub = "heldout_gt" if c["split"] == "heldout" else "train"
        fp = str(REPO / "fixtures" / "selftest" / "images" / sub / f"cam_{c['index']:03d}.png")
        frames.append({"file_path": fp, "split": c["split"],
                       "location_cm": loc.tolist(), "basis_ue": basis.tolist()})
    return {
        "intrinsics": sc["intrinsics"], "background": sc["background"],
        "aabb_min_cm": sc["aabb_min_cm"], "aabb_max_cm": sc["aabb_max_cm"],
        "fiducials": S.FIDUCIALS, "primitives": [], "frames": frames,
    }


def test_ingest_produces_valid_subpixel_dataset(tmp_path):
    up = tmp_path / "ue_poses.json"
    up.write_text(json.dumps(_ue_poses()))
    out = tmp_path / "ds"
    info = ingest.ingest(str(up), str(out))
    assert info["n_frames"] == len(S.cameras())

    doc = json.loads((out / "transforms.json").read_text())
    res = schema.validate(doc, out)
    assert res["ok"], res["issues"]
    assert res["poses_ok"] == res["n_frames"]

    fid = np.array([f["loc_cm"] for f in S.FIDUCIALS], float)
    _, gmean, gmax, n = reproject.evaluate_doc(doc, fid)
    assert n > 100 and gmax < 1e-6, (gmax, n)
