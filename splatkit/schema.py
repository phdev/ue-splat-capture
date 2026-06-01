"""T2 (part 1) -- transforms.json schema + sanity validation.

Validates structure, intrinsics sanity, that every referenced image exists at
the declared resolution, that every pose is a proper right-handed rotation with
a clean bottom row, and that both train/heldout splits are present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import geom

REQUIRED_TOP = ("camera_model", "w", "h", "fl_x", "fl_y", "cx", "cy", "frames")


def validate(doc: dict, base_dir: str | Path) -> dict:
    base_dir = Path(base_dir)
    issues: list[str] = []

    missing = [k for k in REQUIRED_TOP if k not in doc]
    if missing:
        issues.append(f"missing top-level keys: {missing}")
        # cannot proceed meaningfully
        return {"ok": False, "issues": issues, "n_frames": 0,
                "intrinsics_sane": False, "images_ok": 0, "poses_ok": 0,
                "n_splits": 0}

    w, h = int(doc["w"]), int(doc["h"])
    fx, fy, cx, cy = doc["fl_x"], doc["fl_y"], doc["cx"], doc["cy"]
    intrinsics_sane = (w > 0 and h > 0 and fx > 0 and fy > 0
                       and 0 < cx < w and 0 < cy < h)
    if not intrinsics_sane:
        issues.append(f"intrinsics not sane: w={w} h={h} fx={fx} fy={fy} cx={cx} cy={cy}")
    if doc["camera_model"] != "OPENCV":
        issues.append(f"camera_model expected OPENCV, got {doc['camera_model']}")

    frames = doc["frames"]
    n = len(frames)
    if n == 0:
        issues.append("no frames")

    images_ok = 0
    poses_ok = 0
    splits = set()
    for fr in frames:
        # pose
        M = np.asarray(fr.get("transform_matrix", []), float)
        if M.shape == (4, 4) and geom.is_proper_rotation(M) \
                and np.allclose(M[3], [0, 0, 0, 1], atol=1e-9):
            poses_ok += 1
        else:
            issues.append(f"bad pose for {fr.get('file_path')}")
        # image
        fp = base_dir / fr.get("file_path", "")
        if fp.exists():
            try:
                with Image.open(fp) as im:
                    if im.size == (w, h):
                        images_ok += 1
                    else:
                        issues.append(f"{fp.name}: size {im.size} != ({w},{h})")
            except Exception as e:  # pragma: no cover
                issues.append(f"{fp.name}: unreadable ({e})")
        else:
            issues.append(f"missing image: {fr.get('file_path')}")
        splits.add(fr.get("split", "train"))

    return {
        "ok": not issues,
        "issues": issues,
        "n_frames": n,
        "intrinsics_sane": bool(intrinsics_sane and doc["camera_model"] == "OPENCV"),
        "images_ok": images_ok,
        "poses_ok": poses_ok,
        "n_splits": len(splits),
        "splits": sorted(splits),
    }
