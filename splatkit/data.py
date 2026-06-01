"""Load a transforms.json dataset into per-frame camera dicts + images (torch)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .convert import c2w_to_w2c


def load_image(path: Path, device) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)            # (H,W,3) in [0,1]


def load_dataset(transforms_path: str, device: str = "cpu"):
    tp = Path(transforms_path)
    doc = json.loads(tp.read_text())
    base = tp.parent
    fx, fy = float(doc["fl_x"]), float(doc["fl_y"])
    cx, cy = float(doc["cx"]), float(doc["cy"])
    W, H = int(doc["w"]), int(doc["h"])
    bg = torch.tensor(doc.get("background", [0.0, 0.0, 0.0]),
                      dtype=torch.float32, device=device)

    frames = []
    for fr in doc["frames"]:
        M = np.asarray(fr["transform_matrix"], float)
        R_w2c, t_w2c = c2w_to_w2c(M)
        frames.append({
            "file_path": fr["file_path"],
            "split": fr.get("split", "train"),
            "R_w2c": torch.tensor(R_w2c, dtype=torch.float32, device=device),
            "t_w2c": torch.tensor(t_w2c, dtype=torch.float32, device=device),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "W": W, "H": H,
            "image": load_image(base / fr["file_path"], device),
        })
    meta = {
        "bg": bg, "W": W, "H": H,
        "aabb_min": doc.get("aabb_min", [-2, -2, 0]),
        "aabb_max": doc.get("aabb_max", [2, 2, 2]),
    }
    return frames, meta


def split_frames(frames):
    train = [f for f in frames if f["split"] == "train"]
    held = [f for f in frames if f["split"] == "heldout"]
    return train, held
