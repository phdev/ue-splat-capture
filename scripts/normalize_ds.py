"""Normalize a large UE capture to the trainer's native ~2.5 m / origin scale and
downsample to 96 px, so the diorama-tuned defaults (LR, init scale, voxel grid)
all apply and training is fast. Geometry is rescaled rigidly -> identical splat,
just better-conditioned for the optimizer."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

src = Path("out/electric_dreams_ds")
dst = Path("out/electric_dreams_norm")
TARGET = 96
TARGET_EXTENT = 2.5

d = json.load(open(src / "transforms.json"))
amin = np.array(d["aabb_min"], float)
amax = np.array(d["aabb_max"], float)
center = (amin + amax) / 2.0
extent = float((amax - amin).max())
s = TARGET_EXTENT / extent
f = TARGET / d["w"]

d["w"] = d["h"] = TARGET
for k in ("fl_x", "fl_y", "cx", "cy"):
    d[k] *= f
d["aabb_min"] = ((amin - center) * s).tolist()
d["aabb_max"] = ((amax - center) * s).tolist()

for fr in d["frames"]:
    tm = np.array(fr["transform_matrix"], float)
    tm[:3, 3] = (tm[:3, 3] - center) * s            # recenter + rescale camera pos
    fr["transform_matrix"] = tm.tolist()
    rel = fr["file_path"]
    (dst / Path(rel).parent).mkdir(parents=True, exist_ok=True)
    Image.open(src / rel).convert("RGB").resize((TARGET, TARGET), Image.BOX).save(dst / rel)

json.dump(d, open(dst / "transforms.json", "w"), indent=1)
print(f"normalized: scale {s:.4f}, center {center.round(1).tolist()}, "
      f"{TARGET}px, fl {d['fl_x']:.1f}, aabb {np.round(d['aabb_min'],2).tolist()}..{np.round(d['aabb_max'],2).tolist()}")
