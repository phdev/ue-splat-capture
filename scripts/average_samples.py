"""Average the per-pose sample renders from a UE_AVG_SAMPLES capture into one clean
image per pose. Lumen GI / specular / TSR noise is re-randomised per render and is
what 3DGS turns into spiky foliage floaters; averaging N independent samples knocks
it down ~1/sqrt(N), giving each training view a stable appearance.

  average_samples.py <images_dir> [--keep]

Groups cam_IDX_SS.png -> mean -> cam_IDX.png, then deletes the samples (unless --keep).
"""
import glob
import os
import re
import sys

import numpy as np
from PIL import Image

img_dir = sys.argv[1]
keep = "--keep" in sys.argv[2:]

samples = glob.glob(os.path.join(img_dir, "cam_*_*.png"))
groups = {}
pat = re.compile(r"(cam_\d+)_\d+\.png$")
for f in samples:
    m = pat.search(os.path.basename(f))
    if m:
        groups.setdefault(m.group(1), []).append(f)

if not groups:
    print(f"no cam_IDX_SS.png sample groups in {img_dir}; nothing to do")
    sys.exit(0)

for base, fs in sorted(groups.items()):
    fs.sort()
    acc = None
    for f in fs:
        a = np.asarray(Image.open(f).convert("RGB"), np.float32)
        acc = a if acc is None else acc + a
    mean = (acc / len(fs)).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(mean).save(os.path.join(img_dir, base + ".png"))
    if not keep:
        for f in fs:
            os.remove(f)

n_samp = len(next(iter(groups.values())))
print(f"averaged {len(groups)} poses x {n_samp} samples -> cam_IDX.png"
      f"{' (kept samples)' if keep else ' (samples removed)'}")
