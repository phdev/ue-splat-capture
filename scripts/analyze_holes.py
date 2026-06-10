"""Bg-flip hole analysis: pixels that change with viewer background = transparency.
Border-connected changed region = true background; interior changed islands = HOLES.
Usage: analyze_holes.py <scandir> [vizdir]
"""
import glob
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

scandir = sys.argv[1]
vizdir = sys.argv[2] if len(sys.argv) > 2 else None
if vizdir:
    os.makedirs(vizdir, exist_ok=True)

rows = []
for fm in sorted(glob.glob(os.path.join(scandir, "*_m.png"))):
    name = os.path.basename(fm)[:-6]
    fg = os.path.join(scandir, f"{name}_g.png")
    if not os.path.exists(fg):
        continue
    A = np.asarray(Image.open(fm).convert("RGB"), np.float32) / 255.0
    Bm = np.asarray(Image.open(fg).convert("RGB"), np.float32) / 255.0
    if A.shape != Bm.shape:
        continue
    diff = np.abs(A - Bm).max(2)
    changed = diff > 0.25
    lab, nl = ndimage.label(changed)
    border = np.zeros_like(changed)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    bg_labels = np.unique(lab[border & changed])
    bg_labels = bg_labels[bg_labels > 0]
    bgmask = np.isin(lab, bg_labels)
    holes = changed & ~bgmask
    # drop specks < 4 px (AA noise)
    hlab, hn = ndimage.label(holes)
    if hn:
        sizes = ndimage.sum(holes, hlab, range(1, hn + 1))
        keep = np.isin(hlab, np.where(sizes >= 4)[0] + 1)
        holes = keep
    scene_px = int((~changed).sum())
    hole_px = int(holes.sum())
    frac = hole_px / max(scene_px, 1)
    nblobs = int(ndimage.label(holes)[1])
    rows.append((name, hole_px, frac, nblobs))
    if vizdir and hole_px > 0:
        viz = (A * 255).astype(np.uint8).copy()
        viz[holes] = [255, 0, 0]
        Image.fromarray(viz).save(os.path.join(vizdir, f"{name}_holes.png"))

rows.sort(key=lambda r: -r[1])
tot = sum(r[1] for r in rows)
nz = sum(1 for r in rows if r[1] > 0)
print(f"{scandir}: views={len(rows)} views_with_holes={nz} total_hole_px={tot}")
for r in rows[:12]:
    print(f"  {r[0]:8s} hole_px={r[1]:7d} frac={r[2]*100:5.2f}% blobs={r[3]}")
