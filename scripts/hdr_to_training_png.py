"""Linear HDR EXR -> 8-bit training PNGs through OUR tone curve.

WHY: UE's filmic LDR output crushes the shadow band to ~24 8-bit code values
(measured at the user's complaint pose) -> the trainer gets no gradients there ->
under-densified blurry/uneven shadows, unfixable at training time (three measured
failures: denser densify, dual-exposure affines, relative-L1). The fix is upstream:
capture SCS_FINAL_COLOR_HDR linear EXR (UE_HDR_COLOR=1; ~21K distinct shadow values)
and tone-map OURSELVES: a curve FITTED to match UE's filmic in mids/highlights
(shipped look unchanged there) with a LIFTED TOE that gives shadows real code values.

Curve: y(x) = max(filmic_fit(x), toe_lift(x)) where
  filmic_fit = binned-median LUT fitted from a paired (EXR, LDR-PNG) probe capture
  toe_lift(x) = TOE_Y * (x / TOE_X) ** (1/TOE_GAMMA)   (only wins in the crushed toe)
max() keeps everything the filmic curve renders brighter untouched; the handover is
seamless where the curves cross (~x 0.2-0.35).

Usage:
  fit:     hdr_to_training_png.py fit <probe.exr> <probe_ldr.png> <curve.npz>
  convert: hdr_to_training_png.py convert <curve.npz> <dir_with_exrs> [ndirs...]
Writes cam_XXX.png next to each cam_XXX.exr. Idempotent (skips existing pngs).
"""
import glob
import os
import sys

import Imath
import numpy as np
import OpenEXR
from PIL import Image

TOE_X, TOE_Y, TOE_GAMMA = 0.10, 0.16, 2.2   # lift: linear 0.10 -> display 0.16


def read_exr(p):
    f = OpenEXR.InputFile(p)
    dw = f.header()['dataWindow']
    W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    HALF = Imath.PixelType(Imath.PixelType.HALF)
    ch = [np.frombuffer(f.channel(c, HALF), dtype=np.float16).reshape(H, W).astype(np.float32)
          for c in ('R', 'G', 'B')]
    return np.clip(np.stack(ch, -1), 0.0, None)


def fit(exr_path, ldr_path, out_npz):
    hdr = read_exr(exr_path).reshape(-1)
    ldr = (np.asarray(Image.open(ldr_path).convert('RGB'), np.float32) / 255.0).reshape(-1)
    m = hdr > 1e-5
    hdr, ldr = hdr[m], ldr[m]
    # binned median in log-x for a smooth monotone LUT
    xs = np.logspace(np.log10(1e-4), np.log10(max(hdr.max(), 1.0)), 257)
    idx = np.clip(np.searchsorted(xs, hdr) - 1, 0, 255)
    ys = np.zeros(256, np.float32)
    last = 0.0
    for i in range(256):
        sel = ldr[idx == i]
        ys[i] = np.median(sel) if sel.size > 50 else last
        last = ys[i]
    ys = np.maximum.accumulate(ys)  # enforce monotone
    np.savez(out_npz, xs=xs[:-1], ys=ys)
    # report the toe handover
    grid = np.linspace(0, 1, 1001)
    f_fit = np.interp(grid, xs[:-1], ys)
    f_toe = TOE_Y * np.clip(grid / TOE_X, 0, None) ** (1.0 / TOE_GAMMA)
    cross = grid[np.argmin(np.abs(f_fit - f_toe)[50:]) + 50]
    print(f"curve fitted: {out_npz}; toe handover ~x={cross:.3f}; "
          f"shadow mapping: lin 0.0045->{apply_curve(np.array([0.0045]), xs[:-1], ys)[0]:.3f}, "
          f"lin 0.0857->{apply_curve(np.array([0.0857]), xs[:-1], ys)[0]:.3f}")


def apply_curve(x, xs, ys):
    f_fit = np.interp(x, xs, ys)
    f_toe = TOE_Y * np.clip(x / TOE_X, 0, None) ** (1.0 / TOE_GAMMA)
    return np.clip(np.maximum(f_fit, f_toe), 0.0, 1.0)


def convert_dirs(curve_npz, dirs):
    d = np.load(curve_npz)
    xs, ys = d['xs'], d['ys']
    total = 0
    for dd in dirs:
        for e in sorted(glob.glob(os.path.join(dd, '*.exr'))):
            png = e[:-4] + '.png'
            if os.path.exists(png):
                continue
            hdr = read_exr(e)
            out = apply_curve(hdr, xs, ys)
            Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(png)
            total += 1
    print(f"converted {total} exrs -> pngs")


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        fit(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == 'convert':
        convert_dirs(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(__doc__)
