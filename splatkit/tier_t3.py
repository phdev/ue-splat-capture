"""T3 runner (verify-recon): train on TRAIN views, score HELD-OUT views.

Gates (the falsifiable success metric):
    held-out PSNR >= 28 dB, held-out SSIM >= 0.85,
    and held-out PSNR tracks train PSNR (overfit guard).
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse

from . import evaluate as ev, results as R, train as T

PSNR_THRESHOLD = 28.0
SSIM_THRESHOLD = 0.85
OVERFIT_GAP_DB = 8.0     # train_psnr - heldout_psnr must stay below this


def run(transforms_path: str, iters: int, n_gauss: int, seed: int,
        device=None, save_dir: str | None = "out/eval") -> dict:
    model, meta, train_f, held_f, info = T.train(
        transforms_path, iters=iters, n_gauss=n_gauss, seed=seed, device=device)
    bg = meta["bg"]

    held = ev.evaluate(model, held_f, bg, save_dir=save_dir)
    train_eval = ev.evaluate(model, train_f, bg, save_dir=None)
    gap = train_eval["psnr_mean"] - held["psnr_mean"]

    checks = [
        R.check("heldout_psnr_db", held["psnr_mean"], PSNR_THRESHOLD, ">=",
                note=f"{held['n']} held-out views, min {held['psnr_min']:.2f} dB"),
        R.check("heldout_ssim", held["ssim_mean"], SSIM_THRESHOLD, ">=",
                note=f"min {held['ssim_min']:.3f}"),
        R.check("overfit_gap_db", gap, OVERFIT_GAP_DB, "<=",
                note=f"train {train_eval['psnr_mean']:.2f} - heldout "
                     f"{held['psnr_mean']:.2f} dB"),
    ]
    return {
        "checks": checks,
        "heldout": {k: held[k] for k in ("psnr_mean", "ssim_mean", "psnr_min",
                                         "ssim_min", "n")},
        "train": {k: train_eval[k] for k in ("psnr_mean", "ssim_mean")},
        "device": info["device"], "iters": iters, "n_gauss": n_gauss,
        "seed": seed, "train_seconds": info["train_seconds"],
        "per_frame_heldout": held["per_frame"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="T3 reconstruction gate")
    ap.add_argument("--transforms", default="fixtures/selftest/transforms.json")
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--n-gauss", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    res = run(args.transforms, args.iters, args.n_gauss, args.seed, args.device)
    doc = R.write_tier("t3", res["checks"],
                       heldout=res["heldout"], train=res["train"],
                       device=res["device"], iters=res["iters"],
                       n_gauss=res["n_gauss"], seed=res["seed"],
                       train_seconds=res["train_seconds"])
    R.print_tier(doc)
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
