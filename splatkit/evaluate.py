"""Render given poses with a trained splat and score PSNR/SSIM vs ground truth."""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import gsmodel, metrics


@torch.no_grad()
def evaluate(model, frames, bg, save_dir: str | None = None) -> dict:
    psnrs, ssims, per = [], [], []
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    for f in frames:
        pred = gsmodel.render(model, f, bg).clamp(0, 1)
        gt = f["image"]
        p = metrics.psnr(pred, gt)
        s = float(metrics.ssim(pred, gt))
        psnrs.append(p); ssims.append(s)
        per.append({"file_path": f["file_path"], "psnr": p, "ssim": s})
        if save_dir:
            name = Path(f["file_path"]).stem
            comp = torch.cat([gt, pred], dim=1).cpu().numpy()
            Image.fromarray((comp * 255 + 0.5).astype(np.uint8)).save(
                Path(save_dir) / f"{name}_gt_vs_pred.png")
    return {
        "psnr_mean": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim_mean": float(np.mean(ssims)) if ssims else 0.0,
        "psnr_min": float(np.min(psnrs)) if psnrs else 0.0,
        "ssim_min": float(np.min(ssims)) if ssims else 0.0,
        "n": len(frames),
        "per_frame": per,
    }
