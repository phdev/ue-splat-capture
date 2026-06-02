"""Trainer: fit a Gaussian splat to the TRAIN views only (fixed seed/iters).

Backend selection:
  --backend torch  (default) -> in-repo MPS/CPU rasterizer (splatkit.gsmodel)
  --backend msplat          -> only on a CUDA host with msplat installed
                               (`pip install msplat[cli]`); raises here otherwise.

The splat NEVER sees held-out views; those are reserved for the recon gate (T3).
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import time

import numpy as np
import torch

from . import data, gsmodel, initpc, metrics
from .results import env_device_default

DEFAULT_LR = {"means": 1.5e-3, "color": 1.5e-2, "scale": 9.0e-3,
              "quat": 1.0e-3, "opacity": 4.0e-2}


def _seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train(transforms_path: str, iters: int = 1500, n_gauss: int = 6000,
          seed: int = 0, device: str | None = None, lambda_ssim: float = 0.2,
          lr: dict | None = None, log_every: int = 200, backend: str = "torch",
          init: str = "consistency", densify: bool = True, densify_from: int = 150,
          densify_until_frac: float = 0.6, densify_interval: int = 100,
          densify_frac: float = 0.15, scale_split: float = 0.05,
          max_points: int = 18000, sh_degree: int = 0, verbose: bool = True):
    if backend == "msplat":
        raise RuntimeError(
            "msplat backend requires a CUDA host (msplat is CUDA-only). On this "
            "Apple-Silicon machine use --backend torch (the in-repo rasterizer).")
    elif backend != "torch":
        raise ValueError(f"unknown backend {backend!r}")

    device = device or env_device_default()
    lr = lr or DEFAULT_LR
    _seed_everything(seed)

    frames, meta = data.load_dataset(transforms_path, device=device)
    train_f, held_f = data.split_frames(frames)
    bg = meta["bg"]

    if init in ("carve", "consistency"):
        if init == "consistency":
            pts, cols, vox_m = initpc.consistency_point_cloud(
                train_f, meta["aabb_min"], meta["aabb_max"], max_points=n_gauss,
                seed=seed, background=bg.cpu().numpy())
        else:
            pts, cols, vox_m = initpc.carve_point_cloud(
                train_f, bg.cpu().numpy(), meta["aabb_min"], meta["aabb_max"],
                max_points=n_gauss, seed=seed)
        if verbose:
            print(f"  {init} init: {pts.shape[0]} points (voxel {vox_m*100:.1f} cm)",
                  flush=True)
        model = gsmodel.GaussianModel.from_points(
            pts, cols, init_scale=max(vox_m * 1.1, 0.03), device=device,
            init_opacity=0.7, sh_degree=sh_degree)
    else:
        model = gsmodel.init_from_cameras(
            meta["aabb_min"], meta["aabb_max"], n_gauss,
            cams=train_f, gt_images=[f["image"] for f in train_f],
            seed=seed, device=device, sh_degree=sh_degree)
    opt = torch.optim.Adam(model.param_groups(lr), eps=1e-15)

    if verbose:
        with torch.no_grad():
            p0 = np.mean([metrics.psnr(gsmodel.render(model, f, bg).clamp(0, 1),
                                       f["image"]) for f in held_f])
        print(f"  init heldout PSNR {p0:.2f} dB ({model.n} gaussians)", flush=True)

    rng = np.random.RandomState(seed)
    gen = torch.Generator().manual_seed(seed + 12345)
    order = []
    t0 = time.time()
    hist = []
    grad_accum = torch.zeros(model.n, device=device)
    grad_iters = 0
    densify_until = int(densify_until_frac * iters)
    for it in range(1, iters + 1):
        if not order:
            order = list(rng.permutation(len(train_f)))
        fi = order.pop()
        cam = train_f[fi]
        opt.zero_grad(set_to_none=True)
        pred = gsmodel.render(model, cam, bg)
        gt = cam["image"]
        l1 = torch.abs(pred - gt).mean()
        dssim = 1.0 - metrics.ssim(pred, gt)
        loss = (1 - lambda_ssim) * l1 + lambda_ssim * dssim
        loss.backward()
        with torch.no_grad():
            if model.means.grad is not None:
                grad_accum += model.means.grad.norm(dim=1)
                grad_iters += 1
        opt.step()
        with torch.no_grad():
            model.logit_opacity.clamp_(-8.0, 8.0)
            model.log_scales.clamp_(np.log(0.004), np.log(1.5))

        if (densify and densify_from <= it <= densify_until
                and it % densify_interval == 0):
            grad_avg = grad_accum / max(grad_iters, 1)
            model = gsmodel.clone_split_prune(
                model, grad_avg, densify_frac=densify_frac, scale_split=scale_split,
                max_points=max_points, gen=gen)
            opt = torch.optim.Adam(model.param_groups(lr), eps=1e-15)
            grad_accum = torch.zeros(model.n, device=device)
            grad_iters = 0
            if verbose:
                print(f"  iter {it:5d}: densify -> {model.n} gaussians", flush=True)

        if verbose and (it % log_every == 0 or it == 1):
            with torch.no_grad():
                p = metrics.psnr(pred.clamp(0, 1), gt)
            hist.append({"iter": it, "loss": float(loss), "train_psnr_1view": p,
                         "n_gauss": model.n})
            print(f"  iter {it:5d}/{iters}  loss {float(loss):.4f}  "
                  f"train_psnr(1view) {p:.2f}  n={model.n}  ({time.time()-t0:.1f}s)",
                  flush=True)

    info = {"iters": iters, "n_gauss_init": n_gauss, "n_gauss_final": model.n,
            "seed": seed, "device": device, "backend": backend,
            "n_train": len(train_f), "n_heldout": len(held_f), "sh_degree": sh_degree,
            "densify": densify, "lambda_ssim": lambda_ssim,
            "train_seconds": time.time() - t0, "history": hist}
    return model, meta, train_f, held_f, info


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a Gaussian splat")
    ap.add_argument("--transforms", default="fixtures/selftest/transforms.json")
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--n-gauss", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--backend", default="torch", choices=["torch", "msplat"])
    ap.add_argument("--init", default="consistency",
                    choices=["consistency", "carve", "random"])
    ap.add_argument("--sh-degree", type=int, default=0)
    ap.add_argument("--checkpoint", default="out/model.pt")
    args = ap.parse_args()
    model, meta, train_f, held_f, info = train(
        args.transforms, iters=args.iters, n_gauss=args.n_gauss,
        seed=args.seed, device=args.device, backend=args.backend, init=args.init,
        sh_degree=args.sh_degree)
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    torch.save({"state": model.state(), "info": info}, args.checkpoint)
    print(f"saved checkpoint -> {args.checkpoint}  ({info['train_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
