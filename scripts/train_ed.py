"""Train a splat on the Electric Dreams headless capture + eval held-out views.
CONSISTENCY init (now scale-aware via the voxel-grid cap) with a looser variance
tolerance for the real scene. Not gated; saves GT-vs-pred to out/electric_dreams_eval."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["SPLAT_VAR_TOL"] = os.environ.get("SPLAT_VAR_TOL", "0.06")
os.environ["SPLAT_MAX_POINTS"] = os.environ.get("SPLAT_MAX_POINTS", "20000")
from splatkit import train as T, evaluate as ev

model, meta, train_f, held_f, info = T.train(
    os.environ.get("ED_DS", "out/electric_dreams_norm/transforms.json"),
    iters=int(os.environ.get("ED_ITERS", "1000")),
    n_gauss=int(os.environ.get("ED_NGAUSS", "8000")),
    seed=0, sh_degree=1, init="consistency", verbose=True)

bg = meta["bg"]
held = ev.evaluate(model, held_f, bg, save_dir="out/electric_dreams_eval")
tr = ev.evaluate(model, train_f, bg, save_dir=None)
print(f"ED_RESULT heldout PSNR {held['psnr_mean']:.2f} SSIM {held['ssim_mean']:.3f} "
      f"min {held['psnr_min']:.2f}/{held['ssim_min']:.3f} n={held['n']}")
print(f"ED_TRAIN PSNR {tr['psnr_mean']:.2f} SSIM {tr['ssim_mean']:.3f} "
      f"gap {tr['psnr_mean']-held['psnr_mean']:.2f}")
print("ED_DONE")
