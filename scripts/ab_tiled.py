import os, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from splatkit import tier_t3
res = {}
for tiled in ("1", "0"):
    os.environ["SPLAT_TILED"] = tiled
    r = tier_t3.run("fixtures/selftest/transforms.json", iters=400, n_gauss=6000,
                    seed=0, save_dir=None, sh_degree=0)
    res[tiled] = r
    print(f"### TILED={tiled}  heldout_psnr={r['heldout']['psnr_mean']:.2f} dB  "
          f"ssim={r['heldout']['ssim_mean']:.4f}  train_s={r['train_seconds']:.1f}", flush=True)
t, g = res["1"], res["0"]
print(f"### SPEEDUP {g['train_seconds']/t['train_seconds']:.2f}x   "
      f"quality delta {t['heldout']['psnr_mean']-g['heldout']['psnr_mean']:+.2f} dB")
