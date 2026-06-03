"""Forward-only render cost vs resolution (light; no autograd graph)."""
import os, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from splatkit import data, gsmodel, initpc, metrics
from splatkit.results import env_device_default
dev = env_device_default()
frames, meta = data.load_dataset("fixtures/selftest/transforms.json", device=dev)
train_f, _ = data.split_frames(frames)
bg = meta["bg"]; torch.manual_seed(0)
pts, cols, vox = initpc.consistency_point_cloud(train_f, meta["aabb_min"], meta["aabb_max"],
                                                max_points=6000, seed=0, background=bg.cpu().numpy())
model = gsmodel.GaussianModel.from_points(pts, cols, init_scale=max(vox*1.1,0.03),
                                          device=dev, init_opacity=0.7, sh_degree=1)
gen = torch.Generator().manual_seed(12345)
for _ in range(3):
    model = gsmodel.clone_split_prune(model, torch.rand(model.n, device=dev),
                                      densify_frac=0.2, scale_split=0.05, max_points=16000, gen=gen)
print(f"clustered model {model.n} gaussians on {dev}", flush=True)
base = train_f[0]
def cam_at(s):
    c = dict(base); c["fx"]=base["fx"]*s; c["fy"]=base["fy"]*s
    c["cx"]=base["cx"]*s; c["cy"]=base["cy"]*s; c["W"]=int(base["W"]*s); c["H"]=int(base["H"]*s); return c
@torch.no_grad()
def t_ms(fn, cam, n=8):
    for _ in range(2): fn(model, cam, bg)
    if dev=="mps": torch.mps.synchronize()
    t0=time.time()
    for _ in range(n): fn(model, cam, bg)
    if dev=="mps": torch.mps.synchronize()
    return (time.time()-t0)/n*1000
def safe(fn, cam):
    try:
        r=t_ms(fn,cam); torch.mps.empty_cache() if dev=="mps" else None; return f"{r:7.1f}ms"
    except RuntimeError as e:
        torch.mps.empty_cache() if dev=="mps" else None
        return "  OOM   " if "out of memory" in str(e).lower() else "ERR"
for s in (1.0, 2.0, 4.0, 6.0):
    cam = cam_at(s); W,H=int(base["W"]*s),int(base["H"]*s)
    try:
        with torch.no_grad():
            par=f'{metrics.psnr(gsmodel.render_tiled(model,cam,bg).clamp(0,1), gsmodel.render(model,cam,bg).clamp(0,1)):5.1f}dB'
    except RuntimeError: par=" n/a "
    if dev=="mps": torch.mps.empty_cache()
    print(f"res {W:4d}x{H:<4d} ({W*H:7d}px)  parity {par}  global {safe(gsmodel.render,cam)}  tiled {safe(gsmodel.render_tiled,cam)}", flush=True)
