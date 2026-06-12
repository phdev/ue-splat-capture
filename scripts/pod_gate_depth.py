"""POD-SIDE universal gate (MANDATORY before pod delete — CLAUDE.md UNIVERSAL
PIPELINE): render N spread TRAINING views directly from the trained model (no
viewer, no pose conversion — eliminates the viewer pose-ambiguity that made
host-side gating unreliable) -> /root/gate/*.png + inverse-depth MAE vs GT.

Run from /root on the pod after training:
    python3 pod_gate_depth.py -s /root/ed -m /root/ed/out --iteration 30000
Then `tar czf gate.tar.gz gate/` and scp it down to eyeball the renders.

Reading the MAE: island-quality models gate at ~0.009 mean; the canyon draft
gated at ~0.037 (soft but structured). A fog/soup model reads >0.1 on many
views. Any single view >0.1 = under-covered pocket — look at its render.
"""
import os
import sys

sys.path.append("/root/gsv")
import torch
import torchvision
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", type=int, default=40000)
args = get_combined_args(parser)
dataset = model.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
views = scene.getTrainCameras()
step = max(1, len(views) // 12)
views = views[::step][:12]
bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
os.makedirs("/root/gate", exist_ok=True)
pp = pipeline.extract(args)
maes = []
for v in views:
    try:
        pkg = render(v, gaussians, pp, bg, use_trained_exp=False, separate_sh=False)
    except TypeError:
        pkg = render(v, gaussians, pp, bg)
    torchvision.utils.save_image(pkg["render"], f"/root/gate/{v.image_name}_render.png")
    if "depth" in pkg and v.invdepthmap is not None:
        inv = pkg["depth"].squeeze()
        gtinv = v.invdepthmap.cuda().squeeze()
        m = v.depth_mask.cuda().squeeze() if v.depth_mask is not None else torch.ones_like(gtinv)
        mae = float((torch.abs(inv - gtinv) * m).sum() / m.sum())
        maes.append(mae)
        print(f"{v.image_name}: invdepth MAE {mae:.5f}")
if maes:
    print(f"MEAN invdepth MAE: {sum(maes)/len(maes):.5f} over {len(maes)} views")
print("GATE_RENDERS_DONE")
