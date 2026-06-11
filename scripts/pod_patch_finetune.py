"""v2: LOAD_ITER warm-start + exposure init (load_ply skips what create_from_pcd
normally sets up; training_setup then crashes on _exposure)."""
import os
import subprocess

os.chdir("/workspace")
subprocess.run(["git", "-C", "gsv", "checkout", "--", "train.py"], check=True)
p = "gsv/train.py"
s = open(p).read()
anchor = "    scene = Scene(dataset, gaussians)"
assert anchor in s, "Scene anchor not found"
s = s.replace(anchor, """    # LOAD_ITER_PATCH v2: warm-start from a saved iteration's ply + init exposures
    _li = os.environ.get("LOAD_ITER")
    scene = Scene(dataset, gaussians, load_iteration=int(_li)) if _li else Scene(dataset, gaussians)
    if _li:
        _names = [c.image_name for c in scene.getTrainCameras()]
        gaussians.exposure_mapping = {n: i for i, n in enumerate(_names)}
        gaussians.pretrained_exposures = None
        _e = torch.eye(3, 4, device="cuda")[None].repeat(len(_names), 1, 1)
        gaussians._exposure = torch.nn.Parameter(_e.requires_grad_(True))""")
open(p, "w").write(s)
print("train.py patched v2")
