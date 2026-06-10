"""Pod-side: (1) bg masks from GT inverse-depth pngs (bg == exactly 0);
(2) patch gsv/train.py so GT is composited over the SAME per-iteration random
background as the render (--random_background). Any transparency inside the
object silhouette then mismatches GT every iteration -> optimizer seals it.
Run from /workspace AFTER clone+pip, BEFORE train.
"""
import glob
import os

import numpy as np
from PIL import Image

os.chdir("/workspace")

# 1) masks: fg=255 where inverse-depth > 0
os.makedirs("ed/bgmask", exist_ok=True)
n = 0
for f in glob.glob("ed/depths/*.png"):
    d = np.asarray(Image.open(f))
    fg = (d > 0).astype(np.uint8) * 255
    Image.fromarray(fg).save(os.path.join("ed/bgmask", os.path.basename(f)))
    n += 1
print(f"masks: {n}")

# 2) patch train.py
p = "gsv/train.py"
s = open(p).read()
if "ALPHA_COMPOSITE_PATCH" in s:
    print("already patched")
else:
    anchor = "        gt_image = viewpoint_cam.original_image.cuda()"
    assert anchor in s, "gt_image anchor not found"
    patch = anchor + """
        # ALPHA_COMPOSITE_PATCH: composite GT over the same random bg as the render
        # (masks: ed/bgmask/<image_name>.png, fg=255). Punishes any transparency
        # inside the object silhouette; requires --random_background.
        global _ALPHA_MASKS
        try:
            _ALPHA_MASKS
        except NameError:
            _ALPHA_MASKS = {}
            import glob as _glob
            from PIL import Image as _Image
            import numpy as _np
            for _f in _glob.glob(os.path.join(dataset.source_path, "bgmask", "*.png")):
                _k = os.path.splitext(os.path.basename(_f))[0]
                _m = torch.from_numpy((_np.asarray(_Image.open(_f)) > 127).astype(_np.float32))
                _ALPHA_MASKS[_k] = _m
            print(f"[alpha-patch] loaded {len(_ALPHA_MASKS)} bg masks")
        _m = _ALPHA_MASKS.get(viewpoint_cam.image_name) \\
            or _ALPHA_MASKS.get(os.path.splitext(viewpoint_cam.image_name)[0])
        if _m is not None and opt.random_background:
            _mc = _m.cuda().unsqueeze(0)
            if _mc.shape[-2:] == gt_image.shape[-2:]:
                gt_image = gt_image * _mc + bg[:, None, None] * (1.0 - _mc)"""
    s = s.replace(anchor, patch)
    open(p, "w").write(s)
    print("train.py patched")
