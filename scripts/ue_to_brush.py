"""Convert our ingested (OpenCV) transforms.json -> a brush/nerfstudio dataset.
Recenters to origin (f32 precision) and flips OpenCV->OpenGL (--flip gl) or leaves
it (--flip cv). brush reads <dst> (transforms.json + images/)."""
import json, shutil, sys
from pathlib import Path
import numpy as np
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
flip = (sys.argv[3] if len(sys.argv) > 3 else "gl")
d = json.load(open(src / "transforms.json")); frames = d["frames"]
ctr = np.array([np.array(f["transform_matrix"])[:3, 3] for f in frames]).mean(0)
M = np.diag([1.0, -1.0, -1.0, 1.0]) if flip == "gl" else np.eye(4)
(dst / "images").mkdir(parents=True, exist_ok=True)
out = {k: v for k, v in d.items() if k != "frames"}; of = []
for f in frames:
    tm = np.array(f["transform_matrix"], float); tm[:3, 3] -= ctr; tm = tm @ M
    nm = Path(f["file_path"]).name
    shutil.copy(src / f["file_path"], dst / "images" / nm)
    g = dict(f); g["transform_matrix"] = tm.tolist(); g["file_path"] = f"images/{nm}"; of.append(g)
out["frames"] = of
json.dump(out, open(dst / "transforms.json", "w"))
print(f"wrote {dst}/transforms.json: {len(of)} frames, flip={flip}, recentered by {ctr.round(1).tolist()}")
