"""Merge two or more INGESTED datasets (splatkit.ingest output: transforms.json in
world coords + images/{train,heldout_gt}/) into one, so a single training run sees
all passes (e.g. a spire dome + a ground-coverage pass). Merge at the ingested level
(world coords, BEFORE ue_to_brush) so ue_to_brush recenters the union ONCE and the
passes stay aligned. All inputs must share global intrinsics (same hfov/res).

  merge_datasets.py <out_dir> <ds0> <ds1> [<ds2> ...]
"""
import json
import os
import shutil
import sys

out_dir, ins = sys.argv[1], sys.argv[2:]
assert len(ins) >= 2, "need >=2 input datasets"

KEYS = ("w", "h", "fl_x", "fl_y", "cx", "cy", "camera_model", "world_up", "units")
base = json.load(open(os.path.join(ins[0], "transforms.json")))
for d in ins[1:]:
    t = json.load(open(os.path.join(d, "transforms.json")))
    for k in KEYS:
        a, b = base.get(k), t.get(k)
        if isinstance(a, float):
            assert abs(a - b) < 1e-3, f"intrinsic {k} mismatch: {a} vs {b} ({d})"
        else:
            assert a == b, f"intrinsic {k} mismatch: {a} vs {b} ({d})"

os.makedirs(os.path.join(out_dir, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "images", "heldout_gt"), exist_ok=True)
merged_frames = []
for i, d in enumerate(ins):
    t = json.load(open(os.path.join(d, "transforms.json")))
    pre = f"d{i}_"
    for fr in t["frames"]:
        rel = fr["file_path"]                       # e.g. images/train/cam_000.png
        sub, name = rel.split("/")[-2], rel.split("/")[-1]
        new_rel = f"images/{sub}/{pre}{name}"
        shutil.copy2(os.path.join(d, rel), os.path.join(out_dir, new_rel))
        fr = dict(fr); fr["file_path"] = new_rel
        merged_frames.append(fr)
    print(f"  + {d}: {len(t['frames'])} frames (prefix {pre})")

base["frames"] = merged_frames
json.dump(base, open(os.path.join(out_dir, "transforms.json"), "w"))
n_train = sum(f.get("split") == "train" for f in merged_frames)
print(f"merged {len(ins)} datasets -> {out_dir}: {len(merged_frames)} frames "
      f"({n_train} train / {len(merged_frames)-n_train} heldout)")
