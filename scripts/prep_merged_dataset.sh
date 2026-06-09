#!/bin/zsh
# Merge a NUCLEAR foliage-off capture (CAP_A) with the warm-editor foliage-on capture
# (CAP_B = scene21 source) into ONE COLMAP-format depth-supervised dataset. Two captures
# from the same focus/frame indices use the same poses; merging at the dataset level
# doubles the camera supply per-pose AND each pose contributes its own (RGB,depth) pair
# so the trainer gets supervision from BOTH foliage-occluded AND foliage-skipped views.
# Output: out/ed_merged_train/ed/{images,depths,sparse/0}.
set -e
cd "$(dirname "$0")/.."
CAP_A="${CAP_A:-out/ed_nuclear}"          # foliage-off (NEW nuclear pass)
CAP_B="${CAP_B:-out/ed_editor_depth2}"    # foliage-on (scene21 source)
DS_A=out/ed_nuc_ds                         # ingested A
DS_B=out/ed_dep2_ds                        # ingested B
ED=out/ed_merged_train/ed

[ -f "$CAP_A/ue_poses.json" ] || { echo "no $CAP_A/ue_poses.json"; exit 1; }
[ -f "$CAP_B/ue_poses.json" ] || { echo "no $CAP_B/ue_poses.json"; exit 1; }

echo "=== ingest A=$CAP_A ==="
python3 -m splatkit.ingest --ue-poses "$CAP_A/ue_poses.json" --out "$DS_A"
echo "=== ingest B=$CAP_B ==="
python3 -m splatkit.ingest --ue-poses "$CAP_B/ue_poses.json" --out "$DS_B"

echo "=== merge -> $(dirname $ED)/merged ==="
MERGED=out/ed_merged_train/merged_ds
rm -rf "$MERGED"
python3 scripts/merge_datasets.py "$MERGED" "$DS_A" "$DS_B"

echo "=== transforms_to_colmap -> $ED ==="
rm -rf "$ED"; mkdir -p "$ED/images" "$ED/sparse/0"
python3 scripts/transforms_to_colmap.py "$MERGED/transforms.json" "$ED/sparse/0"

echo "=== flatten images ==="
python3 - "$ED" "$MERGED" <<'PY'
import os, sys, glob, shutil
ed, ds = sys.argv[1], sys.argv[2]
n=0
for sub in ("train","heldout_gt"):
    for f in glob.glob(os.path.join(ds,"images",sub,"*.png")):
        shutil.copy2(f, os.path.join(ed,"images",os.path.basename(f))); n+=1
names=[l.split()[-1] for l in open(os.path.join(ed,"sparse/0/images.txt"))
       if l.strip() and not l.startswith('#') and len(l.split())>=10]
miss=[x for x in names if not os.path.exists(os.path.join(ed,"images",x))]
print(f"flattened {n} imgs; colmap poses={len(names)} missing={len(miss)}")
assert not miss, f"MISSING e.g. {miss[:3]}"
PY

echo "=== build merged depth dataset (depths/ + depth_params.json) ==="
# Build a UNIFIED ue_poses-like file from both captures (rewrite file_path to the
# flattened/prefixed image basename so depth_exr_to_inria matches names correctly).
python3 - "$CAP_A" "$CAP_B" "$ED" <<'PY'
import json, os, sys
cap_a, cap_b, ed = sys.argv[1], sys.argv[2], sys.argv[3]
frames=[]
for prefix, cap in (("d0_", cap_a), ("d1_", cap_b)):
    j=json.load(open(os.path.join(cap, "ue_poses.json")))
    for fr in j["frames"]:
        stem=prefix + os.path.splitext(os.path.basename(fr["file_path"]))[0]
        fr2=dict(fr); fr2["file_path"]=os.path.join(ed,"images",stem+".png")
        if fr.get("depth_path"): fr2["depth_path"]=fr["depth_path"]
        frames.append(fr2)
out={"frames":frames}
op=os.path.join(ed,"merged_ue_poses.json")
json.dump(out, open(op,"w"))
print(f"merged ue_poses: {len(frames)} frames (A+B) -> {op}")
PY
python3 scripts/depth_exr_to_inria.py "$ED/merged_ue_poses.json" "$ED"

NI=$(ls "$ED/images" | wc -l | tr -d ' ')
ND=$(ls "$ED/depths" 2>/dev/null | wc -l | tr -d ' ')
HP=$([ -f "$ED/sparse/0/depth_params.json" ] && echo yes || echo NO)
echo "MERGED: imgs=$NI depths=$ND depth_params=$HP -> $ED"
