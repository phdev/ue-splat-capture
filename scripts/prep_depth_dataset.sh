#!/bin/zsh
# Post-(depth)capture -> COLMAP-layout dataset WITH GT depth maps, ready to upload for
# depth-supervised vanilla 3DGS. The in-editor UE_DEPTH=1 capture is a SINGLE session
# (out/ed_editor_depth: images/cam_XXX.png + depth/cam_XXX.exr + ue_poses.json), so no
# averaging/merge -- a direct ingest keeps the cam_XXX names through to COLMAP, so the
# depth maps (depths/cam_XXX.png) line up with the COLMAP image names by construction.
#   1. ingest ue_poses.json -> OpenCV world-coord transforms.json + train/heldout split
#   2. transforms_to_colmap (exact poses -> COLMAP text model, no SfM)
#   3. flatten train+heldout imgs into ed/images
#   4. depth_exr_to_inria: EXR(cm) -> ed/depths/*.png (16-bit inv-depth) + sparse/0/depth_params.json
set -e
cd "$(dirname "$0")/.."
CAP="${CAP:-out/ed_editor_depth}"
DS=out/ed_depth_ds
ED="${ED:-out/ed_depth_train/ed}"

[ -f "$CAP/ue_poses.json" ] || { echo "no $CAP/ue_poses.json -- capture not done?"; exit 1; }
echo "=== ingest $CAP ==="
python3 -m splatkit.ingest --ue-poses "$CAP/ue_poses.json" --out "$DS"

echo "=== transforms_to_colmap -> $ED ==="
rm -rf "$ED"; mkdir -p "$ED/images" "$ED/sparse/0"
python3 scripts/transforms_to_colmap.py "$DS/transforms.json" "$ED/sparse/0"

echo "=== flatten images (train + heldout) ==="
python3 - "$ED" "$DS" <<'PY'
import os, sys, glob, shutil
ed, ds = sys.argv[1], sys.argv[2]
n = 0
for sub in ("train", "heldout_gt"):
    for f in glob.glob(os.path.join(ds, "images", sub, "*.png")):
        shutil.copy2(f, os.path.join(ed, "images", os.path.basename(f))); n += 1
names = [l.split()[-1] for l in open(os.path.join(ed, "sparse/0/images.txt"))
         if l.strip() and not l.startswith('#') and len(l.split()) >= 10]
miss = [x for x in names if not os.path.exists(os.path.join(ed, "images", x))]
print(f"flattened {n} imgs; colmap poses={len(names)} missing={len(miss)}")
assert not miss, f"MISSING e.g. {miss[:3]}"
PY

echo "=== GT depth EXR -> Inria depths/ + depth_params.json ==="
python3 scripts/depth_exr_to_inria.py "$CAP/ue_poses.json" "$ED"

echo "=== verify ==="
NI=$(ls "$ED/images" | wc -l | tr -d ' ')
ND=$(ls "$ED/depths" 2>/dev/null | wc -l | tr -d ' ')
HP=$([ -f "$ED/sparse/0/depth_params.json" ] && echo yes || echo NO)
echo "imgs=$NI depths=$ND depth_params=$HP"
[ "$NI" = "$ND" ] || echo "WARN: imgs != depths ($NI vs $ND) -- some frames lacked depth"
echo "DONE: $ED ready"
