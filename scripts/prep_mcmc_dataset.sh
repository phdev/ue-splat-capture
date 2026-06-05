#!/bin/zsh
# Post-capture -> COLMAP-layout dataset ready to upload for 3DGS-MCMC.
#   1. average the 16 per-pose samples of each pass (denoise)         -> cam_IDX.png
#   2. ingest each pass (UE poses -> OpenCV world-coord transforms.json + split imgs)
#   3. merge the 3 ingested passes (world coords, prefixes d0_/d1_/d2_)
#   4. transforms_to_colmap (our EXACT poses -> COLMAP text model, no SfM)
#   5. assemble out/ed_ns_mcmc/ed/{images/*.png, sparse/0/*.txt} for scp to the pod
# 3DGS-MCMC's --eval then holds out every 8th of the flattened image list.
set -e
cd "$(dirname "$0")/.."
PASSES=(ed_ns_dome ed_ns_ground ed_ns_grid)

for p in $PASSES; do
  echo "=== average $p ==="
  python3 scripts/average_samples.py "out/$p/images"
  echo "=== ingest $p ==="
  python3 -m splatkit.ingest --ue-poses "out/$p/ue_poses.json" --out "out/${p}_ds"
done

echo "=== merge -> out/ed_ns_merged_ds ==="
rm -rf out/ed_ns_merged_ds
python3 scripts/merge_datasets.py out/ed_ns_merged_ds out/ed_ns_dome_ds out/ed_ns_ground_ds out/ed_ns_grid_ds

echo "=== transforms_to_colmap ==="
ED=out/ed_ns_mcmc/ed
rm -rf "$ED"; mkdir -p "$ED/images" "$ED/sparse/0"
python3 scripts/transforms_to_colmap.py out/ed_ns_merged_ds/transforms.json "$ED/sparse/0"

echo "=== flatten images (train + heldout) into $ED/images ==="
# COLMAP images.txt references basenames (d{i}_cam_*.png); 3DGS reads <src>/images flat.
python3 - "$ED" <<'PY'
import os,sys,glob,shutil
ed=sys.argv[1]; src="out/ed_ns_merged_ds/images"
n=0
for sub in ("train","heldout_gt"):
    for f in glob.glob(os.path.join(src,sub,"*.png")):
        shutil.copy2(f, os.path.join(ed,"images",os.path.basename(f))); n+=1
print(f"flattened {n} images -> {ed}/images")
# sanity: every COLMAP image name must exist in images/
names=[l.split()[-1] for l in open(os.path.join(ed,"sparse/0/images.txt")) if l.strip() and not l.startswith('#') and len(l.split())>=10]
miss=[x for x in names if not os.path.exists(os.path.join(ed,"images",x))]
print(f"colmap poses={len(names)} images={n} missing={len(miss)}")
assert not miss, f"MISSING images for {len(miss)} poses e.g. {miss[:3]}"
PY
echo "=== DONE: $ED ready ($(ls $ED/images | wc -l | tr -d ' ') imgs) ==="
