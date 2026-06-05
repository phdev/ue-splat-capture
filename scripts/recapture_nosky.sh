#!/bin/zsh
# Re-capture the full Electric Dreams coverage with the SKY/ATMOSPHERE OFF
# (UE_NOSKY=1) at 1536px + 16-sample temporal averaging, for a 3DGS-MCMC retrain
# that won't waste capacity (or floaters) on sky surfels. Three passes, run
# SEQUENTIALLY (the UE project lock allows only one headless editor at a time),
# each to its own out dir so they can be averaged + ingested + merged after.
#
#   DOME   (240)  spire hemisphere, 6 elevations x 40 az, radius 1800, focus z=spire
#   GROUND (108)  ground-level dome, 3 steep elevations x 36 az, radius 3000
#   GRID   ( 64)  8x8 nadir drone grid, +-26m extent, 16m height, focus z=ground
#
# All three share UE_HFOV=75 + UE_CAP_RES/UE_TRAIN_RES=1536 so the intrinsics
# match and merge_datasets.py will accept the union. EV=10 (validated daylight).
set -e
cd "$(dirname "$0")/.."

COMMON=(UE_NOSKY=1 UE_HFOV=75 UE_CAP_RES=1536 UE_TRAIN_RES=1536
        UE_CAPTURE_EV=10 UE_AVG_SAMPLES=16 UE_CAPS_PER_POSE=3)
FX=89287.5; FY=-5187.4          # scene centre (geometry centroid) in UE cm

run_pass () {  # $1=label  $2=outdir (repo-relative)  $3..=extra env
  local label="$1" out="$PWD/$2"; shift 2   # ABSOLUTE: UE chdir's to its engine
  echo "=== [$label] -> $out ==="           # binaries dir, so a relative path lands there
  env "${COMMON[@]}" "$@" UE_CAPTURE_OUT="$out" UE_HL_LOG="/tmp/${label}.log" \
      scripts/capture_headless_run.sh
  python3 - "$out" "$label" <<'PY'
import json,sys,glob,os
out,label=sys.argv[1],sys.argv[2]
pj=os.path.join(out,"ue_poses.json")
if not os.path.exists(pj): sys.exit(f"!! {label}: no ue_poses.json (capture failed)")
d=json.load(open(pj)); n=len(d["frames"])
ss=len(glob.glob(os.path.join(out,"images","*_00.png")))  # avg sample 0 per pose
print(f"OK {label}: {n} poses, {ss} have sample-00 (expect == poses)")
PY
}

run_pass ed_ns_dome   out/ed_ns_dome \
  UE_ELEVATIONS="8,22,36,50,64,76" UE_N_AZ=40 UE_ORBIT_RADIUS_CM=1800 \
  UE_FOCUS_CM="$FX,$FY,1849"

run_pass ed_ns_ground out/ed_ns_ground \
  UE_ELEVATIONS="28,44,60" UE_N_AZ=36 UE_ORBIT_RADIUS_CM=3000 \
  UE_FOCUS_CM="$FX,$FY,1300"

run_pass ed_ns_grid   out/ed_ns_grid \
  UE_GRID=1 UE_GRID_N=8 UE_GRID_EXTENT_M=26 UE_GRID_HEIGHT_M=16 UE_GRID_CONVERGE=0.25 \
  UE_FOCUS_CM="$FX,$FY,1150"

echo "=== ALL 3 PASSES DONE (dome 240 + ground 108 + grid 64 = 412) ==="
