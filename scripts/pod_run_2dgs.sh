#!/bin/bash
# Pod-side: COLMAP -> 2D Gaussian Splatting (surfels -> cohesive surfaces) on the
# uploaded frames. Same flow as pod_run_3dgs.sh but the 2DGS trainer. Logs staged
# markers to stdout (-> /workspace/run.log); fails fast.
set -uo pipefail
cd /workspace
export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.6"          # A5000 = Ampere sm_86
export QT_QPA_PLATFORM=offscreen           # apt colmap is the Qt-GUI build; headless needs this
ITERS="${ITERS:-30000}"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "STAGE_APT"
apt-get update -qq && apt-get install -y -qq colmap imagemagick >apt.log 2>&1 \
  || { log APT_FAIL; tail -20 apt.log; exit 1; }
log "STAGE_CLONE"
[ -d gs ] || git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git gs >clone.log 2>&1 \
  || { log CLONE_FAIL; tail -10 clone.log; exit 1; }
log "STAGE_PIP"
pip install -q "numpy<2" plyfile tqdm opencv-python-headless matplotlib >pip.log 2>&1
pip install -q gs/submodules/diff-surfel-rasterization gs/submodules/simple-knn >>pip.log 2>&1 \
  || { log PIP_FAIL; tail -30 pip.log; exit 1; }
log "STAGE_COLMAP imgs=$(ls ed/input 2>/dev/null | wc -l)  (GPU exhaustive; the slow step for 412 imgs)"
python3 gs/convert.py -s /workspace/ed --no_gpu >colmap.log 2>&1 \
  || { log COLMAP_FAIL; tail -40 colmap.log; exit 1; }
REG=$(($(wc -l < ed/sparse/0/images.txt 2>/dev/null || echo 8)/2 - 2))
log "COLMAP_REGISTERED imgs=$REG of $(ls ed/input | wc -l)"
log "STAGE_TRAIN iters=$ITERS (2DGS surfels; lambda_dist/normal regularize to surfaces)"
python3 gs/train.py -s /workspace/ed -m /workspace/ed/output \
  --iterations "$ITERS" --test_iterations "$ITERS" --save_iterations "$ITERS" \
  --depth_ratio 1.0 --lambda_dist 100 >train.log 2>&1 \
  || { log TRAIN_FAIL; tail -40 train.log; exit 1; }
PLY="ed/output/point_cloud/iteration_$ITERS/point_cloud.ply"
log "ALL_DONE ply_bytes=$(stat -c%s "$PLY" 2>/dev/null || echo MISSING)"
