#!/bin/bash
# Pod-side: COLMAP -> Inria 3D Gaussian Splatting on the uploaded frames.
# Logs staged markers to stdout (-> /workspace/run.log); fails fast.
set -uo pipefail
cd /workspace
export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.6"          # A5000 = Ampere sm_86
ITERS="${ITERS:-15000}"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "STAGE_APT"
apt-get update -qq && apt-get install -y -qq colmap imagemagick >apt.log 2>&1 \
  || { log APT_FAIL; tail -20 apt.log; exit 1; }
log "STAGE_CLONE"
[ -d gs ] || git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git gs >clone.log 2>&1 \
  || { log CLONE_FAIL; tail -10 clone.log; exit 1; }
log "STAGE_PIP"
pip install -q plyfile tqdm opencv-python-headless >pip.log 2>&1
pip install -q gs/submodules/diff-gaussian-rasterization gs/submodules/simple-knn >>pip.log 2>&1 \
  || { log PIP_FAIL; tail -30 pip.log; exit 1; }
log "STAGE_COLMAP imgs=$(ls ed/input 2>/dev/null | wc -l)"
python3 gs/convert.py -s /workspace/ed --no_gpu >colmap.log 2>&1 \
  || { log COLMAP_FAIL; tail -40 colmap.log; exit 1; }
log "COLMAP_REGISTERED imgs=$(($(wc -l < ed/sparse/0/images.txt 2>/dev/null || echo 8)/2 - 2))"
log "STAGE_TRAIN iters=$ITERS"
python3 gs/train.py -s /workspace/ed -m /workspace/ed/output \
  --iterations "$ITERS" --test_iterations "$ITERS" --save_iterations "$ITERS" >train.log 2>&1 \
  || { log TRAIN_FAIL; tail -40 train.log; exit 1; }
log "STAGE_RENDER"
python3 gs/render.py -m /workspace/ed/output --iteration "$ITERS" --skip_train >render.log 2>&1 \
  || log RENDER_WARN
PLY="ed/output/point_cloud/iteration_$ITERS/point_cloud.ply"
log "ALL_DONE ply_bytes=$(stat -c%s "$PLY" 2>/dev/null || echo MISSING) test_renders=$(ls ed/output/test/ours_$ITERS/renders 2>/dev/null | wc -l)"
