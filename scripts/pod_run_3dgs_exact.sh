#!/bin/bash
# Pod-side: VANILLA Inria 3DGS (graphdeco-inria/gaussian-splatting) on the uploaded
# exact-pose dataset (ed/images + ed/sparse/0) — NO COLMAP. Unbounded densification
# fits hard foliage/detail better than MCMC's fixed cap, so it tends to score HIGHER
# held-out PSNR (brush — a 3DGS impl — hit 19.3 here, vs MCMC's 18.65) at the cost of
# more floaters (cleaned afterwards). python -u so evals stream live (no stdout buffering).
set -uo pipefail
cd /workspace
export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9+PTX}"
ITERS="${ITERS:-30000}"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "STAGE_CLONE"
[ -d gsv ] || git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git gsv >clonev.log 2>&1 \
  || { log CLONE_FAIL; tail -20 clonev.log; exit 1; }
log "STAGE_PIP"
pip install -q "numpy<2" plyfile tqdm opencv-python-headless >pipv.log 2>&1
pip install -q "numpy<2" >>pipv.log 2>&1
pip install -q gsv/submodules/diff-gaussian-rasterization gsv/submodules/simple-knn >>pipv.log 2>&1 \
  || { log PIP_FAIL; tail -40 pipv.log; exit 1; }

NIMG=$(ls ed/images 2>/dev/null | wc -l)
log "DATASET imgs=$NIMG iters=$ITERS (vanilla 3DGS, unbounded densify)"
[ "$NIMG" -gt 0 ] || { log NO_IMAGES; exit 1; }

log "STAGE_TRAIN"
python3 -u gsv/train.py -s /workspace/ed -m /workspace/ed/output_v --eval --data_device cpu \
  --iterations "$ITERS" --test_iterations 7000 15000 30000 "$ITERS" --save_iterations "$ITERS" \
  >trainv.log 2>&1 || { log TRAIN_FAIL; tail -60 trainv.log; exit 1; }
grep -iE "Evaluating|PSNR|\[ITER" trainv.log | tail -10 || true

log "STAGE_RENDER+METRICS"
python3 -u gsv/render.py -m /workspace/ed/output_v --iteration "$ITERS" --skip_train >renderv.log 2>&1 || log RENDER_WARN
python3 -u gsv/metrics.py -m /workspace/ed/output_v >metricsv.log 2>&1 || log METRICS_WARN
cat /workspace/ed/output_v/results.json 2>/dev/null || true
PLY="ed/output_v/point_cloud/iteration_$ITERS/point_cloud.ply"
log "ALL_DONE ply_bytes=$(stat -c%s "$PLY" 2>/dev/null || echo MISSING)"
