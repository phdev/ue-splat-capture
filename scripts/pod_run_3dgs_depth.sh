#!/bin/bash
# Pod-side: VANILLA Inria 3DGS (graphdeco-inria/gaussian-splatting) WITH DEPTH
# REGULARIZATION on the uploaded exact-pose dataset. Same as pod_run_3dgs_exact.sh but
# adds `-d depths` (the GT metric inverse-depth maps from the in-editor UE_DEPTH capture,
# converted by scripts/depth_exr_to_inria.py -> ed/depths/*.png + ed/sparse/0/depth_params.json).
# Depth supervision constrains under-textured/under-observed GROUND that photometric-only
# 3DGS reconstructs in patches -> fewer gray holes. The main-branch rasterizer renders
# inverse depth (render_pkg["depth"]); the loss is depth_l1_weight * |invDepth - mono| * mask.
set -uo pipefail
cd /workspace
export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9+PTX}"
ITERS="${ITERS:-30000}"
GRAD_THRESH="${GRAD_THRESH:-0.00013}"
DENSIFY_UNTIL="${DENSIFY_UNTIL:-20000}"
DEPTH_W_INIT="${DEPTH_W_INIT:-1.0}"        # Inria defaults: init 1.0 -> final 0.01 (exp decay)
DEPTH_W_FINAL="${DEPTH_W_FINAL:-0.01}"
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
NDEP=$(ls ed/depths 2>/dev/null | wc -l)
HASPARAMS=$([ -f ed/sparse/0/depth_params.json ] && echo yes || echo NO)
log "DATASET imgs=$NIMG depths=$NDEP depth_params=$HASPARAMS iters=$ITERS (vanilla 3DGS + depth-reg)"
[ "$NIMG" -gt 0 ] || { log NO_IMAGES; exit 1; }
[ "$NDEP" -gt 0 ] || { log NO_DEPTHS; exit 1; }
[ "$HASPARAMS" = "yes" ] || { log NO_DEPTH_PARAMS; exit 1; }

log "STAGE_TRAIN (depth-reg: -d depths, w $DEPTH_W_INIT->$DEPTH_W_FINAL)"
python3 -u gsv/train.py -s /workspace/ed -m /workspace/ed/output_d -d depths --eval --data_device cpu \
  --densify_grad_threshold "$GRAD_THRESH" --densify_until_iter "$DENSIFY_UNTIL" \
  --depth_l1_weight_init "$DEPTH_W_INIT" --depth_l1_weight_final "$DEPTH_W_FINAL" \
  --iterations "$ITERS" --test_iterations 7000 15000 30000 "$ITERS" --save_iterations 15000 30000 "$ITERS" \
  >traind.log 2>&1
RC=$?
PLY="ed/output_d/point_cloud/iteration_$ITERS/point_cloud.ply"
# The /workspace network volume intermittently throws OSError [Errno 5] on the ply
# stream.close() AFTER the data is fully written (the file is complete on disk). Treat a
# nonzero RC as fatal ONLY if the ply is missing/empty; otherwise it's that benign close.
if [ ! -s "$PLY" ]; then log "TRAIN_FAIL (rc=$RC, no ply)"; tail -60 traind.log; exit 1; fi
[ "$RC" -ne 0 ] && log "TRAIN_RC=$RC but ply present ($(stat -c%s "$PLY") B) -- benign save-close I/O, continuing"
grep -iE "Evaluating|PSNR|Depth|\[ITER" traind.log | tail -12 || true

log "STAGE_RENDER+METRICS"
python3 -u gsv/render.py -m /workspace/ed/output_d --iteration "$ITERS" --skip_train >renderd.log 2>&1 || log RENDER_WARN
python3 -u gsv/metrics.py -m /workspace/ed/output_d >metricsd.log 2>&1 || log METRICS_WARN
cat /workspace/ed/output_d/results.json 2>/dev/null || true
PLY="ed/output_d/point_cloud/iteration_$ITERS/point_cloud.ply"
log "ALL_DONE ply_bytes=$(stat -c%s "$PLY" 2>/dev/null || echo MISSING)"
