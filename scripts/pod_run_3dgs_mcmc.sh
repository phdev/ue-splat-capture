#!/bin/bash
# Pod-side: 3DGS-MCMC (ubc-vision/3dgs-mcmc — "Gaussian Splatting as Markov Chain
# Monte Carlo") on the uploaded frames. MCMC keeps a FIXED budget of gaussians
# (--cap_max) and relocates "dead" ones + injects noise, giving HIGHER PSNR/SSIM
# and far fewer floaters than vanilla 3DGS densification.
#
# We SKIP COLMAP: the dataset is uploaded already in COLMAP layout
#   ed/images/*.png            (the averaged, sky-off frames)
#   ed/sparse/0/{cameras,images,points3D}.txt   (our EXACT UE poses, via
#                                                 scripts/transforms_to_colmap.py)
# so there is no SfM step. --eval holds out every 8th image (llffhold) for a fair,
# standard PSNR/SSIM/LPIPS number directly comparable to the brush 19.3 held-out.
#
# Logs staged markers to stdout (-> /workspace/run.log); fails fast.
set -uo pipefail
cd /workspace
export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9+PTX}"  # A100/Ampere/Ada — multi-arch so the rasterizer runs on whatever card we land on
ITERS="${ITERS:-30000}"
CAP_MAX="${CAP_MAX:-3000000}"            # fixed gaussian budget (MCMC target count)
SCALE_REG="${SCALE_REG:-0.01}"
OPACITY_REG="${OPACITY_REG:-0.01}"
NOISE_LR="${NOISE_LR:-5e5}"
INIT_TYPE="${INIT_TYPE:-random}"         # MCMC works great from random; our points3D is a random cloud anyway
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "STAGE_CLONE"
[ -d gs ] || git clone --recursive https://github.com/ubc-vision/3dgs-mcmc.git gs >clone.log 2>&1 \
  || { log CLONE_FAIL; tail -20 clone.log; exit 1; }
log "STAGE_PIP"
pip install -q "numpy<2" plyfile tqdm opencv-python-headless >pip.log 2>&1
pip install -q "numpy<2" >>pip.log 2>&1   # opencv pulls numpy>=2 back; torch 2.1 ABI needs <2 (else "Numpy is not available" at image load)
# MCMC ships its own diff-gaussian-rasterization fork (adds compute_relocation) + simple-knn
pip install -q gs/submodules/diff-gaussian-rasterization gs/submodules/simple-knn >>pip.log 2>&1 \
  || { log PIP_FAIL; tail -40 pip.log; exit 1; }
# fused-ssim speeds training if present; ignore if the submodule isn't there
[ -d gs/submodules/fused-ssim ] && pip install -q gs/submodules/fused-ssim >>pip.log 2>&1 || true

NIMG=$(ls ed/images 2>/dev/null | wc -l)
REG=$(($(wc -l < ed/sparse/0/images.txt 2>/dev/null || echo 8)/2 - 2))
log "DATASET imgs=$NIMG colmap_poses=$REG cap_max=$CAP_MAX init=$INIT_TYPE iters=$ITERS"
[ "$NIMG" -gt 0 ] || { log NO_IMAGES; exit 1; }

log "STAGE_TRAIN (MCMC: cap_max=$CAP_MAX scale_reg=$SCALE_REG opacity_reg=$OPACITY_REG noise_lr=$NOISE_LR)"
python3 gs/train.py -s /workspace/ed -m /workspace/ed/output --eval --data_device cpu \
  --iterations "$ITERS" --test_iterations 7000 15000 "$ITERS" --save_iterations "$ITERS" \
  --cap_max "$CAP_MAX" --scale_reg "$SCALE_REG" --opacity_reg "$OPACITY_REG" \
  --noise_lr "$NOISE_LR" --init_type "$INIT_TYPE" >train.log 2>&1 \
  || { log TRAIN_FAIL; tail -60 train.log; exit 1; }
# echo the eval rows the trainer prints (PSNR at test iterations)
grep -iE "Evaluating|PSNR|SSIM|L1" train.log | tail -12 || true

log "STAGE_RENDER+METRICS"
python3 gs/render.py -m /workspace/ed/output --iteration "$ITERS" --skip_train >render.log 2>&1 || log RENDER_WARN
python3 gs/metrics.py -m /workspace/ed/output >metrics.log 2>&1 || log METRICS_WARN
cat /workspace/ed/output/results.json 2>/dev/null || true

PLY="ed/output/point_cloud/iteration_$ITERS/point_cloud.ply"
log "ALL_DONE ply_bytes=$(stat -c%s "$PLY" 2>/dev/null || echo MISSING) gaussians_cap=$CAP_MAX"
