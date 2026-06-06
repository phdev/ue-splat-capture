#!/bin/zsh
# Orchestrate DEPTH-SUPERVISED vanilla 3DGS on a provisioned pod: tar+scp the COLMAP
# dataset WITH depth maps (out/ed_depth_train/ed: images/ + depths/ + sparse/0 incl.
# depth_params.json), run pod_run_3dgs_depth.sh (-d depths), poll ALL_DONE, download the
# ply from output_d.
#   train_depth_remote.sh <ip> <port> [ITERS] [GRAD_THRESH] [DENSIFY_UNTIL]
set -euo pipefail
cd "$(dirname "$0")/.."
IP="${1:?ip}"; PORT="${2:?port}"; ITERS="${3:-30000}"; GT="${4:-0.00013}"; DU="${5:-20000}"
ED="${ED:-out/ed_depth_train/ed}"
KEY=~/.ssh/id_ed25519
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"$IP")
SCP(){ scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$@"; }

[ -d "$ED/depths" ] || { echo "no $ED/depths -- run prep_depth_dataset.sh first"; exit 1; }
echo "=== tar dataset (images + depths + sparse) ==="
( cd "$(dirname "$ED")" && COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/ed_depth.tar.gz ed )
echo "ed_depth.tar.gz: $(du -h /tmp/ed_depth.tar.gz | cut -f1)"
echo "=== upload ==="
"${SSH[@]}" 'mkdir -p /workspace'
SCP /tmp/ed_depth.tar.gz root@"$IP":/workspace/
SCP scripts/pod_run_3dgs_depth.sh root@"$IP":/workspace/
"${SSH[@]}" 'cd /workspace && tar --no-same-owner -xzf ed_depth.tar.gz && echo IMGS=$(ls ed/images|wc -l) DEPTHS=$(ls ed/depths|wc -l)'
echo "=== launch depth-reg train ==="
"${SSH[@]}" "cd /workspace && ITERS=$ITERS GRAD_THRESH=$GT DENSIFY_UNTIL=$DU nohup bash pod_run_3dgs_depth.sh > rund.log 2>&1 & echo STARTED pid \$!"
echo "=== poll ==="
for i in {1..360}; do
  sleep 20
  line=$("${SSH[@]}" 'tail -1 /workspace/rund.log' 2>/dev/null || true)
  echo "[$i] $line"
  "${SSH[@]}" 'grep -q ALL_DONE /workspace/rund.log' 2>/dev/null && { echo TRAIN_DONE; break; }
  "${SSH[@]}" 'grep -qE "_FAIL|Traceback|out of memory" /workspace/rund.log' 2>/dev/null && { echo TRAIN_FAIL; "${SSH[@]}" 'tail -40 /workspace/rund.log'; exit 1; }
done
echo "=== download ply + metrics ==="
mkdir -p out/ed_depth_vanilla
PLY="/workspace/ed/output_d/point_cloud/iteration_$ITERS/point_cloud.ply"
SCP root@"$IP":"$PLY" out/ed_depth_vanilla/depth_$ITERS.ply
SCP root@"$IP":/workspace/ed/output_d/results.json out/ed_depth_vanilla/results.json 2>/dev/null || true
echo "=== DONE -> out/ed_depth_vanilla/depth_$ITERS.ply ($(du -h out/ed_depth_vanilla/depth_$ITERS.ply 2>/dev/null|cut -f1)) ==="
cat out/ed_depth_vanilla/results.json 2>/dev/null || true
