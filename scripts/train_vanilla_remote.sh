#!/bin/zsh
# Orchestrate VANILLA Inria 3DGS on a provisioned pod: tar+scp the COLMAP dataset
# (out/ed_ns_mcmc/ed, a symlink to the real ed -> use tar -h), run pod_run_3dgs_exact.sh
# with aggressive densification, poll until ALL_DONE, download the ply from output_v.
#   train_vanilla_remote.sh <ip> <port> [ITERS] [GRAD_THRESH] [DENSIFY_UNTIL]
set -euo pipefail
cd "$(dirname "$0")/.."
IP="${1:?ip}"; PORT="${2:?port}"; ITERS="${3:-30000}"; GT="${4:-0.00013}"; DU="${5:-20000}"
KEY=~/.ssh/id_ed25519
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"$IP")
SCP(){ scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$@"; }

echo "=== tar dataset (-h follows the ed symlink) ==="
( cd out/ed_ns_mcmc && COPYFILE_DISABLE=1 tar -h --no-xattrs -czf /tmp/ed_vanilla.tar.gz ed )
echo "ed.tar.gz: $(du -h /tmp/ed_vanilla.tar.gz | cut -f1)"
echo "=== upload ==="
"${SSH[@]}" 'mkdir -p /workspace'
SCP /tmp/ed_vanilla.tar.gz root@"$IP":/workspace/
SCP scripts/pod_run_3dgs_exact.sh root@"$IP":/workspace/
"${SSH[@]}" 'cd /workspace && tar --no-same-owner -xzf ed_vanilla.tar.gz && echo IMGS=$(ls ed/images|wc -l)'
echo "=== launch vanilla train (aggressive densify) ==="
"${SSH[@]}" "cd /workspace && ITERS=$ITERS GRAD_THRESH=$GT DENSIFY_UNTIL=$DU nohup bash pod_run_3dgs_exact.sh > runv.log 2>&1 & echo STARTED pid \$!"
echo "=== poll ==="
for i in {1..360}; do
  sleep 20
  line=$("${SSH[@]}" 'tail -1 /workspace/runv.log' 2>/dev/null || true)
  echo "[$i] $line"
  "${SSH[@]}" 'grep -q ALL_DONE /workspace/runv.log' 2>/dev/null && { echo TRAIN_DONE; break; }
  "${SSH[@]}" 'grep -qE "_FAIL|Traceback|out of memory" /workspace/runv.log' 2>/dev/null && { echo TRAIN_FAIL; "${SSH[@]}" 'tail -40 /workspace/runv.log'; exit 1; }
done
echo "=== download ply + metrics ==="
mkdir -p out/ed_full2_vanilla
PLY="/workspace/ed/output_v/point_cloud/iteration_$ITERS/point_cloud.ply"
SCP root@"$IP":"$PLY" out/ed_full2_vanilla/vanilla_$ITERS.ply
SCP root@"$IP":/workspace/ed/output_v/results.json out/ed_full2_vanilla/results.json 2>/dev/null || true
echo "=== DONE -> out/ed_full2_vanilla/vanilla_$ITERS.ply ($(du -h out/ed_full2_vanilla/vanilla_$ITERS.ply 2>/dev/null|cut -f1)) ==="
cat out/ed_full2_vanilla/results.json 2>/dev/null || true
