#!/bin/zsh
# Orchestrate a 3DGS-MCMC run on an already-provisioned Runpod pod:
#   tar+scp the COLMAP-layout dataset (out/ed_ns_mcmc/ed) -> /workspace/ed
#   scp pod_run_3dgs_mcmc.sh, run it under nohup, tail run.log until ALL_DONE.
# Then download the trained ply + results.json. Does NOT create/delete the pod
# (use scripts/runpod_pod.py for that) so the caller controls pod lifecycle.
#
#   train_mcmc_remote.sh <ip> <port> [CAP_MAX] [ITERS]
set -euo pipefail
cd "$(dirname "$0")/.."
IP="${1:?ip}"; PORT="${2:?port}"; CAP_MAX="${3:-3000000}"; ITERS="${4:-30000}"
KEY=~/.ssh/id_ed25519
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"$IP")
SCP() { scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$@"; }

echo "=== tar dataset (no AppleDouble/xattrs) ==="
( cd out/ed_ns_mcmc && COPYFILE_DISABLE=1 tar --no-xattrs -czf /tmp/ed_mcmc.tar.gz ed )
echo "ed.tar.gz: $(du -h /tmp/ed_mcmc.tar.gz | cut -f1)"

echo "=== upload ==="
"${SSH[@]}" 'mkdir -p /workspace'
SCP /tmp/ed_mcmc.tar.gz root@"$IP":/workspace/
SCP scripts/pod_run_3dgs_mcmc.sh root@"$IP":/workspace/
"${SSH[@]}" 'cd /workspace && tar --no-same-owner -xzf ed_mcmc.tar.gz && ls ed && echo IMGS=$(ls ed/images|wc -l)'

echo "=== launch training (nohup; survives ssh drop) ==="
"${SSH[@]}" "cd /workspace && CAP_MAX=$CAP_MAX ITERS=$ITERS nohup bash pod_run_3dgs_mcmc.sh > run.log 2>&1 & echo STARTED pid \$!"

echo "=== poll run.log until ALL_DONE / FAIL ==="
for i in {1..240}; do            # up to ~80 min (20s * 240)
  sleep 20
  line=$("${SSH[@]}" 'tail -1 /workspace/run.log' 2>/dev/null || true)
  echo "[$i] $line"
  if "${SSH[@]}" 'grep -q ALL_DONE /workspace/run.log' 2>/dev/null; then echo "TRAIN_DONE"; break; fi
  if "${SSH[@]}" 'grep -qE "_FAIL|Traceback|CUDA out of memory" /workspace/run.log' 2>/dev/null; then
    echo "TRAIN_FAIL"; "${SSH[@]}" 'tail -40 /workspace/run.log'; exit 1; fi
done

echo "=== download ply + metrics ==="
mkdir -p out/ed_ns_mcmc/output
PLY="/workspace/ed/output/point_cloud/iteration_$ITERS/point_cloud.ply"
SCP root@"$IP":"$PLY" out/ed_ns_mcmc/mcmc_$ITERS.ply
SCP root@"$IP":/workspace/ed/output/results.json out/ed_ns_mcmc/results.json 2>/dev/null || true
"${SSH[@]}" 'grep -iE "Evaluating|PSNR|\[ITER" /workspace/train.log | tail -8' 2>/dev/null || true
echo "=== DONE -> out/ed_ns_mcmc/mcmc_$ITERS.ply ($(du -h out/ed_ns_mcmc/mcmc_$ITERS.ply 2>/dev/null|cut -f1)) ==="
cat out/ed_ns_mcmc/results.json 2>/dev/null || true
