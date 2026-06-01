#!/usr/bin/env bash
# Thin wrapper around the verify orchestrator. Args are forwarded
# (e.g. --skip-recon, --iters N, --n-gauss N, --seed N).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec uv run python -m splatkit.verify "$@"
