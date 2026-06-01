#!/usr/bin/env bash
# Regenerate fixtures. Uses Unreal if UnrealEditor-Cmd is found AND $UE_PROJECT
# is set; otherwise falls back (skip-with-warning) to the numpy stand-in so the
# pipeline always has reproducible fixtures.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

UE_CMD="$(uv run python -c 'from ue_capture.detect import find_unreal_cmd; print(find_unreal_cmd() or "")' 2>/dev/null || true)"

if [ -n "$UE_CMD" ] && [ -n "${UE_PROJECT:-}" ]; then
  echo "[capture] Unreal: $UE_CMD"
  echo "[capture] project: $UE_PROJECT"
  OUT="out/ue_capture"
  mkdir -p "$OUT"
  "$UE_CMD" "$UE_PROJECT" -run=pythonscript \
    -script="$ROOT/ue_capture/run_capture.py" -- --out "$OUT"
  uv run python -m splatkit.ingest --ue-poses "$OUT/ue_poses.json" --out fixtures/selftest
  echo "[capture] ingested UE capture -> fixtures/selftest"
else
  if [ -z "$UE_CMD" ]; then
    echo "[capture] WARNING: UnrealEditor-Cmd not found."
  else
    echo "[capture] WARNING: UE found but UE_PROJECT not set (export UE_PROJECT=/path/to.uproject)."
  fi
  echo "[capture] Falling back to numpy stand-in renderer."
  uv run python -m selftest.make_fixtures --ss 2
fi
