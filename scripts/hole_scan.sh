#!/bin/bash
# Bg-flip hole scan — THE deploy gate for transparency. Renders every pose in
# out/site/q31scan/ (42 spire poses x magenta/green background) for one sog and
# screenshots them; scripts/analyze_holes.py then counts interior changed pixels
# (= see-through). Border-connected changed region = true background (excluded).
# Usage: scripts/hole_scan.sh <sog-in-out/site> <outdir>   (serve out/site on :8099)
SOG=$1; OUT=$2
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
B="$HOME/.claude/skills/gstack/browse/dist/browse"
"$B" --headed viewport 700x700 >/dev/null 2>&1
for f in out/site/q31scan/*.json; do
  name=$(basename "$f" .json)
  [ -s "$OUT/$name.png" ] && continue
  TS=$(date +%s%N)
  "$B" --headed goto "http://localhost:8099/index.html?content=$SOG&settings=q31scan/$name.json&noui&noanim&cb=$TS" >/dev/null 2>&1
  "$B" --headed wait --networkidle >/dev/null 2>&1
  "$B" --headed js "new Promise(r=>setTimeout(r,2200))" >/dev/null 2>&1
  "$B" --headed screenshot "$OUT/$name.png" >/dev/null 2>&1
done
ls "$OUT" | grep -c png
echo "SCAN_SHOTS_DONE $OUT"
