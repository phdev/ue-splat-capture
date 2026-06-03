#!/bin/bash
# Pipeline: 3DGS .ply  ->  compressed SOG  ->  static SuperSplat/PlayCanvas viewer
# site, ready to host on GitHub Pages. Uses PlayCanvas splat-transform (npx).
#
#   scripts/make_sog_viewer.sh <input.ply> <out_site_dir>
#
# Then publish (one time):
#   git -C <out_site_dir> init -b main && git -C <out_site_dir> add -A \
#     && git -C <out_site_dir> commit -m "splat viewer"
#   gh repo create <name> --public --source=<out_site_dir> --remote=origin --push
#   echo '{"source":{"branch":"main","path":"/"}}' \
#     | gh api -X POST repos/<owner>/<name>/pages --input -
#   # -> https://<owner>.github.io/<name>/
set -euo pipefail
PLY="${1:?usage: make_sog_viewer.sh <input.ply> <out_site_dir>}"
SITE="${2:?usage: make_sog_viewer.sh <input.ply> <out_site_dir>}"
SOG="${SITE%/}.sog"

echo "[1/2] $PLY -> $SOG  (filter NaN + k-means SH compression; ~16x smaller, full SH kept)"
npx -y @playcanvas/splat-transform "$PLY" -N "$SOG" -w

echo "[2/2] $SOG -> unbundled SuperSplat viewer at $SITE/"
rm -rf "$SITE"; mkdir -p "$SITE"
npx -y @playcanvas/splat-transform "$SOG" "$SITE/index.html" -U -w

echo "DONE -> $SITE/ (index.html + index.sog + index.js). Total: $(du -sh "$SITE" | cut -f1)"
echo "Serve locally: python3 -m http.server 8000 --directory '$SITE'  (needs HTTP, not file://)"
echo "Publish: see header of this script for the gh commands."
# NOTE: true multi-chunk streamed LOD (lod-meta.json) needs an LOD pyramid -- build
# decimated levels tagged with -l 0/1/2 and merge to lod-meta.json, or use SuperSplat's
# export dialog. Overkill below ~5M gaussians (a single SOG streams fine).
