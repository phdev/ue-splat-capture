#!/bin/bash
# Local COLMAP (CPU, Mac) on captured frames -> a COLMAP dataset brush reads.
#   run_colmap_local.sh <images_dir> <work_dir>
# Produces <work_dir>/sparse/0 + <work_dir>/images (symlink). Single shared
# PINHOLE camera (our capture has fixed, distortion-free intrinsics).
set -euo pipefail
IMG="${1:?images dir}"; WORK="${2:?work dir}"
mkdir -p "$WORK/sparse"
DB="$WORK/database.db"; rm -f "$DB"
echo "[colmap] feature_extractor ($(ls "$IMG"/*.png | wc -l | tr -d ' ') imgs)"
colmap feature_extractor --database_path "$DB" --image_path "$IMG" \
  --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE \
  --FeatureExtraction.use_gpu 0 >"$WORK/feat.log" 2>&1
echo "[colmap] exhaustive_matcher (CPU; the slow step)"
colmap exhaustive_matcher --database_path "$DB" --FeatureMatching.use_gpu 0 >"$WORK/match.log" 2>&1
echo "[colmap] mapper"
colmap mapper --database_path "$DB" --image_path "$IMG" --output_path "$WORK/sparse" >"$WORK/map.log" 2>&1
ln -sfn "$IMG" "$WORK/images"
REG=$(ls "$WORK/sparse/0/"*.bin 2>/dev/null | wc -l | tr -d ' ')
echo "COLMAP_DONE model_files=$REG  (images/ + sparse/0 ready for brush at $WORK)"
