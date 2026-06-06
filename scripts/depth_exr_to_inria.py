"""Convert in-editor GT metric-depth EXRs (UE_DEPTH=1 capture, SCS_SCENE_DEPTH, cm)
into the Inria 3DGS depth-regularization format: a 16-bit inverse-depth PNG per image
in <ds>/depths/ + <ds>/sparse/0/depth_params.json with per-image {scale, offset,
med_scale}.

Inria contract (verified against graphdeco-inria/gaussian-splatting main):
  load:  invdepthmap = png/65536            # [0,1]
         invdepthmap[invdepthmap < 0] = 0
         if scale>0: invdepthmap = invdepthmap*scale + offset
  gate:  depth_reliable=False (mask*0) if scale < 0.2*med_scale or scale > 5*med_scale
  loss:  |render_invDepth - invdepthmap| * depth_mask   (inverse depth, in 1/world-units)

The splat trains in METRES (ingest/transforms_to_colmap), so the rendered inverse depth
is 1/Z_m. UE SceneDepth is linear depth in cm, so our target inverse depth is 100/Z_cm
(= 1/Z_m). We store png = clip(invd/INVD_MAX,0,1)*65535 and set scale=INVD_MAX, offset=0
so the loader recovers invd ~exactly. med_scale=INVD_MAX => scale==med_scale passes the
gate for every image. Background/sky (Z<=0 or huge) -> invd 0 -> masked by the <0/zero.

  depth_exr_to_inria.py <ue_poses.json> <out_dataset_dir> [INVD_MAX=auto]

<out_dataset_dir> is the COLMAP dataset root (has images/ + sparse/0); depths/ is written
beside images/. Image stem (cam_000) is matched to its depth EXR via ue_poses frames.
"""
import json
import os
import sys

import numpy as np

try:
    import OpenEXR
    import Imath
except Exception as e:  # pragma: no cover
    sys.exit(f"need OpenEXR python bindings: pip install OpenEXR ({e})")


def read_exr_depth(path):
    """Return HxW float32 depth (UE cm) from an EXR. Picks the depth channel: prefer a
    channel literally named depth/Z, else R (SCS_SceneDepth writes depth into RGB)."""
    f = OpenEXR.InputFile(path)
    hdr = f.header()
    dw = hdr["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    chans = list(hdr["channels"].keys())
    pick = None
    for cand in ("Z", "depth", "Depth", "R"):
        if cand in chans:
            pick = cand
            break
    if pick is None:
        pick = chans[0]
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = f.channel(pick, FLOAT)
    f.close()
    return np.frombuffer(raw, dtype=np.float32).reshape(H, W).copy(), pick, chans


def main():
    poses_json, out_dir = sys.argv[1], sys.argv[2]
    forced = float(sys.argv[3]) if len(sys.argv) > 3 else None
    frames = json.load(open(poses_json))["frames"]
    frames = [fr for fr in frames if fr.get("depth_path") and os.path.exists(fr["depth_path"])]
    if not frames:
        sys.exit("no frames with an existing depth_path in " + poses_json)

    depths_dir = os.path.join(out_dir, "depths")
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    os.makedirs(depths_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # pass 1: read all, compute inverse-metric-depth, find a robust global INVD_MAX
    invds, stems, sample = [], [], None
    for fr in frames:
        z_cm, pick, chans = read_exr_depth(fr["depth_path"])
        if sample is None:
            sample = (z_cm, pick, chans)
        stem = os.path.splitext(os.path.basename(fr["file_path"]))[0]
        # Foreground = real geometry. Background (NOSKY) saturates to the 16f max (~65504cm)
        # or +inf -> treat depth >= 50000cm (500m, far beyond this ~90m scene) as background.
        fg = np.isfinite(z_cm) & (z_cm > 1.0) & (z_cm < 50000.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            invd = np.where(fg, 100.0 / z_cm, 0.0).astype(np.float32)  # 1/m metric inverse depth
        invd[~np.isfinite(invd)] = 0.0
        invds.append(invd)
        stems.append(stem)

    z0, pick0, chans0 = sample
    pos = np.concatenate([d[d > 0] for d in invds]) if any((d > 0).any() for d in invds) else np.array([0.0])
    # Robust INVD_MAX: p99.9 of inverse depth, but CAPPED so the implied min-depth >=
    # DEPTH_MIN_FLOOR_M (default 1m). A few cameras buried in foliage/against geometry see
    # surfaces at ~0.2m; without the cap those rare near pixels stretch the 16-bit range and
    # rob precision from the real 1-90m scene. Capping clamps only those (~2-3%) to png max.
    floor_m = float(os.environ.get("DEPTH_MIN_FLOOR_M", "1.0"))
    auto = float(np.percentile(pos, 99.9))
    invd_max = forced if forced else min(auto, 1.0 / floor_m)
    if invd_max <= 0:
        invd_max = 1.0
    print(f"channels={chans0} depth_channel={pick0}")
    print(f"sample Z(cm): min>{z0[z0>0].min() if (z0>0).any() else 0:.1f} "
          f"median {np.median(z0[z0>0]) if (z0>0).any() else 0:.1f} max {z0.max():.1f} "
          f"frac_bg(<=1cm) {float((z0<=1.0).mean()):.3f}")
    print(f"INVD_MAX (1/m, p99.9) = {invd_max:.5f}  -> min metric depth ~ {1.0/invd_max:.2f} m")

    # pass 2: write 16-bit PNGs + params
    import cv2
    params = {}
    for stem, invd in zip(stems, invds):
        png16 = np.clip(invd / invd_max, 0.0, 1.0) * 65535.0
        cv2.imwrite(os.path.join(depths_dir, stem + ".png"), png16.astype(np.uint16))
        params[stem] = {"scale": invd_max, "offset": 0.0, "med_scale": invd_max}
    json.dump(params, open(os.path.join(sparse_dir, "depth_params.json"), "w"))
    print(f"wrote {len(params)} depth PNGs -> {depths_dir}")
    print(f"wrote depth_params.json (scale=med_scale={invd_max:.5f}, offset=0) -> {sparse_dir}")


if __name__ == "__main__":
    main()
