"""Image-only initial point cloud (no scene geometry; like SfM seeding for 3DGS).

Default method = multi-view PHOTO-CONSISTENCY ("voxel colouring"):
    A voxel that lies on a Lambertian surface projects to the SAME colour in
    every view that sees it (shading here is view-independent), so its projected
    colours have near-zero variance. A voxel floating in air projects to
    different scene points in different views (parallax) -> high variance. We
    keep low-variance voxels and colour them by their mean projection.

    This recovers both the textured ground and the objects, where silhouette
    space-carving fails (a full-frame ground plane leaves almost no background
    to carve against). Uses ONLY the training images + poses.

A silhouette `carve_point_cloud` is also provided for reference/testing.
"""
from __future__ import annotations

import numpy as np


def _voxel_grid(aabb_min, aabb_max, approx_vox_m=0.045):
    aabb_min = np.asarray(aabb_min, float)
    aabb_max = np.asarray(aabb_max, float)
    span = aabb_max - aabb_min
    res = np.maximum(np.round(span / approx_vox_m).astype(int) + 1, 2)
    # Inclusive grid: points land ON the AABB faces, crucially the ground plane
    # (z = aabb_min_z). A surface point reconstructs as photo-consistent; a point
    # floating just above it would see parallax and be rejected.
    axes = [np.linspace(aabb_min[i], aabb_max[i], res[i]) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    return centers, res


def _bilinear(img, u, v):
    """Bilinearly sample img (H,W,3) at float coords u (x), v (y). (N,3)."""
    H, W = img.shape[0], img.shape[1]
    x0 = np.floor(u).astype(int); y0 = np.floor(v).astype(int)
    x1 = x0 + 1; y1 = y0 + 1
    wx = (u - x0)[:, None]; wy = (v - y0)[:, None]
    x0 = np.clip(x0, 0, W - 1); x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1); y1 = np.clip(y1, 0, H - 1)
    c00 = img[y0, x0]; c10 = img[y0, x1]; c01 = img[y1, x0]; c11 = img[y1, x1]
    return (c00 * (1 - wx) * (1 - wy) + c10 * wx * (1 - wy)
            + c01 * (1 - wx) * wy + c11 * wx * wy)


def consistency_point_cloud(frames, aabb_min, aabb_max, approx_vox_m=0.05,
                            var_tol=0.02, min_views=4, near=0.2,
                            max_points=9000, seed=0, background=None,
                            bg_reject_tol=0.06):
    centers, res = _voxel_grid(aabb_min, aabb_max, approx_vox_m)
    V = centers.shape[0]
    n = np.zeros(V)
    csum = np.zeros((V, 3))
    csqsum = np.zeros((V, 3))

    for f in frames:
        R = f["R_w2c"].cpu().numpy(); t = f["t_w2c"].cpu().numpy()
        img = f["image"].cpu().numpy()
        H, W = img.shape[0], img.shape[1]
        Xc = centers @ R.T + t
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = f["fx"] * Xc[:, 0] / z + f["cx"]
            v = f["fy"] * Xc[:, 1] / z + f["cy"]
        inv = (z > near) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        if not inv.any():
            continue
        idx = np.nonzero(inv)[0]
        col = _bilinear(img, u[idx], v[idx])
        n[idx] += 1
        csum[idx] += col
        csqsum[idx] += col * col

    nn = np.maximum(n, 1)[:, None]
    mean = csum / nn
    var_ch = csqsum / nn - mean * mean
    var = np.where(n >= min_views, var_ch.mean(axis=1).clip(min=0), np.inf)

    # Reject empty space that merely SEES the background: a voxel whose mean
    # projected colour is ~background is air in front of the backdrop, not a
    # surface. Critical when the background is flat (e.g. black) -- otherwise
    # photo-consistency seeds a fog of background-coloured floaters.
    candidate = n >= min_views
    if background is not None:
        bg = np.asarray(background, float)
        is_fg = np.linalg.norm(mean - bg[None, :], axis=1) > bg_reject_tol
        candidate = candidate & is_fg

    # Rank voxels by photo-consistency (low variance = surface-like) and keep
    # the most consistent ones up to the budget. var_tol only trims obvious air
    # when there is slack under the budget.
    seen_ok = np.nonzero(candidate)[0]
    order = seen_ok[np.argsort(var[seen_ok])]
    if order.size > max_points:
        order = order[:max_points]
    # drop anything wildly inconsistent even if under budget
    order = order[var[order] < var_tol]
    idx = np.sort(order)

    pts = centers[idx].astype(np.float32)
    cols = np.clip(mean[idx], 0, 1).astype(np.float32)
    vox_m = float(np.mean((np.asarray(aabb_max) - np.asarray(aabb_min)) / res))
    return pts, cols, vox_m


# --------------------------------------------------------------------------- #
# Silhouette space carving (kept for reference / tests; weak when a ground
# plane fills the frame).
# --------------------------------------------------------------------------- #
def carve_point_cloud(frames, bg, aabb_min, aabb_max, approx_vox_m=0.06,
                      bg_tol=0.08, near=0.2, max_points=9000, seed=0):
    centers, res = _voxel_grid(aabb_min, aabb_max, approx_vox_m)
    V = centers.shape[0]
    bg = np.asarray(bg, float)
    carved = np.zeros(V, bool); seen = np.zeros(V, int)
    fg_count = np.zeros(V, int); color_accum = np.zeros((V, 3))
    for f in frames:
        R = f["R_w2c"].cpu().numpy(); t = f["t_w2c"].cpu().numpy()
        img = f["image"].cpu().numpy()
        H, W = img.shape[0], img.shape[1]
        Xc = centers @ R.T + t
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = f["fx"] * Xc[:, 0] / z + f["cx"]
            v = f["fy"] * Xc[:, 1] / z + f["cy"]
        in_view = (z > near) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = np.clip(np.round(u).astype(int), 0, W - 1)
        vi = np.clip(np.round(v).astype(int), 0, H - 1)
        pix = img[vi, ui]
        is_bg = np.linalg.norm(pix - bg[None, :], axis=1) < bg_tol
        carved |= in_view & is_bg
        fg = in_view & (~is_bg)
        seen += in_view; fg_count += fg; color_accum[fg] += pix[fg]
    keep = (~carved) & (fg_count > 0)
    pts = centers[keep].astype(np.float32)
    cols = (color_accum[keep] / np.maximum(fg_count[keep], 1)[:, None]).astype(np.float32)
    if pts.shape[0] > max_points:
        rng = np.random.RandomState(seed)
        sel = np.sort(rng.choice(pts.shape[0], max_points, replace=False))
        pts, cols = pts[sel], cols[sel]
    vox_m = float(np.mean((np.asarray(aabb_max) - np.asarray(aabb_min)) / res))
    return pts, cols, vox_m
