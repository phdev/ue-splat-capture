"""Self-contained differentiable 3D Gaussian splatting (torch, MPS/CPU).

This is the Apple-Silicon trainer backend (msplat is CUDA-only). It implements
EWA-style splatting with a global per-image depth sort and a vectorized
alpha-compositing via cumulative product, so PyTorch autograd supplies the
backward pass -- no custom CUDA/Metal kernels.

Colour is VIEW-DEPENDENT via spherical harmonics (default degree 3): each
gaussian carries an SH DC term + higher-order coefficients, evaluated at the
per-gaussian viewing direction. SH degree 0 (DC only) is the view-independent
special case; the higher orders let the splat model mild view-dependence
(specular highlights, tone-curve shifts) and -- crucially -- GENERALISE it to
held-out views rather than baking per-training-view appearance.

All randomness is seeded; given a fixed seed and CPU device the result is
deterministic.
"""
from __future__ import annotations

import os

import numpy as np
import torch

NEAR = 0.2          # metres; cull gaussians closer than this
SCREEN_MARGIN = 0.25  # fraction of image size to keep off-screen gaussians

# Real spherical-harmonics constants (standard 3DGS convention).
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435)


def sh_coeffs(degree: int) -> int:
    return (degree + 1) ** 2


def eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate SH. sh: (M,K,3) coefficients, dirs: (M,3) unit view dirs.
    Returns (M,3) (before the +0.5 offset)."""
    result = SH_C0 * sh[:, 0]
    if degree >= 1:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
        if degree >= 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result
                      + SH_C2[0] * xy * sh[:, 4]
                      + SH_C2[1] * yz * sh[:, 5]
                      + SH_C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
                      + SH_C2[3] * xz * sh[:, 7]
                      + SH_C2[4] * (xx - yy) * sh[:, 8])
            if degree >= 3:
                result = (result
                          + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9]
                          + SH_C3[1] * xy * z * sh[:, 10]
                          + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
                          + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
                          + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
                          + SH_C3[5] * z * (xx - yy) * sh[:, 14]
                          + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15])
    return result


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-9)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def _color_to_dc(colors: torch.Tensor) -> torch.Tensor:
    """Inverse of the SH render activation for degree 0: dc s.t. SH_C0*dc+0.5=col."""
    return (colors - 0.5) / SH_C0


class GaussianModel:
    def __init__(self, means, sh_dc, sh_rest, log_scales, quats, logit_opacity,
                 device, sh_degree=3):
        self.device = device
        self.sh_degree = sh_degree
        self.means = means.to(device).requires_grad_(True)
        self.sh_dc = sh_dc.to(device).requires_grad_(True)       # (N,3) DC term
        self.sh_rest = sh_rest.to(device).requires_grad_(True)   # (N,K-1,3)
        self.log_scales = log_scales.to(device).requires_grad_(True)  # -> exp
        self.quats = quats.to(device).requires_grad_(True)
        self.logit_opacity = logit_opacity.to(device).requires_grad_(True)

    @property
    def n(self):
        return self.means.shape[0]

    def opacity(self):
        return torch.sigmoid(self.logit_opacity)

    def scales(self):
        return torch.exp(self.log_scales)

    def sh(self):
        return torch.cat([self.sh_dc[:, None, :], self.sh_rest], dim=1)  # (N,K,3)

    def colors_view(self, idx, view_dirs):
        """View-dependent RGB for gaussians `idx` seen along `view_dirs` (M,3)."""
        sh = torch.cat([self.sh_dc[idx][:, None, :], self.sh_rest[idx]], dim=1)
        return torch.clamp(eval_sh(self.sh_degree, sh, view_dirs) + 0.5, 0.0, 1.0)

    def parameters(self):
        return [self.means, self.sh_dc, self.sh_rest, self.log_scales,
                self.quats, self.logit_opacity]

    def param_groups(self, lr):
        return [
            {"params": [self.means], "lr": lr["means"]},
            {"params": [self.sh_dc], "lr": lr["color"]},
            {"params": [self.sh_rest], "lr": lr["color"] * 0.05},  # higher orders slower
            {"params": [self.log_scales], "lr": lr["scale"]},
            {"params": [self.quats], "lr": lr["quat"]},
            {"params": [self.logit_opacity], "lr": lr["opacity"]},
        ]

    def state(self):
        st = {k: getattr(self, k).detach().cpu()
              for k in ("means", "sh_dc", "sh_rest", "log_scales", "quats",
                        "logit_opacity")}
        st["sh_degree"] = self.sh_degree
        return st

    @classmethod
    def from_state(cls, st, device):
        return cls(st["means"], st["sh_dc"], st["sh_rest"], st["log_scales"],
                   st["quats"], st["logit_opacity"], device,
                   sh_degree=int(st.get("sh_degree", 3)))

    @classmethod
    def from_points(cls, points, colors, init_scale, device,
                    init_opacity=0.25, sh_degree=3):
        pts = torch.as_tensor(points, dtype=torch.float32)
        cols = torch.as_tensor(colors, dtype=torch.float32).clamp(0.0, 1.0)
        n = pts.shape[0]
        sh_dc = _color_to_dc(cols)
        sh_rest = torch.zeros(n, sh_coeffs(sh_degree) - 1, 3)
        log_scales = torch.full((n, 3), float(np.log(init_scale)))
        quats = torch.zeros(n, 4); quats[:, 0] = 1.0
        logit_opacity = torch.full((n,), float(np.log(init_opacity / (1 - init_opacity))))
        return cls(pts, sh_dc, sh_rest, log_scales, quats, logit_opacity, device,
                   sh_degree=sh_degree)


@torch.no_grad()
def clone_split_prune(model: "GaussianModel", grad_avg: torch.Tensor,
                      densify_frac=0.10, scale_split=0.06, min_opacity=0.01,
                      max_scale=0.6, max_points=22000, jitter=0.6, gen=None):
    """Adaptive densification + pruning (deterministic given `gen`).

    * prune gaussians with opacity < min_opacity or scale > max_scale (floaters)
    * among survivors, take the top `densify_frac` by accumulated mean-gradient:
        - small ones are CLONED (grow coverage)
        - large ones are SPLIT into two smaller children (add detail)
    Returns a new GaussianModel (caller must rebuild the optimizer).
    """
    dev = model.device
    means = model.means.detach(); logsc = model.log_scales.detach()
    quats = model.quats.detach(); logop = model.logit_opacity.detach()
    sdc = model.sh_dc.detach(); srest = model.sh_rest.detach()
    scales = torch.exp(logsc); opac = torch.sigmoid(logop)

    keep = (opac > min_opacity) & (scales.max(dim=1).values < max_scale)
    M, LS, Q, LO = means[keep], logsc[keep], quats[keep], logop[keep]
    SDC, SR = sdc[keep], srest[keep]
    S = scales[keep]
    ga = grad_avg[keep]
    maxs = S.max(dim=1).values

    k = max(int(densify_frac * ga.numel()), 1)
    thresh = torch.topk(ga.cpu(), min(k, ga.numel())).values.min().to(dev)
    sel = ga >= thresh
    clone = sel & (maxs <= scale_split)
    split = sel & (maxs > scale_split)

    def jit(idx, scl):
        noise = torch.randn(idx.numel(), 3, generator=gen).to(dev) * jitter
        return M[idx] + noise * scl[idx]

    ci = clone.nonzero(as_tuple=False).squeeze(1)
    si = split.nonzero(as_tuple=False).squeeze(1)
    log16 = float(np.log(1.6))

    pm = [M[~split]]; pls = [LS[~split]]; pq = [Q[~split]]
    plo = [LO[~split]]; pdc = [SDC[~split]]; pr = [SR[~split]]
    if ci.numel():
        pm.append(jit(ci, S)); pls.append(LS[ci]); pq.append(Q[ci])
        plo.append(LO[ci]); pdc.append(SDC[ci]); pr.append(SR[ci])
    for _ in range(2):  # two children per split
        if si.numel():
            pm.append(jit(si, S)); pls.append(LS[si] - log16); pq.append(Q[si])
            plo.append(LO[si]); pdc.append(SDC[si]); pr.append(SR[si])

    nm = torch.cat(pm); nls = torch.cat(pls); nq = torch.cat(pq)
    nlo = torch.cat(plo); ndc = torch.cat(pdc); nr = torch.cat(pr)

    if nm.shape[0] > max_points:                       # cap: keep most opaque
        order = torch.argsort(torch.sigmoid(nlo), descending=True)[:max_points]
        nm, nls, nq = nm[order], nls[order], nq[order]
        nlo, ndc, nr = nlo[order], ndc[order], nr[order]
    return GaussianModel(nm, ndc, nr, nls, nq, nlo, dev, sh_degree=model.sh_degree)


def _covariance3d(scales, R):
    S = torch.diag_embed(scales)            # (M,3,3)
    M = R @ S                                # (M,3,3)
    return M @ M.transpose(1, 2)             # R S S^T R^T


def render(model: GaussianModel, cam: dict, bg: torch.Tensor,
           blur: float = 0.12) -> torch.Tensor:
    """Render one camera. cam: R_w2c(3,3), t_w2c(3,), fx,fy,cx,cy,W,H tensors/floats.
    Returns image (H, W, 3) with autograd attached."""
    blur = float(os.environ.get("SPLAT_BLUR", blur))   # 2D antialias floor (px^2)
    device = model.device
    R_w2c = cam["R_w2c"].to(device)
    t_w2c = cam["t_w2c"].to(device)
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H = int(cam["W"]), int(cam["H"])

    mu_c = model.means @ R_w2c.T + t_w2c          # (N,3)
    z = mu_c[:, 2]
    u = fx * (mu_c[:, 0] / z) + cx
    v = fy * (mu_c[:, 1] / z) + cy
    mx = SCREEN_MARGIN * W
    my = SCREEN_MARGIN * H
    keep = (z > NEAR) & (u > -mx) & (u < W + mx) & (v > -my) & (v < H + my)
    if keep.sum() == 0:
        return bg.expand(H, W, 3).clone()

    idx = keep.nonzero(as_tuple=False).squeeze(1)
    mu_c = mu_c[idx]
    z = z[idx]
    mu2d = torch.stack([u[idx], v[idx]], dim=1)             # (M,2)
    scales = model.scales()[idx]
    R = quat_to_rotmat(model.quats[idx])
    opacity = model.opacity()[idx]

    # view-dependent colour: SH evaluated at the camera->gaussian direction
    cam_center = -(R_w2c.T @ t_w2c)                         # world-space eye
    view_dirs = model.means[idx] - cam_center
    view_dirs = view_dirs / (view_dirs.norm(dim=1, keepdim=True) + 1e-9)
    rgb = model.colors_view(idx, view_dirs)                 # (M,3)

    Sigma = _covariance3d(scales, R)                        # (M,3,3)
    zc = z
    J = torch.zeros(mu_c.shape[0], 2, 3, device=device, dtype=mu_c.dtype)
    J[:, 0, 0] = fx / zc
    J[:, 1, 1] = fy / zc
    J[:, 0, 2] = -fx * mu_c[:, 0] / (zc * zc)
    J[:, 1, 2] = -fy * mu_c[:, 1] / (zc * zc)
    JW = J @ R_w2c                                          # (M,2,3)
    cov2d = JW @ Sigma @ JW.transpose(1, 2)                 # (M,2,2)
    cov2d = cov2d + blur * torch.eye(2, device=device, dtype=cov2d.dtype)

    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]
    det = (a * c - b * b).clamp_min(1e-9)
    ca = c / det
    cb = -b / det
    cc = a / det

    # depth sort (front to back); ordering detached from autograd.
    order = torch.argsort(z)
    mu2d = mu2d[order]
    ca, cb, cc = ca[order], cb[order], cc[order]
    opacity = opacity[order]
    rgb = rgb[order]

    # pixel grid (P = H*W), row-major
    ys = torch.arange(H, device=device, dtype=mu_c.dtype) + 0.5
    xs = torch.arange(W, device=device, dtype=mu_c.dtype) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    px = gx.reshape(-1)
    py = gy.reshape(-1)                                     # (P,)

    dx = px[None, :] - mu2d[:, 0:1]                         # (M,P)
    dy = py[None, :] - mu2d[:, 1:2]
    power = -0.5 * (ca[:, None] * dx * dx
                    + 2 * cb[:, None] * dx * dy
                    + cc[:, None] * dy * dy)
    alpha = (opacity[:, None] * torch.exp(power)).clamp(0.0, 0.99)

    one_minus = 1.0 - alpha
    T = torch.cumprod(one_minus, dim=0)
    T_excl = torch.cat([torch.ones(1, T.shape[1], device=device, dtype=T.dtype),
                        T[:-1]], dim=0)
    weight = alpha * T_excl                                 # (M,P)
    color = weight.T @ rgb                                  # (P,3)
    acc = weight.sum(0)                                     # (P,)
    out = color + (1.0 - acc)[:, None] * bg[None, :]
    return out.reshape(H, W, 3)


TILE_SIZE = 16
# Safety ceiling on gaussians composited per tile -- NOT a routine cap. Cost and
# memory scale with the DENSEST tile's occupancy (the padded work tensor is sized
# to it), so this only clamps pathological tiles. Densification clusters hundreds
# of gaussians into a single tile, so a low value (e.g. 384) silently DROPS the
# farthest ones there and wrecks parity (27 dB); 2048 covers normal scenes with
# imperceptible (~42 dB) parity while staying ~5x faster than the global renderer.
# Tune via SPLAT_MAX_PER_TILE (higher = exact but slower/more memory).
MAX_PER_TILE = 2048


def render_tiled(model: GaussianModel, cam: dict, bg: torch.Tensor,
                 blur: float = 0.12, tile_size: int = TILE_SIZE,
                 max_per_tile: int = MAX_PER_TILE) -> torch.Tensor:
    """Tile-based rasterizer (the real 3DGS scheme) -- a faster, more faithful
    replacement for `render`'s global O(M*P) pass.

    Projection / EWA / SH / conic are IDENTICAL to `render`. The difference is
    compositing: each gaussian is binned into the screen tiles its 3-sigma
    footprint covers; every tile is sorted and composited independently over only
    its own pixels; the tiles are stitched back. Cost ~ sum over tiles of
    (gaussians in tile * tile pixels) instead of (every gaussian * every pixel),
    and ordering is per-tile (correct) rather than one global order for the whole
    frame. Differences vs `render` are only the standard 3DGS approximations (the
    3-sigma cutoff and the per-tile cap), so the two images match to within a few
    hundredths of a dB. Returns (H, W, 3) with autograd attached.
    """
    blur = float(os.environ.get("SPLAT_BLUR", blur))
    ts = int(os.environ.get("SPLAT_TILE", tile_size))
    max_per_tile = int(os.environ.get("SPLAT_MAX_PER_TILE", max_per_tile))
    device = model.device
    dt = model.means.dtype
    R_w2c = cam["R_w2c"].to(device)
    t_w2c = cam["t_w2c"].to(device)
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    W, H = int(cam["W"]), int(cam["H"])

    # ---- project + cull (identical to render) ----
    mu_c = model.means @ R_w2c.T + t_w2c
    z = mu_c[:, 2]
    u = fx * (mu_c[:, 0] / z) + cx
    v = fy * (mu_c[:, 1] / z) + cy
    mx = SCREEN_MARGIN * W
    my = SCREEN_MARGIN * H
    keep = (z > NEAR) & (u > -mx) & (u < W + mx) & (v > -my) & (v < H + my)
    if keep.sum() == 0:
        return bg.expand(H, W, 3).clone()
    idx = keep.nonzero(as_tuple=False).squeeze(1)
    mu_c = mu_c[idx]
    z = z[idx]
    mu2d = torch.stack([u[idx], v[idx]], dim=1)             # (M,2)
    scales = model.scales()[idx]
    Rm = quat_to_rotmat(model.quats[idx])
    opacity = model.opacity()[idx]                          # (M,)
    cam_center = -(R_w2c.T @ t_w2c)
    view_dirs = model.means[idx] - cam_center
    view_dirs = view_dirs / (view_dirs.norm(dim=1, keepdim=True) + 1e-9)
    rgb = model.colors_view(idx, view_dirs)                 # (M,3)

    Sigma = _covariance3d(scales, Rm)
    J = torch.zeros(mu_c.shape[0], 2, 3, device=device, dtype=dt)
    J[:, 0, 0] = fx / z
    J[:, 1, 1] = fy / z
    J[:, 0, 2] = -fx * mu_c[:, 0] / (z * z)
    J[:, 1, 2] = -fy * mu_c[:, 1] / (z * z)
    JW = J @ R_w2c
    cov2d = JW @ Sigma @ JW.transpose(1, 2)
    cov2d = cov2d + blur * torch.eye(2, device=device, dtype=dt)
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]
    det = (a * c - b * b).clamp_min(1e-9)
    ca = c / det
    cb = -b / det
    cc = a / det                                            # conic (inverse cov2d)
    M = mu2d.shape[0]

    # ---- bin gaussians into tiles (integer book-keeping; no autograd) ----
    with torch.no_grad():
        mid = 0.5 * (a + c)
        disc = torch.sqrt((0.5 * (a - c)) ** 2 + b * b).clamp_min(0.0)
        lam_max = (mid + disc).clamp_min(1e-6)
        r_px = (3.0 * torch.sqrt(lam_max)).clamp(1.0, float(max(W, H)))
        TW = (W + ts - 1) // ts
        TH = (H + ts - 1) // ts
        T_tiles = TW * TH
        mux, muy = mu2d[:, 0], mu2d[:, 1]
        txmin = torch.floor((mux - r_px) / ts).clamp(0, TW - 1).to(torch.int64)
        txmax = torch.floor((mux + r_px) / ts).clamp(0, TW - 1).to(torch.int64)
        tymin = torch.floor((muy - r_px) / ts).clamp(0, TH - 1).to(torch.int64)
        tymax = torch.floor((muy + r_px) / ts).clamp(0, TH - 1).to(torch.int64)
        nx = txmax - txmin + 1
        ntiles = nx * (tymax - tymin + 1)                   # (M,)
        total = int(ntiles.sum().item())

        # expand to (gaussian, tile) pairs
        g_of_pair = torch.repeat_interleave(torch.arange(M, device=device), ntiles)
        start_excl = torch.cumsum(ntiles, 0) - ntiles
        local_k = torch.arange(total, device=device) - start_excl[g_of_pair]
        nx_p = nx[g_of_pair]
        tile_x = txmin[g_of_pair] + (local_k % nx_p)
        tile_y = tymin[g_of_pair] + (local_k // nx_p)
        tile_id = tile_y * TW + tile_x                      # (total,)

        # sort pairs by (tile, depth front-to-back). Key = tile*(M+1)+depth_rank
        # (both integer): exact in float32 while tile*(M+1) < 2^24, else CPU int64.
        ranks = torch.argsort(torch.argsort(z))             # per-gaussian depth rank
        if T_tiles * (M + 1) < (1 << 24):
            key = tile_id.to(dt) * float(M + 1) + ranks[g_of_pair].to(dt)
            order_p = torch.argsort(key)
        else:
            key = tile_id.cpu() * (M + 1) + ranks[g_of_pair].cpu()   # int64, exact
            order_p = torch.argsort(key).to(device)
        g_sorted = g_of_pair[order_p]
        tile_sorted = tile_id[order_p]

        counts = torch.bincount(tile_sorted, minlength=T_tiles)   # (T,)
        tstart = torch.cumsum(counts, 0) - counts                 # (T,)
        Kmax = max(int(counts.max().clamp(max=max_per_tile).item()), 1)
        krange = torch.arange(Kmax, device=device)
        valid_tk = krange[None, :] < counts.clamp(max=Kmax)[:, None]   # (T,K)
        gather_pos = (tstart[:, None] + krange[None, :]).clamp(max=max(total - 1, 0))
        g_tk = torch.where(valid_tk, g_sorted[gather_pos],
                           torch.zeros_like(gather_pos))            # (T,K)

        # per-tile pixel coords on the padded grid
        t_ids = torch.arange(T_tiles, device=device)
        ox = ((t_ids % TW) * ts)
        oy = ((t_ids // TW) * ts)
        ly, lx = torch.meshgrid(torch.arange(ts, device=device),
                                torch.arange(ts, device=device), indexing="ij")
        lx, ly = lx.reshape(-1), ly.reshape(-1)                    # (ts*ts,)
        px = (ox[:, None] + lx[None, :]).to(dt) + 0.5             # (T, ts*ts)
        py = (oy[:, None] + ly[None, :]).to(dt) + 0.5

    # ---- gather attrs per (tile, slot); autograd flows from here on ----
    mu2d_tk = mu2d[g_tk]                                          # (T,K,2)
    ca_tk, cb_tk, cc_tk = ca[g_tk], cb[g_tk], cc[g_tk]           # (T,K)
    op_tk = opacity[g_tk] * valid_tk.to(dt)                       # (T,K) zero pad
    rgb_tk = rgb[g_tk]                                            # (T,K,3)

    dx = px[:, None, :] - mu2d_tk[:, :, 0:1]                      # (T,K,ts*ts)
    dy = py[:, None, :] - mu2d_tk[:, :, 1:2]
    power = -0.5 * (ca_tk[:, :, None] * dx * dx
                    + 2 * cb_tk[:, :, None] * dx * dy
                    + cc_tk[:, :, None] * dy * dy)
    alpha = (op_tk[:, :, None] * torch.exp(power)).clamp(0.0, 0.99)
    T_cum = torch.cumprod(1.0 - alpha, dim=1)
    T_excl = torch.cat([torch.ones(T_tiles, 1, alpha.shape[2], device=device, dtype=dt),
                        T_cum[:, :-1]], dim=1)
    weight = alpha * T_excl                                       # (T,K,ts*ts)
    color = torch.einsum("tkp,tkc->tpc", weight, rgb_tk)         # (T,ts*ts,3)
    acc = weight.sum(1)                                          # (T,ts*ts)
    out_tiles = color + (1.0 - acc)[:, :, None] * bg[None, None, :]

    # stitch tiles -> padded image, then crop to (H,W)
    img = (out_tiles.reshape(TH, TW, ts, ts, 3)
           .permute(0, 2, 1, 3, 4).reshape(TH * ts, TW * ts, 3))
    return img[:H, :W, :]


def render_auto(model: GaussianModel, cam: dict, bg: torch.Tensor,
                blur: float = 0.12) -> torch.Tensor:
    """Rasterizer used by training/eval. Defaults to the GLOBAL renderer: at the
    96x96 gate resolution it is both faster and slightly more accurate than the
    tiled one (measured: tiled 1.5x SLOWER and -0.35 dB end-to-end -- the per-tile
    bookkeeping outweighs the savings when there are only ~9k pixels). Set
    SPLAT_TILED=1 to use `render_tiled`, which is only worth it at HIGH resolution,
    where the global renderer's O(M*pixels) tensor OOMs (~40 GB at 288px) and the
    tiled one stays within memory. See CLAUDE.md 'Tiled rasterizer'."""
    if os.environ.get("SPLAT_TILED", "0") == "1":
        return render_tiled(model, cam, bg, blur=blur)
    return render(model, cam, bg, blur=blur)


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #
def init_from_cameras(aabb_min, aabb_max, n_gauss, cams, gt_images,
                      seed=0, init_scale=0.06, device="cpu", sh_degree=3):
    """Random means within the AABB; per-gaussian colour seeded by averaging the
    pixels each gaussian projects to across the training views (huge convergence
    head-start vs. random colour)."""
    g = torch.Generator().manual_seed(seed)
    aabb_min = torch.as_tensor(aabb_min, dtype=torch.float32)
    aabb_max = torch.as_tensor(aabb_max, dtype=torch.float32)
    span = aabb_max - aabb_min
    means = aabb_min + torch.rand(n_gauss, 3, generator=g) * span

    colors = torch.full((n_gauss, 3), 0.5)
    counts = torch.zeros(n_gauss)
    accum = torch.zeros(n_gauss, 3)
    for cam, img in zip(cams, gt_images):
        R_w2c = cam["R_w2c"].cpu(); t_w2c = cam["t_w2c"].cpu()
        img = img.cpu()
        H, W = img.shape[0], img.shape[1]
        mu_c = means @ R_w2c.T + t_w2c
        z = mu_c[:, 2]
        u = (cam["fx"] * mu_c[:, 0] / z + cam["cx"])
        v = (cam["fy"] * mu_c[:, 1] / z + cam["cy"])
        ui = u.round().long(); vi = v.round().long()
        ok = (z > NEAR) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        idx = ok.nonzero(as_tuple=False).squeeze(1)
        if idx.numel():
            accum[idx] += img[vi[idx], ui[idx]]
            counts[idx] += 1
    seen = counts > 0
    colors[seen] = accum[seen] / counts[seen][:, None]

    sh_dc = _color_to_dc(colors.clamp(0.0, 1.0))
    sh_rest = torch.zeros(n_gauss, sh_coeffs(sh_degree) - 1, 3)
    log_scales = torch.full((n_gauss, 3), float(np.log(init_scale)))
    quats = torch.zeros(n_gauss, 4); quats[:, 0] = 1.0
    logit_opacity = torch.full((n_gauss,), float(np.log(0.2 / 0.8)))
    return GaussianModel(means, sh_dc, sh_rest, log_scales, quats, logit_opacity,
                         device, sh_degree=sh_degree)
