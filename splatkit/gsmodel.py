"""Self-contained differentiable 3D Gaussian splatting (torch, MPS/CPU).

This is the Apple-Silicon trainer backend (msplat is CUDA-only). It implements
EWA-style splatting with a global per-image depth sort and a vectorized
alpha-compositing via cumulative product, so PyTorch autograd supplies the
backward pass -- no custom CUDA/Metal kernels. Colors are view-independent
(SH degree 0), which is sufficient for the diffuse self-test scene.

All randomness is seeded; given a fixed seed and CPU device the result is
deterministic.
"""
from __future__ import annotations

import numpy as np
import torch

NEAR = 0.2          # metres; cull gaussians closer than this
SCREEN_MARGIN = 0.25  # fraction of image size to keep off-screen gaussians


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-9)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


class GaussianModel:
    def __init__(self, means, colors, log_scales, quats, logit_opacity, device):
        self.device = device
        self.means = means.to(device).requires_grad_(True)
        self.colors_raw = colors.to(device).requires_grad_(True)      # -> sigmoid
        self.log_scales = log_scales.to(device).requires_grad_(True)  # -> exp
        self.quats = quats.to(device).requires_grad_(True)
        self.logit_opacity = logit_opacity.to(device).requires_grad_(True)

    # --- views of activated params ---
    @property
    def n(self):
        return self.means.shape[0]

    def opacity(self):
        return torch.sigmoid(self.logit_opacity)

    def rgb(self):
        return torch.sigmoid(self.colors_raw)

    def scales(self):
        return torch.exp(self.log_scales)

    def parameters(self):
        return [self.means, self.colors_raw, self.log_scales, self.quats,
                self.logit_opacity]

    def param_groups(self, lr):
        return [
            {"params": [self.means], "lr": lr["means"]},
            {"params": [self.colors_raw], "lr": lr["color"]},
            {"params": [self.log_scales], "lr": lr["scale"]},
            {"params": [self.quats], "lr": lr["quat"]},
            {"params": [self.logit_opacity], "lr": lr["opacity"]},
        ]

    def state(self):
        return {k: getattr(self, k).detach().cpu()
                for k in ("means", "colors_raw", "log_scales", "quats", "logit_opacity")}

    @classmethod
    def from_state(cls, st, device):
        return cls(st["means"], st["colors_raw"], st["log_scales"], st["quats"],
                   st["logit_opacity"], device)

    @classmethod
    def from_points(cls, points, colors, init_scale, device,
                    init_opacity=0.25):
        pts = torch.as_tensor(points, dtype=torch.float32)
        cols = torch.as_tensor(colors, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
        n = pts.shape[0]
        colors_raw = torch.logit(cols)
        log_scales = torch.full((n, 3), float(np.log(init_scale)))
        quats = torch.zeros(n, 4); quats[:, 0] = 1.0
        logit_opacity = torch.full((n,), float(np.log(init_opacity / (1 - init_opacity))))
        return cls(pts, colors_raw, log_scales, quats, logit_opacity, device)


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
    col = model.colors_raw.detach()
    scales = torch.exp(logsc); opac = torch.sigmoid(logop)

    keep = (opac > min_opacity) & (scales.max(dim=1).values < max_scale)
    M, LS, Q, LO, C = means[keep], logsc[keep], quats[keep], logop[keep], col[keep]
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

    parts_m = [M[~split]]; parts_ls = [LS[~split]]; parts_q = [Q[~split]]
    parts_lo = [LO[~split]]; parts_c = [C[~split]]
    if ci.numel():
        parts_m.append(jit(ci, S)); parts_ls.append(LS[ci]); parts_q.append(Q[ci])
        parts_lo.append(LO[ci]); parts_c.append(C[ci])
    for _ in range(2):  # two children per split
        if si.numel():
            parts_m.append(jit(si, S)); parts_ls.append(LS[si] - log16)
            parts_q.append(Q[si]); parts_lo.append(LO[si]); parts_c.append(C[si])

    nm = torch.cat(parts_m); nls = torch.cat(parts_ls); nq = torch.cat(parts_q)
    nlo = torch.cat(parts_lo); nc = torch.cat(parts_c)

    if nm.shape[0] > max_points:                       # cap: keep most opaque
        order = torch.argsort(torch.sigmoid(nlo), descending=True)[:max_points]
        nm, nls, nq = nm[order], nls[order], nq[order]
        nlo, nc = nlo[order], nc[order]
    return GaussianModel(nm, nc, nls, nq, nlo, dev)


def _covariance3d(scales, R):
    S = torch.diag_embed(scales)            # (M,3,3)
    M = R @ S                                # (M,3,3)
    return M @ M.transpose(1, 2)             # R S S^T R^T


def render(model: GaussianModel, cam: dict, bg: torch.Tensor,
           blur: float = 0.12) -> torch.Tensor:
    """Render one camera. cam: R_w2c(3,3), t_w2c(3,), fx,fy,cx,cy,W,H tensors/floats.
    Returns image (H, W, 3) with autograd attached."""
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
    rgb = model.rgb()[idx]

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
    # conic = inverse(cov2d)
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


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #
def init_from_cameras(aabb_min, aabb_max, n_gauss, cams, gt_images,
                      seed=0, init_scale=0.06, device="cpu"):
    """Random means within the AABB; per-gaussian colour seeded by averaging the
    pixels each gaussian projects to across the training views (huge convergence
    head-start vs. random colour)."""
    g = torch.Generator().manual_seed(seed)
    aabb_min = torch.as_tensor(aabb_min, dtype=torch.float32)
    aabb_max = torch.as_tensor(aabb_max, dtype=torch.float32)
    span = aabb_max - aabb_min
    means = aabb_min + torch.rand(n_gauss, 3, generator=g) * span

    # colour-from-images init
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
            cols = img[vi[idx], ui[idx]]               # (k,3)
            accum[idx] += cols
            counts[idx] += 1
    seen = counts > 0
    colors[seen] = accum[seen] / counts[seen][:, None]

    colors_raw = torch.logit(colors.clamp(1e-3, 1 - 1e-3))
    log_scales = torch.full((n_gauss, 3), float(np.log(init_scale)))
    quats = torch.zeros(n_gauss, 4); quats[:, 0] = 1.0
    logit_opacity = torch.full((n_gauss,), float(np.log(0.2 / 0.8)))  # ~0.2
    return GaussianModel(means, colors_raw, log_scales, quats, logit_opacity, device)
