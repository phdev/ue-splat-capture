"""PSNR / SSIM, implemented in torch so they serve both as training losses and
as evaluation metrics. No external metric library (supply-chain hygiene)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _to_nchw(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x), dtype=torch.float32)
    if t.dim() == 3:
        if t.shape[-1] in (1, 3):      # HWC -> CHW
            t = t.permute(2, 0, 1)
        t = t.unsqueeze(0)
    return t.float()


def psnr(x, y, data_range: float = 1.0) -> float:
    a, b = _to_nchw(x), _to_nchw(y)
    mse = torch.mean((a - b) ** 2)
    if mse.item() <= 1e-12:
        return 99.0
    return float(10.0 * torch.log10((data_range ** 2) / mse))


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = torch.outer(g, g)
    return w2d


def ssim(x, y, data_range: float = 1.0, window_size: int = 11,
         sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM. Differentiable; returns a 0-d tensor. x,y in [0,data_range]."""
    a, b = _to_nchw(x), _to_nchw(y)
    C = a.shape[1]
    device, dtype = a.device, a.dtype
    win = _gaussian_window(window_size, sigma, device, dtype)
    win = win.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def filt(z):
        return F.conv2d(z, win, padding=pad, groups=C)

    mu_x, mu_y = filt(a), filt(b)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x2 = filt(a * a) - mu_x2
    sig_y2 = filt(b * b) - mu_y2
    sig_xy = filt(a * b) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    smap = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / \
           ((mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2))
    return smap.mean()


def ssim_value(x, y, **kw) -> float:
    with torch.no_grad():
        return float(ssim(x, y, **kw))
