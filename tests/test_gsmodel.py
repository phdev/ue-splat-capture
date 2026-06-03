"""CPU unit tests for the rasterizer + metrics (fast, deterministic, no MPS).

The key check ties the splat renderer to the SAME convention as T1: a single
opaque Gaussian placed at a known world point must light up the pixel the OpenCV
projection predicts. A handedness/axis bug in the renderer would move the blob.
"""
import json
from pathlib import Path

import numpy as np
import torch

from splatkit import data, gsmodel, metrics

FIX = Path(__file__).resolve().parent.parent / "fixtures" / "selftest"


def test_single_gaussian_lands_at_projected_pixel():
    frames, meta = data.load_dataset(str(FIX / "transforms.json"), device="cpu")
    doc = json.loads((FIX / "transforms.json").read_text())
    fr = doc["frames"][0]
    j = fr["fiducials_vis"].index(True)
    u_gt, v_gt = fr["fiducials_px"][j]
    world = np.array(doc["fiducials_world_m"][j], float)

    model = gsmodel.GaussianModel.from_points(
        world[None, :], np.array([[1.0, 1.0, 1.0]]), init_scale=0.02,
        device="cpu", init_opacity=0.99)
    bg = torch.zeros(3)
    with torch.no_grad():
        img = gsmodel.render(model, frames[0], bg)
    W = frames[0]["W"]
    flat = img.sum(-1).reshape(-1).argmax().item()
    v, u = divmod(flat, W)
    assert abs(u - u_gt) <= 3 and abs(v - v_gt) <= 3, (u, v, u_gt, v_gt)


def test_empty_render_returns_background():
    frames, meta = data.load_dataset(str(FIX / "transforms.json"), device="cpu")
    # all gaussians far behind the camera -> culled -> pure background
    model = gsmodel.GaussianModel.from_points(
        np.array([[0.0, 0.0, -1e6]]), np.array([[1.0, 0.0, 0.0]]),
        init_scale=0.02, device="cpu", init_opacity=0.99)
    bg = torch.tensor([0.1, 0.2, 0.3])
    with torch.no_grad():
        img = gsmodel.render(model, frames[0], bg)
    assert torch.allclose(img, bg.expand_as(img), atol=1e-5)


def test_psnr_ssim_identity():
    x = torch.rand(3, 32, 32)
    assert metrics.psnr(x, x) > 90
    assert metrics.ssim(x, x) > 0.999


def test_quat_identity_is_rotation_identity():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    R = gsmodel.quat_to_rotmat(q)
    assert torch.allclose(R[0], torch.eye(3), atol=1e-6)


def _many_gaussian_model(n=300, seed=0):
    g = np.random.RandomState(seed)
    pts = g.uniform(-0.3, 0.3, size=(n, 3)).astype("float32")
    cols = g.uniform(0.0, 1.0, size=(n, 3)).astype("float32")
    return gsmodel.GaussianModel.from_points(
        pts, cols, init_scale=0.05, device="cpu", init_opacity=0.6, sh_degree=1)


def test_tiled_matches_global_render():
    """The tile-based rasterizer must match the reference global renderer to a
    few hundredths of a dB (differences are only the 3-sigma cutoff / per-tile
    cap). A tiling/indexing/ordering bug would smear or shift pixels."""
    frames, _ = data.load_dataset(str(FIX / "transforms.json"), device="cpu")
    model = _many_gaussian_model()
    bg = torch.tensor([0.05, 0.07, 0.10])
    with torch.no_grad():
        a = gsmodel.render(model, frames[0], bg).clamp(0, 1)
        b = gsmodel.render_tiled(model, frames[0], bg).clamp(0, 1)
    assert a.shape == b.shape
    assert metrics.psnr(b, a) > 35.0, f"tiled vs global only {metrics.psnr(b, a):.1f} dB"


def test_tiled_single_gaussian_lands_at_projected_pixel():
    """Same convention check as the global renderer (guards handedness/axes in
    the tiled path): a Gaussian at a known world point lights its projected pixel."""
    frames, _ = data.load_dataset(str(FIX / "transforms.json"), device="cpu")
    doc = json.loads((FIX / "transforms.json").read_text())
    fr = doc["frames"][0]
    j = fr["fiducials_vis"].index(True)
    u_gt, v_gt = fr["fiducials_px"][j]
    world = np.array(doc["fiducials_world_m"][j], float)
    model = gsmodel.GaussianModel.from_points(
        world[None, :], np.array([[1.0, 1.0, 1.0]]), init_scale=0.02,
        device="cpu", init_opacity=0.99)
    with torch.no_grad():
        img = gsmodel.render_tiled(model, frames[0], torch.zeros(3))
    W = frames[0]["W"]
    v, u = divmod(img.sum(-1).reshape(-1).argmax().item(), W)
    assert abs(u - u_gt) <= 3 and abs(v - v_gt) <= 3, (u, v, u_gt, v_gt)


def test_tiled_empty_render_returns_background():
    frames, _ = data.load_dataset(str(FIX / "transforms.json"), device="cpu")
    model = gsmodel.GaussianModel.from_points(
        np.array([[0.0, 0.0, -1e6]]), np.array([[1.0, 0.0, 0.0]]),
        init_scale=0.02, device="cpu", init_opacity=0.99)
    bg = torch.tensor([0.1, 0.2, 0.3])
    with torch.no_grad():
        img = gsmodel.render_tiled(model, frames[0], bg)
    assert torch.allclose(img, bg.expand_as(img), atol=1e-5)
