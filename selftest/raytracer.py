"""Tiny vectorized numpy raytracer -- the ground-truth renderer for fixtures.

Supports a checkerboard plane, spheres, and axis-aligned boxes with simple
view-independent Lambertian shading (key + fill directional lights + ambient,
no shadows). Rays are generated with the SAME intrinsics used everywhere else,
so the rendered pixel of a fiducial matches its UE-native forward projection.
"""
from __future__ import annotations

import numpy as np

TMIN = 1e-3


def _shade(points, normals, albedo, emissive_mask, scene):
    """points,normals,albedo: (M,3). Returns shaded RGB (M,3) in [0,1]."""
    amb = float(scene["ambient"])
    shade = np.full(points.shape[0], amb)
    for lt in scene["lights"]:
        L = np.asarray(lt["dir"], float)
        ndl = np.clip(normals @ L, 0.0, None)
        shade = shade + float(lt["intensity"]) * ndl
    col = albedo * shade[:, None]
    # emissive surfaces ignore lighting
    col = np.where(emissive_mask[:, None], albedo, col)
    return np.clip(col, 0.0, 1.0)


def _checker_albedo(points, mat):
    a1 = np.asarray(mat["albedo"], float)
    chk = mat.get("checker")
    if not chk:
        return np.broadcast_to(a1, (points.shape[0], 3)).copy()
    a2 = np.asarray(chk["color2"], float)
    s = float(chk["scale_cm"])
    parity = (np.floor(points[:, 0] / s) + np.floor(points[:, 1] / s)) % 2
    out = np.where(parity[:, None] > 0.5, a1[None, :], a2[None, :])
    return out


def _intersect_primitive(origins, dirs, prim):
    """Return (t (N,), valid (N,), point (N,3), normal (N,3))."""
    N = origins.shape[0]
    t = np.full(N, np.inf)
    valid = np.zeros(N, bool)
    typ = prim["type"]
    if typ == "plane":
        p0 = np.asarray(prim["point"], float)
        nrm = np.asarray(prim["normal"], float)
        denom = dirs @ nrm
        ok = np.abs(denom) > 1e-9
        tt = np.where(ok, ((p0 - origins) @ nrm) / np.where(ok, denom, 1.0), np.inf)
        valid = ok & (tt > TMIN)
        t = np.where(valid, tt, np.inf)
        point = origins + t[:, None] * dirs
        normal = np.broadcast_to(nrm, (N, 3)).copy()
    elif typ == "sphere":
        c = np.asarray(prim["center"], float)
        r = float(prim["radius"])
        oc = origins - c
        b = np.sum(oc * dirs, axis=1)
        cc = np.sum(oc * oc, axis=1) - r * r
        disc = b * b - cc
        okd = disc > 0
        sq = np.sqrt(np.where(okd, disc, 0.0))
        t0 = -b - sq
        t1 = -b + sq
        tt = np.where(t0 > TMIN, t0, t1)
        valid = okd & (tt > TMIN)
        t = np.where(valid, tt, np.inf)
        point = origins + t[:, None] * dirs
        normal = point - c
        n = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / np.maximum(n, 1e-12)
    elif typ == "box":
        bmin = np.asarray(prim["min"], float)
        bmax = np.asarray(prim["max"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / dirs
            tlo = (bmin - origins) * inv
            thi = (bmax - origins) * inv
        t1 = np.minimum(tlo, thi)
        t2 = np.maximum(tlo, thi)
        tnear = np.max(t1, axis=1)
        tfar = np.min(t2, axis=1)
        hit = (tfar >= tnear) & (tfar > TMIN)
        tt = np.where(tnear > TMIN, tnear, tfar)
        valid = hit & (tt > TMIN) & np.isfinite(tt)
        t = np.where(valid, tt, np.inf)
        point = origins + t[:, None] * dirs
        center = (bmin + bmax) / 2.0
        half = np.maximum((bmax - bmin) / 2.0, 1e-9)
        local = (point - center) / half
        ax = np.argmax(np.abs(local), axis=1)
        normal = np.zeros((N, 3))
        rows = np.arange(N)
        normal[rows, ax] = np.sign(local[rows, ax] + 1e-12)
    else:
        raise ValueError(f"unknown primitive {typ}")
    return t, valid, point, normal


def intersect_scene(origins, dirs, scene):
    """Nearest-hit shading. Returns (color (N,3), best_t (N,), best_prim (N,))."""
    N = origins.shape[0]
    bg = np.asarray(scene["background"], float)
    color = np.broadcast_to(bg, (N, 3)).copy()
    best_t = np.full(N, np.inf)
    best_prim = np.full(N, -1, int)
    # Non-hit rays legitimately produce inf/nan in unused slots (masked out).
    with np.errstate(divide="ignore", invalid="ignore"):
      for pi, prim in enumerate(scene["primitives"]):
        t, valid, point, normal = _intersect_primitive(origins, dirs, prim)
        upd = valid & (t < best_t)
        if not upd.any():
            continue
        mat = prim["mat"]
        pts = point[upd]
        nrm = normal[upd]
        alb = _checker_albedo(pts, mat)
        emis = np.full(pts.shape[0], bool(mat["emissive"]))
        color[upd] = _shade(pts, nrm, alb, emis, scene)
        best_t[upd] = t[upd]
        best_prim[upd] = pi
    return color, best_t, best_prim


def render(scene, loc_cm, basis_ue, intr, ss: int = 2) -> np.ndarray:
    """Render the scene from a UE camera. Returns float image (H,W,3) in [0,1]."""
    W, H = int(intr["w"]), int(intr["h"])
    fx, fy, cx, cy = intr["fl_x"], intr["fl_y"], intr["cx"], intr["cy"]
    fwd, right, up = basis_ue[:, 0], basis_ue[:, 1], basis_ue[:, 2]
    loc = np.asarray(loc_cm, float)

    us = (np.arange(W * ss) + 0.5) / ss        # sample coords in image pixels
    vs = (np.arange(H * ss) + 0.5) / ss
    U, V = np.meshgrid(us, vs)                 # (Hs,Ws)
    a = (U - cx) / fx
    b = -(V - cy) / fy
    dirs = (fwd[None, None, :]
            + a[..., None] * right[None, None, :]
            + b[..., None] * up[None, None, :])
    dirs = dirs.reshape(-1, 3)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.broadcast_to(loc, dirs.shape).copy()

    color, _, _ = intersect_scene(origins, dirs, scene)
    img = color.reshape(H * ss, W * ss, 3)
    if ss > 1:                                  # box-filter downsample (AA)
        img = img.reshape(H, ss, W, ss, 3).mean(axis=(1, 3))
    return img


def first_hit_prim(origin, direction, scene):
    """Index of the nearest primitive hit by a single ray, or -1."""
    o = np.asarray(origin, float).reshape(1, 3)
    d = np.asarray(direction, float).reshape(1, 3)
    d = d / np.linalg.norm(d)
    _, _, prim = intersect_scene(o, d, scene)
    return int(prim[0])


def fiducial_visible(scene, loc_cm, fid_center_cm, prim_index) -> bool:
    """True iff the fiducial's primitive is the first thing the camera sees
    along the ray to its center (i.e. unoccluded)."""
    loc = np.asarray(loc_cm, float)
    d = np.asarray(fid_center_cm, float) - loc
    return first_hit_prim(loc, d, scene) == prim_index
