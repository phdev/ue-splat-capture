"""Camera-rig path generation -- PURE STDLIB (math only).

Runs inside UnrealEditor-Cmd's embedded Python (which has no numpy) AND is
importable/unit-testable in the uv venv. Produces deterministic camera poses in
UE convention (centimetres, Z-up), each a location + a look-at target.

Held-out cameras are interleaved on the orbit rings (every Nth) so held-out
views are always bracketed by nearby training views.
"""
from __future__ import annotations

import math


def orbit_hemisphere(center_cm, radius_cm, elevations_deg=(22.0, 48.0),
                     n_azimuth=24, heldout_every=4, start_index=0):
    cx, cy, cz = center_cm
    poses = []
    idx = start_index
    for elev in elevations_deg:
        er = math.radians(elev)
        for k in range(n_azimuth):
            az = k * (360.0 / n_azimuth)
            ar = math.radians(az)
            loc = [cx + radius_cm * math.cos(er) * math.cos(ar),
                   cy + radius_cm * math.cos(er) * math.sin(ar),
                   cz + radius_cm * math.sin(er)]
            split = "heldout" if (heldout_every and idx % heldout_every == 1) else "train"
            poses.append({
                "index": idx, "kind": "orbit", "split": split,
                "ring_elev_deg": float(elev), "azimuth_deg": float(az),
                "location_cm": loc, "target_cm": [cx, cy, cz],
            })
            idx += 1
    return poses


def grid_nadir(center_cm, extent_cm, height_cm, n_side=7, converge=0.25,
               heldout_every=8, start_index=0):
    """Drone-mapping grid for TERRAIN: n_side x n_side cameras spread over a
    (2*extent_cm)^2 area, each at `height_cm` above the ground plane (center z)
    looking ~straight DOWN at the patch beneath it. Unlike orbit_hemisphere (all
    cameras converge on one hero point -> only the centre ground is covered), this
    gives uniform overlapping coverage of the whole spread ground. `converge`
    (0..1) tilts each camera that fraction toward the centre for some angular
    diversity (0 = pure nadir). Heavy overlap (spacing < footprint) gives the
    stereo baseline 3DGS needs for depth on flat ground."""
    cx, cy, cz = center_cm
    poses = []
    idx = start_index
    n = max(2, n_side)
    for i in range(n):
        for j in range(n):
            x = cx + (i / (n - 1) - 0.5) * 2.0 * extent_cm
            y = cy + (j / (n - 1) - 0.5) * 2.0 * extent_cm
            z = cz + height_cm
            # target: straight below (nadir), nudged toward centre by `converge`
            tgt = [x + (cx - x) * converge, y + (cy - y) * converge, cz]
            split = "heldout" if (heldout_every and idx % heldout_every == 1) else "train"
            poses.append({
                "index": idx, "kind": "grid", "split": split,
                "grid_ij": [i, j], "location_cm": [x, y, z], "target_cm": tgt,
            })
            idx += 1
    return poses


def interior_walk(waypoints_cm, n_steps=8, height_cm=130.0, look_ahead=0.12,
                  start_index=0):
    """Cameras walking a closed polyline at fixed height, looking ahead."""
    closed = list(waypoints_cm) + [waypoints_cm[0]]
    seg = [_dist(closed[i], closed[i + 1]) for i in range(len(closed) - 1)]
    cum = [0.0]
    for s in seg:
        cum.append(cum[-1] + s)
    total = cum[-1]
    poses = []
    for i in range(n_steps):
        s = total * i / n_steps
        p = _poly_at(closed, cum, s)
        pa = _poly_at(closed, cum, (s + look_ahead * total) % total)
        poses.append({
            "index": start_index + i, "kind": "walk", "split": "train",
            "location_cm": [p[0], p[1], height_cm],
            "target_cm": [pa[0], pa[1], height_cm * 0.7],
        })
    return poses


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def _poly_at(closed_pts, cum, s):
    j = 0
    while j < len(cum) - 2 and cum[j + 1] < s:
        j += 1
    seg_len = max(cum[j + 1] - cum[j], 1e-6)
    f = (s - cum[j]) / seg_len
    return [closed_pts[j][k] * (1 - f) + closed_pts[j + 1][k] * f for k in range(3)]


def default_rig(center_cm=(0.0, 0.0, 35.0), radius_cm=360.0, include_walk=False):
    """Canonical self-test rig: a DENSE orbit hemisphere -- 5 rings x 40 azimuths
    (9 deg) = 200 cameras, held out every 5th (interleaved). Denser angular
    coverage means held-out views sit closer to training views (easier to
    reconstruct) and gives enough views to support view-dependent SH without
    overfitting. The interior walk is available (`interior_walk`, unit-tested)
    but OFF by default."""
    poses = orbit_hemisphere(center_cm, radius_cm,
                             elevations_deg=(12.0, 24.0, 36.0, 48.0, 60.0),
                             n_azimuth=40, heldout_every=5)
    if include_walk:
        poses += interior_walk(
            waypoints_cm=[[200, 0, 0], [0, 200, 0], [-200, 0, 0], [0, -200, 0]],
            n_steps=8, height_cm=130.0, start_index=len(poses))
    return poses


if __name__ == "__main__":
    import json
    poses = default_rig()
    n_held = sum(p["split"] == "heldout" for p in poses)
    print(json.dumps({"n_poses": len(poses), "n_heldout": n_held,
                      "kinds": sorted({p["kind"] for p in poses})}, indent=2))
