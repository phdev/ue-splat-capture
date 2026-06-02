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
    """Canonical self-test rig: a DENSE orbit hemisphere -- 4 rings x 30 azimuths
    (12 deg) = 120 cameras, held out every 5th (interleaved). Denser angular
    coverage means held-out views sit closer to training views (easier to
    reconstruct) and gives enough views to support view-dependent SH without
    overfitting. The interior walk is available (`interior_walk`, unit-tested)
    but OFF by default."""
    poses = orbit_hemisphere(center_cm, radius_cm,
                             elevations_deg=(16.0, 30.0, 44.0, 58.0),
                             n_azimuth=30, heldout_every=5)
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
