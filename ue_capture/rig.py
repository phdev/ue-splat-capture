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


def path_fan(waypoints_cm, step_cm=350.0, eye_cm=480.0, look_ahead_cm=700.0,
             fans_deg=((0.0, -8.0), (-30.0, -6.0), (30.0, -6.0), (-58.0, -3.0), (58.0, -3.0),
                       (0.0, -36.0), (0.0, 16.0)),
             heldout_every=8, start_index=0):
    """Cameras following an OPEN polyline (a road/ditch route), emitting a FAN per
    step so corridor surfaces get the multi-view baseline 3DGS needs (forward motion
    = baseline; the yaw/pitch fan = walls/floor/up coverage). Waypoints are [x,y,z]
    with z the LOCAL GROUND height (cm); cameras sit eye_cm above it (terrain-follow).
    fans_deg = (yaw_offset, pitch) per camera relative to the travel direction:
    default = ahead-slightly-down + left + right + floor + up-ahead (spire tops)."""
    pts = [list(p) for p in waypoints_cm]
    seg = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    cum = [0.0]
    for s in seg:
        cum.append(cum[-1] + s)
    total = cum[-1]

    def at(s):  # interpolate [x,y,groundz] at arc-length s (xy metric)
        s = max(0.0, min(total, s))
        j = 0
        while j < len(cum) - 2 and cum[j + 1] < s:
            j += 1
        f = (s - cum[j]) / max(cum[j + 1] - cum[j], 1e-6)
        return [pts[j][k] * (1 - f) + pts[j + 1][k] * f for k in range(3)]

    n_steps = max(2, int(total // step_cm))
    poses = []
    idx = start_index
    for k in range(n_steps + 1):
        s = total * k / n_steps
        p = at(s)
        pa = at(s + look_ahead_cm)
        base_yaw = math.atan2(pa[1] - p[1], pa[0] - p[0])
        cam = [p[0], p[1], p[2] + eye_cm]
        for yaw_off, pitch in fans_deg:
            ya = base_yaw + math.radians(yaw_off)
            pr = math.radians(pitch)
            tgt = [cam[0] + math.cos(pr) * math.cos(ya) * look_ahead_cm,
                   cam[1] + math.cos(pr) * math.sin(ya) * look_ahead_cm,
                   cam[2] + math.sin(pr) * look_ahead_cm]
            split = "heldout" if (heldout_every and idx % heldout_every == 1) else "train"
            poses.append({"index": idx, "kind": "path", "split": split,
                          "location_cm": cam[:], "target_cm": tgt})
            idx += 1
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
