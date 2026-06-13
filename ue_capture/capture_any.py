"""UNIVERSAL splat-capture SCOUT+RIG for ANY loaded Unreal level/region.

Run via remote exec in the warm editor (scripts/any_pipeline.py does this for
you). The level must already be loaded + streamed (warm editor; verify foliage
instance counts first — see the VISIBILITY-PERSISTS law in CLAUDE.md). Phases:
  1. SCOUT — sample ISMC instance transforms (fallback: StaticMeshActor
             locations) -> content extent, ground z, tall-content clusters.
  2. RIG   — a 2D GRID of dome "full" stations covering the extent (enclosed-
             tuned elevations: no rim-grazing 8 deg, no sky-grazing 76 deg) +
             orbit rings at the tall clusters + one far context ring. The grid
             (vs the old 3-focus axis line) is the scene39 fix for the enclosed
             pockets a sparse axis spread starved (CLAUDE.md UNIVERSAL PIPELINE).
  3. EMIT  — writes /tmp/ue_any_plan.json. The HOST (scripts/any_pipeline.py)
             then probes each station for line-of-sight (displacing buried
             stations up/outward), runs the captures, merges with the
             buried-frame filter, preps the depth dataset, and prints the
             depth-primary pod-training commands for the extent tier.

This script only ever takes seconds (no capture is launched from here — the
tick-driven capture can't be awaited from inside a remote exec).

Knobs (env): UA_REGION_CM="x0,y0,x1,y1" to scope a sub-region (else full
content extent), UA_OUT_PREFIX (default ed_any), UA_MAX_STATIONS (default 3),
UA_MAX_ORBITS (default 4).
"""
import json
import os

import unreal

PREFIX = os.environ.get("UA_OUT_PREFIX", "ed_any")
MAX_STATIONS = int(os.environ.get("UA_MAX_STATIONS", "6"))  # dome-grid budget
MAX_ORBITS = int(os.environ.get("UA_MAX_ORBITS", "4"))

# ---------- 1. SCOUT ----------
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
actors = eas.get_all_level_actors()
pts = []
for a in actors:
    for c in a.get_components_by_class(unreal.InstancedStaticMeshComponent):
        cnt = c.get_instance_count()
        if cnt == 0:
            continue
        step = max(1, cnt // 25)
        for i in range(0, cnt, step):
            l = c.get_instance_transform(i, True).translation
            pts.append((l.x, l.y, l.z))
if len(pts) < 1000:  # non-PCG level: fall back to static-mesh actor locations
    for a in actors:
        if a.get_components_by_class(unreal.StaticMeshComponent):
            l = a.get_actor_location()
            pts.append((l.x, l.y, l.z))
if os.environ.get("UA_REGION_CM"):
    x0, y0, x1, y1 = [float(v) for v in os.environ["UA_REGION_CM"].split(",")]
    pts = [p for p in pts if min(x0, x1) <= p[0] <= max(x0, x1)
           and min(y0, y1) <= p[1] <= max(y0, y1)]
n = len(pts)
if n < 50:
    print(f"[any] FATAL: only {n} content samples — level not loaded/streamed, "
          "or UA_REGION_CM is empty. ANY_PLAN_FAIL")
    raise SystemExit
xs = sorted(p[0] for p in pts); ys = sorted(p[1] for p in pts); zs = sorted(p[2] for p in pts)
lo = lambda arr: arr[int(0.005 * n)]
hi = lambda arr: arr[min(n - 1, int(0.995 * n))]
ext_x, ext_y = (hi(xs) - lo(xs)) / 100.0, (hi(ys) - lo(ys)) / 100.0
ground_z = zs[int(0.10 * n)]
top_z = zs[min(n - 1, int(0.99 * n))]
print(f"[any] scout: {n} samples, extent {ext_x:.0f}x{ext_y:.0f}m, "
      f"ground {ground_z/100:.0f}m top {top_z/100:.0f}m")

# tall-content clusters (20m cells with content > ground+22m) -> orbit targets
# (non-max suppression at 3 cells / 60m so one tall mass yields ONE orbit)
cells = {}
for x, y, z in pts:
    if z > ground_z + 2200:
        k = (int(x // 2000), int(y // 2000))
        cells[k] = cells.get(k, 0) + 1
clusters = []
for k, cnt in sorted(cells.items(), key=lambda kv: -kv[1]):
    if any(max(abs(k[0] - q[0]), abs(k[1] - q[1])) < 3 for q, _ in clusters):
        continue
    clusters.append((k, cnt))
    if len(clusters) >= MAX_ORBITS:
        break
orbit_targets = [(cx * 2000 + 1000.0, cy * 2000 + 1000.0) for (cx, cy), _ in clusters]
print(f"[any] tall clusters (m): {[(round(x/100), round(y/100)) for x, y in orbit_targets]}")

# ---------- 2. RIG: dome GRID (even enclosed-pocket coverage) ----------
# scene39 lesson: a 3-focus major-axis line starves enclosed pockets between the
# domes (the s1 pocket fogged at gate MAE 0.31). A 2D GRID of domes scaled to area
# covers the bowl evenly. Pair it with ENCLOSED-TUNED elevations: drop the ~8 deg
# ring (grazes the rim / buries in a bowl) and the ~76 deg (sees sky over the rim);
# keep mids + one steep-down ring for the floor (so no separate ground-dense pass).
cx0, cy0 = (lo(xs) + hi(xs)) / 2.0, (lo(ys) + hi(ys)) / 2.0
span = max(ext_x, ext_y)
station_R = 7000.0 if span > 120 else 6500.0
focus_z = ground_z + 1000.0
x0g, x1g, y0g, y1g = lo(xs), hi(xs), lo(ys), hi(ys)
SPACING_CM = 8000.0  # ~80m between domes -> heavy overlap at R65-70m
nx = max(1, int(round((x1g - x0g) / SPACING_CM)) + 1)
ny = max(1, int(round((y1g - y0g) / SPACING_CM)) + 1)
while nx * ny > MAX_STATIONS:  # trim the longer axis until within the station budget
    if nx >= ny and nx > 1:
        nx -= 1
    elif ny > 1:
        ny -= 1
    else:
        break
_axis = lambda k, a, b: [(a + b) / 2.0] if k == 1 else [a + (b - a) * i / (k - 1) for i in range(k)]
foci = [(x, y) for y in _axis(ny, y0g, y1g) for x in _axis(nx, x0g, x1g)]
DOME_ELEV = "20,36,52,68" if span > 120 else "16,32,50,66"
DOME_NAZ = 30
print(f"[any] dome grid {nx}x{ny} = {len(foci)} domes, elev {DOME_ELEV} naz {DOME_NAZ}")

stations = []
for i, (fx, fy) in enumerate(foci):
    stations.append({"name": f"s{i+1}", "kind": "full",
                     "focus": [fx, fy, focus_z], "radius": station_R,
                     "elev": DOME_ELEV, "naz": DOME_NAZ, "ground": 0,
                     "settle": 120, "converge": 12,
                     "est_poses": len(DOME_ELEV.split(",")) * DOME_NAZ})
for i, (ox, oy) in enumerate(orbit_targets):
    stations.append({"name": f"o{i+1}", "kind": "orbit",
                     "focus": [ox, oy, ground_z + 2500.0], "radius": 2200.0,
                     "elev": "-5,15,35,55", "naz": 24,
                     "settle": 20, "converge": 10, "est_poses": 96})
stations.append({"name": "far", "kind": "orbit",
                 "focus": [cx0, cy0, ground_z + 1500.0],
                 "radius": min(60000.0, max(20000.0, span * 100 * 1.25)),
                 "elev": "12,30", "naz": 24,
                 "settle": 20, "converge": 10, "est_poses": 48})

# ---------- 3. EMIT ----------
plan = {"prefix": PREFIX, "extent_m": [round(ext_x, 1), round(ext_y, 1)],
        "ground_z_cm": ground_z, "top_z_cm": top_z,
        "stations": stations,
        # host probe/displace policy (CLAUDE.md PROBE-DISPLACE law)
        "probe": {"elev": "12,35", "naz": 8, "res": 512, "max_buried": 0.10,
                  "displace_dz_cm": 1000.0, "displace_rmul": 1.2,
                  "retries": 2, "skip_above": 0.60},
        "train_tier": "large" if span > 120 else "standard",
        "init_points": 500000 if span > 120 else 120000}
with open("/tmp/ue_any_plan.json", "w") as f:
    json.dump(plan, f, indent=1)
print(f"[any] plan: {len(stations)} stations, tier={plan['train_tier']}, "
      f"init={plan['init_points']}")
print("ANY_PLAN_DONE")
