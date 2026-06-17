"""Seed a CLOSED Camera Rig Rail loop around the spire, ground-snapped (terrain-follow),
so the user can refine an encircling flythrough route. Companion to path_rail_seed.py;
read back the same way with path_rail_read.py.

Run INSIDE the warm editor:
    python3 scripts/ue_exec.py scripts/path_rail_loop.py 180

Method: find the spire = the tallest terrain near the current view (coarse then fine
down-trace grid -> peak XY), lay N ring points at RADIUS_CM around it, snap each down to
the ground, and build a CLOSED rig-rail spline. Optional /tmp/loop_cfg.json overrides
{"radius_cm":1700,"n":20,"center_cm":[x,y],"fwd_cm":2500} so it can be re-tuned without
editing this file. Labelled CAPTURE_PATH_RAIL + tagged CAPTURE_PATH (destroys any prior).
"""
import json
import math
import os
import unreal

LABEL = "CAPTURE_PATH_RAIL"
TAG = "CAPTURE_PATH"
RAIL_CLASS_PATH = "/Script/CinematicCamera.CameraRig_Rail"
CFG = "/tmp/loop_cfg.json"


def _ground_z(world, x, y, z_start=60000.0):
    start = unreal.Vector(x, y, z_start)
    end = unreal.Vector(x, y, -50000.0)
    try:
        hit = unreal.SystemLibrary.line_trace_single(
            world, start, end, unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, False, [],
            unreal.DrawDebugTrace.NONE, True)
        hr = hit[1] if isinstance(hit, (tuple, list)) else hit
        if hr:
            t = hr.to_tuple()                 # [0]=blocking_hit, [5]=impact_point
            if t[0]:
                return float(t[5].z)
    except Exception:
        pass
    return None


def _peak(world, cx, cy, half, step):
    """Highest surface hit on a grid -> (x, y, z) of the tallest feature (the spire)."""
    best = (cx, cy, -1e9)
    n = int(half // step)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x, y = cx + i * step, cy + j * step
            z = _ground_z(world, x, y)
            if z is not None and z > best[2]:
                best = (x, y, z)
    return best


def _spline(rig):
    comps = rig.get_components_by_class(unreal.SplineComponent)
    return comps[0] if comps else None


def main():
    cfg = {}
    if os.path.exists(CFG):
        try:
            cfg = json.load(open(CFG))
        except Exception:
            cfg = {}
    radius = float(cfg.get("radius_cm", 1700.0))
    n_pts = int(cfg.get("n", 20))
    fwd_cm = float(cfg.get("fwd_cm", 2500.0))

    ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = ues.get_editor_world()

    # Spire center: explicit override, else find the terrain peak ahead of the camera.
    if cfg.get("center_cm"):
        sx, sy = float(cfg["center_cm"][0]), float(cfg["center_cm"][1])
        sz = _ground_z(world, sx, sy) or 0.0
    else:
        cam = ues.get_level_viewport_camera_info()
        cloc, crot = cam[0], cam[1]
        fwd = crot.get_forward_vector()
        m = (fwd.x ** 2 + fwd.y ** 2) ** 0.5 or 1.0
        c0x, c0y = cloc.x + fwd.x / m * fwd_cm, cloc.y + fwd.y / m * fwd_cm
        cx, cy, _ = _peak(world, c0x, c0y, 3500.0, 500.0)     # coarse locate
        sx, sy, sz = _peak(world, cx, cy, 700.0, 175.0)       # refine

    # Ring of ground-snapped points around the spire.
    pts, grounds = [], []
    for k in range(n_pts):
        a = 2.0 * math.pi * k / n_pts
        x = sx + radius * math.cos(a)
        y = sy + radius * math.sin(a)
        gz = _ground_z(world, x, y)
        if gz is None:
            gz = sz
        pts.append(unreal.Vector(x, y, gz))
        grounds.append(gz)

    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for a in eas.get_all_level_actors():
        try:
            if a.actor_has_tag(TAG) or a.get_actor_label() == LABEL:
                eas.destroy_actor(a)
        except Exception:
            pass

    rail_cls = unreal.load_class(None, RAIL_CLASS_PATH)
    rig = eas.spawn_actor_from_class(rail_cls, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    rig.set_actor_label(LABEL)
    try:
        rig.set_editor_property("tags", [unreal.Name(TAG)])
    except Exception:
        pass

    sp = _spline(rig)
    sp.set_spline_points(pts, unreal.SplineCoordinateSpace.WORLD, True)
    sp.set_closed_loop(True, True)                            # CLOSED loop around the spire
    try:
        eas.set_selected_level_actors([rig])
    except Exception:
        pass

    gmin, gmax = min(grounds), max(grounds)
    print("SPIRE=(%.0f,%.0f,%.0f) radius=%.0fcm n=%d closed=1 ground=(%.0f..%.0f) loop_len=%.0fcm"
          % (sx, sy, sz, radius, n_pts, gmin, gmax, sp.get_spline_length()))


main()
