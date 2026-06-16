"""Read the CAPTURE_PATH_RAIL spline the user shaped (see scripts/path_rail_seed.py)
and write a waypoints route for scripts/capture_path.py.

Run INSIDE the warm editor:
    python3 scripts/ue_exec.py scripts/path_rail_read.py 120

Samples the rig-rail spline densely by arc-length (preserving the curve the user
shaped), snaps each sample DOWN to the terrain (the path_fan rig treats waypoint z
as LOCAL GROUND and sits the camera eye_cm above it), and writes
/tmp/ed_path_rail.json = {"waypoints": [[x,y,groundz]_cm], "ground_cm": median}.
Then capture:  python3 scripts/capture_path.py --path /tmp/ed_path_rail.json --prefix ed_path
"""
import json
import unreal

TAG = "CAPTURE_PATH"
LABEL = "CAPTURE_PATH_RAIL"
OUT = "/tmp/ed_path_rail.json"
SAMPLE_CM = 150.0      # dense arc-length sampling -> faithful to the shaped curve


def _ground_z(world, x, y, z_hint):
    start = unreal.Vector(x, y, z_hint + 20000.0)
    end = unreal.Vector(x, y, z_hint - 200000.0)
    try:
        hit = unreal.SystemLibrary.line_trace_single(
            world, start, end, unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, False, [],
            unreal.DrawDebugTrace.NONE, True)
        hr = hit[1] if isinstance(hit, (tuple, list)) else hit
        if hr:
            t = hr.to_tuple()           # (blocking, overlap, time, dist, location, impact_point, ...)
            if t[0]:
                return float(t[5].z)
    except Exception:
        pass
    return z_hint


def _find_rail(eas):
    for a in eas.get_all_level_actors():
        try:
            if a.actor_has_tag(TAG) or a.get_actor_label() == LABEL:
                return a
        except Exception:
            pass
    return None


def main():
    ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = ues.get_editor_world()
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

    rig = _find_rail(eas)
    if not rig:
        print("NO_RAIL: place one first with scripts/path_rail_seed.py")
        return
    comps = rig.get_components_by_class(unreal.SplineComponent)
    if not comps:
        print("NO_SPLINE on the rail actor")
        return
    sp = comps[0]
    L = sp.get_spline_length()
    if L < 1.0:
        print("EMPTY_SPLINE length=%.1f" % L)
        return

    n = int(L // SAMPLE_CM)
    wps, grounds = [], []
    for i in range(n + 1):
        d = min(L, i * SAMPLE_CM)
        loc = sp.get_location_at_distance_along_spline(d, unreal.SplineCoordinateSpace.WORLD)
        gz = _ground_z(world, loc.x, loc.y, loc.z)
        wps.append([round(loc.x, 1), round(loc.y, 1), round(gz, 1)])
        grounds.append(gz)

    grounds.sort()
    gmid = grounds[len(grounds) // 2]
    json.dump({"waypoints": wps, "ground_cm": round(gmid, 1)}, open(OUT, "w"))

    xs = [w[0] for w in wps]
    ys = [w[1] for w in wps]
    print("ROUTE_OK out=%s n=%d length=%.0fcm x=(%.0f..%.0f) y=(%.0f..%.0f) ground~%.0f"
          % (OUT, len(wps), L, min(xs), max(xs), min(ys), max(ys), gmid))


main()
