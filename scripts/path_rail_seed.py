"""Seed a CineCameraRigRail in the OPEN editor's CURRENT scene/view so the user can
SHAPE the capture route by dragging spline control points. Later
scripts/path_rail_read.py samples the spline back into a waypoints route for
capture_path.py.

Run INSIDE the warm editor:
    python3 scripts/ue_exec.py scripts/path_rail_seed.py 120

Seeds a SHORT starter spline (a few sparse control points — easy to drag; the curve
interpolates and the reader re-samples it) laid along the ground in front of the
CURRENT viewport camera, in whatever level is loaded right now. No external coords.
Destroys any prior CAPTURE_PATH rail first (idempotent). Prints a confirmation line.
"""
import unreal

LABEL = "CAPTURE_PATH_RAIL"
TAG = "CAPTURE_PATH"
SPAN_CM = 2400.0     # starter path length along view-forward (~24 m)
N_PTS = 6            # starter control points
TRACE_UP = 20000.0   # start the down-trace this far above the sample
TRACE_DN = 200000.0  # trace this far down to hit ground


def _ground_z(world, x, y, z_hint):
    """Down-trace at (x,y) -> ground Z; fall back to z_hint on a miss."""
    start = unreal.Vector(x, y, z_hint + TRACE_UP)
    end = unreal.Vector(x, y, z_hint - TRACE_DN)
    try:
        hit = unreal.SystemLibrary.line_trace_single(
            world, start, end, unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, False, [],
            unreal.DrawDebugTrace.NONE, True)
        # HitResult fields are protected in Python; to_tuple() unpacks them:
        # (blocking_hit, initial_overlap, time, distance, location, impact_point, ...)
        hr = hit[1] if isinstance(hit, (tuple, list)) else hit
        if hr:
            t = hr.to_tuple()
            if t[0]:                      # blocking_hit
                return float(t[5].z)      # impact_point.z = ground
    except Exception as e:
        print("trace_miss %s" % e)
    return z_hint


# The Camera Rig Rail (classic cinematic spline-path actor) isn't in the unreal
# Python namespace in this build, but its UClass loads by path. Its editable
# RailSplineComponent is a USplineComponent we fetch generically.
RAIL_CLASS_PATH = "/Script/CinematicCamera.CameraRig_Rail"


def _spline(rig):
    try:
        comps = rig.get_components_by_class(unreal.SplineComponent)
        if comps and len(comps) > 0:
            return comps[0]
    except Exception as e:
        print("spline_find_err %s" % e)
    try:
        return rig.get_rail_spline_component()
    except Exception:
        return None


def main():
    ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    cam = ues.get_level_viewport_camera_info()        # (location, rotation)
    cloc, crot = cam[0], cam[1]
    fwd = crot.get_forward_vector()
    # Flatten forward onto the ground plane so the path lays along terrain.
    fx, fy = fwd.x, fwd.y
    m = (fx * fx + fy * fy) ** 0.5 or 1.0
    fx, fy = fx / m, fy / m

    world = ues.get_editor_world()
    # Ground directly under the camera anchors the starter Z.
    g0 = _ground_z(world, cloc.x, cloc.y, cloc.z - 300.0)
    # Lay points starting a touch in front of the camera so they're on-screen.
    base_x = cloc.x + fx * 400.0
    base_y = cloc.y + fy * 400.0

    pts = []
    for i in range(N_PTS):
        d = (SPAN_CM / (N_PTS - 1)) * i
        x = base_x + fx * d
        y = base_y + fy * d
        z = _ground_z(world, x, y, g0)
        pts.append((x, y, z))

    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for a in eas.get_all_level_actors():
        try:
            if a.actor_has_tag(TAG) or a.get_actor_label() == LABEL:
                eas.destroy_actor(a)
        except Exception:
            pass

    rail_cls = unreal.load_class(None, RAIL_CLASS_PATH)
    rig = eas.spawn_actor_from_class(rail_cls,
                                     unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    rig.set_actor_label(LABEL)
    try:
        rig.set_editor_property("tags", [unreal.Name(TAG)])
    except Exception:
        pass

    spline = _spline(rig)
    vecs = [unreal.Vector(*p) for p in pts]
    spline.set_spline_points(vecs, unreal.SplineCoordinateSpace.WORLD, True)
    try:
        spline.set_closed_loop(False, True)
    except Exception:
        pass

    # Frame + select the new rail so it's obvious in the viewport.
    try:
        eas.set_selected_level_actors([rig])
    except Exception:
        pass

    n = spline.get_number_of_spline_points()
    length = spline.get_spline_length()
    print("CAM=(%.0f,%.0f,%.0f) fwd=(%.2f,%.2f) ground0=%.0f"
          % (cloc.x, cloc.y, cloc.z, fx, fy, g0))
    print("RAIL_OK label=%s tag=%s points=%d length=%.0fcm start=(%.0f,%.0f,%.0f) end=(%.0f,%.0f,%.0f)"
          % (rig.get_actor_label(), TAG, n, length,
             pts[0][0], pts[0][1], pts[0][2], pts[-1][0], pts[-1][1], pts[-1][2]))


main()
