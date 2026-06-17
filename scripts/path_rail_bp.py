"""Build a graph-free Blueprint that flies a camera along CAPTURE_PATH_RAIL on Play
(BeginPlay), so the capture route can be previewed in PIE.

Run INSIDE the warm editor (keep the UE window FOREGROUND):
    python3 scripts/ue_exec.py scripts/path_rail_bp.py 150

BP = subclass of CineCameraActor + an InterpToMovementComponent whose control points are
the (ground-snapped, eye-height) path samples, Duration=DUR, Loop. The placed instance has
auto_activate_for_player = Player0, so on Play the player view becomes this camera and the
InterpToMovementComponent moves it along the path -- no event-graph nodes (which Python
can't author). Idempotent. Optional /tmp/preview_cfg.json {dur,eye}.
"""
import json
import os
import unreal

BP_DIR = "/Game/CapturePath"
BP_NAME = "BP_PathFly"
BP_FULL = BP_DIR + "/" + BP_NAME
TAG_RAIL = "CAPTURE_PATH"
TAG_FLY = "PATH_FLY_CAM"


def gz(world, x, y, zh):
    hit = unreal.SystemLibrary.line_trace_single(
        world, unreal.Vector(x, y, zh + 20000), unreal.Vector(x, y, zh - 200000),
        unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, False, [], unreal.DrawDebugTrace.NONE, True)
    hr = hit[1] if isinstance(hit, (tuple, list)) else hit
    if hr:
        t = hr.to_tuple()
        if t[0]:
            return float(t[5].z)
    return zh


def main():
    cfg = {}
    if os.path.exists("/tmp/preview_cfg.json"):
        try:
            cfg = json.load(open("/tmp/preview_cfg.json"))
        except Exception:
            cfg = {}
    DUR = float(cfg.get("dur", 30.0))
    EYE = float(cfg.get("eye", 480.0))

    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = ues.get_editor_world()
    rail = None
    for a in eas.get_all_level_actors():
        try:
            if a.actor_has_tag(TAG_RAIL):
                rail = a
                break
        except Exception:
            pass
    if not rail:
        print("NO_RAIL")
        return

    sp = rail.get_components_by_class(unreal.SplineComponent)[0]
    L = sp.get_spline_length()
    N = max(24, min(60, int(L / 250)))
    WS = unreal.SplineCoordinateSpace.WORLD
    world_pts = []
    for i in range(N + 1):
        loc = sp.get_location_at_distance_along_spline(L * i / N, WS)
        world_pts.append(unreal.Vector(loc.x, loc.y, gz(world, loc.x, loc.y, loc.z) + EYE))
    origin = world_pts[0]
    print("PTS %d origin=(%.0f,%.0f,%.0f)" % (len(world_pts), origin.x, origin.y, origin.z))

    # Fresh BP.
    aes = unreal.get_editor_subsystem(unreal.AssetEditorSubsystem)
    if unreal.EditorAssetLibrary.does_asset_exist(BP_FULL):
        try:
            aes.close_all_editors_for_asset(unreal.EditorAssetLibrary.load_asset(BP_FULL))
        except Exception:
            pass
        unreal.EditorAssetLibrary.delete_asset(BP_FULL)
    f = unreal.BlueprintFactory()
    f.set_editor_property("parent_class", unreal.CineCameraActor)
    at = unreal.AssetToolsHelpers.get_asset_tools()
    bp = at.create_asset(BP_NAME, BP_DIR, None, f)
    print("BP_CREATED %s" % (bp.get_name() if bp else None))

    # Add InterpToMovementComponent via SubobjectDataSubsystem.
    sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    roots = sds.k2_gather_subobject_data_for_blueprint(bp)
    print("ROOTS %d" % len(roots))
    params = unreal.AddNewSubobjectParams()
    params.set_editor_property("parent_handle", roots[0])
    params.set_editor_property("new_class", unreal.InterpToMovementComponent)
    params.set_editor_property("blueprint_context", bp)
    new_handle, fail = sds.add_new_subobject(params)
    print("ADDSUB fail=%s" % str(fail))
    try:
        sds.rename_subobject(new_handle, unreal.Text("InterpMove"))
    except Exception as e:
        print("rename_err", str(e)[:40])
    data = sds.k2_find_subobject_data_from_handle(new_handle)
    comp = unreal.SubobjectDataBlueprintFunctionLibrary.get_object(data)
    print("COMP %s" % type(comp).__name__)

    # Configure InterpToMovement: relative control points, duration, loop.
    cps = []
    for v in world_pts:
        cp = unreal.InterpControlPoint()
        try:
            cp.set_editor_property("position_control_point", unreal.Vector(v.x - origin.x, v.y - origin.y, v.z - origin.z))
        except Exception as e:
            print("cp_err", str(e)[:50])
        try:
            cp.set_editor_property("position_is_relative", True)
        except Exception:
            pass
        cps.append(cp)
    for prop, val in [("control_points", cps), ("duration", DUR)]:
        try:
            comp.set_editor_property(prop, val)
        except Exception as e:
            print("set_err", prop, str(e)[:50])
    beh = None
    for nm in ("LOOP_RESET", "LOOP", "PING_PONG", "ONE_SHOT"):
        if hasattr(unreal.InterpToBehaviourType, nm):
            beh = getattr(unreal.InterpToBehaviourType, nm)
            break
    if beh is not None:
        try:
            comp.set_editor_property("behaviour_type", beh)
            print("BEH set %s" % nm)
        except Exception as e:
            print("beh_err", str(e)[:50])
    try:
        print("CP_CHECK n=%d" % len(comp.get_editor_property("control_points")))
    except Exception as e:
        print("cp_check_err", str(e)[:50])

    unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    unreal.EditorAssetLibrary.save_asset(BP_FULL)
    print("BP_SAVED")

    # Place an instance, auto-activate for player 0.
    for a in list(eas.get_all_level_actors()):
        try:
            if a.actor_has_tag(TAG_FLY):
                eas.destroy_actor(a)
        except Exception:
            pass
    gen = unreal.EditorAssetLibrary.load_blueprint_class(BP_FULL)
    inst = eas.spawn_actor_from_class(gen, origin, unreal.Rotator(0, 0, 0))
    inst.set_actor_label("PATH_FLY_CAM")
    try:
        inst.set_editor_property("tags", [unreal.Name(TAG_FLY)])
    except Exception:
        pass
    try:
        inst.set_editor_property("auto_activate_for_player", unreal.AutoReceiveInput.PLAYER0)
    except Exception as e:
        print("autoact_err", str(e)[:50])
    print("INSTANCE_PLACED label=%s" % inst.get_actor_label())

    # Spire-facing: CineCamera Look-at Tracking aimed at a target at the loop center
    # (InterpToMovement only translates). Fallback to a fixed spire-facing yaw.
    cz = gz(world, 89712.0, -5226.0, 2000.0)
    spire = unreal.Vector(89712.0, -5226.0, cz + 900.0)
    for a in list(eas.get_all_level_actors()):
        try:
            if a.actor_has_tag("PATH_FLY_TARGET"):
                eas.destroy_actor(a)
        except Exception:
            pass
    tgt = eas.spawn_actor_from_class(unreal.TargetPoint, spire, unreal.Rotator(0, 0, 0))
    tgt.set_actor_label("PATH_FLY_TARGET")
    try:
        tgt.set_editor_property("tags", [unreal.Name("PATH_FLY_TARGET")])
    except Exception:
        pass
    laset = False
    try:
        cc = inst.get_cine_camera_component()
        for pn in ("look_at_tracking_settings", "lookat_tracking_settings"):
            try:
                lat = cc.get_editor_property(pn)
                lat.set_editor_property("enable_look_at_tracking", True)
                lat.set_editor_property("actor_to_track", tgt)
                cc.set_editor_property(pn, lat)
                laset = True
                print("LOOKAT set via %s" % pn)
                break
            except Exception as e:
                print("lookat_try %s %s" % (pn, str(e)[:40]))
    except Exception as e:
        print("lookat_err", str(e)[:50])
    if not laset:
        try:
            inst.set_actor_rotation(unreal.Rotator(yaw=180.0), False)
            print("LOOKAT fallback fixed yaw=180")
        except Exception as e:
            print("rot_err", str(e)[:40])
    print("BP_DONE %s dur=%.0fs pts=%d" % (BP_FULL, DUR, len(world_pts)))


main()
