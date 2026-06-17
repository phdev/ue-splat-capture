"""Set up a PIE preview of the capture rail: a Cine Camera flies along
CAPTURE_PATH_RAIL when you Play, so you can check the route before capturing.

Run INSIDE the warm editor (keep the UE window FOREGROUND):
    python3 scripts/ue_exec.py scripts/path_rail_preview.py 120

Builds: a CineCameraActor attached to the rail (rides the spline), a LevelSequence
(/Game/CapturePath/PathPreview) that animates the rail's CurrentPositionOnRail 0->1
over DUR seconds with a Camera Cut bound to that camera, and an auto-play
LevelSequenceActor -> hitting Play flies the camera around the loop. Because the
camera FOLLOWS the rail, it always reflects your latest spline edits (no re-bake).
Idempotent: re-running rebuilds the camera + sequence. Optional /tmp/preview_cfg.json
{"dur":30,"eye":480}.
"""
import json
import os
import unreal

TAG_RAIL = "CAPTURE_PATH"
TAG_CAM = "PATH_PREVIEW_CAM"
TAG_SEQACT = "PATH_PREVIEW_SEQ"
SEQ_DIR = "/Game/CapturePath"
SEQ_NAME = "PathPreview"
SEQ_FULL = SEQ_DIR + "/" + SEQ_NAME


def find_by_tag(eas, tag):
    for a in eas.get_all_level_actors():
        try:
            if a.actor_has_tag(tag):
                return a
        except Exception:
            pass
    return None


def main():
    cfg = {}
    if os.path.exists("/tmp/preview_cfg.json"):
        try:
            cfg = json.load(open("/tmp/preview_cfg.json"))
        except Exception:
            cfg = {}
    DUR = float(cfg.get("dur", 30.0))
    EYE = float(cfg.get("eye", 480.0))
    FPS = 30

    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    rail = find_by_tag(eas, TAG_RAIL)
    if not rail:
        print("NO_RAIL")
        return

    # Clean prior preview actors.
    for t in (TAG_CAM, TAG_SEQACT):
        a = find_by_tag(eas, t)
        while a:
            eas.destroy_actor(a)
            a = find_by_tag(eas, t)

    rail.set_editor_property("lock_orientation_to_rail", True)
    sp = rail.get_components_by_class(unreal.SplineComponent)[0]
    start = sp.get_location_at_distance_along_spline(0.0, unreal.SplineCoordinateSpace.WORLD)

    cam = eas.spawn_actor_from_class(unreal.CineCameraActor,
                                     unreal.Vector(start.x, start.y, start.z + EYE),
                                     unreal.Rotator(0, 0, 0))
    cam.set_actor_label("PATH_PREVIEW_CAM")
    try:
        cam.set_editor_property("tags", [unreal.Name(TAG_CAM)])
    except Exception:
        pass
    cam.attach_to_actor(rail, "", unreal.AttachmentRule.KEEP_WORLD,
                        unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD, False)
    cam.set_actor_relative_location(unreal.Vector(0, 0, EYE), False, False)

    # Validate the rail moves the attached camera: nudge position, see if it travels.
    rail.set_editor_property("current_position_on_rail", 0.0)
    p0 = cam.get_actor_location()
    rail.set_editor_property("current_position_on_rail", 0.5)
    p1 = cam.get_actor_location()
    rail.set_editor_property("current_position_on_rail", 0.0)
    moved = ((p0.x - p1.x) ** 2 + (p0.y - p1.y) ** 2 + (p0.z - p1.z) ** 2) ** 0.5
    print("ATTACH_MOVE=%.0fcm (rail drives camera)" % moved)

    # Build the level sequence.
    # Always create a FRESH, uniquely-named sequence: create_asset returns None on a
    # name collision, and deleting an asset that's open in a Sequencer tab is unreliable.
    # Best-effort clean the old one; if it won't delete, just use the next free name.
    aes = unreal.get_editor_subsystem(unreal.AssetEditorSubsystem)
    name = SEQ_NAME
    i = 0
    while unreal.EditorAssetLibrary.does_asset_exist(SEQ_DIR + "/" + name):
        try:
            aes.close_all_editors_for_asset(unreal.EditorAssetLibrary.load_asset(SEQ_DIR + "/" + name))
            unreal.EditorAssetLibrary.delete_asset(SEQ_DIR + "/" + name)
        except Exception:
            pass
        if unreal.EditorAssetLibrary.does_asset_exist(SEQ_DIR + "/" + name):
            i += 1
            name = "%s_%d" % (SEQ_NAME, i)
    seq_full = SEQ_DIR + "/" + name
    at = unreal.AssetToolsHelpers.get_asset_tools()
    seq = at.create_asset(name, SEQ_DIR, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    seq.set_display_rate(unreal.FrameRate(FPS, 1))
    seq.set_playback_start_seconds(0.0)
    seq.set_playback_end_seconds(DUR)
    endf = int(FPS * DUR)

    # Rail CurrentPositionOnRail 0 -> 1.
    rb = seq.add_possessable(rail)
    ft = rb.add_track(unreal.MovieSceneFloatTrack)
    ft.set_property_name_and_path("CurrentPositionOnRail", "CurrentPositionOnRail")
    fs = ft.add_section()
    fs.set_start_frame_seconds(0.0)
    fs.set_end_frame_seconds(DUR)
    tr = seq.get_tick_resolution()
    tps = tr.numerator / tr.denominator           # ticks per second
    ch = fs.get_all_channels()[0]
    ch.add_key(unreal.FrameNumber(0), 0.0)
    ch.add_key(unreal.FrameNumber(int(round(DUR * tps))), 1.0)
    print("RAIL_TRACK_OK")

    # Camera + camera cut bound to it.
    cb = seq.add_possessable(cam)
    try:
        cut = seq.add_track(unreal.MovieSceneCameraCutTrack)
    except Exception:
        cut = seq.add_master_track(unreal.MovieSceneCameraCutTrack)
    cs = cut.add_section()
    cs.set_start_frame_seconds(0.0)
    cs.set_end_frame_seconds(DUR)
    try:
        bid = unreal.MovieSceneSequenceExtensions.get_binding_id(seq, cb)
    except Exception:
        bid = cb.get_binding_id()
    cs.set_camera_binding_id(bid)
    print("CUT_OK")

    unreal.EditorAssetLibrary.save_asset(seq_full)

    # Auto-play actor.
    lsa = eas.spawn_actor_from_class(unreal.LevelSequenceActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    lsa.set_actor_label("PATH_PREVIEW_SEQ")
    try:
        lsa.set_editor_property("tags", [unreal.Name(TAG_SEQACT)])
    except Exception:
        pass
    try:
        lsa.set_sequence(seq)
    except Exception:
        lsa.set_editor_property("level_sequence", unreal.SoftObjectPath(seq_full))
    ps = lsa.get_editor_property("playback_settings")
    ps.set_editor_property("auto_play", True)
    try:                                              # loop forever for inspection
        lc = ps.get_editor_property("loop_count")
        try:
            lc.set_editor_property("value", -1)       # struct form
            ps.set_editor_property("loop_count", lc)
        except Exception:
            ps.set_editor_property("loop_count", -1)   # plain-int form
    except Exception:
        pass
    lsa.set_editor_property("playback_settings", ps)

    print("PREVIEW_DONE seq=%s dur=%.0fs eye=%.0f autoplay=1" % (seq_full, DUR, EYE))


try:
    main()
except Exception:
    import traceback
    print("PREVIEW_EXC\n" + traceback.format_exc())
