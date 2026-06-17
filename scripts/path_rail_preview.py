"""Set up a PIE preview of the capture rail: a Cine Camera flies along
CAPTURE_PATH_RAIL when you Play, so you can check the route before capturing.

Run INSIDE the warm editor (keep the UE window FOREGROUND):
    python3 scripts/ue_exec.py scripts/path_rail_preview.py 150

Approach: BAKE the flythrough into the sequence. We sample the rail spline by arc
length, snap each sample to the terrain, and key a CineCamera's world transform
(position at eye height, yaw facing forward along travel) over DUR seconds, with a
Camera Cut bound to it and an auto-play + looping LevelSequenceActor. Baking (vs
attaching the camera to the rail + animating CurrentPositionOnRail) is what actually
moves the camera in PIE -- the rail only repositions attached actors in-editor.
Re-run after you reshape the loop to re-bake. Optional /tmp/preview_cfg.json {dur,eye}.
"""
import json
import math
import os
import unreal

TAG_RAIL = "CAPTURE_PATH"
TAG_CAM = "PATH_PREVIEW_CAM"
TAG_SEQACT = "PATH_PREVIEW_SEQ"
SEQ_DIR = "/Game/CapturePath"
SEQ_NAME = "PathPreview"


def find_by_tag(eas, tag):
    for a in eas.get_all_level_actors():
        try:
            if a.actor_has_tag(tag):
                return a
        except Exception:
            pass
    return None


def ground_z(world, x, y, z_hint):
    start = unreal.Vector(x, y, z_hint + 20000.0)
    end = unreal.Vector(x, y, z_hint - 200000.0)
    try:
        hit = unreal.SystemLibrary.line_trace_single(
            world, start, end, unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, False, [],
            unreal.DrawDebugTrace.NONE, True)
        hr = hit[1] if isinstance(hit, (tuple, list)) else hit
        if hr:
            t = hr.to_tuple()
            if t[0]:
                return float(t[5].z)
    except Exception:
        pass
    return z_hint


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
    ues = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = ues.get_editor_world()
    rail = find_by_tag(eas, TAG_RAIL)
    if not rail:
        print("NO_RAIL")
        return

    for t in (TAG_CAM, TAG_SEQACT):
        a = find_by_tag(eas, t)
        while a:
            eas.destroy_actor(a)
            a = find_by_tag(eas, t)

    sp = rail.get_components_by_class(unreal.SplineComponent)[0]
    L = sp.get_spline_length()
    N = max(48, min(160, int(L / 100.0)))             # ~1 key/m
    WS = unreal.SplineCoordinateSpace.WORLD
    samples, prev_yaw = [], None
    for i in range(N + 1):
        d = L * i / N                                  # 0..L (closed loop -> last≈first)
        loc = sp.get_location_at_distance_along_spline(d, WS)
        dirc = sp.get_direction_at_distance_along_spline(d, WS)
        gz = ground_z(world, loc.x, loc.y, loc.z)
        yaw = math.degrees(math.atan2(dirc.y, dirc.x))
        if prev_yaw is not None:                        # unwrap so it doesn't spin at ±180
            while yaw - prev_yaw > 180.0:
                yaw -= 360.0
            while yaw - prev_yaw < -180.0:
                yaw += 360.0
        prev_yaw = yaw
        samples.append(((loc.x, loc.y, gz + EYE), yaw))

    x0, y0, z0 = samples[0][0]

    # Fresh, uniquely-named sequence (create_asset auto-opens -> delete is unreliable).
    aes = unreal.get_editor_subsystem(unreal.AssetEditorSubsystem)
    name, i = SEQ_NAME, 0
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
    tr = seq.get_tick_resolution()
    tps = tr.numerator / tr.denominator

    # SPAWNABLE camera: a possessable bound to a Python-spawned level actor does not
    # reliably resolve/drive in PIE; a spawnable is owned by the sequence so it always binds.
    tmpl = eas.spawn_actor_from_class(unreal.CineCameraActor, unreal.Vector(x0, y0, z0),
                                      unreal.Rotator(0.0, samples[0][1], 0.0))
    tmpl.set_actor_label("PATH_PREVIEW_CAM_TMPL")
    try:
        cb = seq.add_spawnable_from_instance(tmpl)
    except Exception:
        cb = unreal.MovieSceneSequenceExtensions.add_spawnable_from_instance(seq, tmpl)
    eas.destroy_actor(tmpl)                            # spawnable keeps its own copy
    print("SPAWNABLE_OK %s" % cb.get_display_name())
    tt = cb.add_track(unreal.MovieScene3DTransformTrack)
    ts = tt.add_section()
    ts.set_start_frame_seconds(0.0)
    ts.set_end_frame_seconds(DUR)
    ch = ts.get_all_channels()                          # [locX,Y,Z, rotRoll,Pitch,Yaw, sclX,Y,Z]
    print("CHANNELS=%d" % len(ch))
    for k, (loc, yaw) in enumerate(samples):
        f = unreal.FrameNumber(int(round(DUR * k / (len(samples) - 1) * tps)))
        ch[0].add_key(f, loc[0])
        ch[1].add_key(f, loc[1])
        ch[2].add_key(f, loc[2])
        ch[3].add_key(f, 0.0)
        ch[4].add_key(f, 0.0)
        ch[5].add_key(f, yaw)
    for s in (6, 7, 8):                                  # keep scale = 1 (unkeyed can default to 0)
        ch[s].add_key(unreal.FrameNumber(0), 1.0)
    print("BAKE_OK keys=%d" % len(samples))

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
    unreal.EditorAssetLibrary.save_asset(seq_full)

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
    try:
        lc = ps.get_editor_property("loop_count")
        try:
            lc.set_editor_property("value", -1)
            ps.set_editor_property("loop_count", lc)
        except Exception:
            ps.set_editor_property("loop_count", -1)
    except Exception:
        pass
    lsa.set_editor_property("playback_settings", ps)

    print("PREVIEW_DONE seq=%s dur=%.0fs eye=%.0f keys=%d autoplay=1" % (seq_full, DUR, EYE, len(samples)))


try:
    main()
except Exception:
    import traceback
    print("PREVIEW_EXC\n" + traceback.format_exc())
