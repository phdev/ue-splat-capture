"""Capture an ORBIT around a hero spot in an ALREADY-OPEN level (live editor).

Unlike run_capture.py (which spawns the matte self-test diorama and kills Lumen),
this captures a real, already-streamed scene -- e.g. Electric Dreams (World
Partition + PCG) -- by orbiting wherever your viewport is aimed. Run it from the
editor's Python console:

    py "/Users/peterhowell/ue-splat-capture/ue_capture/capture_hero_orbit.py"

It (1) reads the active viewport camera, (2) ray-traces forward to find the hero
surface you're looking at (-> orbit centre + radius), (3) orbits a camera dome
around it, (4) renders FINAL_COLOR_LDR with the scene's real lighting (pinned
exposure for cross-view consistency; SSR/motion-blur off as they are the worst
view-dependent offenders a splat cannot fit), and (5) writes ue_poses.json.

Tunable via env (re-run if framing/exposure is off):
    UE_CAPTURE_OUT     output dir (default <repo>/out/electric_dreams)
    UE_ORBIT_RADIUS_CM force orbit radius (else auto from the ray hit distance)
    UE_FOCUS_DIST_CM   fallback focus distance if the ray misses (default 600)
    UE_CAP_RES         render resolution, px (default 512; ingest downsamples)
    UE_TRAIN_RES       target/train resolution the intrinsics describe (default 128)
    UE_HFOV            horizontal FOV degrees (default 70)
    UE_N_AZ            azimuth samples per ring (default 28 -> 4*28=112 cams)
    UE_CAPS_PER_POSE   captures per pose for Lumen/TSR convergence (default 4)
    UE_CAPTURE_EV      pinned exposure target (default 1.0; higher = brighter)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ue_capture import export, rig  # noqa: E402


def _flag(unreal, nm, on):
    s = unreal.EngineShowFlagsSetting()
    s.set_editor_property("show_flag_name", nm)
    s.set_editor_property("enabled", on)
    return s


def _impact(hit):
    for k in ("impact_point", "location"):
        try:
            v = hit.get_editor_property(k)
            if v is not None:
                return v
        except Exception:
            pass
    return None


def _focus_and_radius(unreal, world, ed):
    """Return (focus_cm[list3], radius_cm) from the live viewport + a forward ray."""
    cam_loc, cam_rot = ed.get_level_viewport_camera_info()
    if cam_loc is None:
        raise RuntimeError("No active level viewport -- click in the 3D viewport, then re-run.")
    fwd = unreal.MathLibrary.get_forward_vector(cam_rot)
    far = 200000.0  # 2 km
    start = cam_loc
    end = unreal.Vector(cam_loc.x + fwd.x * far, cam_loc.y + fwd.y * far, cam_loc.z + fwd.z * far)
    focus, dist = None, None
    try:
        res = unreal.SystemLibrary.line_trace_single(
            world, start, end, unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [],
            unreal.DrawDebugType.NONE, True)
        hit = res[1] if isinstance(res, (tuple, list)) else res
        ok = (res[0] if isinstance(res, (tuple, list)) else getattr(hit, "blocking_hit", False))
        imp = _impact(hit) if ok else None
        if imp is not None:
            focus = [imp.x, imp.y, imp.z]
            dist = ((imp.x - cam_loc.x) ** 2 + (imp.y - cam_loc.y) ** 2 + (imp.z - cam_loc.z) ** 2) ** 0.5
            unreal.log(f"[orbit] ray hit hero surface at {focus}, {dist/100:.1f} m away")
    except Exception as e:
        unreal.log_warning(f"[orbit] line trace failed ({e}); using fallback distance")
    if focus is None:
        dist = float(os.environ.get("UE_FOCUS_DIST_CM", "600"))
        focus = [cam_loc.x + fwd.x * dist, cam_loc.y + fwd.y * dist, cam_loc.z + fwd.z * dist]
        unreal.log(f"[orbit] no ray hit; focusing {dist/100:.1f} m ahead at {focus}")
    radius = float(os.environ.get("UE_ORBIT_RADIUS_CM", str(max(150.0, min(dist, 4000.0)))))
    return focus, radius


def main():
    import unreal

    out_dir = os.environ.get("UE_CAPTURE_OUT") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "electric_dreams")
    cap_res = int(os.environ.get("UE_CAP_RES", "512"))
    train_res = int(os.environ.get("UE_TRAIN_RES", "128"))
    hfov = float(os.environ.get("UE_HFOV", "70"))
    n_az = int(os.environ.get("UE_N_AZ", "28"))
    caps = int(os.environ.get("UE_CAPS_PER_POSE", "4"))
    ev = float(os.environ.get("UE_CAPTURE_EV", "1.0"))
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    ed = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = ed.get_editor_world()
    focus, radius = _focus_and_radius(unreal, world, ed)
    poses = rig.orbit_hemisphere(focus, radius, elevations_deg=(8.0, 22.0, 38.0, 55.0),
                                 n_azimuth=n_az, heldout_every=6)
    n_held = sum(p["split"] == "heldout" for p in poses)
    unreal.log(f"[orbit] focus={focus} radius={radius/100:.1f}m -> {len(poses)} cams "
               f"({len(poses)-n_held} train / {n_held} heldout), {cap_res}px, hfov {hfov}")

    # SceneCapture2D: real lighting kept; pin exposure; drop SSR/motion-blur/bloom.
    actor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
        unreal.SceneCapture2D, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    comp = actor.capture_component2d
    comp.fov_angle = hfov
    comp.capture_source = unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
    for k, v in [("capture_every_frame", False), ("capture_on_movement", False),
                 ("always_persist_rendering_state", True)]:
        try: comp.set_editor_property(k, v)
        except Exception: pass
    try:
        comp.set_editor_property("show_flag_settings", [
            _flag(unreal, "MotionBlur", False), _flag(unreal, "ScreenSpaceReflections", False),
            _flag(unreal, "Bloom", False)])
    except Exception: pass
    try:
        pp = comp.post_process_settings
        for k, v in [("auto_exposure_min_brightness", ev), ("auto_exposure_max_brightness", ev),
                     ("override_auto_exposure_min_brightness", True),
                     ("override_auto_exposure_max_brightness", True),
                     ("bloom_intensity", 0.0), ("override_bloom_intensity", True)]:
            pp.set_editor_property(k, v)
        comp.set_editor_property("post_process_settings", pp)
    except Exception: pass

    rt = unreal.RenderingLibrary.create_render_target2d(
        actor, cap_res, cap_res, unreal.TextureRenderTargetFormat.RTF_RGBA8)
    comp.texture_target = rt

    frames = []
    for i, p in enumerate(poses):
        loc = unreal.Vector(*p["location_cm"])
        rot = unreal.MathLibrary.find_look_at_rotation(loc, unreal.Vector(*p["target_cm"]))
        actor.set_actor_location_and_rotation(loc, rot, False, False)
        for _ in range(caps):           # converge Lumen/TSR at the new view
            comp.capture_scene()
        name = f"cam_{p['index']:03d}"
        unreal.RenderingLibrary.export_render_target(world, rt, out_dir + "/images", name)
        raw = os.path.join(out_dir, "images", name)
        png = raw + ".png"
        if os.path.exists(raw) and not os.path.exists(png):
            os.replace(raw, png)
        frames.append({"file_path": png, "split": p["split"],
                       "location_cm": export.location_from_actor(unreal, actor),
                       "basis_ue": export.basis_from_actor(unreal, actor)})
        if (i + 1) % 20 == 0:
            unreal.log(f"[orbit] rendered {i+1}/{len(poses)}")

    ext = radius * 0.85
    ue_poses = os.path.join(out_dir, "ue_poses.json")
    export.write_ue_poses(ue_poses, train_res, train_res, hfov, frames, scene_meta={
        "background": [0.0, 0.0, 0.0],
        "aabb_min_cm": [focus[0] - ext, focus[1] - ext, focus[2] - ext],
        "aabb_max_cm": [focus[0] + ext, focus[1] + ext, focus[2] + ext],
        "fiducials": [], "primitives": []})

    try:
        unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
    except Exception:
        pass
    unreal.log(f"[orbit] DONE -> {ue_poses}")
    print(f"WROTE {ue_poses}  ({len(frames)} frames, focus {focus}, radius {radius/100:.1f} m)")
    print(f"CHECK a sample render: {out_dir}/images/cam_000.png")


main()
