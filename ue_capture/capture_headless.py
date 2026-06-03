"""HEADLESS capture of an already-authored level (World Partition + PCG aware).

Launched by capture_headless_run.sh via UnrealEditor-Cmd. Unlike run_capture.py
(matte self-test) and capture_hero_orbit.py (live editor), this drives a closed
editor: it loads the level, streams in the World-Partition actors, (re)generates
PCG, auto-frames a hero spot from the actor bounds (there is no viewport to read
headless), orbits a camera dome, and writes ue_poses.json.

Two modes via env UE_PROBE:
  UE_PROBE=1  -> load + report actor/geometry counts + bounds + a few OVERVIEW
                 frames (cheap; validates that geometry actually loaded headless).
  UE_PROBE=0  -> full orbit capture (writes ue_poses.json).

Env: UE_CAPTURE_OUT, UE_LEVEL, UE_FOCUS_CM="x,y,z", UE_ORBIT_RADIUS_CM,
     UE_CAP_RES (512), UE_TRAIN_RES (128), UE_HFOV (70), UE_N_AZ (28),
     UE_CAPS_PER_POSE (4), UE_CAPTURE_EV (1.0), UE_MAX_LOAD (4000).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ue_capture import export, rig  # noqa: E402

LEVEL = os.environ.get("UE_LEVEL", "/Game/Levels/PCG/ElectricDreams_PCGCloseRange")


def _flag(unreal, nm, on):
    s = unreal.EngineShowFlagsSetting()
    s.set_editor_property("show_flag_name", nm)
    s.set_editor_property("enabled", on)
    return s


def _box_center_ext(b):
    """FBox -> (center[3], half-diagonal). Defensive about validity."""
    mn, mx = b.min, b.max
    c = [(mn.x + mx.x) * 0.5, (mn.y + mx.y) * 0.5, (mn.z + mx.z) * 0.5]
    half = (((mx.x - mn.x) ** 2 + (mx.y - mn.y) ** 2 + (mx.z - mn.z) ** 2) ** 0.5) * 0.5
    return c, half


def _get_descs(unreal):
    try:
        res = unreal.WorldPartitionBlueprintLibrary.get_actor_descs()
        descs = res[1] if isinstance(res, (tuple, list)) else res
        return list(descs)
    except Exception as e:
        unreal.log_warning(f"[hl] get_actor_descs failed: {e}")
        return []


def _focus_radius(unreal):
    """Auto-frame: centroid of WP actor bounds (where the geometry is), radius
    from the spread. Overridable via env."""
    descs = _get_descs(unreal)
    unreal.log(f"[hl] world-partition actor descriptors: {len(descs)}")
    centers, valids = [], []
    for d in descs:
        try:
            b = d.get_editor_property("bounds")
            c, half = _box_center_ext(b)
            if half > 1.0 and half < 5.0e5:        # skip degenerate / world-sized
                centers.append(c); valids.append((d, c, half))
        except Exception:
            pass
    env_focus = os.environ.get("UE_FOCUS_CM")
    if env_focus:
        focus = [float(x) for x in env_focus.split(",")]
    elif centers:
        focus = [sum(c[i] for c in centers) / len(centers) for i in range(3)]
    else:
        try:
            wb = unreal.WorldPartitionBlueprintLibrary.get_editor_world_bounds()
            focus, _ = _box_center_ext(wb)
        except Exception:
            focus = [0.0, 0.0, 0.0]
    if centers:
        dists = sorted(((c[0]-focus[0])**2+(c[1]-focus[1])**2+(c[2]-focus[2])**2)**0.5 for c in centers)
        spread = dists[len(dists)//2]              # median distance from centroid
    else:
        spread = 600.0
    radius = float(os.environ.get("UE_ORBIT_RADIUS_CM", str(max(250.0, min(spread, 4000.0)))))
    return focus, radius, valids


def _load_region(unreal, focus, radius, valids):
    """Stream in the WP actors near the capture region (cap to avoid loading a
    giant world into a RAM-tight machine)."""
    max_load = int(os.environ.get("UE_MAX_LOAD", "4000"))
    valids.sort(key=lambda t: ((t[1][0]-focus[0])**2+(t[1][1]-focus[1])**2+(t[1][2]-focus[2])**2))
    near = valids[:max_load]
    guids = []
    for d, _c, _h in near:
        try:
            guids.append(d.get_editor_property("guid"))
        except Exception:
            pass
    unreal.log(f"[hl] loading {len(guids)} actors near focus (of {len(valids)} valid)")
    try:
        unreal.WorldPartitionBlueprintLibrary.load_actors(guids)   # synchronous
    except Exception as e:
        unreal.log_warning(f"[hl] load_actors failed: {e}")


def _generate_pcg(unreal):
    n = 0
    try:
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    except Exception:
        actors = []
    for a in actors:
        try:
            pcg = a.get_component_by_class(unreal.PCGComponent)
            if pcg:
                pcg.generate_local(True); n += 1
        except Exception:
            pass
    unreal.log(f"[hl] triggered PCG generate on {n} components (cached results load "
               "synchronously; async regen can't complete in a blocking script)")
    return n


def _count_geometry(unreal):
    sm = ism = 0
    try:
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    except Exception:
        actors = []
    for a in actors:
        try:
            if a.get_component_by_class(unreal.StaticMeshComponent):
                sm += 1
            if a.get_component_by_class(unreal.InstancedStaticMeshComponent):
                ism += 1
        except Exception:
            pass
    return len(actors), sm, ism


def _setup_capture(unreal, res, hfov, ev):
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
        actor, res, res, unreal.TextureRenderTargetFormat.RTF_RGBA8)
    comp.texture_target = rt
    return actor, comp, rt


def _render(unreal, world, poses, comp, rt, actor, caps, out_dir):
    frames = []
    for i, p in enumerate(poses):
        loc = unreal.Vector(*p["location_cm"])
        rot = unreal.MathLibrary.find_look_at_rotation(loc, unreal.Vector(*p["target_cm"]))
        actor.set_actor_location_and_rotation(loc, rot, False, False)
        for _ in range(caps):
            comp.capture_scene()
        name = f"cam_{p['index']:03d}"
        unreal.RenderingLibrary.export_render_target(world, rt, out_dir + "/images", name)
        raw = os.path.join(out_dir, "images", name); png = raw + ".png"
        if os.path.exists(raw) and not os.path.exists(png):
            os.replace(raw, png)
        frames.append({"file_path": png, "split": p["split"],
                       "location_cm": export.location_from_actor(unreal, actor),
                       "basis_ue": export.basis_from_actor(unreal, actor)})
        if (i + 1) % 20 == 0:
            unreal.log(f"[hl] rendered {i+1}/{len(poses)}")
    return frames


def main():
    import unreal
    out_dir = os.environ.get("UE_CAPTURE_OUT") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "electric_dreams")
    probe = os.environ.get("UE_PROBE", "0") == "1"
    cap_res = int(os.environ.get("UE_CAP_RES", "512"))
    train_res = int(os.environ.get("UE_TRAIN_RES", "128"))
    hfov = float(os.environ.get("UE_HFOV", "70"))
    n_az = int(os.environ.get("UE_N_AZ", "28"))
    caps = int(os.environ.get("UE_CAPS_PER_POSE", "4"))
    ev = float(os.environ.get("UE_CAPTURE_EV", "1.0"))
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    unreal.log(f"[hl] loading level {LEVEL}")
    try:
        unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
    except Exception as e:
        unreal.log_warning(f"[hl] load_map failed (may already be loaded): {e}")
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()

    focus, radius, valids = _focus_radius(unreal)
    _load_region(unreal, focus, radius, valids)
    _generate_pcg(unreal)
    na, sm, ism = _count_geometry(unreal)
    unreal.log(f"[hl] focus={focus} radius={radius/100:.1f}m  loaded actors={na} "
               f"(static-mesh={sm}, instanced={ism})")

    actor, comp, rt = _setup_capture(unreal, cap_res, hfov, ev)

    if os.environ.get("UE_EXPO_SWEEP") == "1":
        # one representative overview pose, rendered at a range of pinned
        # exposures (higher target = DARKER image) so we can pick the right EV.
        p = rig.orbit_hemisphere(focus, radius * 1.3, elevations_deg=(30.0,),
                                 n_azimuth=1, heldout_every=0)[0]
        loc = unreal.Vector(*p["location_cm"])
        rot = unreal.MathLibrary.find_look_at_rotation(loc, unreal.Vector(*p["target_cm"]))
        actor.set_actor_location_and_rotation(loc, rot, False, False)
        for ev_val in (0.5, 2.0, 8.0, 24.0, 64.0):
            pp = comp.post_process_settings
            pp.set_editor_property("auto_exposure_min_brightness", ev_val)
            pp.set_editor_property("auto_exposure_max_brightness", ev_val)
            comp.set_editor_property("post_process_settings", pp)
            for _ in range(caps):
                comp.capture_scene()
            nm = f"expo_{ev_val}"
            unreal.RenderingLibrary.export_render_target(world, rt, out_dir + "/images", nm)
            raw = os.path.join(out_dir, "images", nm)
            if os.path.exists(raw) and not os.path.exists(raw + ".png"):
                os.replace(raw, raw + ".png")
            unreal.log(f"[hl] expo {ev_val} captured")
        print(f"EXPO_SWEEP_DONE out={out_dir}/images")
    elif probe:
        # a few wide OVERVIEW frames from a high ring, to eyeball geometry+framing
        poses = rig.orbit_hemisphere(focus, radius * 1.4, elevations_deg=(25.0, 50.0),
                                     n_azimuth=4, heldout_every=0)
        frames = _render(unreal, world, poses, comp, rt, actor, caps, out_dir)
        unreal.log(f"[hl] PROBE wrote {len(frames)} overview frames")
        print(f"PROBE_DONE actors={na} sm={sm} ism={ism} focus={focus} "
              f"radius_m={radius/100:.1f} frames={len(frames)} out={out_dir}/images")
    else:
        poses = rig.orbit_hemisphere(focus, radius, elevations_deg=(8.0, 22.0, 38.0, 55.0),
                                     n_azimuth=n_az, heldout_every=6)
        unreal.log(f"[hl] full orbit: {len(poses)} cams")
        frames = _render(unreal, world, poses, comp, rt, actor, caps, out_dir)
        ext = radius * 0.85
        ue_poses = os.path.join(out_dir, "ue_poses.json")
        export.write_ue_poses(ue_poses, train_res, train_res, hfov, frames, scene_meta={
            "background": [0.0, 0.0, 0.0],
            "aabb_min_cm": [focus[0]-ext, focus[1]-ext, focus[2]-ext],
            "aabb_max_cm": [focus[0]+ext, focus[1]+ext, focus[2]+ext],
            "fiducials": [], "primitives": []})
        print(f"WROTE {ue_poses} ({len(frames)} frames, focus {focus}, radius {radius/100:.1f}m)")

    try:
        unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
    except Exception:
        pass
    # ensure the headless process exits so automation knows we're done
    try:
        unreal.SystemLibrary.quit_editor()
    except Exception:
        pass


main()
