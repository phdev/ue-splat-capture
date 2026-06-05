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


def _setup_capture(unreal, res, hfov, ev, depth=False):
    actor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
        unreal.SceneCapture2D, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    comp = actor.capture_component2d
    comp.fov_angle = hfov
    # depth: SCS_SCENE_DEPTH writes per-pixel linear depth (UE cm) -> needs a FLOAT render
    # target (export_render_target then writes EXR). Ground-truth depth for depth-supervised
    # training. Else the normal lit LDR colour capture.
    comp.capture_source = (unreal.SceneCaptureSource.SCS_SCENE_DEPTH if depth
                           else unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    for k, v in [("capture_every_frame", False), ("capture_on_movement", False),
                 ("always_persist_rendering_state", True)]:
        try: comp.set_editor_property(k, v)
        except Exception: pass
    flags = [_flag(unreal, "MotionBlur", False), _flag(unreal, "ScreenSpaceReflections", False),
             _flag(unreal, "Bloom", False)]
    if os.environ.get("UE_NOSKY") == "1":
        # hide the visible sky so 3DGS doesn't waste capacity (and floaters) on it; keep
        # the directional sun + skylight ambient so the terrain stays lit.
        for nm in ("Atmosphere", "Fog", "VolumetricFog", "Cloud"):
            flags.append(_flag(unreal, nm, False))
    try:
        comp.set_editor_property("show_flag_settings", flags)
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
    fmt = unreal.TextureRenderTargetFormat.RTF_RGBA8
    if depth:                                          # need a FLOAT RT (enum name varies by UE ver)
        for nm in ("RTF_R32f", "RTF_RGBA32f", "RTF_RGBA16f", "RTF_R16f", "RTF_RG16f",
                   "RTF_R32F", "RTF_RGBA32F", "RTF_RGBA16F"):
            if hasattr(unreal.TextureRenderTargetFormat, nm):
                fmt = getattr(unreal.TextureRenderTargetFormat, nm)
                unreal.log(f"[hl] depth RT format = {nm}")
                break
    rt = unreal.RenderingLibrary.create_render_target2d(actor, res, res, fmt)
    comp.texture_target = rt
    return actor, comp, rt


def _render(unreal, world, poses, comp, rt, actor, caps, out_dir, avg_samples=1):
    """Render each pose. When avg_samples>1, export N independent renders per pose
    (cam_IDX_SS.png) -- Lumen GI / specular / TSR noise is re-randomised each render
    and is what 3DGS turns into spiky foliage floaters; averaging the N samples
    (scripts/average_samples.py, run after) denoises each training view. `caps`
    renders after each camera move flush stale temporal history before sampling."""
    frames = []
    for i, p in enumerate(poses):
        loc = unreal.Vector(*p["location_cm"])
        rot = unreal.MathLibrary.find_look_at_rotation(loc, unreal.Vector(*p["target_cm"]))
        actor.set_actor_location_and_rotation(loc, rot, False, False)
        for _ in range(caps):                       # warm-up after the camera move
            comp.capture_scene()
        base = f"cam_{p['index']:03d}"
        if avg_samples > 1:
            for s in range(avg_samples):
                comp.capture_scene()                # one independent noisy sample
                nm = f"{base}_{s:02d}"
                unreal.RenderingLibrary.export_render_target(world, rt, out_dir + "/images", nm)
                raw = os.path.join(out_dir, "images", nm)
                if os.path.exists(raw) and not os.path.exists(raw + ".png"):
                    os.replace(raw, raw + ".png")
            png = os.path.join(out_dir, "images", base + ".png")   # produced by averaging
        else:
            comp.capture_scene()
            unreal.RenderingLibrary.export_render_target(world, rt, out_dir + "/images", base)
            raw = os.path.join(out_dir, "images", base); png = raw + ".png"
            if os.path.exists(raw) and not os.path.exists(png):
                os.replace(raw, png)
        frames.append({"file_path": png, "split": p["split"],
                       "location_cm": export.location_from_actor(unreal, actor),
                       "basis_ue": export.basis_from_actor(unreal, actor)})
        if (i + 1) % 20 == 0:
            unreal.log(f"[hl] rendered {i+1}/{len(poses)} ({avg_samples}x samples)")
    return frames


def _principal_axis_xy(valids):
    """Dominant horizontal direction of the loaded geometry (the canyon's length),
    from the PCA of actor-bounds centers. No viewport headless, so this is how we
    orient the corridor."""
    import math
    cs = [c for (_d, c, _h) in valids]
    if len(cs) < 3:
        return [1.0, 0.0, 0.0]
    mx = sum(c[0] for c in cs) / len(cs); my = sum(c[1] for c in cs) / len(cs)
    sxx = sxy = syy = 0.0
    for c in cs:
        dx = c[0] - mx; dy = c[1] - my
        sxx += dx * dx; sxy += dx * dy; syy += dy * dy
    tr = sxx + syy; det = sxx * syy - sxy * sxy
    lam = tr / 2.0 + math.sqrt(max((tr / 2.0) ** 2 - det, 0.0))
    v = [lam - syy, sxy] if abs(sxy) > 1e-6 else ([1.0, 0.0] if sxx >= syy else [0.0, 1.0])
    n = math.hypot(v[0], v[1]) or 1.0
    return [v[0] / n, v[1] / n, 0.0]


def _canyon_poses(focus, valids, length_cm, n_fwd, n_lat, n_h, yaws, fan_deg, pitches, heldout_every):
    """Grid of positions along the principal axis (capped to length_cm, centered on
    focus), each with a yaw/pitch fan -- dense multi-position canyon coverage."""
    import math
    fwd_h = _principal_axis_xy(valids)
    right_h = [fwd_h[1], -fwd_h[0], 0.0]
    cs = [c for (_d, c, _h) in valids]
    zs = sorted(c[2] for c in cs); zmid = zs[len(zs) // 2]
    projl = [(c[0] - focus[0]) * right_h[0] + (c[1] - focus[1]) * right_h[1] for c in cs]
    width = (max(projl) - min(projl)) if projl else length_cm * 0.4
    step_fwd = length_cm / max(n_fwd - 1, 1)
    step_lat = (min(width, length_cm) * 0.5) / max(n_lat - 1, 1)
    step_h = max(width, length_cm) * 0.10
    start = [focus[0] - fwd_h[0] * length_cm / 2.0, focus[1] - fwd_h[1] * length_cm / 2.0, zmid]
    poses = []; idx = 0
    for i in range(n_fwd):
        for j in range(n_lat):
            for k in range(n_h):
                lat = (j - (n_lat - 1) / 2.0) * step_lat
                pos = [start[0] + fwd_h[0] * (i * step_fwd) + right_h[0] * lat,
                       start[1] + fwd_h[1] * (i * step_fwd) + right_h[1] * lat,
                       start[2] + k * step_h]
                for a_i in range(yaws):
                    a = math.radians((-fan_deg / 2.0) + fan_deg * (a_i / max(yaws - 1, 1)))
                    horiz = [math.cos(a) * fwd_h[c] + math.sin(a) * right_h[c] for c in range(3)]
                    for p_deg in pitches:
                        p = math.radians(p_deg)
                        d = [math.cos(p) * horiz[0], math.cos(p) * horiz[1], math.sin(p)]
                        tgt = [pos[c] + d[c] * 2000.0 for c in range(3)]
                        split = "heldout" if (heldout_every and idx % heldout_every == 1) else "train"
                        poses.append({"index": idx, "split": split, "location_cm": pos, "target_cm": tgt})
                        idx += 1
    return poses


def main():
    import unreal
    out_dir = os.environ.get("UE_CAPTURE_OUT") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "electric_dreams")
    probe = os.environ.get("UE_PROBE", "0") == "1"
    canyon = os.environ.get("UE_CANYON", "0") == "1"
    cap_res = int(os.environ.get("UE_CAP_RES", "512"))
    train_res = int(os.environ.get("UE_TRAIN_RES", "512" if canyon else "128"))
    hfov = float(os.environ.get("UE_HFOV", "75" if canyon else "70"))
    n_az = int(os.environ.get("UE_N_AZ", "28"))
    caps = int(os.environ.get("UE_CAPS_PER_POSE", "3" if canyon else "4"))
    avg_samples = int(os.environ.get("UE_AVG_SAMPLES", "1"))   # temporal denoise per pose
    depth = os.environ.get("UE_DEPTH") == "1"                  # capture ground-truth depth (EXR)
    if depth:
        avg_samples = 1                                        # depth is geometric, no denoise needed
    ev = float(os.environ.get("UE_CAPTURE_EV", "10" if canyon else "1.0"))
    cyn_len = float(os.environ.get("UE_CANYON_LEN_M", "50")) * 100.0
    cyn = dict(n_fwd=int(os.environ.get("UE_FWD_STEPS", "5")),
               n_lat=int(os.environ.get("UE_LAT_STEPS", "3")),
               n_h=int(os.environ.get("UE_HEIGHT_STEPS", "1")),
               yaws=int(os.environ.get("UE_YAWS", "6")),
               fan_deg=float(os.environ.get("UE_FAN_DEG", "200")),
               pitches=[float(x) for x in os.environ.get("UE_PITCHES", "-28,-8,12").split(",")],
               heldout_every=int(os.environ.get("UE_HELDOUT_EVERY", "8")))
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

    actor, comp, rt = _setup_capture(unreal, cap_res, hfov, ev, depth=depth)

    if os.environ.get("UE_DIAG") == "1":
        # Stability probe: render ONE foliage-heavy pose N times back-to-back, export
        # each. Diffing them reveals per-view instability (Lumen GI not converged /
        # TSR noise / animated foliage) that 3DGS turns into spiky floaters.
        diag_n = int(os.environ.get("UE_DIAG_N", "12"))
        p = rig.orbit_hemisphere(focus, radius, elevations_deg=(14.0,), n_azimuth=1,
                                 heldout_every=0)[0]
        loc = unreal.Vector(*p["location_cm"])
        rot = unreal.MathLibrary.find_look_at_rotation(loc, unreal.Vector(*p["target_cm"]))
        actor.set_actor_location_and_rotation(loc, rot, False, False)
        for i in range(diag_n):
            comp.capture_scene()
            name = f"diag_{i:03d}"
            unreal.RenderingLibrary.export_render_target(world, rt, out_dir + "/images", name)
            raw = os.path.join(out_dir, "images", name); png = raw + ".png"
            if os.path.exists(raw) and not os.path.exists(png):
                os.replace(raw, png)
        unreal.log(f"[diag] wrote {diag_n} back-to-back renders of one pose")
        print(f"DIAG_DONE n={diag_n} out={out_dir}/images")
        try: unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
        except Exception: pass
        try: unreal.SystemLibrary.quit_editor()
        except Exception: pass
        return

    if os.environ.get("UE_GRID") == "1":
        # Drone-mapping nadir grid for TERRAIN: uniform down-looking coverage of the
        # spread ground (a converging dome only covers the centre). Merge with the
        # spire dome afterwards. Pin UE_FOCUS_CM to the GROUND centre/level.
        gn = int(os.environ.get("UE_GRID_N", "7"))
        gext = float(os.environ.get("UE_GRID_EXTENT_M", "24")) * 100.0
        ght = float(os.environ.get("UE_GRID_HEIGHT_M", "16")) * 100.0
        gconv = float(os.environ.get("UE_GRID_CONVERGE", "0.25"))
        gposes = rig.grid_nadir(focus, gext, ght, n_side=gn, converge=gconv, heldout_every=8)
        unreal.log(f"[hl] terrain grid: {len(gposes)} cams ({gn}x{gn}), extent {gext/100:.0f}m, "
                   f"height {ght/100:.0f}m, converge {gconv}, avg_samples={avg_samples}")
        frames = _render(unreal, world, gposes, comp, rt, actor, caps, out_dir, avg_samples)
        m = gext * 1.2
        export.write_ue_poses(os.path.join(out_dir, "ue_poses.json"), train_res, train_res, hfov,
                              frames, scene_meta={"background": [0.0, 0.0, 0.0],
                              "aabb_min_cm": [focus[0]-m, focus[1]-m, focus[2]-ght],
                              "aabb_max_cm": [focus[0]+m, focus[1]+m, focus[2]+ght],
                              "fiducials": [], "primitives": []})
        print(f"WROTE {out_dir}/ue_poses.json ({len(frames)} grid frames, focus {focus})")
        try: unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
        except Exception: pass
        try: unreal.SystemLibrary.quit_editor()
        except Exception: pass
        return

    if canyon:
        cposes = _canyon_poses(focus, valids, cyn_len, **cyn)
        if probe:
            sub = cposes[::max(1, len(cposes) // 12)][:12]
            frames = _render(unreal, world, sub, comp, rt, actor, caps, out_dir)
            print(f"CANYON_PROBE_DONE actors={na} sm={sm} ism={ism} frames={len(frames)} "
                  f"planned={len(cposes)} len_m={cyn_len/100:.0f} out={out_dir}/images")
        else:
            unreal.log(f"[hl] canyon: {len(cposes)} cams over {cyn_len/100:.0f} m")
            frames = _render(unreal, world, cposes, comp, rt, actor, caps, out_dir)
            xs = [p['location_cm'][0] for p in cposes]; ys = [p['location_cm'][1] for p in cposes]
            zs = [p['location_cm'][2] for p in cposes]; m = cyn_len * 0.35
            export.write_ue_poses(os.path.join(out_dir, "ue_poses.json"), train_res, train_res, hfov,
                                  frames, scene_meta={"background": [0.0, 0.0, 0.0],
                                  "aabb_min_cm": [min(xs) - m, min(ys) - m, min(zs) - m],
                                  "aabb_max_cm": [max(xs) + m, max(ys) + m, max(zs) + m],
                                  "fiducials": [], "primitives": []})
            print(f"WROTE {out_dir}/ue_poses.json ({len(frames)} canyon frames)")
    elif os.environ.get("UE_EXPO_SWEEP") == "1":
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
        elev = tuple(float(x) for x in os.environ.get("UE_ELEVATIONS", "8,22,38,55").split(","))
        poses = rig.orbit_hemisphere(focus, radius, elevations_deg=elev,
                                     n_azimuth=n_az, heldout_every=6)
        unreal.log(f"[hl] full orbit: {len(poses)} cams, elevations={elev}, avg_samples={avg_samples}")
        frames = _render(unreal, world, poses, comp, rt, actor, caps, out_dir, avg_samples)
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
