"""Capture a CANYON-sized region from the LIVE editor (PCG foliage + assets present).

Unlike capture_hero_orbit.py (one orbit around a single point), this walks a GRID
of camera positions ALONG the canyon (forward x lateral x height from where your
viewport is aimed) and at each position shoots a YAW/PITCH FAN -- so every wall,
floor and feature is seen from many positions AND angles (the dense multi-view
coverage that removes 3DGS floaters). Real scene lighting; exposure pinned DARKER
(EV~12) than the hero capture so the sun-lit rock faces don't blow out.

Run it in the OPEN editor's Python console, aimed DOWN the canyon stretch you want:
    py "/Users/peterhowell/ue-splat-capture/ue_capture/capture_canyon_live.py"

Tunables (env): UE_CAPTURE_OUT, UE_CAPTURE_EV (12), UE_CAP_RES (512), UE_TRAIN_RES
(512), UE_HFOV (75), UE_FWD_STEPS (5), UE_LAT_STEPS (3), UE_HEIGHT_STEPS (2),
UE_STEP_M (auto from the forward ray hit, else 8), UE_YAWS (6), UE_FAN_DEG (200),
UE_PITCHES ("-28,-8,12"), UE_CAPS_PER_POSE (3). Re-run with overrides if framing/
exposure is off (check out/electric_dreams_canyon/images/cam_000.png first).
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ue_capture import export  # noqa: E402


def _flag(unreal, nm, on):
    s = unreal.EngineShowFlagsSetting()
    s.set_editor_property("show_flag_name", nm)
    s.set_editor_property("enabled", on)
    return s


def _norm(v):
    m = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return [v[0] / m, v[1] / m, v[2] / m]


def _forward_ray_dist(unreal, world, cam_loc, fwd):
    """Distance to the geometry the viewport is aimed at (sets the corridor scale)."""
    far = 300000.0
    end = unreal.Vector(cam_loc.x + fwd[0] * far, cam_loc.y + fwd[1] * far, cam_loc.z + fwd[2] * far)
    try:
        res = unreal.SystemLibrary.line_trace_single(
            world, cam_loc, end, unreal.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [],
            unreal.DrawDebugType.NONE, True)
        hit = res[1] if isinstance(res, (tuple, list)) else res
        ok = res[0] if isinstance(res, (tuple, list)) else getattr(hit, "blocking_hit", False)
        if ok:
            for k in ("impact_point", "location"):
                try:
                    p = hit.get_editor_property(k)
                    if p is not None:
                        return ((p.x - cam_loc.x) ** 2 + (p.y - cam_loc.y) ** 2 + (p.z - cam_loc.z) ** 2) ** 0.5
                except Exception:
                    pass
    except Exception as e:
        unreal.log_warning(f"[canyon] ray failed: {e}")
    return None


def _build_poses(cam_loc, fwd, step_cm, n_fwd, n_lat, n_h, yaws, fan_deg, pitches, heldout_every):
    """Grid of positions along the canyon, each with a yaw/pitch fan of view dirs."""
    fwd_h = _norm([fwd[0], fwd[1], 0.0])
    if fwd_h[0] == 0 and fwd_h[1] == 0:           # viewport looking straight down
        fwd_h = [1.0, 0.0, 0.0]
    right_h = _norm([fwd_h[1], -fwd_h[0], 0.0])   # horizontal right
    base = [cam_loc.x, cam_loc.y, cam_loc.z]
    poses = []
    idx = 0
    for i in range(n_fwd):
        for j in range(n_lat):
            for k in range(n_h):
                lat = (j - (n_lat - 1) / 2.0) * step_cm
                pos = [base[c] + fwd_h[c] * (i * step_cm) + right_h[c] * lat
                       + (0.0 if c < 2 else k * step_cm * 0.6) for c in range(3)]
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
    return poses, fwd_h, right_h


def main():
    import unreal
    out_dir = os.environ.get("UE_CAPTURE_OUT") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "electric_dreams_canyon")
    cap_res = int(os.environ.get("UE_CAP_RES", "512"))
    train_res = int(os.environ.get("UE_TRAIN_RES", "512"))
    hfov = float(os.environ.get("UE_HFOV", "75"))
    n_fwd = int(os.environ.get("UE_FWD_STEPS", "5"))
    n_lat = int(os.environ.get("UE_LAT_STEPS", "3"))
    n_h = int(os.environ.get("UE_HEIGHT_STEPS", "1"))
    yaws = int(os.environ.get("UE_YAWS", "6"))
    fan_deg = float(os.environ.get("UE_FAN_DEG", "200"))
    pitches = [float(x) for x in os.environ.get("UE_PITCHES", "-28,-8,12").split(",")]
    caps = int(os.environ.get("UE_CAPS_PER_POSE", "3"))
    ev = float(os.environ.get("UE_CAPTURE_EV", "12"))
    heldout_every = int(os.environ.get("UE_HELDOUT_EVERY", "8"))
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    ed = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = ed.get_editor_world()
    cam_loc, cam_rot = ed.get_level_viewport_camera_info()
    if cam_loc is None:
        raise RuntimeError("No active viewport -- click in the 3D viewport, aim down the canyon, re-run.")
    fwd_v = unreal.MathLibrary.get_forward_vector(cam_rot)
    fwd = [fwd_v.x, fwd_v.y, fwd_v.z]

    dist = _forward_ray_dist(unreal, world, cam_loc, fwd)
    step_cm = float(os.environ.get("UE_STEP_M", str((dist / 100.0 * 1.3 / max(n_fwd, 1)) if dist else 8.0))) * 100.0
    poses, fwd_h, right_h = _build_poses(cam_loc, fwd, step_cm, n_fwd, n_lat, n_h,
                                         yaws, fan_deg, pitches, heldout_every)
    n_held = sum(p["split"] == "heldout" for p in poses)
    unreal.log(f"[canyon] start={[round(cam_loc.x),round(cam_loc.y),round(cam_loc.z)]} "
               f"step={step_cm/100:.1f}m grid={n_fwd}x{n_lat}x{n_h} fan={yaws}x{len(pitches)} "
               f"-> {len(poses)} cams ({len(poses)-n_held} train/{n_held} held), EV={ev}, {cap_res}px")

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
        for _ in range(caps):
            comp.capture_scene()
        name = f"cam_{p['index']:04d}"
        unreal.RenderingLibrary.export_render_target(world, rt, out_dir + "/images", name)
        raw = os.path.join(out_dir, "images", name)
        if os.path.exists(raw) and not os.path.exists(raw + ".png"):
            os.replace(raw, raw + ".png")
        frames.append({"file_path": raw + ".png", "split": p["split"],
                       "location_cm": export.location_from_actor(unreal, actor),
                       "basis_ue": export.basis_from_actor(unreal, actor)})
        if (i + 1) % 40 == 0:
            unreal.log(f"[canyon] rendered {i+1}/{len(poses)}")

    xs = [p["location_cm"][0] for p in poses]; ys = [p["location_cm"][1] for p in poses]
    zs = [p["location_cm"][2] for p in poses]
    m = step_cm * 2.0
    export.write_ue_poses(os.path.join(out_dir, "ue_poses.json"), train_res, train_res, hfov, frames,
                          scene_meta={"background": [0.0, 0.0, 0.0],
                                      "aabb_min_cm": [min(xs) - m, min(ys) - m, min(zs) - m],
                                      "aabb_max_cm": [max(xs) + m, max(ys) + m, max(zs) + m],
                                      "fiducials": [], "primitives": []})
    try:
        unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(actor)
    except Exception:
        pass
    unreal.log(f"[canyon] DONE {len(frames)} frames")
    print(f"WROTE {out_dir}/ue_poses.json ({len(frames)} frames). CHECK {out_dir}/images/cam_0000.png")


main()
