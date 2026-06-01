"""Render rig cameras to disk from inside Unreal.

Two backends:
  * SceneCapture2D (default, robust to script headlessly): captures final colour
    to PNG and SceneDepth to EXR per pose.
  * Movie Render Queue (`--mrq`): the production path named in the spec; requires
    the Movie Render Queue plugin. Implemented best-effort; SceneCapture is used
    if MRQ is unavailable.

`unreal` only; no third-party deps (runs in UnrealEditor-Cmd's Python).
"""
from __future__ import annotations

import os

from . import export


def _make_capture(unreal, w, h, hfov_deg, capture_source, rtf=None):
    sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = sub.spawn_actor_from_class(unreal.SceneCapture2D,
                                       unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    comp = actor.capture_component2d
    comp.fov_angle = hfov_deg
    comp.capture_source = capture_source
    # Clean background: drop atmosphere/fog/clouds so the empty void renders as
    # the black clear colour instead of a noisy bright horizon. SceneCapture2D
    # exposes overrides via `show_flag_settings` (NOT a `show_flags` attribute).
    try:
        def _flag(nm):
            s = unreal.EngineShowFlagsSetting()
            s.set_editor_property("show_flag_name", nm)
            s.set_editor_property("enabled", False)
            return s
        comp.set_editor_property("show_flag_settings",
                                 [_flag(n) for n in (
                                     "Atmosphere", "Fog", "VolumetricFog", "Cloud",
                                     "VolumetricCloud", "GlobalIllumination",
                                     "AmbientOcclusion", "ScreenSpaceAO",
                                     "ScreenSpaceReflections", "LumenGlobalIllumination",
                                     "LumenReflections", "ReflectionEnvironment",
                                     "DistanceFieldAO")])
    except Exception:
        pass
    # Pin exposure (no eye adaptation) so a surface has the SAME brightness in
    # every view -- essential for multi-view consistency. B=5 well-exposes the
    # lit scene (validated). Bloom off.
    try:
        pp = comp.post_process_settings
        for k, v in [("auto_exposure_min_brightness", 5.0),
                     ("auto_exposure_max_brightness", 5.0),
                     ("override_auto_exposure_min_brightness", True),
                     ("override_auto_exposure_max_brightness", True),
                     ("bloom_intensity", 0.0), ("override_bloom_intensity", True)]:
            pp.set_editor_property(k, v)
        comp.set_editor_property("post_process_settings", pp)
    except Exception:
        pass
    # 8-bit RT so export writes PNG (the float default writes EXR).
    rtf = rtf or unreal.TextureRenderTargetFormat.RTF_RGBA8
    rt = unreal.RenderingLibrary.create_render_target2d(actor, w, h, rtf)
    comp.texture_target = rt
    return actor, comp, rt


def _export_png(unreal, world, rt, out_images_dir, name):
    """export_render_target writes the file verbatim (no extension). Normalize
    it to <name>.png so the rest of the pipeline sees a real .png path."""
    import os
    unreal.RenderingLibrary.export_render_target(world, rt, out_images_dir, name)
    raw = os.path.join(out_images_dir, name)
    png = raw + ".png"
    if os.path.exists(raw) and not os.path.exists(png):
        os.replace(raw, png)
    return png


def render_cameras(unreal, poses, w, h, hfov_deg, out_dir, want_depth=True):
    """Render each pose's colour (+depth). Returns frames metadata list.

    poses: list of dicts with location_cm + target_cm + split (from rig).
    """
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()

    # FXAA (no temporal noise), no auto-exposure/bloom/motion-blur, and crucially
    # NO view-dependent global illumination / screen-space effects: UE5's Lumen GI
    # + SSAO/SSR make the same surface look different from different views (and add
    # noise), which a view-independent (SH0) splat cannot fit. Direct lighting only
    # (our 6 directional lights) is view-consistent.
    try:
        for c in ("r.AntiAliasingMethod 2", "r.DefaultFeature.AutoExposure 0",
                  "r.DefaultFeature.Bloom 0", "r.DefaultFeature.MotionBlur 0",
                  "r.DynamicGlobalIlluminationMethod 0", "r.ReflectionMethod 0",
                  "r.AmbientOcclusionLevels 0", "r.SSGI.Enable 0", "r.SSR.Quality 0",
                  "r.Lumen.DiffuseIndirect.Allow 0"):
            unreal.SystemLibrary.execute_console_command(world, c)
    except Exception:
        pass

    # Supersample: render at ss*res; splatkit.ingest box-downsamples to (w,h).
    # Single-sample FXAA leaves per-view aliasing (jaggies shift between views ->
    # view-inconsistent -> caps the splat); SSAA matches the synthetic ss=2 path.
    ss = int(os.environ.get("UE_CAPTURE_SS", "2"))
    rw, rh = w * ss, h * ss

    # Lit FinalColorLDR: real shading (depth cues) with matte materials (so it
    # stays view-independent) + pinned exposure for cross-view consistency.
    col_actor, col_comp, col_rt = _make_capture(
        unreal, rw, rh, hfov_deg, unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    dep_actor = dep_comp = dep_rt = None
    if want_depth:
        dep_actor, dep_comp, dep_rt = _make_capture(
            unreal, w, h, hfov_deg, unreal.SceneCaptureSource.SCS_SCENE_DEPTH,
            rtf=unreal.TextureRenderTargetFormat.RTF_RGBA16F)

    frames = []
    for p in poses:
        loc = unreal.Vector(*p["location_cm"])
        tgt = unreal.Vector(*p["target_cm"])
        rot = unreal.MathLibrary.find_look_at_rotation(loc, tgt)
        for comp_actor in (col_actor, dep_actor):
            if comp_actor:
                comp_actor.set_actor_location_and_rotation(loc, rot, False, False)
        col_comp.capture_scene()
        name = f"cam_{p['index']:03d}"
        png = _export_png(unreal, world, col_rt, out_dir + "/images", name)
        depth_path = None
        if want_depth:
            dep_comp.capture_scene()
            unreal.RenderingLibrary.export_render_target(world, dep_rt, out_dir + "/images",
                                                         name + "_depth")
            depth_path = os.path.join(out_dir, "images", name + "_depth")
        # authoritative pose, read off the actor UE actually rendered with
        frames.append({
            "file_path": png,
            "split": p["split"],
            "location_cm": export.location_from_actor(unreal, col_actor),
            "basis_ue": export.basis_from_actor(unreal, col_actor),
            "depth_path": depth_path,
        })
    return frames, (col_actor, dep_actor)
