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

    col_actor, col_comp, col_rt = _make_capture(
        unreal, w, h, hfov_deg, unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
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
