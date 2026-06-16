"""TICK-DRIVEN capture inside a LAUNCHED GUI editor (UnrealEditor, NOT -Cmd).

Headless (-Cmd + a blocking python script) can't generate PCG foliage or converge
TSR/Lumen, because a blocking script never yields the engine ticks those need. This
script instead registers a slate post-tick callback and RETURNS -- the editor keeps
ticking (PCG generates, temporal AA accumulates) while a small state machine walks the
poses one capture per few ticks. Launch it with scripts/capture_editor_run.sh.

Env: UE_CAPTURE_OUT (abs), UE_FOCUS_CM, UE_ORBIT_RADIUS_CM, UE_ELEVATIONS, UE_N_AZ,
     UE_HFOV, UE_CAP_RES, UE_TRAIN_RES, UE_CAPTURE_EV, UE_SETTLE_TICKS (wait for
     level+PCG, default 600), UE_CONVERGE_TICKS (per-pose TSR settle, default 10),
     UE_PROBE (1 = a few overview poses to validate foliage),
     UE_DEPTH (1 = also export GT metric depth EXR per pose into depth/, via a second
     lockstep SceneCapture2D using SCS_SCENE_DEPTH -> RTF_RGBA32f; for depth-reg training).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ue_capture import export, rig  # noqa: E402

LEVEL = os.environ.get("UE_LEVEL", "/Game/Levels/PCG/ElectricDreams_PCGCloseRange")

# module-level state the tick callback advances (UE python tick callbacks are plain fns)
_S = {"phase": "settle", "i": 0, "wait": 0, "poses": [], "frames": [],
      "comp": None, "rt": None, "actor": None, "world": None, "out_dir": None,
      "handle": None, "caps": 0, "train_res": 512, "hfov": 75.0, "focus": None,
      "radius": 0.0}


def _flag(unreal, nm, on):
    s = unreal.EngineShowFlagsSetting()
    s.set_editor_property("show_flag_name", nm)
    s.set_editor_property("enabled", on)
    return s


def _count_foliage(unreal):
    """How many instanced-static-mesh (PCG/foliage) components are live right now --
    rises as PCG generates, so we can log whether the live editor actually populated it."""
    n_ism = n_inst = 0
    try:
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    except Exception:
        actors = []
    for a in actors:
        try:
            for c in a.get_components_by_class(unreal.InstancedStaticMeshComponent):
                n_ism += 1
                try: n_inst += c.get_instance_count()
                except Exception: pass
        except Exception:
            pass
    return n_ism, n_inst


def _scene_top(unreal, focus, radius):
    """Top-Z (cm) of the tallest NON-INSTANCED static-mesh actor within `radius` (xy) of
    focus, its count, and that tallest actor's label. EXCLUDES instanced meshes (PCG
    foliage/trees subclass StaticMeshComponent and would mask a missing rock) so this
    actually detects the hero ROCK SPIRE -- which silently failed to stream into the depth
    capture. We gate the capture on this so it can't shoot a spire-less scene again."""
    fx, fy = focus[0], focus[1]
    top = -1e18; n = 0; name = ""
    try:
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    except Exception:
        actors = []
    for a in actors:
        try:
            smcs = a.get_components_by_class(unreal.StaticMeshComponent)
            if not any(not isinstance(c, unreal.InstancedStaticMeshComponent) for c in smcs):
                continue                               # only foliage/instanced here -> skip
            o, e = a.get_actor_bounds(False)           # (origin, box_extent)
            if ((o.x - fx) ** 2 + (o.y - fy) ** 2) ** 0.5 > radius:
                continue
            n += 1
            t = o.z + e.z
            if t > top:
                top = t
                try: name = a.get_actor_label()
                except Exception: name = str(a.get_name())
        except Exception:
            pass
    return (top if n else 0.0), n, name


def _dump_scene(unreal, focus):
    """One-shot diagnostic: what tall geometry IS loaded? Logs the 10 tallest non-instanced
    static-mesh actors anywhere (name, top-Z, distance from focus) + Landscape actor bounds +
    anything named like rock/spire/cliff. Tells us whether the spire is a not-streaming static
    mesh, part of the Landscape (streams differently), or genuinely absent."""
    fx, fy, fz = focus
    try:
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    except Exception:
        actors = []
    rows = []; land = []; named = []
    for a in actors:
        try:
            cls = a.get_class().get_name()
            try: lbl = a.get_actor_label()
            except Exception: lbl = a.get_name()
            if "Landscape" in cls:
                o, e = a.get_actor_bounds(False)
                land.append((lbl, cls, o.z + e.z, e.z * 2))
                continue
            smcs = a.get_components_by_class(unreal.StaticMeshComponent)
            if any(not isinstance(c, unreal.InstancedStaticMeshComponent) for c in smcs):
                o, e = a.get_actor_bounds(False)
                d = ((o.x - fx) ** 2 + (o.y - fy) ** 2) ** 0.5
                rows.append((o.z + e.z, lbl, d, e.z * 2))
            if any(s in lbl.lower() for s in ("rock", "spire", "cliff", "stone", "boulder", "mesa")):
                o, e = a.get_actor_bounds(False)
                named.append((lbl, cls, o.z + e.z, ((o.x-fx)**2+(o.y-fy)**2)**0.5))
        except Exception:
            pass
    rows.sort(reverse=True)
    unreal.log(f"[ed] DUMP: total actors={len(actors)} focus_z={fz:.0f}")
    for top, lbl, d, h in rows[:10]:
        unreal.log(f"[ed] DUMP tall-SM: top={top:.0f}cm h={h:.0f}cm dist={d:.0f}cm '{lbl}'")
    for lbl, cls, top, h in land:
        unreal.log(f"[ed] DUMP LANDSCAPE: '{lbl}' ({cls}) top={top:.0f}cm h={h:.0f}cm")
    for lbl, cls, top, d in named[:10]:
        unreal.log(f"[ed] DUMP named-rock: '{lbl}' ({cls}) top={top:.0f}cm dist={d:.0f}cm")


def _apply_show_only(unreal, comp, mesh_keywords, foliage_blocklist, include_classes=()):
    """MESH-NAME-FILTERED show-only: render only StaticMesh primitives whose mesh name
    contains a terrain keyword (Boulder/Cliff/Embankment/.../Rock) and NOT a foliage
    blocklist substring. Walks ALL actors, ALL StaticMeshComponents (including ISMCs
    inside PCG BPs which mix terrain + foliage in the same actor). Also unconditionally
    includes any component on actors whose class matches include_classes (e.g. Landscape).
    The actor-level allow ('Rock_*_BP' etc.) was inadequate -- PCGDemo_DitchBP holds 4500
    SM_ForestGround_01 (terrain) AND 2300 SM_Amaryllis_04 (foliage) in the same actor.
    Returns (n_components_added, [sample_mesh_names])."""
    try:
        comp.primitive_render_mode = unreal.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST
    except Exception as e:
        unreal.log_warning(f"[ed] show-only set mode failed: {e}")
        return 0, []
    w = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    try:
        acts = unreal.GameplayStatics.get_all_actors_of_class(w, unreal.Actor)
    except Exception:
        acts = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    n_comp = 0; samples = []
    for a in acts:
        try:
            cls = a.get_class().get_name()
            # Unconditional include for whole actor (e.g., Landscape)
            if any(s in cls for s in include_classes):
                for c in a.get_components_by_class(unreal.PrimitiveComponent):
                    try: comp.show_only_component(c); n_comp += 1
                    except Exception: pass
                continue
            # Otherwise walk StaticMeshComponents (incl. ISMCs) and filter by mesh name
            for c in a.get_components_by_class(unreal.StaticMeshComponent):
                try:
                    sm = c.get_editor_property("static_mesh")
                    if not sm: continue
                    mn = sm.get_name()
                    if any(b in mn for b in foliage_blocklist): continue
                    if not any(k in mn for k in mesh_keywords): continue
                    comp.show_only_component(c); n_comp += 1
                    if len(samples) < 10 and mn not in samples: samples.append(mn)
                except Exception: pass
        except Exception:
            pass
    return n_comp, samples


def _rerun_construction(unreal, focus, radius):
    """Re-run Blueprint construction scripts on actors near the focus. The hero rock is a
    Rock_*_BP whose mesh is ASSIGNED in its construction script; when WP loads the actor
    shell without (re)running construction, the mesh is unset -> collapsed bounds -> spire
    invisible. rerun_construction_scripts() forces the BP to (re)build its mesh. Returns #run."""
    fx, fy = focus[0], focus[1]
    k = 0
    try:
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    except Exception:
        actors = []
    for a in actors:
        try:
            o, e = a.get_actor_bounds(False)
            if ((o.x - fx) ** 2 + (o.y - fy) ** 2) ** 0.5 > radius:
                continue
            a.rerun_construction_scripts(); k += 1
        except Exception:
            pass
    return k


def _wp_load(unreal):
    """Pin ALL World-Partition actor descriptors loaded (re-callable; streaming can evict,
    so we re-pin during settle until the scene height stabilizes). Returns #descriptors."""
    try:
        res = unreal.WorldPartitionBlueprintLibrary.get_actor_descs()
        descs = res[1] if isinstance(res, (tuple, list)) else res
        guids = []
        for d in (descs or []):
            try: guids.append(d.get_editor_property("guid"))
            except Exception: pass
        if guids:
            unreal.WorldPartitionBlueprintLibrary.load_actors(guids)
        return len(guids)
    except Exception as e:
        unreal.log_warning(f"[ed] WP load: {e}")
        return 0


def _tick(delta_seconds):
    import unreal
    try:
        if _S["wait"] > 0:
            _S["wait"] -= 1
            if _S["phase"] == "settle" and _S["wait"] % 120 == 0:
                _wp_load(unreal)                          # re-pin WP (streaming can evict)
                _rerun_construction(unreal, _S["focus"], _S["radius"] * 2.0)  # build Rock_BP meshes
                ism, inst = _count_foliage(unreal)
                top, nt, nm = _scene_top(unreal, _S["focus"], _S["radius"] * 1.5)
                unreal.log(f"[ed] settling... {_S['wait']} left; foliage ISM={ism} inst={inst}; "
                           f"rock_top={top:.0f}cm ({nt} rocks near focus; tallest='{nm}')")
            return

        if _S["phase"] == "settle":
            top, nt, nm = _scene_top(unreal, _S["focus"], _S["radius"] * 1.5)
            ism, inst = _count_foliage(unreal)
            # COMPLETENESS GATE: don't capture until the tall hero geometry (spire) has
            # streamed in. If a min height is set and not yet met, re-pin WP + wait more
            # (up to max_ext extensions) -- this is the guard that stops a silent
            # spire-less capture like scene20. min_top<=0 disables the gate (probe/measure).
            if _S["min_top"] > 0 and top < _S["min_top"] and _S["ext"] < _S["max_ext"]:
                _S["ext"] += 1; _S["wait"] = 150; _wp_load(unreal)
                unreal.log(f"[ed] GATE: scene_top={top:.0f}cm < min={_S['min_top']:.0f}cm "
                           f"-> re-pin WP + wait (ext {_S['ext']}/{_S['max_ext']})")
                return
            gate = "OK" if (_S["min_top"] <= 0 or top >= _S["min_top"]) else "PROCEED-ANYWAY(maxed)"
            unreal.log(f"[ed] settled [{gate}]. foliage ISM={ism} inst={inst}; "
                       f"rock_top={top:.0f}cm ({nt} rocks; tallest='{nm}'). capturing {len(_S['poses'])} poses.")
            _dump_scene(unreal, _S["focus"])             # one-shot: what tall geometry IS loaded
            _S["phase"] = "move"; _S["i"] = 0

        if _S["phase"] == "move":
            p = _S["poses"][_S["i"]]
            loc = unreal.Vector(*p["location_cm"])
            rot = unreal.MathLibrary.find_look_at_rotation(loc, unreal.Vector(*p["target_cm"]))
            _S["actor"].set_actor_location_and_rotation(loc, rot, False, False)
            if _S.get("dactor"):
                _S["dactor"].set_actor_location_and_rotation(loc, rot, False, False)
            for _ in range(_S["caps"]):
                _S["comp"].capture_scene()                  # prime; let TSR converge over the wait
            _S["wait"] = int(os.environ.get("UE_CONVERGE_TICKS", "10"))
            _S["phase"] = "shoot"
            return

        if _S["phase"] == "shoot":
            p = _S["poses"][_S["i"]]
            base = f"cam_{p['index']:03d}"
            avg = _S["avg"]
            hdr = os.environ.get("UE_HDR_COLOR") == "1"
            # per-sample color export (cam_IDX_SS when averaging, else cam_IDX)
            sbase = f"{base}_{_S['sample']:02d}" if avg > 1 else base
            _S["comp"].capture_scene()
            unreal.RenderingLibrary.export_render_target(_S["world"], _S["rt"], _S["out_dir"] + "/images", sbase)
            sraw = os.path.join(_S["out_dir"], "images", sbase)
            if hdr:
                # float RT exports a true EXR; keep it .exr (HDR averaging not supported
                # with multi-sample -- use avg=1 with UE_HDR_COLOR). Record future .png.
                if os.path.exists(sraw) and not os.path.exists(sraw + ".exr"):
                    os.replace(sraw, sraw + ".exr")
            elif os.path.exists(sraw) and not os.path.exists(sraw + ".png"):
                os.replace(sraw, sraw + ".png")
            # more color samples for this pose? capture next on the following tick so
            # Lumen GI / TSR noise re-randomises between samples.
            if avg > 1 and _S["sample"] + 1 < avg:
                _S["sample"] += 1
                _S["wait"] = 1
                return
            _S["sample"] = 0
            # geometric depth: one capture is enough (no averaging)
            depth_path = None
            if _S.get("dcomp"):
                _S["dcomp"].capture_scene()
                unreal.RenderingLibrary.export_render_target(_S["world"], _S["drt"], _S["out_dir"] + "/depth", base)
                draw = os.path.join(_S["out_dir"], "depth", base); dexr = draw + ".exr"
                if os.path.exists(draw) and not os.path.exists(dexr):
                    os.replace(draw, dexr)
                depth_path = dexr
            # file_path records the FINAL averaged cam_IDX.png (average_samples.py writes
            # it post-capture from the cam_IDX_SS samples; for avg=1 it already exists).
            png = os.path.join(_S["out_dir"], "images", base + ".png")
            _S["frames"].append({"file_path": png, "split": p["split"], "depth_path": depth_path,
                                 "location_cm": export.location_from_actor(unreal, _S["actor"]),
                                 "basis_ue": export.basis_from_actor(unreal, _S["actor"])})
            if (_S["i"] + 1) % 10 == 0:
                unreal.log(f"[ed] captured {_S['i']+1}/{len(_S['poses'])} (avg {avg})")
            _S["i"] += 1
            _S["phase"] = "move" if _S["i"] < len(_S["poses"]) else "done"
            return

        if _S["phase"] == "done":
            ext = _S["radius"] * 0.85
            ue_poses = os.path.join(_S["out_dir"], "ue_poses.json")
            f = _S["focus"]
            export.write_ue_poses(ue_poses, _S["train_res"], _S["train_res"], _S["hfov"], _S["frames"],
                                  scene_meta={"background": [0.0, 0.0, 0.0],
                                              "aabb_min_cm": [f[0]-ext, f[1]-ext, f[2]-ext],
                                              "aabb_max_cm": [f[0]+ext, f[1]+ext, f[2]+ext],
                                              "fiducials": [], "primitives": []})
            unreal.log(f"[ed] DONE wrote {len(_S['frames'])} frames -> {ue_poses}")
            print(f"EDITOR_CAPTURE_DONE frames={len(_S['frames'])} out={_S['out_dir']}")
            try: unreal.unregister_slate_post_tick_callback(_S["handle"])
            except Exception: pass
            try: unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(_S["actor"])
            except Exception: pass
            if _S.get("dactor"):
                try: unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(_S["dactor"])
                except Exception: pass
            # UE_NO_QUIT=1 (warm-editor capture run from the user's open editor console) ->
            # leave their editor open instead of closing it when the capture finishes.
            if os.environ.get("UE_NO_QUIT") != "1":
                try: unreal.SystemLibrary.quit_editor()
                except Exception: pass
            else:
                unreal.log("[ed] UE_NO_QUIT=1 -> capture done, leaving editor open")
    except Exception as e:
        unreal.log_error(f"[ed] tick error: {e}")
        try: unreal.unregister_slate_post_tick_callback(_S["handle"])
        except Exception: pass
        if os.environ.get("UE_NO_QUIT") != "1":
            try: unreal.SystemLibrary.quit_editor()
            except Exception: pass


def main():
    import unreal
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.environ.get("UE_CAPTURE_OUT") or os.path.join(
        repo, "out", "ed_editor_probe" if os.environ.get("UE_PROBE") == "1" else "ed_editor")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    cap_res = int(os.environ.get("UE_CAP_RES", "1536"))
    train_res = int(os.environ.get("UE_TRAIN_RES", str(cap_res)))
    hfov = float(os.environ.get("UE_HFOV", "75"))
    ev = float(os.environ.get("UE_CAPTURE_EV", "10"))
    caps = int(os.environ.get("UE_CAPS_PER_POSE", "3"))
    probe = os.environ.get("UE_PROBE", "0") == "1"

    # UE_SKIP_LOAD=1 (warm-editor capture): DON'T reload the map -- the user's open editor
    # already has the level + the PCG-generated spire (BP_PCG_LargeAssembly) built; a reload
    # would un-generate it (the spire only appears after PCG runs, which a fresh launch doesn't
    # reliably do). Use the currently-open world instead.
    if os.environ.get("UE_SKIP_LOAD") == "1":
        unreal.log("[ed] UE_SKIP_LOAD=1 -> using already-open level (warm editor)")
    else:
        try:
            unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
        except Exception as e:
            unreal.log_warning(f"[ed] load_map: {e}")
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()

    focus = [float(x) for x in os.environ["UE_FOCUS_CM"].split(",")] if os.environ.get("UE_FOCUS_CM") \
        else [89287.5, -5187.4, 1849.0]
    radius = float(os.environ.get("UE_ORBIT_RADIUS_CM", "1800"))

    # World Partition streams by SOURCE (the viewport). A fresh bridge-launched editor leaves
    # the viewport wherever it booted -> the focus region's big rock (spire) never streams,
    # even though PCG foliage (which we generate_local) is present. So: (1) move the editor
    # viewport TO the focus so WP streams that region; (2) activate all data layers (the spire
    # may be on a layer that's off by default -> its descriptor wouldn't even be returned);
    # (3) pin all actor descriptors. Then settle re-pins + the gate verifies the rock loaded.
    try:
        unreal.EditorLevelLibrary.set_level_viewport_camera_info(
            unreal.Vector(focus[0] - 3000, focus[1] - 3000, focus[2] + 2500),
            unreal.Rotator(-25.0, 45.0, 0.0))
        unreal.log("[ed] moved editor viewport to focus (WP streaming source)")
    except Exception as e:
        unreal.log_warning(f"[ed] viewport move: {e}")
    try:
        dls = unreal.get_editor_subsystem(unreal.DataLayerEditorSubsystem)
        layers = unreal.DataLayerEditorSubsystem.get_all_data_layers(dls) if hasattr(unreal.DataLayerEditorSubsystem, "get_all_data_layers") else dls.get_all_data_layers()
        k = 0
        for dl in (layers or []):
            for fn in ("set_data_layer_is_loaded_in_editor", "set_data_layer_visibility"):
                try:
                    getattr(dls, fn)(dl, True, True); k += 1; break
                except Exception:
                    try: getattr(dls, fn)(dl, True); k += 1; break
                    except Exception: pass
        unreal.log(f"[ed] data layers activated: {k}/{len(layers or [])}")
    except Exception as e:
        unreal.log_warning(f"[ed] data layers: {e}")
    ng = _wp_load(unreal)
    kc = _rerun_construction(unreal, focus, radius * 2.0)
    top0, nt0, nm0 = _scene_top(unreal, focus, radius * 1.5)
    unreal.log(f"[ed] WP: pinned {ng} descriptors; reran construction on {kc} actors; "
               f"initial rock_top={top0:.0f}cm ({nt0} rocks; tallest='{nm0}')")
    # UE_SKIP_PCG=1 (warm editor): the spire assembly + foliage are already generated; don't
    # re-run generate_local (it could disturb the loaded assembly). Otherwise generate PCG.
    if os.environ.get("UE_SKIP_PCG") == "1":
        unreal.log("[ed] UE_SKIP_PCG=1 -> not regenerating PCG (warm editor already has it)")
    else:
        try:
            for a in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
                pcg = a.get_component_by_class(unreal.PCGComponent)
                if pcg: pcg.generate_local(True)
        except Exception as e:
            unreal.log_warning(f"[ed] PCG: {e}")

    actor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
        unreal.SceneCapture2D, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    comp = actor.capture_component2d
    comp.fov_angle = hfov
    # UE_HDR_COLOR=1: capture LINEAR pre-tonemap color (SCS_FINAL_COLOR_HDR) into a
    # float RT -> true EXR export. The 8-bit LDR path crushes the shadow band to ~24
    # code values (measured); linear half-float holds ~21K distinct values there. A
    # local converter (scripts/hdr_to_training_png.py) then applies OUR tone curve
    # (filmic-matched mids/highlights, lifted toe) to make 8-bit training PNGs.
    hdr_color = os.environ.get("UE_HDR_COLOR") == "1"
    comp.capture_source = (unreal.SceneCaptureSource.SCS_FINAL_COLOR_HDR if hdr_color
                           else unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    for k, v in [("capture_every_frame", False), ("capture_on_movement", False),
                 ("always_persist_rendering_state", True)]:
        try: comp.set_editor_property(k, v)
        except Exception: pass
    flags = [_flag(unreal, "MotionBlur", False), _flag(unreal, "ScreenSpaceReflections", False),
             _flag(unreal, "Bloom", False)]
    if os.environ.get("UE_NOSKY") == "1":
        for nm in ("Atmosphere", "Fog", "VolumetricFog", "Cloud"):
            flags.append(_flag(unreal, nm, False))
    try: comp.set_editor_property("show_flag_settings", flags)
    except Exception: pass
    try:
        pp = comp.post_process_settings
        for k, v in [("auto_exposure_min_brightness", ev), ("auto_exposure_max_brightness", ev),
                     ("override_auto_exposure_min_brightness", True),
                     ("override_auto_exposure_max_brightness", True)]:
            pp.set_editor_property(k, v)
        comp.set_editor_property("post_process_settings", pp)
    except Exception: pass
    rt = unreal.RenderingLibrary.create_render_target2d(
        actor, cap_res, cap_res,
        unreal.TextureRenderTargetFormat.RTF_RGBA16F if hdr_color
        else unreal.TextureRenderTargetFormat.RTF_RGBA8)
    comp.texture_target = rt

    # UE_SHOW_ONLY=1 (NUCLEAR ground-only mode): MESH-NAME filter -- render only terrain
    # mesh instances, skip foliage. The PCGDemo BPs (DitchBP/GroundBP) mix terrain meshes
    # (Boulder/Cliff/Embankment/RockFormation/ForestGround/...) with foliage meshes
    # (MoneyPlant/Elderberry/Fern/...) IN THE SAME ACTOR, so actor-level filtering can't
    # separate them. Per-mesh-name does. Pairs with a foliage-on capture (scene21); the
    # merge gives the trainer ground supervision under what was previously canopy occlusion.
    if os.environ.get("UE_SHOW_ONLY") == "1":
        terrain_kw = tuple(os.environ.get("UE_TERRAIN_KEYWORDS",
            "Boulder,Cliff,Embankment,Beach,Pebbles,Stones,Formation,Rocky,Rock,ForestGround,ForestTerrain,Sandstone").split(","))
        foliage_block = tuple(os.environ.get("UE_FOLIAGE_BLOCK",
            "Plant,Grass,Fern,Ivy,Leaves,Berry,Archangel,Ginger,Periwinkle,Castor,Clover,Palm,Kikuyu,Arrow,Branch,Roots,Sphere,Icon,Cordyline,Amaryllis,Oak,Cover,WaterPlane,Money,YellowArch").split(","))
        incl_classes = tuple(os.environ.get("UE_SHOW_ONLY_CLASSES", "Landscape").split(","))
        nc, samples = _apply_show_only(unreal, comp, terrain_kw, foliage_block, incl_classes)
        unreal.log(f"[ed] UE_SHOW_ONLY=1 (RGB, mesh-filter): {nc} primitives kept "
                   f"(terrain={terrain_kw[:6]}+...); sample_meshes={samples[:6]}")

    # Optional GT metric-depth capture (UE_DEPTH=1): a SECOND SceneCapture2D bound to the
    # SAME view each pose, SCS_SCENE_DEPTH -> FLOAT RT (RTF_RGBA32f) -> EXR. SCS_SceneDepth
    # is LINEAR depth in world units (cm); a 32f RT keeps full precision (16f loses it past
    # a few thousand cm). export_render_target writes EXR for a float RT. Metric depth needs
    # NO SfM scale alignment (our COLMAP init is random) -> directly usable for depth-reg.
    dactor = dcomp = drt = None
    if os.environ.get("UE_DEPTH") == "1":
        try:
            os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)
            dactor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
                unreal.SceneCapture2D, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
            dcomp = dactor.capture_component2d
            dcomp.fov_angle = hfov
            # SCS_SceneDepth -> SCS_SCENE_DEPTH in the Python binding (matches SCS_FINAL_COLOR_LDR above)
            dcomp.capture_source = unreal.SceneCaptureSource.SCS_SCENE_DEPTH
            for k, v in [("capture_every_frame", False), ("capture_on_movement", False),
                         ("always_persist_rendering_state", True)]:
                try: dcomp.set_editor_property(k, v)
                except Exception: pass
            # HALF-float RT (RTF_RGBA16F) is the one export_render_target writes as a true
            # float EXR. RTF_RGBA32F instead exports a 16-bit PNG that treats the RT as [0,1]
            # COLOR, so depth-in-cm (>>1) saturates to white -> metric LOST (verified). 16f
            # holds our depth range (cm up to ~65504, precision ~0.5cm there); far/background
            # overflows to +inf -> a clean background marker. So PREFER 16F; never 32F here.
            fmt = fmtname = None
            for nm in ("RTF_RGBA16F", "RTF_R16F"):
                fmt = getattr(unreal.TextureRenderTargetFormat, nm, None)
                if fmt is not None: fmtname = nm; break
            if fmt is None:
                raise RuntimeError("no float TextureRenderTargetFormat found")
            drt = unreal.RenderingLibrary.create_render_target2d(dactor, cap_res, cap_res, fmt)
            dcomp.texture_target = drt
            unreal.log(f"[ed] UE_DEPTH=1: depth SceneCapture2D armed (SCS_SCENE_DEPTH, {fmtname}, fov={hfov})")
            # mirror the show-only allowlist onto the depth capture so it matches RGB
            if os.environ.get("UE_SHOW_ONLY") == "1":
                tkw = tuple(os.environ.get("UE_TERRAIN_KEYWORDS",
                    "Boulder,Cliff,Embankment,Beach,Pebbles,Stones,Formation,Rocky,Rock,ForestGround,ForestTerrain,Sandstone").split(","))
                fbl = tuple(os.environ.get("UE_FOLIAGE_BLOCK",
                    "Plant,Grass,Fern,Ivy,Leaves,Berry,Archangel,Ginger,Periwinkle,Castor,Clover,Palm,Kikuyu,Arrow,Branch,Roots,Sphere,Icon,Cordyline,Amaryllis,Oak,Cover,WaterPlane,Money,YellowArch").split(","))
                icl = tuple(os.environ.get("UE_SHOW_ONLY_CLASSES", "Landscape").split(","))
                ndc, _ = _apply_show_only(unreal, dcomp, tkw, fbl, icl)
                unreal.log(f"[ed] UE_SHOW_ONLY=1 (depth, mesh-filter): {ndc} primitives kept")
        except Exception as e:
            unreal.log_error(f"[ed] UE_DEPTH setup FAILED ({e}) -- continuing RGB-only")
            dactor = dcomp = drt = None

    if os.environ.get("UE_POSES_FILE"):
        # explicit pose list [{location_cm:[x,y,z], target_cm:[x,y,z]}] -> capture exactly
        # these (used by the VANTAGE rig: a converging slab from a fixed viewpoint). focus/
        # radius derived from the targets/locations for the settle + aabb machinery.
        import json as _json
        _pl = _json.load(open(os.environ["UE_POSES_FILE"]))
        poses = []
        for i, p in enumerate(_pl):
            poses.append({"index": i, "kind": "vantage",
                          "split": p.get("split", "heldout" if i % 8 == 1 else "train"),
                          "location_cm": p["location_cm"], "target_cm": p["target_cm"]})
        _tg = [p["target_cm"] for p in poses]; _lo = [p["location_cm"] for p in poses]
        focus = [sum(c[k] for c in _tg) / len(_tg) for k in range(3)]
        _allx = [c[0] for c in _lo] + [c[0] for c in _tg]; _ally = [c[1] for c in _lo] + [c[1] for c in _tg]
        radius = max(max(_allx) - min(_allx), max(_ally) - min(_ally)) / 2.0 + 2000.0
        unreal.log(f"[ed] POSES_FILE: {len(poses)} explicit poses; focus={[round(c) for c in focus]} r={radius/100:.0f}m")
    elif os.environ.get("UE_PATH_FILE"):
        # PATH flythrough: a road/ditch polyline (waypoints carry local ground z) ->
        # a forward+side+floor camera fan per step. focus/radius are derived from the
        # path bbox so the settle/WP-stream/aabb machinery covers the whole corridor.
        import json as _json
        _pf = _json.load(open(os.environ["UE_PATH_FILE"]))
        _wps = _pf["waypoints"]
        poses = rig.path_fan(_wps, step_cm=float(os.environ.get("UE_PATH_STEP", "350")),
                             eye_cm=float(os.environ.get("UE_PATH_EYE", "480")))
        for i, p in enumerate(poses):
            p["index"] = i
        _xs = [w[0] for w in _wps]; _ys = [w[1] for w in _wps]; _zs = [w[2] for w in _wps]
        focus = [sum(_xs) / len(_xs), sum(_ys) / len(_ys), sum(_zs) / len(_zs)]
        radius = max(max(_xs) - min(_xs), max(_ys) - min(_ys)) / 2.0 + 3000.0
        unreal.log(f"[ed] PATH: {len(_wps)} waypoints -> {len(poses)} fan poses; "
                   f"focus={[round(c) for c in focus]} radius={radius/100:.0f}m")
    elif probe:
        # LOW full-azimuth ring so the probe actually frames the hero spire silhouette
        # (the old elev 20/45 x az0/90/180/270 looked down at the pit and missed it).
        pelev = tuple(float(x) for x in os.environ.get("UE_PROBE_ELEV", "8,22").split(","))
        poses = rig.orbit_hemisphere(focus, radius * 1.6, elevations_deg=pelev,
                                     n_azimuth=int(os.environ.get("UE_PROBE_NAZ", "8")), heldout_every=0)
    elif os.environ.get("UE_SPIRE_ORBIT") == "1":
        # TIGHT orbit around the SPIRE (BP_PCG_LargeAssembly at ~(90250, -4360, ~3700)),
        # NOT the focus. Dome cameras orbit around focus; the spire is 12-14m east+north of
        # focus, so dome cameras see only its front -> back of spire is uncovered (visible
        # as a void from above). This pass orbits the spire directly to fill that gap.
        # UE_SPIRE_CENTER_CM=x,y,z overrides the default spire location.
        sc = os.environ.get("UE_SPIRE_CENTER_CM")
        if sc:
            sx, sy, sz = [float(v) for v in sc.split(",")]
        else:
            sx, sy, sz = 90250.0, -4360.0, 3700.0
        elev = tuple(float(v) for v in os.environ.get("UE_SPIRE_ELEV", "-5,15,35,55").split(","))
        radius_cm = float(os.environ.get("UE_SPIRE_RADIUS_CM", "2000"))
        naz = int(os.environ.get("UE_SPIRE_NAZ", "24"))
        poses = list(rig.orbit_hemisphere([sx, sy, sz], radius_cm, elevations_deg=elev,
                                          n_azimuth=naz, heldout_every=6))
        for i, p in enumerate(poses): p["index"] = i
        unreal.log(f"[ed] SPIRE_ORBIT: center=({sx:.0f},{sy:.0f},{sz:.0f}) r={radius_cm/100:.0f}m "
                   f"elev={elev} n_az={naz} -> {len(poses)} poses")
    elif os.environ.get("UE_FULL") == "1":
        # FULL-COVERAGE of the whole ~45m-radius terrain ISLAND (the extent probe showed a
        # finite ~90m patch; the first scene16 only reached ~25m -> floating-island edge).
        # close dome (central detail) + WIDE ring (outer patch + sides, low-mid angles) +
        # big nadir grid (the whole patch from above). One editor session (one settle).
        fx, fy = focus[0], focus[1]
        dome = rig.orbit_hemisphere([fx, fy, 1849.0], 3000.0,
                                    elevations_deg=(8.0, 22.0, 36.0, 50.0, 64.0, 76.0),
                                    n_azimuth=40, heldout_every=6)             # ~240, central detail
        wide = rig.orbit_hemisphere([fx, fy, 1500.0], 6500.0,
                                    elevations_deg=(5.0, 16.0, 28.0), n_azimuth=40, heldout_every=6)  # ~120, full patch + sides
        grid = rig.grid_nadir([fx, fy, 1150.0], 4800.0, 3500.0, n_side=9, converge=0.2, heldout_every=8)  # 81, whole patch nadir
        passes = list(dome) + list(wide) + list(grid)
        # UE_GROUND_DENSE=1: a denser, LOWER, more-converged nadir grid to add ground coverage +
        # GT-depth maps where the patches show (paired with UE_DEPTH for depth-supervised fill).
        # converge 0.4 tilts cams so they also see obliquely INTO gaps between rock masses (a flat
        # nadir can't see under canopy -- that ground is occluded from every camera -- but the
        # oblique tilt + lower height recover more of the between-rock ground than the 11.5m grid).
        gdense = []
        if os.environ.get("UE_GROUND_DENSE") == "1":
            gdense = rig.grid_nadir([fx, fy, 850.0], 5000.0, 3800.0, n_side=11, converge=0.4, heldout_every=8)  # 121
            passes = passes + list(gdense)
        poses = []
        for i, p in enumerate(passes):
            q = dict(p); q["index"] = i; poses.append(q)
        unreal.log(f"[ed] FULL: dome {len(dome)} + wide {len(wide)} + grid {len(grid)}"
                   f" + gdense {len(gdense)} = {len(poses)}")
    else:
        elev = tuple(float(x) for x in os.environ.get("UE_ELEVATIONS", "8,22,36,50,64,76").split(","))
        poses = rig.orbit_hemisphere(focus, radius, elevations_deg=elev,
                                     n_azimuth=int(os.environ.get("UE_N_AZ", "40")), heldout_every=6)

    _S.update({"phase": "settle", "i": 0, "wait": int(os.environ.get("UE_SETTLE_TICKS", "600")),
               "poses": poses, "frames": [], "comp": comp, "rt": rt, "actor": actor, "world": world,
               "out_dir": out_dir, "caps": caps, "train_res": train_res, "hfov": hfov,
               "focus": focus, "radius": radius,
               # UE_AVG_SAMPLES>1: export N color samples per pose (cam_IDX_SS.png),
               # one per tick so Lumen/TSR noise re-randomises; scripts/average_samples.py
               # folds them -> cam_IDX.png post-capture (kills foliage spike-floaters).
               "avg": max(1, int(os.environ.get("UE_AVG_SAMPLES", "1"))), "sample": 0,
               "dactor": dactor, "dcomp": dcomp, "drt": drt,
               # completeness gate: require tall geometry (spire) loaded before capturing.
               "min_top": float(os.environ.get("UE_MIN_SCENE_TOP_CM", "0")), "ext": 0,
               "max_ext": int(os.environ.get("UE_MAX_SETTLE_EXT", "8"))})
    _S["handle"] = unreal.register_slate_post_tick_callback(_tick)
    unreal.log(f"[ed] registered tick callback; {len(poses)} poses, settle {_S['wait']} ticks. "
               f"focus={focus} radius={radius/100:.1f}m res={cap_res} ev={ev}")
    print(f"EDITOR_CAPTURE_ARMED poses={len(poses)} (returning to let the editor tick)")


main()
