"""TICK-DRIVEN capture inside a LAUNCHED GUI editor (UnrealEditor, NOT -Cmd).

Headless (-Cmd + a blocking python script) can't generate PCG foliage or converge
TSR/Lumen, because a blocking script never yields the engine ticks those need. This
script instead registers a slate post-tick callback and RETURNS -- the editor keeps
ticking (PCG generates, temporal AA accumulates) while a small state machine walks the
poses one capture per few ticks. Launch it with scripts/capture_editor_run.sh.

Env: UE_CAPTURE_OUT (abs), UE_FOCUS_CM, UE_ORBIT_RADIUS_CM, UE_ELEVATIONS, UE_N_AZ,
     UE_HFOV, UE_CAP_RES, UE_TRAIN_RES, UE_CAPTURE_EV, UE_SETTLE_TICKS (wait for
     level+PCG, default 600), UE_CONVERGE_TICKS (per-pose TSR settle, default 10),
     UE_PROBE (1 = a few overview poses to validate foliage).
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


def _tick(delta_seconds):
    import unreal
    try:
        if _S["wait"] > 0:
            _S["wait"] -= 1
            if _S["phase"] == "settle" and _S["wait"] % 120 == 0:
                ism, inst = _count_foliage(unreal)
                unreal.log(f"[ed] settling... {_S['wait']} ticks left; foliage ISM={ism} instances={inst}")
            return

        if _S["phase"] == "settle":
            ism, inst = _count_foliage(unreal)
            unreal.log(f"[ed] settled. foliage ISM={ism} instances={inst}. capturing {len(_S['poses'])} poses.")
            _S["phase"] = "move"; _S["i"] = 0

        if _S["phase"] == "move":
            p = _S["poses"][_S["i"]]
            loc = unreal.Vector(*p["location_cm"])
            rot = unreal.MathLibrary.find_look_at_rotation(loc, unreal.Vector(*p["target_cm"]))
            _S["actor"].set_actor_location_and_rotation(loc, rot, False, False)
            for _ in range(_S["caps"]):
                _S["comp"].capture_scene()                  # prime; let TSR converge over the wait
            _S["wait"] = int(os.environ.get("UE_CONVERGE_TICKS", "10"))
            _S["phase"] = "shoot"
            return

        if _S["phase"] == "shoot":
            p = _S["poses"][_S["i"]]
            _S["comp"].capture_scene()
            base = f"cam_{p['index']:03d}"
            unreal.RenderingLibrary.export_render_target(_S["world"], _S["rt"], _S["out_dir"] + "/images", base)
            raw = os.path.join(_S["out_dir"], "images", base); png = raw + ".png"
            if os.path.exists(raw) and not os.path.exists(png):
                os.replace(raw, png)
            _S["frames"].append({"file_path": png, "split": p["split"],
                                 "location_cm": export.location_from_actor(unreal, _S["actor"]),
                                 "basis_ue": export.basis_from_actor(unreal, _S["actor"])})
            if (_S["i"] + 1) % 10 == 0:
                unreal.log(f"[ed] captured {_S['i']+1}/{len(_S['poses'])}")
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
            try: unreal.SystemLibrary.quit_editor()
            except Exception: pass
    except Exception as e:
        unreal.log_error(f"[ed] tick error: {e}")
        try: unreal.unregister_slate_post_tick_callback(_S["handle"])
        except Exception: pass
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

    try:
        unreal.EditorLoadingAndSavingUtils.load_map(LEVEL)
    except Exception as e:
        unreal.log_warning(f"[ed] load_map: {e}")
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()

    focus = [float(x) for x in os.environ["UE_FOCUS_CM"].split(",")] if os.environ.get("UE_FOCUS_CM") \
        else [89287.5, -5187.4, 1849.0]
    radius = float(os.environ.get("UE_ORBIT_RADIUS_CM", "1800"))

    # nudge World Partition + PCG to populate (the GUI editor ticks will finish it).
    # Use get_actor_descs() (the headless-proven API; get_actor_descriptor_instances
    # doesn't exist) -> load all descriptors so the capture region is streamed in.
    try:
        res = unreal.WorldPartitionBlueprintLibrary.get_actor_descs()
        descs = res[1] if isinstance(res, (tuple, list)) else res
        guids = []
        for d in (descs or []):
            try: guids.append(d.get_editor_property("guid"))
            except Exception: pass
        if guids:
            unreal.WorldPartitionBlueprintLibrary.load_actors(guids)
            unreal.log(f"[ed] WP: loading {len(guids)} actor descriptors")
    except Exception as e:
        unreal.log_warning(f"[ed] WP load: {e}")
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
    comp.capture_source = unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
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
    rt = unreal.RenderingLibrary.create_render_target2d(actor, cap_res, cap_res,
                                                        unreal.TextureRenderTargetFormat.RTF_RGBA8)
    comp.texture_target = rt

    if probe:
        poses = rig.orbit_hemisphere(focus, radius * 1.3, elevations_deg=(20.0, 45.0),
                                     n_azimuth=4, heldout_every=0)
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
        poses = []
        for i, p in enumerate(list(dome) + list(wide) + list(grid)):
            q = dict(p); q["index"] = i; poses.append(q)
        unreal.log(f"[ed] FULL: dome {len(dome)} + wide {len(wide)} + grid {len(grid)} = {len(poses)}")
    else:
        elev = tuple(float(x) for x in os.environ.get("UE_ELEVATIONS", "8,22,36,50,64,76").split(","))
        poses = rig.orbit_hemisphere(focus, radius, elevations_deg=elev,
                                     n_azimuth=int(os.environ.get("UE_N_AZ", "40")), heldout_every=6)

    _S.update({"phase": "settle", "i": 0, "wait": int(os.environ.get("UE_SETTLE_TICKS", "600")),
               "poses": poses, "frames": [], "comp": comp, "rt": rt, "actor": actor, "world": world,
               "out_dir": out_dir, "caps": caps, "train_res": train_res, "hfov": hfov,
               "focus": focus, "radius": radius})
    _S["handle"] = unreal.register_slate_post_tick_callback(_tick)
    unreal.log(f"[ed] registered tick callback; {len(poses)} poses, settle {_S['wait']} ticks. "
               f"focus={focus} radius={radius/100:.1f}m res={cap_res} ev={ev}")
    print(f"EDITOR_CAPTURE_ARMED poses={len(poses)} (returning to let the editor tick)")


main()
