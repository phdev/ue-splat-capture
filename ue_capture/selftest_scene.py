"""Self-test scene definition + Unreal spawner (pure stdlib + lazy `unreal`).

The geometry constants MUST stay in sync with ``selftest/scene.py`` (the numpy
stand-in) so a UE capture reproduces the same scene/fiducials/rig. They are kept
here as plain dicts (no numpy) because this runs inside UnrealEditor-Cmd.
"""
from __future__ import annotations

# --- canonical scene (UE convention: centimetres, Z-up) -------------------- #
INTRINSICS = {"w": 96, "h": 96, "hfov_deg": 55.0}
BACKGROUND = [0.12, 0.14, 0.18]
AABB_MIN_CM = [-150.0, -150.0, 0.0]
AABB_MAX_CM = [150.0, 150.0, 160.0]
TARGET_CENTER_CM = [0.0, 0.0, 35.0]
ORBIT_RADIUS_CM = 360.0

PLATFORM = {"min": [-150.0, -150.0, -12.0], "max": [150.0, 150.0, 0.0],
            "color": [0.8, 0.8, 0.83]}

OBJECTS = [
    {"shape": "sphere", "center": [70.0, 40.0, 45.0], "radius": 45.0, "color": [0.85, 0.25, 0.20]},
    {"shape": "sphere", "center": [-80.0, -30.0, 35.0], "radius": 35.0, "color": [0.20, 0.45, 0.85]},
    {"shape": "sphere", "center": [10.0, 90.0, 28.0], "radius": 28.0, "color": [0.95, 0.80, 0.20]},
    {"shape": "box", "min": [-130.0, 40.0, 0.0], "max": [-70.0, 100.0, 60.0], "color": [0.30, 0.70, 0.40]},
    {"shape": "box", "min": [60.0, -110.0, 0.0], "max": [120.0, -50.0, 40.0], "color": [0.70, 0.50, 0.85]},
]

FIDUCIALS = [
    {"id": "F0", "loc_cm": [0.0, 0.0, 8.0], "radius": 7.0, "color": [1.0, 1.0, 1.0]},
    {"id": "F1", "loc_cm": [120.0, 120.0, 15.0], "radius": 7.0, "color": [1.0, 0.0, 1.0]},
    {"id": "F2", "loc_cm": [-120.0, 120.0, 90.0], "radius": 7.0, "color": [0.0, 1.0, 1.0]},
    {"id": "F3", "loc_cm": [120.0, -120.0, 60.0], "radius": 7.0, "color": [1.0, 0.5, 0.0]},
    {"id": "F4", "loc_cm": [-120.0, -120.0, 30.0], "radius": 7.0, "color": [0.2, 1.0, 0.2]},
    {"id": "F5", "loc_cm": [0.0, -140.0, 110.0], "radius": 7.0, "color": [1.0, 1.0, 0.0]},
    {"id": "F6", "loc_cm": [140.0, 0.0, 100.0], "radius": 7.0, "color": [0.5, 0.5, 1.0]},
    {"id": "F7", "loc_cm": [-30.0, 60.0, 150.0], "radius": 7.0, "color": [1.0, 0.3, 0.3]},
]

# Engine basic-shape meshes are 100 cm across with a default 50 cm radius / 100 cm box.
_SPHERE_MESH = "/Engine/BasicShapes/Sphere.Sphere"
_CUBE_MESH = "/Engine/BasicShapes/Cube.Cube"
_BASE_MAT = "/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"


def primitives_for_scene_json():
    """Geometry list (for scene.json / T2 camera-in-geometry), UE cm."""
    prims = [{"type": "box", "min": PLATFORM["min"], "max": PLATFORM["max"]}]
    for o in OBJECTS:
        if o["shape"] == "sphere":
            prims.append({"type": "sphere", "center": o["center"], "radius": o["radius"]})
        else:
            prims.append({"type": "box", "min": o["min"], "max": o["max"]})
    return prims


# --------------------------------------------------------------------------- #
# Unreal spawning (only runs inside UnrealEditor-Cmd)
# --------------------------------------------------------------------------- #
def _colored_material(unreal, rgb, emissive=False):
    base = unreal.load_object(None, _BASE_MAT)
    mid = unreal.MaterialInstanceDynamic.create(base, None)
    col = unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0)
    for pname in ("Color", "BaseColor", "Tint"):
        try:
            mid.set_vector_parameter_value(pname, col)
        except Exception:
            pass
    if emissive:
        for pname in ("Emissive", "EmissiveColor", "Emissive Color"):
            try:
                mid.set_vector_parameter_value(pname, col)
            except Exception:
                pass
    return mid


def _spawn_mesh(unreal, actors_sys, mesh_path, location_cm, scale, rgb, emissive=False):
    mesh = unreal.load_object(None, mesh_path)
    loc = unreal.Vector(*location_cm)
    actor = actors_sys.spawn_actor_from_object(mesh, loc, unreal.Rotator(0, 0, 0))
    comp = actor.static_mesh_component
    comp.set_world_scale3d(unreal.Vector(*scale))
    try:
        comp.set_material(0, _colored_material(unreal, rgb, emissive))
    except Exception as e:  # pragma: no cover - UE only
        unreal.log_warning(f"material set failed: {e}")
    return actor


def spawn_scene(unreal):
    """Spawn platform, objects, and emissive fiducials. Returns spawned actors."""
    actors_sys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    spawned = []

    # platform: scale the 100cm cube to the platform extents
    pmin, pmax = PLATFORM["min"], PLATFORM["max"]
    size = [(pmax[i] - pmin[i]) / 100.0 for i in range(3)]
    center = [(pmax[i] + pmin[i]) / 2.0 for i in range(3)]
    spawned.append(_spawn_mesh(unreal, actors_sys, _CUBE_MESH, center, size, PLATFORM["color"]))

    for o in OBJECTS:
        if o["shape"] == "sphere":
            s = [o["radius"] / 50.0] * 3   # basic sphere is 50 cm radius
            spawned.append(_spawn_mesh(unreal, actors_sys, _SPHERE_MESH, o["center"], s, o["color"]))
        else:
            size = [(o["max"][i] - o["min"][i]) / 100.0 for i in range(3)]
            center = [(o["max"][i] + o["min"][i]) / 2.0 for i in range(3)]
            spawned.append(_spawn_mesh(unreal, actors_sys, _CUBE_MESH, center, size, o["color"]))

    for f in FIDUCIALS:
        s = [f["radius"] / 50.0] * 3
        spawned.append(_spawn_mesh(unreal, actors_sys, _SPHERE_MESH, f["loc_cm"], s,
                                   f["color"], emissive=True))
    return spawned
