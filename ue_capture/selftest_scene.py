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
BG_COLOR = [0.30, 0.33, 0.42]   # emissive backdrop dome colour


def _fresh_material(unreal, name):
    p = "/Game/" + name
    if unreal.EditorAssetLibrary.does_asset_exist(p):
        unreal.EditorAssetLibrary.delete_asset(p)
    return unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, "/Game", unreal.Material, unreal.MaterialFactoryNew())


def ensure_color_material(unreal):
    """Author a MATTE material: 'Color' VectorParameter -> BaseColor, Roughness=1,
    Specular=0. Matte = view-independent, so a lit FinalColorLDR capture stays
    SH0-friendly (no moving specular highlights). Validated live on UE 5.7."""
    mel = unreal.MaterialEditingLibrary
    mat = _fresh_material(unreal, "M_SplatMatte")
    vp = mel.create_material_expression(mat, unreal.MaterialExpressionVectorParameter, -350, 0)
    vp.set_editor_property("parameter_name", "Color")
    mel.connect_material_property(vp, "", unreal.MaterialProperty.MP_BASE_COLOR)
    cr = mel.create_material_expression(mat, unreal.MaterialExpressionConstant, -350, 200)
    cr.set_editor_property("r", 1.0)
    mel.connect_material_property(cr, "", unreal.MaterialProperty.MP_ROUGHNESS)
    cs = mel.create_material_expression(mat, unreal.MaterialExpressionConstant, -350, 320)
    cs.set_editor_property("r", 0.0)
    mel.connect_material_property(cs, "", unreal.MaterialProperty.MP_SPECULAR)
    mel.recompile_material(mat)
    return mat


def ensure_bg_material(unreal):
    """Unlit, two-sided, constant-emissive material for the backdrop dome -> a
    clean solid background from every orbit view (the RT clear colour does NOT
    fill FinalColorLDR empties, and the empty void is noisy/black otherwise)."""
    mel = unreal.MaterialEditingLibrary
    mat = _fresh_material(unreal, "M_Bg")
    mat.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    mat.set_editor_property("two_sided", True)
    c3 = mel.create_material_expression(mat, unreal.MaterialExpressionConstant3Vector, -350, 0)
    c3.set_editor_property("constant", unreal.LinearColor(*BG_COLOR, 1.0))
    mel.connect_material_property(c3, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(mat)
    return mat


def _spawn_mesh(unreal, actors_sys, mat, mesh_path, location_cm, scale, rgb):
    """Spawn a basic-shape mesh with a per-object coloured instance of `mat`."""
    mesh = unreal.load_asset(mesh_path)
    actor = actors_sys.spawn_actor_from_object(mesh, unreal.Vector(*location_cm),
                                               unreal.Rotator(0, 0, 0))
    comp = actor.static_mesh_component
    comp.set_world_scale3d(unreal.Vector(*scale))
    try:
        mid = unreal.MaterialLibrary.create_dynamic_material_instance(actor, mat)
        mid.set_vector_parameter_value("Color", unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0))
        try:  # BasicShapeMaterial (textured platform) -> force matte
            mid.set_scalar_parameter_value("Roughness", 1.0)
        except Exception:
            pass
        comp.set_material(0, mid)
    except Exception as e:  # pragma: no cover - UE only
        unreal.log_warning(f"material set failed: {e}")
    return actor


def spawn_scene(unreal):
    """Spawn lights, backdrop dome, platform, objects, and fiducials.

    Lit FinalColorLDR capture: a strong top key + all-axis fill directionals give
    every face illumination (cameras orbit, so fixed lights alone would leave
    camera-facing sides black) plus a top-biased shading gradient for depth cues;
    matte materials keep it view-independent. An unlit emissive dome provides a
    solid background. Exposure is pinned on the SceneCapture (render.py).
    """
    actors_sys = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    mat = ensure_color_material(unreal)
    bg_mat = ensure_bg_material(unreal)
    spawned = []

    # lights: strong top key + 4 side fills + weak bottom fill
    for (pitch, yaw, inten) in [(-90, 0, 3.5), (-15, 0, 1.6), (-15, 180, 1.6),
                                (-15, 90, 1.6), (-15, -90, 1.6), (60, 0, 0.8)]:
        lt = actors_sys.spawn_actor_from_class(
            unreal.DirectionalLight, unreal.Vector(0, 0, 500), unreal.Rotator(pitch, yaw, 0))
        try:
            lt.directional_light_component.set_intensity(inten)
        except Exception:
            pass

    # backdrop dome (big two-sided unlit emissive sphere, no shadow)
    dome = actors_sys.spawn_actor_from_object(
        unreal.load_asset(_SPHERE_MESH), unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    dome.static_mesh_component.set_world_scale3d(unreal.Vector(40, 40, 40))
    try:
        dome.static_mesh_component.set_material(
            0, unreal.MaterialLibrary.create_dynamic_material_instance(dome, bg_mat))
        dome.static_mesh_component.set_cast_shadow(False)
    except Exception as e:  # pragma: no cover
        unreal.log_warning(f"dome material: {e}")

    # platform: scale the 100cm cube to the platform extents
    pmin, pmax = PLATFORM["min"], PLATFORM["max"]
    size = [(pmax[i] - pmin[i]) / 100.0 for i in range(3)]
    center = [(pmax[i] + pmin[i]) / 2.0 for i in range(3)]
    # platform uses the matte material too (BasicShapeMaterial's specular is not
    # zeroable and its view-dependent highlights blow up the held-out gap).
    spawned.append(_spawn_mesh(unreal, actors_sys, mat, _CUBE_MESH, center, size, PLATFORM["color"]))

    for o in OBJECTS:
        if o["shape"] == "sphere":
            s = [o["radius"] / 50.0] * 3   # basic sphere is 50 cm radius
            spawned.append(_spawn_mesh(unreal, actors_sys, mat, _SPHERE_MESH, o["center"], s, o["color"]))
        else:
            size = [(o["max"][i] - o["min"][i]) / 100.0 for i in range(3)]
            center = [(o["max"][i] + o["min"][i]) / 2.0 for i in range(3)]
            spawned.append(_spawn_mesh(unreal, actors_sys, mat, _CUBE_MESH, center, size, o["color"]))

    for f in FIDUCIALS:
        s = [f["radius"] / 50.0] * 3
        spawned.append(_spawn_mesh(unreal, actors_sys, mat, _SPHERE_MESH, f["loc_cm"], s, f["color"]))
    return spawned
