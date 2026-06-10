"""Add (or remove) the spire back-face FILL lights in the warm editor — the scene29 fix.

Run via:  python3 scripts/ue_exec.py scripts/ue_add_fill_lights.py
Remove:   UE_FILL_REMOVE=1 python3 ... (env is read INSIDE the editor process, so set it
          there via a wrapper, or just edit REMOVE below)

WHY (the black-on-black law, measured on scene28): the spire's shadowed back face
renders ~0.04 luminance against the pure-black UE_NOSKY background. Photometrically a
deleted surface and the background are nearly identical, so 3DGS training is FREE to
delete/smear the face (4.4x fewer gaussians than the sunlit front; every opacity reset
crushes the faint survivors). No post-hoc filter can fix it — scene29a-e tried signature
strips, depth-truth culling, scale clamps; the no-reset run saved the face but grew
un-cullable fog. The DATA fix: light the face IN THE SCENE (scene-consistent across all
views, unlike per-camera EV tricks), capture, train with DEFAULT opacity resets.

Numbers that worked (scene29): 4 spotlights = 2 azimuth columns straddling the anti-sun
direction (sun yaw +/-45 deg) x 2 heights (z 5200/3000 aiming 4600/2400), R=14m from the
column, 2,500,000 cd (300k cd was INVISIBLE at the pinned-EV10 exposure calibrated for a
100k-lux sun), outer cone 50, shadows ON (protects the sunlit front), specular 0.25.
Result: back-face p50 0.04 -> 0.08-0.15, p90 ~0.5; front face byte-identical bright.
"""
import math
import unreal

REMOVE = False  # flip to True (or copy to /tmp and edit) to clean the lights out

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
for a in list(eas.get_all_level_actors()):
    if a.get_actor_label().startswith("FILL_BACK_"):
        eas.destroy_actor(a)
if REMOVE:
    print("FILL lights removed")
else:
    sun = next(a for a in eas.get_all_level_actors()
               if "DirectionalLight" in a.get_class().get_name())
    fwd = sun.get_actor_rotation().get_forward_vector()
    base_az = math.degrees(math.atan2(fwd.y, fwd.x))  # anti-sun azimuth

    SPIRE_X, SPIRE_Y = 90250.0, -4360.0
    R = 1400.0
    INTENS = 2500000.0
    n = 0
    for daz in (-45.0, 45.0):
        az = math.radians(base_az + daz)
        px = SPIRE_X + math.cos(az) * R
        py = SPIRE_Y + math.sin(az) * R
        for lz, aimz in ((5200.0, 4600.0), (3000.0, 2400.0)):
            pos = unreal.Vector(px, py, lz)
            tgt = unreal.Vector(SPIRE_X, SPIRE_Y, aimz)
            rot = unreal.MathLibrary.find_look_at_rotation(pos, tgt)
            actor = eas.spawn_actor_from_class(unreal.SpotLight, pos, rot)
            actor.set_actor_label(f"FILL_BACK_{n}")
            comp = actor.get_component_by_class(unreal.SpotLightComponent)
            comp.set_mobility(unreal.ComponentMobility.MOVABLE)
            comp.set_editor_property("intensity_units", unreal.LightUnits.CANDELAS)
            comp.set_editor_property("intensity", INTENS)
            comp.set_editor_property("attenuation_radius", 6000.0)
            comp.set_editor_property("outer_cone_angle", 50.0)
            comp.set_editor_property("inner_cone_angle", 25.0)
            comp.set_editor_property("cast_shadows", True)
            comp.set_editor_property("specular_scale", 0.25)
            n += 1
    print(f"added {n} FILL_BACK lights @ {INTENS:.0f} cd (anti-sun az {base_az:.1f})")
