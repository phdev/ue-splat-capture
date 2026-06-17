"""Reliable rail PREVIEW: attach a Cine Camera to CAPTURE_PATH_RAIL so you can fly the
capture path by scrubbing the rail's "Current Position on Rail" (0..1) slider in the
Details panel. This sidesteps Sequencer entirely.

Run INSIDE the warm editor (keep the UE window FOREGROUND):
    python3 scripts/ue_exec.py scripts/path_rail_pilot.py 90

Why this and not an auto-play LevelSequence: in this project a baked
MovieScene3DTransformTrack (possessable AND spawnable) did NOT drive the camera in
PIE or on Sequencer scrub -- the keys/section/range all read back correct, but the
spawned/possessed camera stayed at the start. Animating the rail's
CurrentPositionOnRail also fails at RUNTIME (the rail only repositions attached actors
in-editor). What DOES work, verified (camera travels ~3415cm as the slider goes
0->0.5): attach the camera to the rail and move the slider in-editor. So the preview
is a manual editor scrub, not PIE.

USE IT: 1) right-click PATH_PREVIEW_CAM in the Outliner -> "Pilot 'PATH_PREVIEW_CAM'"
(viewport looks through it). 2) select CAPTURE_PATH_RAIL; in Details drag
"Current Position on Rail" 0 -> 1. The viewport flies the path. Eject when done.
"""
import math
import unreal

TAG_RAIL = "CAPTURE_PATH"
TAG_CAM = "PATH_PREVIEW_CAM"
TAG_SEQ = "PATH_PREVIEW_SEQ"
EYE = 480.0


def main():
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    rail = None
    for a in eas.get_all_level_actors():
        try:
            if a.actor_has_tag(TAG_RAIL):
                rail = a
                break
        except Exception:
            pass
    if not rail:
        print("NO_RAIL")
        return

    # Clear any prior preview cameras + (non-working) auto-play sequence actors.
    killed = 0
    for a in list(eas.get_all_level_actors()):
        try:
            lbl = a.get_actor_label()
            if (a.actor_has_tag(TAG_CAM) or a.actor_has_tag(TAG_SEQ)
                    or lbl.startswith("PATH_PREVIEW_CAM") or isinstance(a, unreal.LevelSequenceActor)):
                eas.destroy_actor(a)
                killed += 1
        except Exception:
            pass

    rail.set_editor_property("lock_orientation_to_rail", True)
    sp = rail.get_components_by_class(unreal.SplineComponent)[0]
    s = sp.get_location_at_distance_along_spline(0.0, unreal.SplineCoordinateSpace.WORLD)
    cam = eas.spawn_actor_from_class(unreal.CineCameraActor,
                                     unreal.Vector(s.x, s.y, s.z + EYE), unreal.Rotator(0, 0, 0))
    cam.set_actor_label("PATH_PREVIEW_CAM")
    try:
        cam.set_editor_property("tags", [unreal.Name(TAG_CAM)])
    except Exception:
        pass
    cam.attach_to_actor(rail, "", unreal.AttachmentRule.KEEP_WORLD,
                        unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD, False)
    cam.set_actor_relative_location(unreal.Vector(0, 0, EYE), False, False)

    rail.set_editor_property("current_position_on_rail", 0.5)
    p1 = cam.get_actor_location()
    rail.set_editor_property("current_position_on_rail", 0.0)
    p0 = cam.get_actor_location()
    moved = math.sqrt((p0.x - p1.x) ** 2 + (p0.y - p1.y) ** 2 + (p0.z - p1.z) ** 2)
    eas.set_selected_level_actors([rail])
    print("PILOT_READY killed=%d attach_move=%.0fcm (slider 0->0.5)" % (killed, moved))


main()
