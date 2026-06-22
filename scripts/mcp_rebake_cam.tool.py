import json, math

ACTOR = ("/Game/Levels/PCG/ElectricDreams_PCGCloseRange."
         "ElectricDreams_PCGCloseRange:PersistentLevel."
         "BP_PathFly_C_UAID_F02F4B078FB99FE502_1282371579")
IM = ACTOR + ".InterpMove"

CX, CY = 89712.0, -5226.0   # scene center (cm) — matches the rail
R = 3000.0                  # radius 30 m — matches the rail
FLAT_Z = 2300.0             # constant ring height — matches the rail
N = 72                      # control points around the ring (smooth linear interp)


def set_actor_xform(loc, rot):
    return execute_tool(
        "editor_toolset.toolsets.actor.ActorTools.set_actor_transform",
        json.dumps({"actor": {"refPath": ACTOR},
                    "xform": {"location": loc, "rotation": rot}}))["returnValue"]


def set_props(refpath, values):
    return execute_tool(
        "editor_toolset.toolsets.object.ObjectTools.set_properties",
        json.dumps({"instance": {"refPath": refpath},
                    "values": json.dumps(values)}))["returnValue"]


def cp_count():
    r = execute_tool(
        "editor_toolset.toolsets.object.ObjectTools.get_properties",
        json.dumps({"instance": {"refPath": IM},
                    "properties": ["controlPoints"]}))["returnValue"]
    return len(json.loads(r)["controlPoints"])


def run():
    # ring point i (i in 0..N, last == first so the loop closes seamlessly)
    def w(i):
        a = 2.0 * math.pi * i / N
        return (CX + R * math.cos(a), CY + R * math.sin(a), FLAT_Z)

    w0 = w(0)  # start point, +X side of center
    # Root rotation MUST be identity: InterpToMovement applies its control points
    # through the actor's root rotation (world_pos = RootRotation . ControlPoint),
    # so any non-zero root yaw rotates the whole path around the origin. Facing is
    # handled separately by rotating the CameraComponent in the event graph.
    moved = set_actor_xform(
        {"x": w0[0], "y": w0[1], "z": w0[2]},
        {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})

    # ABSOLUTE world control points (bPositionIsRelative=false): the camera flies
    # the exact world ring regardless of where it spawns. Relative points orbit a
    # shifted center if the play-start location differs from the editor position
    # (it does in Simulate's streamed/duplicated world).
    cps = []
    for i in range(N + 1):
        wi = w(i)
        cps.append({"positionControlPoint": {"x": wi[0], "y": wi[1], "z": wi[2]},
                    "bPositionIsRelative": False})

    # The setter only appends ONE element per call when growing, so clear to
    # empty then send growing prefixes — each call appends exactly one point.
    set_props(IM, {"controlPoints": []})
    for k in range(len(cps)):
        set_props(IM, {"controlPoints": cps[:k + 1]})
    final = cp_count()

    # verify world positions land on the flat ring (points are absolute world)
    bad = 0
    for i, c in enumerate(cps):
        p = c["positionControlPoint"]
        wx, wy, wz = p["x"], p["y"], p["z"]
        if abs(math.hypot(wx - CX, wy - CY) - R) > 1.0 or abs(wz - FLAT_Z) > 1.0:
            bad += 1
    return {"moved": moved, "n_set": final, "n_expected": len(cps),
            "off_ring": bad, "start_world": list(w0),
            "radius_cm": R, "flat_z": FLAT_Z}
