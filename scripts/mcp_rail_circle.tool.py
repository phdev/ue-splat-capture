import json, math

SPLINE = ("/Game/Levels/PCG/ElectricDreams_PCGCloseRange."
          "ElectricDreams_PCGCloseRange:PersistentLevel."
          "CameraRig_Rail_UAID_F02F4B078FB96FE502_2112490110.RailSplineComponent")

CX, CY = 89712.0, -5226.0   # scene center (cm)
R = 3000.0                  # radius 30 m
N = 20                      # points (matches existing mesh segments)
FLAT_Z = 2300.0             # constant ring height (cm); None = ground-snap
H = 150.0                   # eye height above ground (ground-snap mode only)
TOP, BOT = 8000.0, -3000.0  # vertical trace span


def trace(x, y):
    r = execute_tool(
        "editor_toolset.toolsets.scene.SceneTools.trace_world",
        json.dumps({"start": {"x": x, "y": y, "z": TOP},
                    "end": {"x": x, "y": y, "z": BOT}}))
    return r["returnValue"]  # distance from TOP, or None


def set_props(values):
    return execute_tool(
        "editor_toolset.toolsets.object.ObjectTools.set_properties",
        json.dumps({"instance": {"refPath": SPLINE},
                    "values": json.dumps(values)}))["returnValue"]


def run():
    pts, ground_zs, hits = [], [], 0
    last_gz = 1750.0
    for i in range(N):
        a = 2.0 * math.pi * i / N
        x = CX + R * math.cos(a)
        y = CY + R * math.sin(a)
        d = trace(x, y)
        if d is None:
            gz = last_gz
        else:
            gz = TOP - d
            last_gz = gz
            hits += 1
        ground_zs.append(gz)
        z = FLAT_Z if FLAT_Z is not None else gz + H
        pts.append((x, y, z))

    points = []
    for i in range(N):
        p0, p1, p2 = pts[(i - 1) % N], pts[i], pts[(i + 1) % N]
        tan = {"x": (p2[0] - p0[0]) * 0.5,
               "y": (p2[1] - p0[1]) * 0.5,
               "z": (p2[2] - p0[2]) * 0.5}
        points.append({
            "inVal": i,
            "outVal": {"x": p1[0], "y": p1[1], "z": p1[2]},
            "arriveTangent": tan,
            "leaveTangent": tan,
            "interpMode": "CIM_CurveAuto",
        })

    ok = set_props({"SplineCurves": {"position": {"points": points}}})
    return {"ok": ok, "n": N, "radius_cm": R, "center": [CX, CY],
            "flat_z": FLAT_Z, "ground_hits": hits,
            "ground_min": min(ground_zs), "ground_max": max(ground_zs),
            "clearance_min": FLAT_Z - max(ground_zs) if FLAT_Z else None,
            "clearance_max": FLAT_Z - min(ground_zs) if FLAT_Z else None}
