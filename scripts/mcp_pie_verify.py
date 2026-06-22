"""Simulate, sample the BP_PathFly camera (position + yaw), then report:
  (a) path: fitted circle center/radius/residual  (is it the intended ring?)
  (b) facing: does reported yaw match the path tangent at each sample?

Usage: python3 scripts/mcp_pie_verify.py [n_samples]
"""
import json
import math
import sys

import mcp_call

APP = "EditorToolset.EditorAppToolset"
SCN = "editor_toolset.toolsets.scene.SceneTools"
ACT = "editor_toolset.toolsets.actor.ActorTools"
BP_CLASS = "/Game/CapturePath/BP_PathFly.BP_PathFly_C"
CX, CY, R = 89712.0, -5226.0, 3000.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def call(toolset, tool, args, sid, timeout=40):
    raw = mcp_call._curl({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                          "params": {"name": "call_tool", "arguments": {
                              "toolset_name": toolset, "tool_name": tool,
                              "arguments": args}}}, sid, timeout=timeout)
    return json.loads(mcp_call._parse(raw)["result"]["content"][0]["text"])["returnValue"]


def solve3(M, V):
    import copy
    M = copy.deepcopy(M); V = list(V)
    for i in range(3):
        p = M[i][i]
        for j in range(3):
            M[i][j] /= p
        V[i] /= p
        for k in range(3):
            if k != i:
                f = M[k][i]
                for j in range(3):
                    M[k][j] -= f * M[i][j]
                V[k] -= f * V[i]
    return V


def main():
    sid = mcp_call.session()
    call(APP, "SetCameraTransform", {"transform": {
        "location": {"x": 92712, "y": -5226, "z": 3200},
        "rotation": {"pitch": -90, "yaw": 0, "roll": 0},
        "scale": {"x": 1, "y": 1, "z": 1}}}, sid)
    call(APP, "StartPIE", {"options": {"bSimulate": True,
         "playMode": "PlayMode_Simulate", "warmupSeconds": 1.5}}, sid, timeout=90)
    refs = call(SCN, "find_actors", {"name": "", "tag": "", "collision_channels": [],
                "actor_type": {"refPath": BP_CLASS}}, sid)
    actor = [x["refPath"] for x in refs if "UEDPIE" in x["refPath"]][0]

    pts = []
    for _ in range(N):
        t = call(ACT, "get_actor_transform", {"actor": {"refPath": actor}}, sid)
        pts.append((t["location"]["x"], t["location"]["y"], t["location"]["z"],
                    t["rotation"]["yaw"]))
    call(APP, "StopPIE", {}, sid)

    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    M = [[sum(p[0]**2 for p in pts), sum(p[0]*p[1] for p in pts), sx],
         [sum(p[0]*p[1] for p in pts), sum(p[1]**2 for p in pts), sy],
         [sx, sy, n]]
    V = [sum(p[0]*(p[0]**2+p[1]**2) for p in pts),
         sum(p[1]*(p[0]**2+p[1]**2) for p in pts),
         sum(p[0]**2+p[1]**2 for p in pts)]
    A, B, C = solve3(M, V)
    fcx, fcy = A/2, B/2
    fr = math.sqrt(C + fcx*fcx + fcy*fcy)
    res = [abs(math.hypot(p[0]-fcx, p[1]-fcy)-fr) for p in pts]

    print(f"PATH: fitted center=({fcx:.0f},{fcy:.0f}) radius={fr:.0f}cm "
          f"resid_rms={math.sqrt(sum(r*r for r in res)/n):.0f}  "
          f"target=({CX:.0f},{CY:.0f}) r={R:.0f}")
    print(f"      center offset = {math.hypot(fcx-CX, fcy-CY):.0f}cm  "
          f"z range {min(p[2] for p in pts):.0f}..{max(p[2] for p in pts):.0f}")

    # FACING: compare reported yaw to circle tangent (try both orbit directions)
    print("\nFACING (reported yaw vs path tangent at each sample):")
    err_ccw = []; err_cw = []
    for (x, y, z, yaw) in pts:
        th = math.degrees(math.atan2(y - fcy, x - fcx))
        e_ccw = (yaw - (th + 90) + 180) % 360 - 180
        e_cw = (yaw - (th - 90) + 180) % 360 - 180
        err_ccw.append(abs(e_ccw)); err_cw.append(abs(e_cw))
    use, errs = ("CCW", err_ccw) if sum(err_ccw) <= sum(err_cw) else ("CW", err_cw)
    print(f"  orbit dir={use}  mean facing err={sum(errs)/n:.1f}deg  max={max(errs):.1f}deg")
    print(f"  per-sample err: {[round(e) for e in errs]}")
    print("  VERDICT:", "FACES TRAVEL ✓" if sum(errs)/n < 15 else "NOT facing travel ✗")


main()
