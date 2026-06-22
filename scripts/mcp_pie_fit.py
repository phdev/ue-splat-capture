"""Simulate, rapidly sample the BP_PathFly camera position, fit a circle to the
path, and report center / radius / fit residual + the spawn location.

Usage: python3 scripts/mcp_pie_fit.py [n_samples]
"""
import json
import math
import sys

import mcp_call

APP = "EditorToolset.EditorAppToolset"
SCN = "editor_toolset.toolsets.scene.SceneTools"
ACT = "editor_toolset.toolsets.actor.ActorTools"
BP_CLASS = "/Game/CapturePath/BP_PathFly.BP_PathFly_C"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 14


def call(toolset, tool, args, sid, timeout=40):
    raw = mcp_call._curl({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                          "params": {"name": "call_tool", "arguments": {
                              "toolset_name": toolset, "tool_name": tool,
                              "arguments": args}}}, sid, timeout=timeout)
    r = mcp_call._parse(raw)
    return json.loads(r["result"]["content"][0]["text"])["returnValue"]


def main():
    sid = mcp_call.session()
    call(APP, "SetCameraTransform", {"transform": {
        "location": {"x": 92712, "y": -5226, "z": 3200},
        "rotation": {"pitch": -90, "yaw": 0, "roll": 0},
        "scale": {"x": 1, "y": 1, "z": 1}}}, sid)
    call(APP, "StartPIE", {"options": {"bSimulate": True,
         "playMode": "PlayMode_Simulate", "warmupSeconds": 1.5}}, sid, timeout=90)
    refs = call(SCN, "find_actors", {"name": "", "tag": "",
                "collision_channels": [], "actor_type": {"refPath": BP_CLASS}}, sid)
    actor = [x["refPath"] for x in refs if "UEDPIE" in x["refPath"]][0]

    pts = []
    for _ in range(N):
        t = call(ACT, "get_actor_transform", {"actor": {"refPath": actor}}, sid)
        pts.append((t["location"]["x"], t["location"]["y"], t["location"]["z"]))
    call(APP, "StopPIE", {}, sid)

    print("spawn(first sample):", tuple(round(v) for v in pts[0]))
    # Kasa algebraic circle fit on x,y
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0]**2 for p in pts); syy = sum(p[1]**2 for p in pts)
    sxy = sum(p[0]*p[1] for p in pts)
    sxz = sum(p[0]*(p[0]**2+p[1]**2) for p in pts)
    syz = sum(p[1]*(p[0]**2+p[1]**2) for p in pts)
    sz = sum(p[0]**2+p[1]**2 for p in pts)
    # solve [[sxx,sxy,sx],[sxy,syy,sy],[sx,sy,n]] [A,B,C] = [sxz,syz,sz]
    M = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]]
    V = [sxz, syz, sz]

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
                        M[k][j] -= f*M[i][j]
                    V[k] -= f*V[i]
        return V
    A, B, C = solve3(M, V)
    cx, cy = A/2, B/2
    R = math.sqrt(C + cx*cx + cy*cy)
    res = [abs(math.hypot(p[0]-cx, p[1]-cy)-R) for p in pts]
    zs = [p[2] for p in pts]
    print(f"fitted center=({cx:.0f},{cy:.0f})  radius={R:.0f}cm  "
          f"residual rms={math.sqrt(sum(r*r for r in res)/n):.0f} max={max(res):.0f}")
    print(f"z range: {min(zs):.0f}..{max(zs):.0f}")
    print(f"target:  center=(89712,-5226) radius=3000")


main()
