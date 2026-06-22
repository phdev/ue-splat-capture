"""Simulate-In-Editor and sample the BP_PathFly camera transform over time to
verify the delta-position facing graph: does the actor's yaw track its travel
direction?

Usage: python3 scripts/mcp_pie_sample.py [n_samples] [interval_s]
"""
import json
import math
import sys
import time

import mcp_call

APP = "EditorToolset.EditorAppToolset"
SCN = "editor_toolset.toolsets.scene.SceneTools"
ACT = "editor_toolset.toolsets.actor.ActorTools"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 7
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7


def call(toolset, tool, args, sid, timeout=40):
    raw = mcp_call._curl({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                          "params": {"name": "call_tool", "arguments": {
                              "toolset_name": toolset, "tool_name": tool,
                              "arguments": args}}}, sid, timeout=timeout)
    r = mcp_call._parse(raw)
    txt = r["result"]["content"][0]["text"]
    try:
        return json.loads(txt)["returnValue"]
    except Exception:
        return {"_text": txt}


BP_CLASS = "/Game/CapturePath/BP_PathFly.BP_PathFly_C"


def find_by_class(sid, cls):
    a = call(SCN, "find_actors",
             {"name": "", "tag": "", "collision_channels": [],
              "actor_type": {"refPath": cls}}, sid)
    return [x["refPath"] for x in a] if isinstance(a, list) else []


def main():
    sid = mcp_call.session()
    # SIE uses the editor viewport camera as a WP streaming source — park it
    # centrally so the ring cell loads in the sim world.
    call(APP, "SetCameraTransform", {"transform": {
        "location": {"x": 89712, "y": -5226, "z": 4500},
        "rotation": {"pitch": -90, "yaw": 0, "roll": 0},
        "scale": {"x": 1, "y": 1, "z": 1}}}, sid)
    print("StartPIE (Simulate)...")
    call(APP, "StartPIE", {"options": {"bSimulate": True,
                                       "playMode": "PlayMode_Simulate",
                                       "warmupSeconds": 1.5}}, sid, timeout=90)
    refs = find_by_class(sid, BP_CLASS)
    pie = [r for r in refs if "UEDPIE" in r] or refs
    print("actors (by class):", refs)
    if not pie:
        call(APP, "StopPIE", {}, sid)
        print("NO ACTOR FOUND"); return
    actor = pie[0]

    samples = []
    for k in range(N):
        t = call(ACT, "get_actor_transform", {"actor": {"refPath": actor}}, sid)
        loc, rot = t["location"], t["rotation"]
        samples.append((loc["x"], loc["y"], loc["z"], rot["yaw"]))
        time.sleep(DT)

    call(APP, "StopPIE", {}, sid)

    print("\n  sample (x, y, z, reported_yaw) | move_yaw=atan2(dy,dx) | err")
    moved = 0.0
    for i in range(len(samples) - 1):
        x0, y0, z0, yaw0 = samples[i]
        x1, y1, _, _ = samples[i + 1]
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        moved += dist
        if dist < 1.0:
            print(f"  {i}: ({x0:.0f},{y0:.0f},{z0:.0f}) yaw={yaw0:6.1f} | stationary ({dist:.2f}cm)")
            continue
        myaw = math.degrees(math.atan2(dy, dx))
        err = (yaw0 - myaw + 180) % 360 - 180
        print(f"  {i}: ({x0:.0f},{y0:.0f},{z0:.0f}) yaw={yaw0:6.1f} | move={myaw:6.1f} | err={err:+.1f}")
    print(f"\n  total horiz movement: {moved:.0f} cm over {N} samples")


main()
