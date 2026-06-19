"""Capture the UE level viewport via MCP and save the PNG.

Usage: python3 scripts/mcp_capture.py <out.png> <x> <y> <z> <pitch> <yaw> [gridSpacing] [gridHeight]
If gridSpacing>0, draws a ground grid annotation at gridHeight.
"""
import base64
import json
import sys

import mcp_call

out = sys.argv[1]
x, y, z, pitch, yaw = map(float, sys.argv[2:7])
grid = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
gh = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0

args = {"captureTransform": {
    "location": {"x": x, "y": y, "z": z},
    "rotation": {"pitch": pitch, "yaw": yaw, "roll": 0.0},
    "scale": {"x": 1.0, "y": 1.0, "z": 1.0}}}
args["annotations"] = {"gridSpacing": grid, "gridExtent": 20000.0,
                       "gridHeight": gh, "maxLabelDistance": 0.0,
                       "classFilter": None, "maxLabels": 0}

sid = mcp_call.session()
rv = None
for attempt in range(5):
    raw = mcp_call._curl({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                          "params": {"name": "call_tool", "arguments": {
                              "toolset_name": "EditorToolset.EditorAppToolset",
                              "tool_name": "CaptureViewport",
                              "arguments": args}}}, sid, timeout=120)
    r = mcp_call._parse(raw)
    try:
        rv = json.loads(r["result"]["content"][0]["text"])["returnValue"]
        break
    except Exception:
        print(f"  attempt {attempt+1} empty/err ({len(raw)} bytes), retrying")
if rv is None:
    raise SystemExit("capture failed after retries")
data = rv["image"]["data"]
open(out, "wb").write(base64.b64decode(data))
print("saved", out, len(data), "b64 bytes  cam=",
      rv["cameraLocation"], "fov", rv.get("cameraFOV"))
