"""Minimal Streamable-HTTP client for the Unreal MCP plugin (UE 5.8), driven via `curl -N`.

WHY curl -N (hard-won): the Unreal MCP server streams tool-CALL results on the POST response
as Server-Sent Events AND rejects the GET SSE channel (HTTP 405). Python urllib's
`resp.read()` returns 0 bytes for those calls (premature EOF on the held-open stream), so a
pure-urllib client silently gets empty tool results. `curl -N` reads the held-open SSE stream
correctly. Protocol methods (initialize, tools/list) DO return inline JSON; only tool calls
stream. So every request here shells to `curl -N` and parses the `data:` line.

Usage:
  python3 scripts/mcp_call.py list                          # list_toolsets
  python3 scripts/mcp_call.py describe <Toolset.Name>       # describe_toolset
  python3 scripts/mcp_call.py meta <metaTool> '<json-args>' # call any tool-search meta-tool
  python3 scripts/mcp_call.py raw  <method>   '<json>'      # raw JSON-RPC method
Env: MCP_URL (default http://127.0.0.1:8000/mcp).
"""
import json
import os
import subprocess
import sys

URL = os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp")
HDR = "/tmp/.mcp_hdr.txt"


def _curl(payload, sid=None, hdr=None, timeout=40):
    cmd = ["curl", "-sS", "-N", "-m", str(timeout)]
    if hdr:
        cmd += ["-D", hdr]
    cmd += ["-X", "POST", URL, "-H", "Content-Type: application/json",
            "-H", "Accept: application/json, text/event-stream"]
    if sid:
        cmd += ["-H", "Mcp-Session-Id: " + sid]
    cmd += ["-d", json.dumps(payload)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5).stdout


def _parse(out):
    for line in out.splitlines():
        if line.startswith("data:") and line[5:].strip():
            try:
                return json.loads(line[5:].strip())
            except Exception:
                pass
    try:
        return json.loads(out)
    except Exception:
        return {"_raw": out[:400]}


def session():
    _curl({"jsonrpc": "2.0", "id": 1, "method": "initialize",
           "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                      "clientInfo": {"name": "mcp_call", "version": "0"}}}, hdr=HDR)
    sid = None
    for line in open(HDR):
        if line.lower().startswith("mcp-session-id:"):
            sid = line.split(":", 1)[1].strip()
    _curl({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    return sid


def tool_call(name, args, sid, rid=9):
    return _parse(_curl({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                         "params": {"name": name, "arguments": args}}, sid))


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        return
    sid = session()
    if a[0] == "list":
        r = tool_call("list_toolsets", {}, sid)
    elif a[0] == "describe":
        r = tool_call("describe_toolset", {"toolset_name": a[1]}, sid)
    elif a[0] == "call":   # call <toolset_name> <tool_name> '<json-args>'
        r = tool_call("call_tool", {"toolset_name": a[1], "tool_name": a[2],
                                    "arguments": json.loads(a[3]) if len(a) > 3 else {}}, sid)
    elif a[0] == "meta":
        r = tool_call(a[1], json.loads(a[2]) if len(a) > 2 else {}, sid)
    elif a[0] == "raw":
        r = _parse(_curl({"jsonrpc": "2.0", "id": 9, "method": a[1],
                          "params": json.loads(a[2]) if len(a) > 2 else {}}, sid))
    else:
        print("unknown cmd; see --help")
        return
    print(json.dumps(r, indent=2))


main()
