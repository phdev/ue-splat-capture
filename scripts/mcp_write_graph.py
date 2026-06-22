"""Write a Blueprint graph from a DSL file via the Unreal MCP server.

Usage: python3 scripts/mcp_write_graph.py <graph_refPath> <dsl_file>
Calls BlueprintTools.write_graph_dsl (which also compiles the Blueprint).
"""
import json
import sys

import mcp_call

graph = sys.argv[1]
code = open(sys.argv[2]).read()

sid = mcp_call.session()
raw = mcp_call._curl({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                      "params": {"name": "call_tool", "arguments": {
                          "toolset_name": "editor_toolset.toolsets.blueprint.BlueprintTools",
                          "tool_name": "write_graph_dsl",
                          "arguments": {"graph": {"refPath": graph},
                                        "code": code}}}}, sid, timeout=120)
r = mcp_call._parse(raw)
try:
    print(r["result"]["content"][0]["text"])
except Exception:
    print(json.dumps(r, indent=2))
