"""Run a ProgrammaticToolset script file via the Unreal MCP server.

Usage: python3 scripts/mcp_run_tool.py <script.tool.py>
Reads the file, wraps it as execute_tool_script, prints the run() return dict.
"""
import json
import sys

import mcp_call  # same dir

TOOLSET = "editor_toolset.toolsets.programmatic.ProgrammaticToolset"


def main():
    script = open(sys.argv[1]).read()
    sid = mcp_call.session()
    r = mcp_call.tool_call("call_tool", {
        "toolset_name": TOOLSET,
        "tool_name": "execute_tool_script",
        "arguments": {"script": script},
    }, sid)
    try:
        txt = r["result"]["content"][0]["text"]
    except Exception:
        print(json.dumps(r, indent=2))
        return
    try:
        inner = json.loads(txt)
        # execute_tool_script returns {"returnValue": "<json string>"}
        rv = inner.get("returnValue", inner)
        print(json.dumps(json.loads(rv) if isinstance(rv, str) else rv, indent=2))
    except Exception:
        print(txt)


main()
