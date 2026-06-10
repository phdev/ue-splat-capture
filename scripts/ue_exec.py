"""Run a Python file inside the ALREADY-OPEN UnrealEditor via Python Remote Execution.

Usage:  python3 scripts/ue_exec.py <file.py> [timeout_s]

Requires in the editor (once per editor session): Project Settings > Plugins > Python >
"Enable Remote Execution", multicast group 239.0.0.1:6766 (defaults). The client is UE's
own remote_execution module from the engine install. GOTCHAS (cost us hours):
  - multicast needs TTL>=1 AND a bind address that actually routes; we try 127.0.0.1,
    0.0.0.0, then the host IP until a node answers.
  - MODE_EXEC_FILE executes the file's whole body (multi-line OK). MODE_EXEC_STATEMENT
    chokes on anything multi-line.
  - output comes back as a list of {type, output} log records; UE buffers ~1MB. Print
    sparingly inside the remote script.
"""
import sys
import time
import socket

ENGINE_PY = ("/Users/Shared/Epic Games/UE_5.7/Engine/Plugins/Experimental/"
             "PythonScriptPlugin/Content/Python")
sys.path.insert(0, ENGINE_PY)
import remote_execution as re  # noqa: E402


def run_file(path, timeout=120):
    code = open(path).read()
    last_err = None
    for bind in ("127.0.0.1", "0.0.0.0", socket.gethostbyname(socket.gethostname())):
        try:
            cfg = re.RemoteExecutionConfig()
            cfg.multicast_group_endpoint = ("239.0.0.1", 6766)
            cfg.multicast_bind_address = bind
            r = re.RemoteExecution(cfg)
            r.start()
            nodes = []
            deadline = time.time() + 12
            while time.time() < deadline and not nodes:
                time.sleep(0.4)
                nodes = list(r.remote_nodes)
            if not nodes:
                r.stop()
                continue
            r.open_command_connection(nodes[0]["node_id"])
            out = r.run_command(code, unattended=True,
                                exec_mode=re.MODE_EXEC_FILE,
                                raise_on_failure=False)
            r.stop()
            return out
        except Exception as e:  # try next bind
            last_err = e
    raise SystemExit(f"no UE node found (last err: {last_err})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    res = run_file(sys.argv[1], t)
    for rec in res.get("output", []):
        print(f"[{rec.get('type','?')}] {rec.get('output','').rstrip()}")
    print("SUCCESS:", res.get("success"))
