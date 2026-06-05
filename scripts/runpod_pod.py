"""Minimal Runpod pod lifecycle for ue-splat CUDA training (3DGS / 2DGS / MCMC).
No deps (urllib). API key read from ~/.config/ue-splat-capture/runpod_api_key
(gitignored, NEVER commit). Proven REST body from the quake pod-create.

  runpod_pod.py create [--name N] [--disk 60] [--vol 80] [--gpu "A5000,A40,..."]
  runpod_pod.py status <id>        # prints JSON: desiredStatus, publicIp, ssh port
  runpod_pod.py ssh <id>           # prints "root@<ip> -p <port>" once reachable
  runpod_pod.py wait <id>          # block until SSH endpoint is mapped, print it
  runpod_pod.py delete <id>        # DELETE the pod (always run when done!)
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://rest.runpod.io/v1"
KEY = open(os.path.expanduser("~/.config/ue-splat-capture/runpod_api_key")).read().strip()
# 48GB cards FIRST for headroom (cap_max 3M @1536px); fall back to 24GB, then 80GB.
GPUS = ["NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA L40S",
        "NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX A5000",
        "NVIDIA GeForce RTX 4090", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
# 3DGS-MCMC's diff-gaussian-rasterization fork is 3DGS-era (torch 2.0 / CUDA 11.8);
# the cuda11.8 devel image builds it cleanly (cuda12.4 hits deprecated-API errors).
IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"


def _req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(API + path, data=data, method=method,
                               headers={"Authorization": "Bearer " + KEY,
                                        "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r) as resp:
            t = resp.read().decode()
            return resp.status, (json.loads(t) if t.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()}


def _pubkey():
    p = os.path.expanduser("~/.ssh/id_ed25519.pub")
    return open(p).read().strip()


def create(args):
    name = _opt(args, "--name", "ue-splat-mcmc")
    disk = int(_opt(args, "--disk", "70"))
    vol = int(_opt(args, "--vol", "90"))
    gpus = _opt(args, "--gpu", "")
    gpu_ids = [g.strip() for g in gpus.split(",")] if gpus else GPUS
    pk = _pubkey()
    body = {"cloudType": os.environ.get("RUNPOD_CLOUD_TYPE", "SECURE"),
            "computeType": "GPU", "name": name, "imageName": IMAGE,
            "gpuCount": 1, "gpuTypeIds": gpu_ids, "gpuTypePriority": "availability",
            "containerDiskInGb": disk, "volumeInGb": vol, "volumeMountPath": "/workspace",
            "ports": ["22/tcp", "8888/http"], "supportPublicIp": True,
            "env": {"PUBLIC_KEY": pk, "SSH_PUBLIC_KEY": pk}}
    st, p = _req("POST", "/pods", body)
    if st >= 300:
        print(f"CREATE_FAIL {st}: {p.get('error')}"); sys.exit(1)
    print(p["id"])


def status(args):
    st, p = _req("GET", "/pods/" + args[0])
    ip = p.get("publicIp") or (p.get("machine") or {}).get("publicIp")
    pm = p.get("portMappings") or {}
    print(json.dumps({"id": p.get("id"), "desiredStatus": p.get("desiredStatus"),
                      "publicIp": ip, "ssh22": pm.get("22") or pm.get(22),
                      "gpu": (p.get("machine") or {}).get("gpuTypeId") or p.get("gpuTypeIds")}, indent=1))


def _endpoint(pid):
    st, p = _req("GET", "/pods/" + pid)
    ip = p.get("publicIp") or (p.get("machine") or {}).get("publicIp")
    pm = p.get("portMappings") or {}
    port = pm.get("22") or pm.get(22)
    return (ip, port) if ip and port else (None, None)


def wait(args):
    pid = args[0]; deadline = time.time() + 20 * 60
    while time.time() < deadline:
        ip, port = _endpoint(pid)
        if ip and port:
            print(f"root@{ip} -p {port}"); return
        time.sleep(15)
    print("WAIT_TIMEOUT"); sys.exit(1)


def ssh(args):
    ip, port = _endpoint(args[0])
    print(f"root@{ip} -p {port}" if ip else "NOT_READY")


def delete(args):
    st, p = _req("DELETE", "/pods/" + args[0])
    print(f"DELETED {args[0]} (status {st})")


def _opt(args, k, d):
    return args[args.index(k) + 1] if k in args else d


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"create": create, "status": status, "ssh": ssh, "wait": wait,
     "delete": delete}.get(cmd, lambda a: print(__doc__))(sys.argv[2:])
