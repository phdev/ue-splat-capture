"""UNIVERSAL host-side pipeline: scout -> probe/displace -> capture -> merge ->
prep. Folds the hand-run island/canyon pipelines into one driver for ANY level.

    python3 scripts/any_pipeline.py [--prefix ed_any] [--region x0,y0,x1,y1]
        [--scout-only] [--skip-scout] [--probe-only] [--skip-probe]
        [--stations s1,o2] [--from-merge]

Stages (each resumable — the plan at /tmp/ue_any_plan.json carries state):
  SCOUT   remote-exec ue_capture/capture_any.py in the WARM editor (level must
          already be loaded/streamed; verify foliage counts first).
  PROBE   per station: 16-pose 512px ring; if >10% of frames are buried
          (black_frac>0.40 = camera inside rock/foliage), displace the station
          up (+10m) and out (radius x1.2) and retry (2 tries; >60% buried after
          retries -> skip the station). Kills the 35-55% buried-pose waste that
          enclosed regions otherwise produce.
  CAPTURE per station sequentially (one SceneCapture in the editor): "full" =
          UE_FULL dome+ground rig; "orbit" = UE_SPIRE_ORBIT ring. 1536px, EV10,
          NOSKY, GT depth. Completion = ue_poses.json appears.
  MERGE   union all stations, DROP frames with black_frac>0.40, prefix names.
  PREP    prep_depth_dataset.sh -> COLMAP layout + inv-depth pngs ->
          re-init sparse points by extent tier -> jpg-pack -> tar.
Then PRINTS the depth-primary pod commands for the tier (see CLAUDE.md
"UNIVERSAL PIPELINE" — DEPTH_W 1.0->0.5 always; densify knobs by extent).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAN = "/tmp/ue_any_plan.json"

# every key any wrapper may set — popped first in each wrapper (the editor's
# os.environ PERSISTS across remote execs; stale keys corrupt later captures)
POP_KEYS = ("UE_FULL", "UE_GROUND_DENSE", "UE_GRID", "UE_PROBE", "UE_SHOW_ONLY",
            "UE_HDR_COLOR", "UE_DEPTH", "UE_SPIRE_ORBIT", "UE_SPIRE_CENTER_CM",
            "UE_SPIRE_ELEV", "UE_SPIRE_NAZ", "UE_SPIRE_RADIUS_CM", "UE_FOCUS_CM",
            "UE_ORBIT_RADIUS_CM", "UE_ELEVATIONS", "UE_N_AZ", "UE_PROBE_ELEV",
            "UE_PROBE_NAZ", "UE_AVG_SAMPLES", "UE_DIAG", "UE_CAPTURE_OUT",
            "UA_REGION_CM", "UA_OUT_PREFIX", "UA_MAX_STATIONS", "UA_MAX_ORBITS")

WRAPPER = '''import os
for k in {pops!r}:
    os.environ.pop(k, None)
for k, v in {cfg!r}.items():
    os.environ[k] = v
import sys
if {repo!r} not in sys.path:
    sys.path.insert(0, {repo!r})
_f = {target!r}
exec(compile(open(_f).read(), _f, "exec"), {{"__file__": _f, "__name__": "__main__"}})
'''

BASE_CFG = {"UE_NOSKY": "1", "UE_CAPTURE_EV": "10", "UE_HFOV": "75",
            "UE_SKIP_LOAD": "1", "UE_SKIP_PCG": "1",
            "UE_MIN_SCENE_TOP_CM": "0", "UE_NO_QUIT": "1"}


def ue_exec(wrapper_path, timeout=600):
    r = subprocess.run([sys.executable, f"{REPO}/scripts/ue_exec.py",
                        wrapper_path, str(timeout)],
                       capture_output=True, text=True, timeout=timeout + 60)
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip())
    return out


def send_capture(name, cfg, target="ue_capture/capture_editor.py"):
    path = f"/tmp/ue_any_{name}.py"
    with open(path, "w") as f:
        f.write(WRAPPER.format(pops=POP_KEYS, cfg=cfg, repo=REPO,
                               target=os.path.join(REPO, target)))
    return ue_exec(path)


def wait_capture(out_dir, est_poses, label):
    """Capture completion = ue_poses.json written (capture_editor's last act)."""
    deadline = time.time() + max(1200, est_poses * 25 + 600)
    while time.time() < deadline:
        if os.path.exists(f"{out_dir}/ue_poses.json"):
            n = len(glob.glob(f"{out_dir}/images/*.png"))
            print(f"[{label}] DONE: {n} frames")
            return True
        n = len(glob.glob(f"{out_dir}/images/*.png"))
        print(f"[{label}] {n}/{est_poses} frames...", flush=True)
        time.sleep(30)
    print(f"[{label}] TIMEOUT waiting for {out_dir}/ue_poses.json")
    return False


def black_frac(p):
    im = np.asarray(Image.open(p).convert("L").resize((128, 128)), np.float32) / 255.0
    return float((im < 0.04).mean())


def buried_frac(out_dir):
    imgs = sorted(glob.glob(f"{out_dir}/images/*.png"))
    if not imgs:
        return 1.0
    return float(np.mean([black_frac(p) > 0.40 for p in imgs]))


def station_cfg(st, plan, probe=False):
    fx, fy, fz = st["focus"]
    cfg = dict(BASE_CFG)
    if probe:
        pr = plan["probe"]
        cfg.update({"UE_PROBE": "1", "UE_PROBE_ELEV": pr["elev"],
                    "UE_PROBE_NAZ": str(pr["naz"]),
                    "UE_FOCUS_CM": f"{fx},{fy},{fz}",
                    "UE_ORBIT_RADIUS_CM": str(st["radius"]),
                    "UE_CAP_RES": str(pr["res"]), "UE_TRAIN_RES": str(pr["res"]),
                    "UE_CONVERGE_TICKS": "8", "UE_SETTLE_TICKS": "60",
                    "UE_CAPTURE_OUT": f"{REPO}/out/{plan['prefix']}_probe_{st['name']}"})
        return cfg
    cfg.update({"UE_DEPTH": "1", "UE_CAP_RES": "1536", "UE_TRAIN_RES": "1536",
                "UE_SETTLE_TICKS": str(st["settle"]),
                "UE_CONVERGE_TICKS": str(st["converge"]),
                "UE_CAPTURE_OUT": f"{REPO}/out/{plan['prefix']}_{st['name']}"})
    if st["kind"] == "full":
        cfg.update({"UE_FULL": "1", "UE_GROUND_DENSE": "1",
                    "UE_FOCUS_CM": f"{fx},{fy},{fz}",
                    "UE_ORBIT_RADIUS_CM": str(st["radius"])})
    else:
        cfg.update({"UE_SPIRE_ORBIT": "1",
                    "UE_SPIRE_CENTER_CM": f"{fx},{fy},{fz}",
                    "UE_SPIRE_ELEV": st["elev"], "UE_SPIRE_NAZ": str(st["naz"]),
                    "UE_SPIRE_RADIUS_CM": str(st["radius"])})
    return cfg


def probe_station(st, plan):
    pr = plan["probe"]
    n_probe = pr["naz"] * len(pr["elev"].split(","))
    for attempt in range(pr["retries"] + 1):
        out_dir = f"{REPO}/out/{plan['prefix']}_probe_{st['name']}"
        shutil.rmtree(out_dir, ignore_errors=True)
        send_capture(f"probe_{st['name']}", station_cfg(st, plan, probe=True))
        if not wait_capture(out_dir, n_probe, f"probe {st['name']} try{attempt}"):
            st["probe"] = "timeout"
            return st
        bf = buried_frac(out_dir)
        print(f"[probe {st['name']}] buried {bf:.2f} "
              f"(focus_z {st['focus'][2]/100:.0f}m R {st['radius']/100:.0f}m)")
        if bf <= pr["max_buried"]:
            st["probe"] = f"ok buried={bf:.2f} tries={attempt}"
            return st
        if attempt < pr["retries"]:  # displace UP and OUT, retry
            st["focus"][2] += pr["displace_dz_cm"]
            st["radius"] *= pr["displace_rmul"]
            print(f"[probe {st['name']}] displacing -> z {st['focus'][2]/100:.0f}m "
                  f"R {st['radius']/100:.0f}m")
    if bf > pr["skip_above"]:
        st["probe"] = f"SKIP buried={bf:.2f}"
        print(f"[probe {st['name']}] SKIPPED (buried {bf:.2f} after displacement)")
    else:  # capture anyway; the merge-stage 0.40 filter drops the buried frames
        st["probe"] = f"warn buried={bf:.2f}"
    return st


def merge(plan):
    dst = f"{REPO}/out/{plan['prefix']}_on"
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(f"{dst}/images")
    os.makedirs(f"{dst}/depth")
    out_meta, frames = None, []
    for st in plan["stations"]:
        if str(st.get("probe", "")).startswith("SKIP"):
            continue
        src = f"{REPO}/out/{plan['prefix']}_{st['name']}"
        if not os.path.exists(f"{src}/ue_poses.json"):
            print(f"[merge] {st['name']}: no capture, skipping")
            continue
        j = json.load(open(f"{src}/ue_poses.json"))
        out_meta = out_meta or dict(j)
        kept = dropped = 0
        for fr in j["frames"]:
            if black_frac(fr["file_path"]) > 0.40:
                dropped += 1
                continue
            stem = st["name"] + "_" + os.path.splitext(os.path.basename(fr["file_path"]))[0]
            shutil.copy2(fr["file_path"], f"{dst}/images/{stem}.png")
            new = dict(fr)
            new["file_path"] = f"{dst}/images/{stem}.png"
            if fr.get("depth_path") and os.path.exists(fr["depth_path"]):
                shutil.copy2(fr["depth_path"], f"{dst}/depth/{stem}.exr")
                new["depth_path"] = f"{dst}/depth/{stem}.exr"
            frames.append(new)
            kept += 1
        print(f"[merge] {st['name']}: kept {kept}, dropped {dropped} buried")
    out_meta["frames"] = frames
    json.dump(out_meta, open(f"{dst}/ue_poses.json", "w"))
    print(f"[merge] TOTAL {len(frames)} frames -> {dst}")
    return dst


def prep(plan, cap_dir):
    ed = f"{REPO}/out/{plan['prefix']}_train/ed"
    shutil.rmtree(f"{REPO}/out/ed_depth_ds", ignore_errors=True)
    shutil.rmtree(f"{REPO}/out/{plan['prefix']}_train", ignore_errors=True)
    subprocess.run(["zsh", f"{REPO}/scripts/prep_depth_dataset.sh"],
                   env={**os.environ, "CAP": cap_dir, "ED": ed},
                   cwd=REPO, check=True)
    subprocess.run([sys.executable, f"{REPO}/scripts/transforms_to_colmap.py",
                    f"{REPO}/out/ed_depth_ds/transforms.json", f"{ed}/sparse/0",
                    str(plan["init_points"])], cwd=REPO, check=True)
    pngs = glob.glob(f"{ed}/images/*.png")
    for p in pngs:
        Image.open(p).convert("RGB").save(p[:-4] + ".jpg", "JPEG", quality=92)
        os.remove(p)
    ip = f"{ed}/sparse/0/images.txt"
    txt = open(ip).read()
    open(ip, "w").write(txt.replace(".png", ".jpg"))
    print(f"[prep] {len(pngs)} pngs -> jpgs")
    tar = f"/tmp/{plan['prefix']}.tar.gz"
    subprocess.run(f"cd {REPO}/out/{plan['prefix']}_train && "
                   f"COPYFILE_DISABLE=1 tar --no-xattrs -cf - ed | gzip -1 > {tar}",
                   shell=True, check=True)
    print(f"[prep] dataset tar: {tar} ({os.path.getsize(tar)/1e9:.2f} GB)")
    return tar


def pod_instructions(plan, tar):
    large = plan["train_tier"] == "large"
    print(f"""
=== POD TRAINING (tier={plan['train_tier']}, {plan['extent_m'][0]:.0f}x{plan['extent_m'][1]:.0f}m) ===
1. A100-80GB pod (scripts/runpod_pod.py create/wait), upload {tar} to /root (NEVER /workspace).
2. clone Inria gsv + pip (numpy<2 AFTER opencv) + scripts/pod_patch_alpha.py (alpha-seal masks;
   verify "[alpha-patch] loaded N bg masks" at iter 0).
3. python3 -u gsv/train.py -s /root/ed -m /root/ed/out -r 1 --data_device cuda \\
     --random_background -d depths \\
     --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.5 \\
     --iterations {40000 if large else 30000} --densify_until_iter {26000 if large else 20000} \\
     --densify_grad_threshold {0.00006 if large else 0.00013} \\
     --percent_dense {0.003 if large else 0.01} \\
     --port 6021 --test_iterations -1 --save_iterations 15000 {40000 if large else 30000}
   (DEPTH-PRIMARY is NON-NEGOTIABLE for enclosed scenes; harmless+helpful on open ones.)
4. Mid-train: check iteration_15000 ply size (severe under/over-densification shows here).
   pkill ONLY with [b]racketed patterns; network_gui port lingers -> --port NNNN on relaunch.
5. Gate BEFORE pod delete: python3 scripts/pod_gate_depth.py (pod-side render at training
   poses + invdepth MAE; island-quality ~0.009, draft ~0.04) AND stat the final ply size.
6. gzip+split chunked parallel scp down; ALWAYS delete the pod.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="ed_any")
    ap.add_argument("--region", default="", help="x0,y0,x1,y1 world-cm scope")
    ap.add_argument("--stations", default="", help="comma subset of station names")
    ap.add_argument("--scout-only", action="store_true")
    ap.add_argument("--skip-scout", action="store_true")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--from-merge", action="store_true",
                    help="captures already on disk; merge+prep only")
    args = ap.parse_args()

    if not args.skip_scout and not args.from_merge:
        if os.path.exists(PLAN):
            os.remove(PLAN)
        cfg = {"UA_OUT_PREFIX": args.prefix}
        if args.region:
            cfg["UA_REGION_CM"] = args.region
        send_capture("scout", cfg, target="ue_capture/capture_any.py")
        for _ in range(24):
            if os.path.exists(PLAN):
                break
            time.sleep(5)
        if not os.path.exists(PLAN):
            sys.exit("scout produced no plan (editor down / level not loaded?)")
    plan = json.load(open(PLAN))
    print(f"[plan] {json.dumps({k: v for k, v in plan.items() if k != 'stations'})}")
    for st in plan["stations"]:
        print(f"[plan]   {st}")
    if args.scout_only:
        return
    wanted = set(args.stations.split(",")) if args.stations else None
    stations = [s for s in plan["stations"] if not wanted or s["name"] in wanted]

    if not args.from_merge:
        if not args.skip_probe:
            for st in stations:
                probe_station(st, plan)
                json.dump(plan, open(PLAN, "w"), indent=1)
        if args.probe_only:
            return
        for st in stations:
            if str(st.get("probe", "")).startswith("SKIP"):
                continue
            out_dir = f"{REPO}/out/{plan['prefix']}_{st['name']}"
            if os.path.exists(f"{out_dir}/ue_poses.json"):
                print(f"[capture {st['name']}] already done, skipping")
                continue
            shutil.rmtree(out_dir, ignore_errors=True)
            send_capture(st["name"], station_cfg(st, plan))
            wait_capture(out_dir, st["est_poses"], f"capture {st['name']}")

    cap_dir = merge(plan)
    tar = prep(plan, cap_dir)
    pod_instructions(plan, tar)


if __name__ == "__main__":
    main()
