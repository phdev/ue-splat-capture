"""Capture a FLYTHROUGH along a road/ditch polyline (the path-fan rig) -> depth dataset.

    python3 scripts/capture_path.py --path /tmp/ed_path_road.json --prefix ed_path \
        --avg 8 --step 400 --eye 480

One UE_PATH capture in the WARM editor (forward+side+floor fan per step, eye_cm above the
local ground), temporal-averaged, then prepped into a depth-supervised COLMAP dataset and
tarred. Prints the depth-primary pod-training commands. NO buried-frame filter — open-route
frames frame the (black NOSKY) sky above the ground, which is valid background, not burial.
The editor caches ue_capture.rig, so the wrapper clears sys.modules['ue_capture*'] to pick
up path_fan.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WRAPPER = '''import os, sys
for m in list(sys.modules):
    if m.startswith("ue_capture"): del sys.modules[m]   # fresh rig (path_fan)
for k in ("UE_FULL","UE_GROUND_DENSE","UE_PROBE","UE_SHOW_ONLY","UE_HDR_COLOR","UE_FOCUS_CM",
          "UE_ORBIT_RADIUS_CM","UE_ELEVATIONS","UE_N_AZ","UE_SPIRE_ORBIT","UE_SPIRE_CENTER_CM",
          "UE_PROBE_ELEV","UE_PROBE_NAZ"):
    os.environ.pop(k, None)
for k, v in {cfg!r}.items():
    os.environ[k] = v
if {repo!r} not in sys.path:
    sys.path.insert(0, {repo!r})
_f = {repo!r} + "/ue_capture/capture_editor.py"
exec(compile(open(_f).read(), _f, "exec"), {{"__file__": _f, "__name__": "__main__"}})
'''


def wait_capture(out_dir, label="path"):
    STALL_S, HARD = 900, 28800   # avg=8 x hundreds of fan poses is slow (~5-7h)
    t0 = last_change = time.time(); last_n = -1
    while time.time() - t0 < HARD:
        if os.path.exists(f"{out_dir}/ue_poses.json"):
            print(f"[{label}] DONE: {len(glob.glob(f'{out_dir}/images/*.png'))} frames"); return True
        n = len(glob.glob(f"{out_dir}/images/*.png")); now = time.time()
        if n != last_n: last_n = n; last_change = now
        idle = now - last_change
        print(f"[{label}] {n} sample frames (idle {int(idle)}s)...", flush=True)
        if idle > STALL_S:
            print(f"[{label}] STALL — giving up"); return False
        time.sleep(30)
    print(f"[{label}] HARD-CAP"); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/tmp/ed_path_road.json")
    ap.add_argument("--prefix", default="ed_path")
    ap.add_argument("--avg", type=int, default=8)
    ap.add_argument("--step", default="400")
    ap.add_argument("--eye", default="480")
    ap.add_argument("--init", default="300000")
    ap.add_argument("--from-capture", action="store_true", help="capture on disk; average+prep only")
    args = ap.parse_args()
    out_dir = f"{REPO}/out/{args.prefix}"

    if not args.from_capture:
        cfg = {"UE_PATH_FILE": args.path, "UE_PATH_STEP": args.step, "UE_PATH_EYE": args.eye,
               "UE_DEPTH": "1", "UE_NOSKY": "1", "UE_CAP_RES": "1536", "UE_TRAIN_RES": "1536",
               "UE_CAPTURE_EV": "10", "UE_CONVERGE_TICKS": "12", "UE_HFOV": "75",
               "UE_SKIP_LOAD": "1", "UE_SKIP_PCG": "1", "UE_SETTLE_TICKS": "120",
               "UE_MIN_SCENE_TOP_CM": "0", "UE_NO_QUIT": "1", "UE_CAPTURE_OUT": out_dir}
        if args.avg > 1:
            cfg["UE_AVG_SAMPLES"] = str(args.avg)
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        wp = f"/tmp/ue_{args.prefix}_wrap.py"
        open(wp, "w").write(WRAPPER.format(cfg=cfg, repo=REPO))
        r = subprocess.run([sys.executable, f"{REPO}/scripts/ue_exec.py", wp, "600"],
                           capture_output=True, text=True, timeout=660)
        print((r.stdout or "")[-500:])
        if not wait_capture(out_dir):
            sys.exit("path capture failed")
        if args.avg > 1:
            r = subprocess.run([sys.executable, f"{REPO}/scripts/average_samples.py", f"{out_dir}/images"],
                               capture_output=True, text=True)
            print((r.stdout or "").strip().splitlines()[-1] if r.stdout else "[avg] done")

    ed = f"{REPO}/out/{args.prefix}_train/ed"
    import shutil
    shutil.rmtree(f"{REPO}/out/ed_depth_ds", ignore_errors=True)
    shutil.rmtree(f"{REPO}/out/{args.prefix}_train", ignore_errors=True)
    subprocess.run(["zsh", f"{REPO}/scripts/prep_depth_dataset.sh"],
                   env={**os.environ, "CAP": out_dir, "ED": ed}, cwd=REPO, check=True)
    subprocess.run([sys.executable, f"{REPO}/scripts/transforms_to_colmap.py",
                    f"{REPO}/out/ed_depth_ds/transforms.json", f"{ed}/sparse/0", args.init], cwd=REPO, check=True)
    from PIL import Image
    pngs = glob.glob(f"{ed}/images/*.png")
    for p in pngs:
        Image.open(p).convert("RGB").save(p[:-4] + ".jpg", "JPEG", quality=92); os.remove(p)
    ip = f"{ed}/sparse/0/images.txt"; open(ip, "w").write(open(ip).read().replace(".png", ".jpg"))
    tar = f"/tmp/{args.prefix}.tar.gz"
    subprocess.run(f"cd {REPO}/out/{args.prefix}_train && COPYFILE_DISABLE=1 tar --no-xattrs -cf - ed | gzip -1 > {tar}",
                   shell=True, check=True)
    print(f"[prep] {len(pngs)} frames -> jpgs; dataset tar {tar} ({os.path.getsize(tar)/1e9:.2f} GB)")
    print(f"""
=== POD: depth-primary flythrough train ===
upload {tar} to /root; clone gsv + pip(numpy<2 after opencv) + pod_patch_alpha.py;
python3 -u gsv/train.py -s /root/ed -m /root/ed/out -r 1 --data_device cuda --random_background -d depths \\
  --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.5 --iterations 40000 --densify_until_iter 26000 \\
  --densify_grad_threshold 0.00006 --percent_dense 0.003 --port 6021 --test_iterations -1 --save_iterations 15000 40000
then pod_gate_depth.py; pull; clean; SOG; deploy with the viewer auto-flying {args.path}.""")


if __name__ == "__main__":
    main()
