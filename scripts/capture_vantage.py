"""Capture a VANTAGE 'window' splat from a fixed viewpoint + look direction (e.g. a
PlayerStart): a dense slab of cameras perpendicular to the look dir, converging on a few
forward depth foci, so the player's opening view reconstructs with parallax.

    python3 scripts/capture_vantage.py --pos 90080,-7390,1992 --fwd -0.139,0.989,-0.045 \
        --prefix ed_vantage --avg 4 --ev 10

Slab: cols x rows positions (h/v steps) around the viewpoint; each looks at foci P+F*d for
d in --foci. -> /tmp/ue_<prefix>_poses.json -> UE_POSES_FILE capture -> average -> prep ->
tar. No buried filter (front-facing frames carry NOSKY sky = valid bg). Single vantage =
limited baseline -> near content sharp, distant softer (by design).
"""
import argparse
import glob
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WRAPPER = '''import os, sys
for m in list(sys.modules):
    if m.startswith("ue_capture"): del sys.modules[m]
for k in ("UE_FULL","UE_GROUND_DENSE","UE_PROBE","UE_SHOW_ONLY","UE_HDR_COLOR","UE_FOCUS_CM",
          "UE_ORBIT_RADIUS_CM","UE_ELEVATIONS","UE_N_AZ","UE_SPIRE_ORBIT","UE_SPIRE_CENTER_CM",
          "UE_PROBE_ELEV","UE_PROBE_NAZ","UE_PATH_FILE"):
    os.environ.pop(k, None)
for k, v in {cfg!r}.items():
    os.environ[k] = v
if {repo!r} not in sys.path:
    sys.path.insert(0, {repo!r})
_f = {repo!r} + "/ue_capture/capture_editor.py"
exec(compile(open(_f).read(), _f, "exec"), {{"__file__": _f, "__name__": "__main__"}})
'''


def norm(v):
    m = math.sqrt(sum(c * c for c in v)) or 1.0
    return [c / m for c in v]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def build_poses(P, F, cols, rows, hstep, vstep, foci):
    F = norm(F)
    R = norm(cross(F, [0.0, 0.0, 1.0]))   # right
    U = norm(cross(R, F))                  # up
    poses = []
    for j in range(-(rows // 2), rows // 2 + 1):
        for i in range(-(cols // 2), cols // 2 + 1):
            loc = [P[k] + i * hstep * R[k] + j * vstep * U[k] for k in range(3)]
            for d in foci:
                tgt = [P[k] + F[k] * d for k in range(3)]
                poses.append({"location_cm": [round(c, 1) for c in loc],
                              "target_cm": [round(c, 1) for c in tgt]})
    return poses


def wait_capture(out_dir, label="vantage"):
    import time
    STALL, HARD = 900, 21600
    t0 = last = time.time(); ln = -1
    while time.time() - t0 < HARD:
        if os.path.exists(f"{out_dir}/ue_poses.json"):
            print(f"[{label}] DONE: {len(glob.glob(f'{out_dir}/images/*.png'))} frames"); return True
        n = len(glob.glob(f"{out_dir}/images/*.png")); now = time.time()
        if n != ln: ln = n; last = now
        print(f"[{label}] {n} frames (idle {int(now-last)}s)...", flush=True)
        if now - last > STALL: print(f"[{label}] STALL"); return False
        time.sleep(30)
    print(f"[{label}] HARD-CAP"); return False


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True, help="viewpoint x,y,z (world cm)")
    ap.add_argument("--fwd", required=True, help="forward dir fx,fy,fz")
    ap.add_argument("--prefix", default="ed_vantage")
    ap.add_argument("--avg", type=int, default=4)
    ap.add_argument("--ev", default="10")
    ap.add_argument("--cols", type=int, default=11)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--hstep", type=float, default=200.0)   # cm
    ap.add_argument("--vstep", type=float, default=200.0)
    ap.add_argument("--foci", default="1500,3000,6000")     # forward depths (cm) to converge on
    ap.add_argument("--init", default="200000")
    ap.add_argument("--from-capture", action="store_true")
    args = ap.parse_args()
    P = [float(x) for x in args.pos.split(",")]
    F = [float(x) for x in args.fwd.split(",")]
    foci = [float(x) for x in args.foci.split(",")]
    out_dir = f"{REPO}/out/{args.prefix}"
    pf = f"/tmp/ue_{args.prefix}_poses.json"

    if not args.from_capture:
        poses = build_poses(P, F, args.cols, args.rows, args.hstep, args.vstep, foci)
        json.dump(poses, open(pf, "w"))
        print(f"[vantage] {len(poses)} poses ({args.cols}x{args.rows} slab x {len(foci)} foci)")
        cfg = {"UE_POSES_FILE": pf, "UE_DEPTH": "1", "UE_NOSKY": "1",
               "UE_CAP_RES": "1536", "UE_TRAIN_RES": "1536", "UE_CAPTURE_EV": args.ev,
               "UE_CONVERGE_TICKS": "12", "UE_HFOV": "75", "UE_SKIP_LOAD": "1", "UE_SKIP_PCG": "1",
               "UE_SETTLE_TICKS": "120", "UE_MIN_SCENE_TOP_CM": "0", "UE_NO_QUIT": "1",
               "UE_CAPTURE_OUT": out_dir}
        if args.avg > 1:
            cfg["UE_AVG_SAMPLES"] = str(args.avg)
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        wp = f"/tmp/ue_{args.prefix}_wrap.py"
        open(wp, "w").write(WRAPPER.format(cfg=cfg, repo=REPO))
        r = subprocess.run([sys.executable, f"{REPO}/scripts/ue_exec.py", wp, "600"],
                           capture_output=True, text=True, timeout=660)
        print((r.stdout or "")[-400:])
        if not wait_capture(out_dir):
            sys.exit("vantage capture failed")
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
    ip = f"{ed}/sparse/0/images.txt"
    _t = open(ip).read(); open(ip, "w").write(_t.replace(".png", ".jpg"))
    tar = f"/tmp/{args.prefix}.tar.gz"
    subprocess.run(f"cd {REPO}/out/{args.prefix}_train && COPYFILE_DISABLE=1 tar --no-xattrs -cf - ed | gzip -1 > {tar}",
                   shell=True, check=True)
    print(f"[prep] {len(pngs)} frames; tar {tar} ({os.path.getsize(tar)/1e9:.2f} GB)")
    print("=== POD: depth-primary train, STANDARD densify (grad 0.00013 / percent_dense 0.01 /")
    print("    densify_until 20000 / 30000 iters) + --data_device cpu. A vantage is FOCUSED ->")
    print("    large-tier densify (0.00006) explodes the gaussian count + OOMs an 80GB card by")
    print("    ~iter 12K. Then gate/pull/voxel-clean/SOG; viewer OPENS at the viewpoint pose. ===")


if __name__ == "__main__":
    main()
