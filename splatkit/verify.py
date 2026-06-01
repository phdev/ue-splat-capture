"""`make verify` orchestrator: run T0..T3, print a summary table, compare to the
committed baseline, and exit 0 ONLY if every gate passes.

UE-dependent capture is NOT part of verify -- all tiers run against the committed
fixtures so verification reproduces without an Unreal install. UE detection is
reported (skip-with-warning) for visibility only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import reproject, results as R, tier_t0, tier_t2, tier_t3

# Per-metric regression tolerance (absolute), applied vs baseline in the
# "worse" direction implied by the comparison op.
_TOL = {
    "max_pose_mean_reproj_px": 0.25,
    "global_mean_reproj_px": 0.25,
    "aabb_frustum_coverage_frac": 0.01,
    "heldout_psnr_db": 1.5,
    "heldout_ssim": 0.03,
    "overfit_gap_db": 1.5,
    "_default": 1e-6,
}


def _regressed(op, value, base, tol):
    if op in (">=", ">"):
        return value < base - tol
    if op in ("<=", "<"):
        return value > base + tol
    return abs(value - base) > tol


def run_all(transforms, scene, iters, n_gauss, seed, device=None,
            with_recon=True):
    docs = {}

    # T0 -- pure-math convert tests (pytest), writes results/t0.json itself.
    tier_t0.main()
    docs["t0"] = json.loads((R.RESULTS_DIR / "t0.json").read_text())

    # T1 -- reprojection
    res = reproject.run(transforms, scene)
    docs["t1"] = R.write_tier("t1", res["checks"], n_poses=res["n_poses"],
                              n_observations=res["n_observations"])

    # T2 -- dataset schema + coverage
    res = tier_t2.run(transforms, scene)
    docs["t2"] = R.write_tier("t2", res["checks"], schema_issues=res["schema_issues"])

    # T3 -- reconstruction (optional; the slow one)
    if with_recon:
        res = tier_t3.run(transforms, iters, n_gauss, seed, device=device)
        docs["t3"] = R.write_tier(
            "t3", res["checks"], heldout=res["heldout"], train=res["train"],
            device=res["device"], iters=iters, n_gauss=n_gauss, seed=seed,
            train_seconds=res["train_seconds"])
    return docs


def _load_baseline():
    p = R.RESULTS_DIR / "baseline.json"
    return json.loads(p.read_text()) if p.exists() else None


def summarize(docs, baseline) -> bool:
    print("\n" + "=" * 74)
    print(f"{'TIER':<5}{'METRIC':<30}{'VALUE':>11} {'OP':^3}{'THRESH':>9}  RESULT")
    print("-" * 74)
    all_pass = True
    regressions = []
    for tier in ("t0", "t1", "t2", "t3"):
        if tier not in docs:
            continue
        doc = docs[tier]
        for c in doc["checks"]:
            mark = "PASS" if c["pass"] else "FAIL"
            all_pass &= c["pass"]
            print(f"{tier:<5}{c['metric']:<30}{c['value']:>11.4g} {c['op']:^3}"
                  f"{c['threshold']:>9.4g}  {mark}")
            if baseline and tier in baseline:
                base = {x["metric"]: x for x in baseline[tier]["checks"]}
                if c["metric"] in base:
                    b = base[c["metric"]]["value"]
                    tol = _TOL.get(c["metric"], _TOL["_default"])
                    if _regressed(c["op"], c["value"], b, tol):
                        regressions.append(
                            f"{tier}.{c['metric']}: {c['value']:.4g} vs baseline "
                            f"{b:.4g} (tol {tol})")
    print("=" * 74)
    if regressions:
        print("REGRESSIONS beyond tolerance vs baseline:")
        for r in regressions:
            print("  !", r)
    else:
        print("no regressions beyond tolerance" if baseline else
              "(no baseline.json committed yet)")
    print(f"\nOVERALL: {'PASS -- all gates green' if all_pass else 'FAIL'}\n")
    return all_pass


def main() -> int:
    from ue_capture import detect
    ap = argparse.ArgumentParser(description="Run all verification tiers")
    ap.add_argument("--transforms", default="fixtures/selftest/transforms.json")
    ap.add_argument("--scene", default="fixtures/selftest/scene.json")
    ap.add_argument("--iters", type=int, default=int(os.environ.get("VERIFY_ITERS", 1500)))
    ap.add_argument("--n-gauss", type=int, default=int(os.environ.get("VERIFY_NGAUSS", 6000)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("VERIFY_SEED", 0)))
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-recon", action="store_true",
                    help="run T0-T2 only (skip the slow recon gate)")
    args = ap.parse_args()

    ue = detect.find_unreal_cmd()
    if ue:
        print(f"[capture] UnrealEditor-Cmd: {ue}")
    else:
        print("[capture] UnrealEditor-Cmd NOT FOUND -- skip-with-warning; "
              "`make capture` falls back to the numpy stand-in. Verify runs on "
              "committed fixtures regardless.")
    print(f"[verify] running tiers on committed fixtures: {args.transforms}")

    docs = run_all(args.transforms, args.scene, args.iters, args.n_gauss,
                   args.seed, device=args.device, with_recon=not args.skip_recon)
    ok = summarize(docs, _load_baseline())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
