"""T2 runner (verify-dataset): schema + frustum coverage + camera-in-geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import coverage, results as R, schema


def run(transforms_path: str, scene_path: str,
        coverage_threshold: float = 0.99) -> dict:
    tp = Path(transforms_path)
    doc = json.loads(tp.read_text())
    scene = json.loads(Path(scene_path).read_text())

    sch = schema.validate(doc, tp.parent)
    n = sch["n_frames"]
    cov = coverage.frustum_coverage(doc)
    geo = coverage.cameras_inside_geometry(doc, scene)

    checks = [
        R.check("aabb_frustum_coverage_frac", cov["fraction"], coverage_threshold, ">=",
                note=f"{cov['n_samples']} samples, min {cov['min_views_per_point']} views/pt"),
        R.check("min_views_per_aabb_point", cov["min_views_per_point"], 1, ">="),
        R.check("cameras_inside_geometry", geo["n_inside"], 0, "=="),
        R.check("n_frames", n, 1, ">="),
        R.check("intrinsics_sane", 1.0 if sch["intrinsics_sane"] else 0.0, 1.0, ">="),
        R.check("images_exist_at_declared_res", sch["images_ok"], n, "==",
                note=f"{sch['images_ok']}/{n}"),
        R.check("poses_proper_rotation", sch["poses_ok"], n, "=="),
        R.check("splits_present", sch["n_splits"], 2, ">=",
                note=",".join(sch.get("splits", []))),
    ]
    return {"checks": checks, "schema_issues": sch["issues"][:10],
            "coverage": cov, "inside": geo}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="T2 dataset verifier")
    ap.add_argument("--transforms", default="fixtures/selftest/transforms.json")
    ap.add_argument("--scene", default="fixtures/selftest/scene.json")
    ap.add_argument("--coverage-threshold", type=float, default=0.99)
    args = ap.parse_args()
    res = run(args.transforms, args.scene, args.coverage_threshold)
    doc = R.write_tier("t2", res["checks"], schema_issues=res["schema_issues"])
    R.print_tier(doc)
    if res["schema_issues"]:
        print("  schema issues:", res["schema_issues"])
    raise SystemExit(0 if doc["pass"] else 1)
