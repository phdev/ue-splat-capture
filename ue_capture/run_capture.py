"""Entrypoint executed INSIDE UnrealEditor-Cmd:

    UnrealEditor-Cmd <project> -run=pythonscript \
        -script=".../ue_capture/run_capture.py" -- --out out/ue_capture

Spawns the self-test scene, generates the rig, renders each pose (colour+depth),
and writes a neutral `ue_poses.json`. Convert it to a verifiable dataset from the
uv venv with:  `python -m splatkit.ingest --ue-poses <out>/ue_poses.json --out fixtures/selftest`
"""
from __future__ import annotations

import argparse
import os
import sys

# allow `import ue_capture.*` when run as a loose script inside UE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ue_capture import export, render, rig, selftest_scene as S  # noqa: E402


def main(out_dir: str, want_depth: bool = True):
    import unreal  # only available inside UnrealEditor-Cmd

    unreal.log("ue-splat-capture: spawning self-test scene")
    S.spawn_scene(unreal)

    poses = rig.default_rig(center_cm=tuple(S.TARGET_CENTER_CM),
                            radius_cm=S.ORBIT_RADIUS_CM)
    unreal.log(f"ue-splat-capture: rendering {len(poses)} cameras")
    intr = S.INTRINSICS
    frames, _actors = render.render_cameras(
        unreal, poses, intr["w"], intr["h"], intr["hfov_deg"], out_dir, want_depth)

    os.makedirs(out_dir, exist_ok=True)
    ue_poses_path = os.path.join(out_dir, "ue_poses.json")
    export.write_ue_poses(
        ue_poses_path, intr["w"], intr["h"], intr["hfov_deg"], frames,
        scene_meta={
            "background": S.BACKGROUND,
            "aabb_min_cm": S.AABB_MIN_CM, "aabb_max_cm": S.AABB_MAX_CM,
            "fiducials": S.FIDUCIALS,
            "primitives": S.primitives_for_scene_json(),
        })
    unreal.log(f"ue-splat-capture: wrote {ue_poses_path}")
    print(f"WROTE {ue_poses_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/ue_capture")
    ap.add_argument("--no-depth", action="store_true")
    # UnrealEditor-Cmd passes script args after a literal `--`
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = ap.parse_args(argv)
    main(args.out, want_depth=not args.no_depth)
