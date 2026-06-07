"""WARM-EDITOR depth capture -- paste this into an ALREADY-OPEN UnrealEditor's Python
console (Window > Output Log > Cmd dropdown set to "Python", or the Python console):

    py "/Users/peterhowell/ue-splat-capture/ue_capture/capture_editor_warm.py"

Use this when the automated (fresh-launch) capture drops the hero spire: a fresh
bridge-launched editor never streams the Rock_*_BP mesh in, but YOUR open editor already
has it loaded. This sets the full depth-capture config (UE_FULL + UE_DEPTH + dense ground,
no-sky, 1536px) and runs the SAME tick-driven capture in your editor, WITHOUT closing it
(UE_NO_QUIT=1). The completeness GATE (UE_MIN_SCENE_TOP_CM) verifies the tall spire is
actually loaded before it captures, so it can't silently shoot a spire-less scene again.

Before running: make sure the spire is visible in your viewport (navigate to it so it's
streamed). Output -> out/ed_editor_depth2/{images,depth}/ + ue_poses.json. ~560 poses,
~45-60 min; the editor will tick through it and stay open when done.
"""
import os

REPO = "/Users/peterhowell/ue-splat-capture"
_CFG = {
    "UE_FULL": "1", "UE_DEPTH": "1", "UE_GROUND_DENSE": "1", "UE_NOSKY": "1",
    "UE_CAP_RES": "1536", "UE_TRAIN_RES": "1536", "UE_CAPTURE_EV": "10",
    "UE_CONVERGE_TICKS": "12",
    "UE_SKIP_LOAD": "1",               # DON'T reload the map -- keep the warm editor's loaded spire
    "UE_SKIP_PCG": "1",                # don't regenerate PCG (spire assembly already built)
    "UE_SETTLE_TICKS": "120",          # scene is already loaded -> short warmup for the SceneCapture
    "UE_FOCUS_CM": "89287.5,-5187.4,1849", "UE_ORBIT_RADIUS_CM": "6500",
    "UE_MIN_SCENE_TOP_CM": "0",        # gate off: spire (BP_PCG_LargeAssembly, 45.6m) verified loaded
    "UE_NO_QUIT": "1",                 # leave the user's editor open when done
    "UE_CAPTURE_OUT": REPO + "/out/ed_editor_depth2",
}
for k, v in _CFG.items():
    os.environ.setdefault(k, v)

import sys
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# exec the capture module with __file__/__name__ set so its sys.path bootstrap + bottom
# main() run exactly once, with our env in place (avoids import-cache double-run issues).
_f = os.path.join(REPO, "ue_capture", "capture_editor.py")
exec(compile(open(_f).read(), _f, "exec"), {"__file__": _f, "__name__": "__main__"})
