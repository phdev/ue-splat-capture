# CLAUDE.md — ue-splat-capture

Guide for AI agents working in this repo. Keep this file updated when you change
behavior (per the user's standing instruction).

## What this is
UE scene → posed images → local Gaussian splat (Apple Silicon, no CUDA) → gated
verification. The one falsifiable claim and the gate live in
`splatkit/verify.py` / `make verify` (exits 0 iff all tiers pass):
held-out PSNR ≥ 28 dB, SSIM ≥ 0.85, fiducial reprojection < 1 px.

## Run it
```bash
make setup            # uv sync (pinned: torch 2.5.1 MPS, numpy 2.2.1, pillow, pytest)
make verify           # T0..T3 + summary table; the headline gate
make test             # full pytest suite
make test-convert     # T0   make verify-poses  # T1   make verify-dataset # T2
make verify-recon     # T3 (slow; trains a splat)  ITERS=… NGAUSS=… SEED=…
make fixtures         # regenerate committed fixtures via the numpy stand-in
make baseline         # freeze results/*.json -> results/baseline.json
```
Always invoke via `uv run` (Makefile does). Set `PYTORCH_ENABLE_MPS_FALLBACK=1`
(Makefile/scripts export it) so any MPS-unsupported op falls back to CPU.

## Architecture / where things are
- `splatkit/convert.py` — THE coordinate math (UE LH cm → OpenCV RH m). Do not
  "simplify" the `diag(1,-1,1)` flip; T0/T1 exist to protect it.
- `splatkit/reproject.py` (T1), `schema.py`+`coverage.py` (T2),
  `gsmodel.py` (3DGS rasterizer + densification), `train.py`, `evaluate.py`,
  `initpc.py` (image-only point-cloud init), `metrics.py` (PSNR/SSIM, torch),
  `ingest.py` (ue_poses.json → transforms.json), `verify.py` (orchestrator),
  `tier_t0/2/3.py` (gate runners), `make_baseline.py`.
- `selftest/` — numpy raytracer + canonical scene; `make_fixtures.py` writes the
  committed fixtures. This is the **UE stand-in** so CI needs no Unreal.
- `ue_capture/` — runs inside UnrealEditor-Cmd (UE 5.7 detected at
  `/Users/Shared/Epic Games/UE_5.7/...`). **Pure stdlib only** (UE's Python has
  no numpy). `rig.py`/`detect.py` are venv-importable + unit-tested.
- `fixtures/selftest/` committed: images + transforms.json + scene.json.
- `tests/` pytest; root `conftest.py` puts repo root on sys.path.

## Key facts / gotchas
- **msplat is CUDA-only** → cannot run here. Default trainer is the in-repo torch
  rasterizer (`--backend torch`). `--backend msplat` errors on this Mac by design.
  Do not add msplat to the lockfile (it makes resolution fail).
- transforms.json convention: `camera_model: OPENCV`, world Z-up RH metres,
  `transform_matrix` = camera-to-world. Frames also carry `location_cm`,
  `basis_ue`, `fiducials_px`, `fiducials_vis` (used by T1 and negative tests).
- T1's GT pixels come from `selftest.scene.project_ue_native`, written
  independently of `convert.py`, so T1 is a real cross-check (not a tautology).
- The recon trainer: consistency (photo-consistency / voxel-colouring) init →
  Adam → grad-driven densify/prune. Fixed seeds; deterministic given device.
  Tunables: `ITERS`, `NGAUSS`, `--scale-split`, densify schedule in `train.py`.
- **Buffered output:** when running training to a file, use `python -u` and do
  NOT pipe through `tail` (tail only flushes at EOF — you'll see nothing live).
- Training is slow-ish on MPS (full-image O(N·P) rasteriser, no tiling). 96×96
  fixtures keep it tractable. Don't run two MPS jobs at once (they contend).

## UE 5.7 capture — validated live on this machine
A real headless capture was run: `UnrealEditor-Cmd <proj> -ExecutePythonScript=…
-unattended -nosplash -nop4 -RenderOffScreen -stdout`. Findings baked into the code:
- **Class names in 5.7 Python**: `unreal.RenderingLibrary` (create_render_target2d,
  export_render_target) and `unreal.MathLibrary` (find_look_at_rotation) — NOT the
  `Kismet*` names. `unreal.load_asset("/Engine/BasicShapes/Cube.Cube")` works.
- **PNG output**: create the RT as `TextureRenderTargetFormat.RTF_RGBA8` (the float
  default writes EXR), and `export_render_target` writes the file with NO extension
  → `render._export_png` renames it to `<name>.png`.
- **Lighting required**: an empty level renders black; `selftest_scene.spawn_scene`
  spawns two directional lights.
- `MaterialInstanceDynamic.create` is absent in 5.7 Python, so object colours are
  best-effort (they render neutral/gray) — cosmetic; doesn't affect poses/gates.
- Offscreen Metal rendering works headless on Apple Silicon. Live result: 80 frames
  rendered; ingest → **T1 reprojection 3e-14 px, T2 all PASS** on authentic UE poses.
- Output goes to `out/` (gitignored); committed fixtures stay the numpy stand-in.
  Drive it with `UE_PROJECT=… make capture` (or `UE_CAPTURE_OUT=… UnrealEditor-Cmd …
  -ExecutePythonScript=ue_capture/run_capture.py`).

## Reproducibility
Fixed seeds (rig, init, optim, densify RNG), deterministic ordering, committed
`results/baseline.json`. `make verify` flags regressions beyond per-metric
tolerance (`_TOL` in `verify.py`). Re-run on a clean checkout reproduces pass/fail
and metrics within margin.
