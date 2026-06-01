# ue-splat-capture

Render a synthetic Unreal Engine scene to posed images, train a 3D Gaussian
splat locally on **Apple Silicon (no CUDA)**, and prove correctness with
automated gates.

## The one falsifiable claim

> A splat trained **only** from UE renders reconstructs held-out UE views at
> **≥ 28 dB PSNR** and **≥ 0.85 SSIM** on the bundled self-test scene, **and**
> every exported camera pose reprojects known 3D fiducials to **< 1.0 px** mean
> error.

`make verify` exits `0` only if all of that holds.

```bash
make setup     # create the pinned uv venv (torch 2.5.1 w/ MPS, numpy, pillow)
make verify    # run T0..T3 on committed fixtures; exits 0 iff every gate passes
```

## Constraints honoured

- **Apple Silicon, no CUDA.** The trainer named in the spec, `msplat`, is a
  CUDA-only rasterizer and cannot run here. The default trainer is an in-repo
  differentiable 3DGS rasterizer in pure PyTorch (MPS/CPU), `splatkit/gsmodel.py`,
  selected by `train.py --backend torch`. `--backend msplat` is wired for a CUDA
  host (`pip install msplat[cli]`) but errors out on this machine. The success
  metric is trainer-agnostic — it is about reconstruction quality, not which
  kernel computes it.
- **All dependencies pinned**; `uv.lock` committed. No cloud calls at runtime.
- **UE side** (`ue_capture/`) runs in UnrealEditor-Cmd's embedded Python (pure
  stdlib, no numpy); the pure-Python side (`splatkit/`) runs in a uv venv.
- **Output is Nerfstudio/instant-ngp `transforms.json`.**

## Coordinate convention (the part that must be right)

Source — **Unreal**: left-handed, **Z-up**, X-forward, Y-right, **centimetres**.
Target — **transforms.json**: right-handed, **metres**, camera axes **OpenCV**
(+X right, +Y down, +Z forward); `transform_matrix` is camera-to-world.

Handedness is changed by negating exactly one world axis: `D = diag(1, -1, 1)`,
times `0.01` for cm→m (`splatkit/convert.py`). Two independent guards make a
wrong convention **fail a test rather than silently mirror**:

1. **Proper-rotation gate (T0):** the converted camera-to-world rotation must
   have `det ≈ +1`. The classic "treat UE as already right-handed" bug yields
   `det = −1` and is rejected.
2. **Reprojection gate (T1):** known fiducials projected through the exported
   OpenCV intrinsics+extrinsics must match an *independently written* UE-native
   projector to < 1 px. An asymmetric flip (points but not the camera basis)
   diverges by many pixels and is caught.

On the committed fixtures T1 reprojects to ~1e-14 px.

## Verification tiers

Each writes `results/<tier>.json` as `{metric, value, threshold, pass, checks}`
and returns non-zero on failure.

| Tier | Make target          | What it gates |
|------|----------------------|---------------|
| T0   | `make test-convert`  | Pure-math pytest: world round-trip, known-good fixture, reprojection vs an independent projector, **handedness negatives**. No UE. |
| T1   | `make verify-poses`  | Reproject known fiducials → **< 1 px** mean error per pose. Catches handedness/axis bugs. |
| T2   | `make verify-dataset`| transforms.json schema-valid; intrinsics sane; images exist at declared res; **frustum union covers the AABB**; **no camera inside geometry**. |
| T3   | `make verify-recon`  | Train splat on the **train** split (fixed seed/iters), render the **held-out** split, assert PSNR/SSIM meet the metric, plus an **overfit guard** (held-out tracks train). |

`make verify` runs T0..T3, prints a summary table, compares to
`results/baseline.json` (flags regressions beyond per-metric tolerance), and
exits 0 only if every gate passes.

## Reproducibility

Fixed seeds everywhere (rig, point-cloud init, splat optimisation, densification
RNG), deterministic file ordering, and a committed `results/baseline.json`.
`SPLAT_CPU=1` forces CPU for bit-more-deterministic runs; the gate thresholds
keep a healthy margin over typical metrics so MPS non-determinism never flips a
result.

## Capture from Unreal

The committed fixtures come from a numpy raytracer stand-in (`selftest/`) so
verification reproduces **without** an Unreal install. To (re)capture from real
Unreal Engine 5.x:

```bash
export UE_PROJECT=/path/to/Your.uproject     # UnrealEditor-Cmd is auto-detected
make capture                                  # runs UE -> ue_poses.json -> ingest
```

`make capture` auto-detects `UnrealEditor-Cmd` inside the installed `.app`
bundle; if UE (or `$UE_PROJECT`) is absent it regenerates the numpy fixtures
with a warning. The UE side spawns fiducials at known coords + the orbit rig,
renders color+depth, and writes a neutral `ue_poses.json`; `splatkit.ingest`
performs the coordinate conversion in the venv (the two interpreters never share
packages).

## Layout

```
ue_capture/   UE-side (UnrealEditor-Cmd Python): detect, rig, selftest_scene,
              render (Movie Render Queue / SceneCapture), export, run_capture
splatkit/     pure Python: convert (the math), schema, coverage, reproject,
              initpc (image-only point cloud), gsmodel (3DGS), train, evaluate,
              metrics, ingest, verify, results
selftest/     numpy scene + raytracer that stands in for UE to make fixtures
fixtures/selftest/  committed images + transforms.json + scene.json
tests/        pytest (convert/reproject/dataset/rig/ingest)
results/      baseline.json (committed) + per-run tier JSON (gitignored)
```

## Out of scope (separate goals)

In-engine splat rendering / plugin choice, SOG conversion + streamed LOD,
collision-mesh export, GI/light-probe bake. This goal is the trustworthy posed
dataset + train + verify core that everything downstream depends on.
