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
- **SH colour**: `gsmodel` supports view-dependent spherical-harmonics colour;
  `--sh-degree` (default **0** = view-independent). Degree 0 is best for these
  sparse (~54-view) captures. Higher degrees OVERFIT sparse captures: on the UE
  capture, SH degree 3 made held-out WORSE (22.3 vs 23.7 dB) and grew the
  train→held-out gap (4.5→6.7) — the extra view-dependent capacity memorizes
  per-training-view appearance instead of generalizing. (Synthetic, being
  view-independent, is unaffected: 33.0 @ deg0, 33.5 @ deg3.) Raise the degree
  only with many more views.
- **UE capture PASSES ROBUSTLY.** Held-out PSNR/SSIM progression on the live UE
  capture: 13.3 (base-colour) -> 17.8 (lit) -> 22.4 (Lumen/AO off) -> 23.7
  (supersample+fl) -> 26.4 (120-cam rig) -> 28.58/0.854 (120-cam + SH1 + 2500it)
  -> **30.56 dB / 0.887 SSIM** (200-cam rig + SH1 + 3500it + 24k gaussians,
  overfit gap 0.62, WORST view 27.9 dB). **VIEW COUNT was the decisive lever**:
  more views lower the held-out gap, raise the worst views, and let SH degree 1
  generalise the platform specular (SH overfits at <~100 views; SH degree 2-3
  and blur<0.1 made it WORSE -- see run "A"). Winning recipe: `make capture`
  (200-cam dense rig) -> ingest (apply ~0.989 UE fl calibration) -> `SPLAT_MAX_POINTS=24000
  tier_t3 --sh-degree 1 --iters 3500`. Keep default blur 0.12, lambda_ssim 0.2.
- **Buffered output:** when running training to a file, use `python -u` and do
  NOT pipe through `tail` (tail only flushes at EOF — you'll see nothing live).
- Training is slow-ish on MPS. 96×96 fixtures keep it tractable. Don't run two
  MPS jobs at once (they contend -- a concurrent job inflates per-iter time
  several-fold and can look like the renderer's fault; verify in isolation).
- **Tiled rasterizer (`render_tiled`, opt-in via `SPLAT_TILED=1`).** There are
  two rasterizers: the default GLOBAL one (`render`, evaluates every gaussian
  against every pixel) and a tile-based one (`render_tiled`, bins gaussians into
  screen tiles + composites each tile independently). `render_auto` dispatches;
  **default is GLOBAL.** Measured reality (don't re-litigate): at the 96×96 gate
  the tiled path is a NET LOSS end-to-end -- 1.5x SLOWER and -0.35 dB (clean
  `train()` A/B, 400 iters) -- because its backward over the padded
  (tiles x max_per_tile x tile_px) tensor outweighs the savings when there are
  only ~9k pixels. Forward-only benchmarks MISLEAD (they show tiled "winning" at
  96px); always measure fwd+bwd end-to-end. Where tiled IS the only option: HIGH
  resolution -- the global renderer's O(M*pixels) tensor OOMs (~40 GB at 288px,
  hard-OOM by 384px) while tiled stays bounded and its parity vs global actually
  improves with resolution (42 dB @96px -> 71 dB @192px). Gotcha: `MAX_PER_TILE`
  (default 2048, env `SPLAT_MAX_PER_TILE`) is a SAFETY ceiling, not a routine
  cap; densification piles 1000+ gaussians into one tile, and a low value (e.g.
  384) silently DROPS the farthest ones there -> parity craters to 27 dB. Parity
  is guarded by `tests/test_gsmodel.py::test_tiled_*`. So: high-res training is
  the real quality lever and `render_tiled` is its enabler, but it does NOT
  improve the 96px result -- leave it off there.

## UE 5.7 capture — validated live on this machine
A real headless capture was run: `UnrealEditor-Cmd <proj> -ExecutePythonScript=…
-unattended -nosplash -nop4 -RenderOffScreen -stdout`. Findings baked into the code:
- **Class names in 5.7 Python**: `unreal.RenderingLibrary` (create_render_target2d,
  export_render_target) and `unreal.MathLibrary` (find_look_at_rotation) — NOT the
  `Kismet*` names. `unreal.load_asset("/Engine/BasicShapes/Cube.Cube")` works.
- **PNG output**: create the RT as `TextureRenderTargetFormat.RTF_RGBA8` (the float
  default writes EXR), and `export_render_target` writes the file with NO extension
  → `render._export_png` renames it to `<name>.png`.
- **Capture mode = lit `SCS_FINAL_COLOR_LDR`** with the recipe below. (BASE_COLOR
  gives flat albedo with NO shading → no depth cues → the splat smears depth, ~13 dB.
  Lighting provides the depth cues; matte materials keep it SH0-friendly.)
- **Materials**: author `M_SplatMatte` (VectorParameter "Color"→BaseColor, Constant
  1→Roughness, Constant 0→Specular) so surfaces are matte/view-independent (default
  material roughness is 0 = mirror-glossy → moving highlights → bad). MIDs via
  `MaterialLibrary.create_dynamic_material_instance` (`MaterialInstanceDynamic.create`
  is ABSENT in 5.7). Assets created headless aren't saved unless `EditorAssetLibrary.
  save_asset`, so author fresh each run.
- **Lighting**: cameras orbit, so fixed lights leave camera-facing sides black and
  SkyLight captured-scene ambient doesn't fill in a one-shot. Use 6 explicit
  DirectionalLights (strong top key + 4 side fills + weak bottom) → every face lit +
  top-biased gradient.
- **Background**: SceneCapture2D has NO `show_flags` attr (use `show_flag_settings`
  with EngineShowFlagsSetting to drop Atmosphere/Fog/Cloud) AND the RT clear colour
  does NOT fill FinalColor empties → add a big two-sided UNLIT emissive `M_Bg` dome.
- **Exposure**: pin `auto_exposure_min/max_brightness = 5` (+overrides) on the capture
  post-process so a surface has the SAME brightness in every view (no eye adaptation).
- `splatkit.ingest` auto-detects the dataset background from image corners so the
  trainer composites/inits against the real backdrop colour.
- Offscreen Metal rendering works headless on Apple Silicon. Live result: 80 frames
  rendered; ingest → **T1 reprojection 3e-14 px, T2 all PASS** on authentic UE poses;
  a splat then trains on the UE images (see out/ue_eval2).
- Output goes to `out/` (gitignored); committed fixtures stay the numpy stand-in.
  Drive it with `UE_PROJECT=… make capture` (or `UE_CAPTURE_OUT=… UnrealEditor-Cmd …
  -ExecutePythonScript=ue_capture/run_capture.py`).

## Capturing an ARBITRARY UE level (not the self-test diorama)
Two entry points capture any already-authored level into the same pipeline:
- **Live editor** `ue_capture/capture_hero_orbit.py` -- run `py "<path>"` in the
  OPEN editor's Python console. Reads the viewport camera, ray-traces to the hero
  spot, orbits it, keeps the scene's real lighting. Best for World-Partition/PCG
  scenes (geometry is already streamed/generated in memory; no re-stream needed).
- **Headless** `ue_capture/capture_headless.py` via `scripts/capture_headless_run.sh`
  (editor must be CLOSED -- it holds the project lock). Loads the level, streams WP
  actors (`WorldPartitionBlueprintLibrary.get_actor_descs`/`load_actors`), auto-frames
  from actor bounds (no viewport headless), orbits. `UE_PROBE=1` = load + report
  geometry counts + a few overview frames FIRST (cheap validation); `UE_EXPO_SWEEP=1`
  = render one pose at several exposures to pick `UE_CAPTURE_EV`.

## Capture COVERAGE for a hero object (dome vs canyon) — fixes "missing gaps"
A 3DGS reconstruction can ONLY have geometry where enough camera views saw the
surface. Under-covered surfaces either don't reconstruct (black holes) or reconstruct
faint+sparse (and then floater-cleaning strips them -> holes). **Match the rig to the
subject:**
- **`UE_CANYON=1`** = a flat grid of positions with shallow pitch fans along the
  principal axis. Great for a corridor/canyon FLYTHROUGH; **wrong for a single hero
  object** — it only sees it from near-level angles, so the object's TOP and BACK are
  never captured. (The first Electric Dreams splat used this -> the spire's back was a
  black void when you orbited to it. A flat capture's holes hide from a level-angle QA.)
- **Orbit DOME** (default headless path) = `rig.orbit_hemisphere`: full 360deg azimuth
  at each elevation ring. For a hero object set **`UE_ELEVATIONS`** (comma-deg, default
  `8,22,38,55`) to a WIDE sweep e.g. `8,22,36,50,64,76` and **`UE_N_AZ=40`** -> 240 cams
  covering sides through top. This FILLED the spire holes. Match intrinsics if you want
  to merge datasets (`UE_HFOV`, `UE_CAP_RES`/`UE_TRAIN_RES`); ingest writes GLOBAL
  intrinsics so different HFOVs can't share one transforms.json.
- **QA from a full low-angle ORBIT, never 2 angles** (`scripts/orbit_poses.py` writes N
  `?settings=poseK.json` files; loop the headed browser over them). Holes hide from the
  hero angle. Render the OLD splat at the same poses to tell capture-gap from over-clean.
- **Coverage changes the clean:** with complete coverage the real surface is dense+opaque
  (high SOR-neighbour count, high opacity), so the SAME aggressive `despike_ply.py` that
  HOLED an under-covered splat is now SAFE (floaters are still sparse/faint, surface is
  not). Re-capture beat every cleaning knob. Dome result: 2M->582K, `scene6.sog` live.
- Validated cmd: `UE_PROBE=0 UE_ELEVATIONS="8,22,36,50,64,76" UE_N_AZ=40 UE_HFOV=75
  UE_CAP_RES=512 UE_TRAIN_RES=512 UE_ORBIT_RADIUS_CM=1800 UE_CAPTURE_EV=10
  UE_CAPTURE_OUT=out/ed_dome scripts/capture_headless_run.sh` (close the GUI editor first).

## Capture STABILITY — the real cause of "spotty" foliage (temporal averaging)
If a splat is **spiky/spotty on vegetation + wet surfaces but the static rock is clean**,
it is almost certainly NOT thin-geometry or a cleaning problem — it is **per-view render
noise**. `SceneCapture2D` does NOT temporally accumulate Lumen GI / specular / TSR the way
the live viewport does, so each captured view has a *different* noise realisation on
foliage and glossy/wet surfaces. 3DGS assumes every view agrees; when those pixels disagree
view-to-view it cannot fit a surface and sprays **spiky floaters exactly there**.
- **Diagnose it (`UE_DIAG=1 UE_DIAG_N=12`)**: renders ONE pose N times back-to-back. Diff
  the frames: a *constant* (non-shrinking) frame-to-frame delta concentrated on foliage/
  specular = stochastic noise (the static rock diffs to ~0). Measured here: per-pixel
  temporal std mean ~0.8 but **p99 ~9 and max ~94 / 255** — all on vegetation + wet rock.
- **Noise vs motion**: average the N samples and compare sharpness (laplacian variance) to
  a single frame. ~unchanged (943→924) = stochastic NOISE (averaging denoises, no blur).
  A big sharpness drop would mean real foliage MOTION (wind WPO) → then freeze wind/time
  instead. Here it was noise.
- **Fix = temporal averaging (`UE_AVG_SAMPLES=N`)**: `_render` exports N independent renders
  per pose (`cam_IDX_SS.png`); `scripts/average_samples.py <imgs>` folds each group into one
  clean `cam_IDX.png`. Noise falls ~1/sqrt(N): N=16 → ~25%. This is the lever that actually
  removes foliage spottiness — re-capture beats every post-clean knob, AGAIN.
- Validated cmd: add `UE_AVG_SAMPLES=16` to the dome capture (keep `UE_CAPS_PER_POSE=3` as a
  warm-up flush after each camera move), then `python3 scripts/average_samples.py
  out/<cap>/images` before ingest. ~16x the renders (still ~30 min at 1024px; renders are
  ~0.37s each). Then ingest → ue_to_brush → brush as usual.

## TERRAIN gaps (ground reconstructs in patches) — add a ground-coverage pass
A spire **dome** converges every camera on the hero -> the flat ground only appears at the
bottom edge of frame at grazing angles -> it reconstructs in disconnected patches with gray
gaps ("floating island"). Same coverage logic as the spire holes, pointed at the ground:
add a **second pass that looks DOWN at the terrain**, then MERGE with the dome and train on
the union. No code change — drive it with env: lower the focus to ground level + bigger
radius + steeper-down rings:
`UE_FOCUS_CM="<x>,<y>,<groundZ>" UE_ORBIT_RADIUS_CM=3000 UE_ELEVATIONS="28,44,60"
UE_N_AZ=36 UE_AVG_SAMPLES=16` (same UE_HFOV/UE_CAP_RES as the dome so intrinsics match).
- **Merge at the INGESTED level** (`scripts/merge_datasets.py <out> <ds0> <ds1>`): ingest
  keeps WORLD coords (no recenter), so merging there and letting `ue_to_brush` recenter the
  UNION once keeps the passes aligned. It asserts the global intrinsics match and prefixes
  filenames (`d0_`,`d1_`) to avoid collisions. Then `ue_to_brush` on the merged dir → brush.
- Sanity-check the ground frames first (some azimuths face shadow and come out dark — a few
  are fine; if most are black, raise exposure / lower EV). Validated: dome 240 + ground 108
  -> 348 views, merged + retrained to fill the terrain.
- **For the SPREAD ground (a converging ground-dome only covers the centre), use a NADIR
  GRID** (`UE_GRID=1`, `rig.grid_nadir`): NxN cameras spread over the terrain (`UE_GRID_N`,
  `UE_GRID_EXTENT_M`), each at `UE_GRID_HEIGHT_M` above the ground looking ~straight DOWN at
  the patch beneath it (`UE_GRID_CONVERGE` 0..1 tilts toward centre for angular diversity) —
  drone-mapping style, uniform overlapping coverage. Pin `UE_FOCUS_CM` to the GROUND level.
  Keep height/spacing so footprint (2*h*tan(hfov/2)) >> grid spacing (heavy overlap = the
  stereo baseline 3DGS needs on flat ground). Merge it as a 3rd pass with the dome + ground
  dome. NOTE: truly flat featureless ground stays hard for 3DGS (few features to triangulate)
  even with coverage — coverage closes the big gaps, not every last hole.

Gotchas (learned the hard way on Electric Dreams):
- **A C++ game module must be REBUILT first** for headless: a project with `Source/`
  (e.g. `ElectricDreamsSample`) aborts at boot with "game module could not be loaded"
  because the GUI's Live-Coding-patched dylib on disk can't load in a fresh process.
  Fix: `"$UE/Build/BatchFiles/Mac/Build.sh" <Target>Editor Mac Development
  -project=<uproject>` then relaunch. (`<Target>` from `Binaries/Mac/*.target`.)
- **Exposure is inverted from intuition**: a HIGHER pinned
  `auto_exposure_min/max_brightness` = DARKER image. A bright daylight scene needs
  ~8 (1.0 blows pure white); the diorama used 5. Sweep to choose.
- **PCG foliage will NOT generate** in a blocking headless script (async needs engine
  ticks a Python script never yields) -> expect bare terrain + hero meshes.
- Non-destructive: one transient SceneCapture2D, deleted after; no global cvars.

## IN-EDITOR (non-headless) capture — the BIG quality lever (PCG foliage + GT depth)
The headless `-Cmd` captures are MISSING the PCG foliage: a probe showed the live GUI
editor has **39,369 foliage instances vs 4 headless** (`instanced=4`). Every headless
splat (scene6-15) reconstructed a near-bare rock scene; the real level is densely
vegetated. So for a faithful splat, capture in the LIVE editor. `ue_capture/capture_editor.py`
+ `scripts/capture_editor_run.sh` do this **TICK-DRIVEN** (register a slate post-tick
callback + return; a state machine walks the poses while the editor ticks -> PCG generates,
TSR converges). Same env as headless (UE_NOSKY/ELEVATIONS/N_AZ/CAP_RES/FOCUS_CM/...) plus
UE_SETTLE_TICKS (wait for WP+PCG, ~450) and UE_CONVERGE_TICKS (per-pose TSR, ~12).
- **Three launch blockers (all fixed in capture_editor_run.sh), learned the hard way:**
  1. **`Binaries/Mac/UnrealEditor` is a STUB** that re-execs the `.app` copy and exits
     (exit 1 in ~4s, detaching the real editor) -> launch the **`.app` binary directly**
     (`UnrealEditor.app/Contents/MacOS/UnrealEditor`).
  2. **`-ExecutePythonScript` is run-then-QUIT** automation: the editor calls QUIT_EDITOR
     the instant the script returns, before any tick fires -> use **`-ExecCmds="py <script>"`**
     (console context) so the editor stays open and ticks.
  3. The `ps aux | grep` abort-guard **self-matched** other processes' command lines ->
     **`pgrep -x UnrealEditor`** (exact exe name; headless is `-Cmd`, not matched).
  Also: NO `-RenderOffScreen` (want real GPU rendering); WP load via `get_actor_descs()`
  (not `get_actor_descriptor_instances`, which doesn't exist).
- **The editor must be launched from an interactive GUI session** (window-server/Metal).
  Claude's automation shell, `!` in chat, and `open` from it all FAIL (editor exits early,
  no render context). The bridge that WORKS once the user has **Terminal open**:
  `osascript -e 'tell application "Terminal" to do script "<path-to-.command>"'` — runs in
  their GUI session. Keep the AppleScript arg a quote-free FILE PATH (a `.command`); inline
  commands with `"` break AppleScript string parsing. The `.command` does `exec >
  /tmp/bridge_run.txt 2>&1` so its output is readable. After that, captures can be driven
  programmatically via the bridge (no per-launch user action).
- **GT depth (TODO, now feasible in-editor):** headless `export_render_target` is 8-bit and
  saturates (depth in cm > 255 -> flat 255); the in-editor High-Res Screenshot buffer-viz
  dumps `SceneDepth` as float EXR. Metric depth needs no SfM alignment (good, since our
  exact-pose COLMAP has only a random init cloud, so monocular depth-reg's `make_depth_scale`
  can't align).

## Trainer scale limits (important, do not re-litigate)
The trainer targets the ~1 m diorama and does NOT generalise to a large photoreal
scene. Hardening added so it at least doesn't crash: `initpc._voxel_grid` caps
per-axis resolution (`max_res=96`) so a big AABB doesn't explode to 100s of millions
of voxels; `SPLAT_VAR_TOL` loosens the consistency variance gate for view-variant
(Lumen) surfaces; `train.py` random-init auto-sizes init scale to the AABB;
`scripts/normalize_ds.py` rescales a capture to ~2.5 m so the diorama-tuned LR/init
apply. BUT even with all that, a 40 m photoreal capture (Electric Dreams,
`out/electric_dreams_ds`) does NOT reconstruct: the sky is unmodellable by the
single-background-colour compositor, the texture exceeds a few-thousand gaussians,
and the optimisation diverges (loss rises). **The capture (`transforms.json` +
images) is the deliverable -- feed it to a real CUDA 3DGS trainer (Inria/gsplat/
Nerfstudio).** Our in-repo trainer only handles small/simple hero regions.

## Getting a real splat from a capture (VALIDATED -- cloud AND local)
The CAPTURE is a standard 3DGS dataset; real trainers reconstruct it faithfully
where ours can't. Both validated on the Electric Dreams capture (112 frames ->
COLMAP registered 91 views):
- **Cloud (fast): `scripts/pod_run_3dgs.sh`** on a Runpod CUDA pod. Provisions via
  REST `POST https://rest.runpod.io/v1/pods` (image `runpod/pytorch:2.1.0-...cuda11.8`,
  SSH key `~/.ssh/id_ed25519`, key gitignored at `~/.config/ue-splat-capture/runpod_api_key`;
  proven request body in the quake repo's `runpod/create_capture_pod.mjs`). Script
  apt-installs COLMAP, clones Inria 3DGS, COLMAP -> train -> render. Result: **27 dB,
  92 MB .ply, ~15 min, ~$0.20**. GOTCHA: `pip install "numpy<2"` on the pod (torch 2.1
  can't interop with numpy 2.x -> "Numpy is not available" in PILtoTorch). ALWAYS
  `DELETE /v1/pods/<id>` when done.
- **Local (free): `brush`** (`~/brush/target/release/brush`) -- a real wgpu/**Metal**
  3DGS trainer, native on Apple Silicon. `brush <colmap_dir> --total-train-iters 15000
  --eval-split-every 8 --eval-save-to-disk --export-every 5000 --export-path <dir>
  --export-name <name>`. Comparable quality, fully local, ~12 min. CAVEATS: cap
  `--max-splats 2e6` or it pressures unified RAM and exits early (~14k/15k).
- **FULLY-LOCAL path (VALIDATED, no COLMAP, no cloud):** capture -> `splatkit.ingest`
  -> `scripts/ue_to_brush.py <ds> <out> gl` (recenter + OpenCV->OpenGL flip) -> `brush
  <out> --total-train-iters 15000 --eval-split-every 8 --eval-save-to-disk --max-splats
  2000000`. The **gl** flip is correct (render-vs-GT PSNR gl 12.78 vs cv 7.52 @4k). We
  feed brush the EXACT UE poses and SKIP COLMAP -- the brew COLMAP 4.0.4 matcher
  SIGABRT-crashes on this Mac (`--FeatureMatching.use_gpu 0`, the renamed 4.x flag,
  doesn't help), and exact poses beat SfM-on-sky anyway. Capture at **EV=10** (8 blows,
  12 crushes). "Option A" result: EV=10 dense 200-view orbit -> 19.5 dB held-out,
  natural exposure + fewer floaters than the first 91-view run.
**brush is the standing LOCAL real-3DGS trainer; Runpod is only faster.** View a .ply
in SuperSplat (browser) or `brush <ply> --with-viewer`.

## FOLIAGE spikiness — Mip-Splatting ON MAC (brush `--render-mode mip`)
Thin vegetation renders as aliased spikes in vanilla 3DGS. The fix is Mip-Splatting (3D
smoothing + screen-space anti-alias filter) — and brush has it **built in and working on
Metal**: add `--render-mode mip` (+ `--split-at-screen-size 0.1` to force-split oversized
spiky gaussians). No CUDA/cloud, no different trainer, no re-capture — just retrain the
same dataset. Output is a normal .ply -> same clean/SOG/viewer pipeline. Softened the
foliage + best SSIM (0.665 vs 0.645). GOTCHAS: `--lpips-loss-weight` PANICS on the Metal
backend (burn op unimplemented — leave it 0); standalone 2DGS/Mip-Splatting GitHub repos
are CUDA-only. brush also has an intermittent cubecl `unwrap()` GPU panic mid-train — a
late checkpoint (metrics plateau ~22k) is fine. Live `scene11.sog` used this.

## TERRAIN cohesion (patchy fuzz -> surfaces) — 2D Gaussian Splatting on Runpod
3DGS (brush) makes free-floating BLOBS with no surface constraint -> ground is fuzzy/
patchy. **2DGS** makes flat SURFELS that lie on surfaces -> cohesive connected terrain.
2DGS is CUDA-only (no Mac) -> Runpod. Pipeline (the long, hard one):
- **Provision:** `runpod/create_capture_pod.mjs` request body in the quake repo is the proven
  template — `POST /v1/pods` (image `runpod/pytorch:2.1.0-...cuda11.8`, gpuTypeIds list,
  `env.PUBLIC_KEY`=`~/.ssh/id_ed25519.pub`, ports `22/tcp`). Poll `GET /v1/pods/<id>` for
  `publicIp`+`portMappings["22"]`. SSH/scp with `~/.ssh/id_ed25519`. **DELETE /v1/pods/<id>**
  when done. SSH GOTCHAS: (1) **zsh doesn't word-split `$OPTS`** — inline the ssh flags;
  (2) long foreground SSH commands drop (255) — use `nohup bash <script-file> >log 2>&1 &`
  then poll the log (the `setsid bash -c '...'` inline form silently failed); (3) macOS tar
  -> extract with `--no-same-owner`, `COPYFILE_DISABLE=1 tar --no-xattrs` to avoid `._*`.
- **SKIP COLMAP** (412-img exhaustive CPU matching ~12 h; GPU SIFT crashes headless — Qt
  needs `QT_QPA_PLATFORM=offscreen`, and `--no_gpu`). Instead feed EXACT poses:
  `scripts/transforms_to_colmap.py out/ed_full_ds/transforms.json <sparse/0>` writes
  cameras/images/points3D.txt (auto-detects OpenCV vs OpenGL via fwd·to_focus dot; random
  init cloud). VALIDATE with a 500-1000 iter run (`-r 2`, PSNR must climb) BEFORE the 30k.
- **Train:** `python3 -u gs/train.py -s ed -m ed/output -r 1 --data_device cpu
  --iterations 30000 --lambda_dist 100` (hbb1/2d-gaussian-splatting; needs matplotlib).
  2DGS ply = **2 scales + normals**; `scripts/twodgs_to_3dgs.py` adds a thin scale_2 (flat
  disk) + drops normals for the viewer. The 2DGS splat is in OpenCV WORLD coords (offset
  ~892m) -> RECENTER by median before the box-crop clean; clean GENTLY (surfels are flat by
  design). Result: PSNR 21.95, cohesive terrain. Live `scene13.sog`. `scripts/pod_run_2dgs.sh`
  is the COLMAP-on-pod variant (kept but slow); exact-poses is the fast path.

## Higher PSNR/SSIM: no-sky re-capture + 3DGS-MCMC (the scene15 pipeline)
To push held-out metrics + kill sky floaters, re-capture with the SKY OFF and train
**3DGS-MCMC** (ubc-vision/3dgs-mcmc — fixed gaussian budget + relocation, far fewer
floaters than vanilla densify). One driver: `scripts/recapture_nosky.sh` runs all three
passes (dome 240 + ground 108 + grid 64 = 412) at **1536px** with `UE_NOSKY=1` +
`UE_AVG_SAMPLES=16`.
- **`UE_NOSKY=1`** (in `_setup_capture`) disables the `Atmosphere/Fog/VolumetricFog/Cloud`
  show flags but KEEPS the directional sun + skylight -> terrain stays lit, background is
  pure black. No sky surfels for the trainer to waste capacity (or floaters) on. Probe it
  first (a dome `UE_PROBE=1` + a `UE_GRID=1` nadir) — confirm black bg + lit rock.
- **UE chdir's to its engine-binaries dir mid-startup**, so a RELATIVE `UE_CAPTURE_OUT`
  lands captures OUTSIDE the repo (`/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/out/`).
  ALWAYS pass an ABSOLUTE `UE_CAPTURE_OUT` (recapture_nosky.sh prefixes `$PWD`). The prior
  good passes' `ue_poses.json` already had absolute `file_path`s — that's the tell.
- **Pipeline** `scripts/prep_mcmc_dataset.sh`: average_samples -> splatkit.ingest (per pass)
  -> merge_datasets -> transforms_to_colmap -> flatten all imgs into `ed/images/` + `ed/sparse/0`
  (COLMAP layout, no SfM). `--eval` then holds out every 8th (llffhold) for a comparable number.
- **Pod** `scripts/runpod_pod.py {create,wait,status,delete}` (REST, key gitignored; prefers
  48GB cards, **cuda11.8 image** — the MCMC diff-gaussian-rasterization fork is 3DGS-era and
  fails to build on cuda12.4). `scripts/train_mcmc_remote.sh <ip> <port> [CAP_MAX] [ITERS]`
  tars (`COPYFILE_DISABLE=1 --no-xattrs`) + scp's `ed/`, runs `scripts/pod_run_3dgs_mcmc.sh`
  (`--cap_max` + `--scale_reg/--opacity_reg/--noise_lr` + `--data_device cpu`). **PIN
  `numpy<2`** AFTER the opencv pip (opencv pulls numpy>=2 back; torch 2.1 ABI breaks ->
  "Numpy is not available" at image load). **ALWAYS `runpod_pod.py delete <id>` after.**
- **Result (scene15):** RTX 4090, cap_max 2.5M, 30k iters -> held-out **PSNR 18.65 / SSIM
  0.7175 / LPIPS 0.245**. SSIM beats brush (0.665); PSNR is ~flat but at 1.5x the resolution
  (1536 vs ~1024 — higher res systematically lowers PSNR). More 3D detail than the 2DGS
  scene14, at the cost of 3DGS's edge spikes + soft foreground floaters.
- **Clean (MCMC ply = standard 3DGS, ~892m offset):** `despike_ply.py IN clean.ply 0.3 5 0.3
  0.25 0.06 <box±50m> 1.0 1.0 0.4 20 1.0` — spatial-only: spikes + haze + op-floor + SOR +
  keep-largest-CC, **glint DISABLED** (`sat_thr>=1.0`, the new guard — the hardcoded pure-hue/
  white clauses eat stylized terrain). MCMC parks ~30% of its budget at low opacity (the
  relocation "dead" pool) so the op-floor drops ~800k harmlessly. Then RECENTER to origin in
  numpy (splat-transform `-t` mis-parses leading-minus values) — **the viewer renders BLANK
  if content is ~892m from origin**; scene14/scene15 settings target the ORIGIN.
- **SOG:** `set_viewer_camera.py centered.ply settings.json` (it applies the `-r -90` itself)
  then `splat-transform centered.ply -N -G 0.15,0.15,0.02 -H 0 -r -90,0,0 scene15.sog -w`.
  Keep `-G` GENTLE (0.15,0.15,0.02) — `0.2,0.22,0.05` punched holes through real terrain.
  `-H 0` (drop SH) keeps the SOG small + matte (fine for this diffuse rock); full SH was 24MB.
- **Viewer A/B:** index.html now reads `?content=<file>` (default scene15.sog) so you can
  compare live: `…/?content=scene14.sog` vs the default. Both SOGs ship in out/site.

## HIGHEST-CONFIDENCE Runpod path: VANILLA Inria 3DGS (the scene17 winner)
When you want a SHARP, complete-coverage splat on novel/dense data and need it to
just-work, train **vanilla Inria 3DGS** (graphdeco-inria/gaussian-splatting) on the
exact-pose dataset — NOT MCMC. MCMC's noise-injection DIVERGES on dense PCG foliage
(scene16 attempt: held-out PSNR ~10.5, train ~9.6 — never fit), and brush completes but
is SOFT (~16 on the full-coverage 441-view data). Vanilla's unbounded densify fits hard
foliage/detail and is STABLE (no noise injection). **Result (scene17): held-out PSNR
19.32 / SSIM 0.619 / LPIPS 0.365, 1.17M gaussians, ~41 min on an A40 — +3.3 dB over
brush on the SAME complete-coverage data, and visibly sharper foliage/rock.**
- **Runners:** `scripts/pod_run_3dgs_exact.sh` (pod-side: clones Inria, pins `numpy<2`
  AFTER opencv, builds diff-gaussian-rasterization + simple-knn, `train.py --eval
  --data_device cpu`, then render + metrics). Aggressive-densify knobs as env:
  `GRAD_THRESH` (default 0.00013, < the 0.0002 default = more splits/clones) and
  `DENSIFY_UNTIL` (default 20000, > 15000 default = keep densifying longer) — these grew
  it to 1.17M (the conservative defaults left an earlier run at ~700K). `python -u` so the
  per-`--test_iterations` PSNR streams live.
- **Orchestrator:** `scripts/train_vanilla_remote.sh <ip> <port> [ITERS=30000]
  [GRAD_THRESH=0.00013] [DENSIFY_UNTIL=20000]` — tars `out/ed_ns_mcmc/ed` with **`tar -h`**
  (it's a symlink to the real COLMAP dataset), scp's it + the runner, launches under
  `nohup`, polls runv.log for `ALL_DONE`, downloads the ply →
  `out/ed_full2_vanilla/vanilla_<ITERS>.ply`. GOTCHA: the poll/download step can stall on a
  flaky SSH (the orchestrator hung after `ALL_DONE` once with an empty local dir + no scp
  running) — if so, just `scp` the ply down manually
  (`/workspace/ed/output_v/point_cloud/iteration_<ITERS>/point_cloud.ply`) and
  `runpod_pod.py delete <id>`. The ply is ready on the pod regardless; verify via
  `ssh … tail runv.log` (ALL_DONE + ply_bytes) before taking over.
- **Vanilla ply = full SH degree 3 + normals (62 props)**, ~248 bytes/gaussian, in OpenCV
  WORLD coords (~892m X offset). Clean → recenter → SOG exactly like MCMC/2DGS:
  - **Clean GENTLY — over-cleaning STRIPS the ground (this bit scene17 → fixed in scene18).**
    The terrain reconstructs as FAINT + SPARSE gaussians (oblique/grazing capture); the
    aggressive recipe `0.4 5 1.0 0.25 0.03 <box> 1.0 1.0 0.5 16 1.5` (op-floor 0.03 + SOR 0.5
    + keep-largest-CC 1.5) plus a `splat-transform -G` floater filter read that real ground as
    floaters and punched **splotchy gray holes** (the user's "splotchy / gaps"). DIAGNOSE by
    rendering the RAW (uncleaned, no `-G`) SOG — if its ground is solid (it was: complete-
    coverage capture + vanilla densify fill the ground fine), the CLEAN is the culprit, not the
    reconstruction. **scene18 winning recipe:** `despike_ply.py IN clean.ply 0.45 5 2.5 0.12
    0.01 <box±~50m around median> 1.0 1.0 1.3 16 0` — spikes (0.45, vanilla edge needles) +
    HUGE-haze only (2.5m) + **op-floor 0.01** (NOT 0.03 — ground is faint) + box + glint OFF +
    **lenient SOR 1.3** (peels only isolated AIR floaters; 0.5 ate ground — at 1.3 it drops 3K
    vs 25K) + **NO keep-largest-CC** (cc=0; CC can drop disconnected-but-real ground patches).
    Kept 1.12M of 1.17M (96%). Residual faint dark air-haze (no-sky background) is medium-sized
    (not big+faint), so the haze filter can't catch it without risking ground — left in.
  - **Recenter to origin IN NUMPY** (subtract median xyz; splat-transform `-t` mis-parses
    leading-minus; viewer renders BLANK far from origin) — there is NO recenter script,
    inline it. Then `set_viewer_camera.py centered.ply sceneN.json` (it **requires an
    EXISTING json to update** — `cp` a prior sceneN.json first; it only rewrites `cameras`).
  - **SOG:** `splat-transform centered.ply -N -H 0 -r -90,0,0 scene18.sog -w` — **NO `-G`
    floater filter** (it voxel-drops sparse gaussians = thin ground = MORE holes; the despike
    SOR already handles floaters). `-H 0` drops SH → ~14MB matte. The site default settings.json
    must mirror the default scene's json (`contents` defaults to sogName but `settings` defaults
    to ./settings.json) — `cp scene18.json settings.json` after fitting the camera.
- **Full-orbit QA (the user-mandated gate — never 2-3 angles):** `scripts/orbit_poses.py
  <sceneN.json> <dir> 0,0,0 <dist> <elev> <n>` writes pose JSONs around the ORIGIN (content
  is recentered); load each `?content=scene17.sog&settings=<dir>/poseK.json&noui` in the
  **headed** browse binary (`~/.claude/skills/gstack/browse/dist/browse --headed`; headless
  has no WebGPU). 8 mid (elev 18) + 6 low (elev 6) confirmed no floating-island gap, foliage
  all around. For the brush A/B use scene16's OWN center/scale (different coords) — generate
  matched orbit poses at scene16's target. rm the qa* pose dirs before the Pages commit.
- **Deploy:** `out/site/` is its OWN git repo (`phdev/electric-dreams-splat`, the Pages
  site) — commit+push there to deploy (the code repo is `phdev/ue-splat-capture`; `out/` is
  gitignored in it). BUMP the SOG filename each deploy (scene17→scene18→…) so the Pages CDN
  + browser cache miss — the user sees the change without a manual hard-refresh of the SOG.
  Run `python3 scripts/test_viewer_consistency.py` BEFORE every Pages commit — it catches
  deployed-but-not-in-dropdown sogs, dangling SCENES entries, and default/label drift (this
  class of bug shipped scene25 invisible and orphaned scene17 for days).

## LAYERED splats: the scene25/26 recipe (multi-pass concat — what works and what doesn't)
The current LIVE default (**scene26**, 2.18M gauss) is a CONCAT of three independently
trained splats, not one training run. Key findings, learned expensively (scenes 19-26):
- **VISIBILITY STATE PERSISTS in the editor across sessions.** Foliage-off experiments that
  `set_visibility(False)` on ISMCs silently corrupted EVERY later capture in that editor
  (and survived restarts via unsaved-state). ALWAYS run the restore-visibility sweep (set
  actor + every SceneComponent visible/not-hidden) + verify instance counts (~49.5K when
  fully loaded, 738 ISMs) BEFORE any capture. "Streaming Disabled" badge / "No loaded
  region" → WP not streaming; force-load via `WorldPartitionBlueprintLibrary.get_actor_descs
  → load_actors + pin_actors` + move the viewport near the scene (it's the streaming source).
- **Joint training of mixed passes FAILS.** Duplicate poses with conflicting content
  (foliage-on RGB + foliage-off RGB at the same camera) → optimizer averages → severe
  under-densification (scene22/23: 263-735K vs the 1.5M+ healthy runs) + PSNR=inf data leak
  in eval. Split poses (on for dome, off for nadir) ALSO fails — still geometrically
  contradictory along shared sightlines (PSNR ~15-17). Mixed-EV joint training fails the
  same way (v3: EV=10 dome + EV=12 spire orbit → 591K mush).
- **The recipe that WORKS: train each internally-consistent pass STANDALONE, then concat
  the gaussian arrays + spatial dedup.** `np.concatenate` of the raw ply rows (same 62-prop
  layout) is a valid splat; alpha-blending handles overlap. Dedup: kdtree on the base layer,
  drop added gaussians with a base neighbor within 0.5m (0.35m for dense close-orbit
  layers) — ~89% of a foliage-off layer dropped as redundant; the survivors are exactly the
  under-canopy/gap content. Clean AFTER concat with the tight knobs (`0.4 5 2.0 0.18 0.03
  <box> 1.0 1.0 0.8 16 0`) and compute the crop box from the BASE layer's median (a
  close-orbit layer skews the combined median → box clips the island edge).
- **REPAIR layers need NEWEST-WINS, not dedup (scene27; `scripts/concat_layers.py --repair`).**
  Oldest-wins dedup keeps the OLD fat dark smudge and drops the NEW crisp re-shoot — a
  base-of-spire repair layer contributed only 26K/668K gaussians until flipped. In the
  repair zone (back half-plane of the feature, self-calibrated from layer medians — the
  ingest FLIPS Y vs UE world, never hand-type world coords), delete existing gaussians
  within 0.4m of new content and insert the new layer wholesale; oldest-wins outside.
  Also: re-shoot the GAP'S elevation band — spire-orbit elevations 10-80° never saw the
  LOWER back face; the repair orbit centered at the BASE (z=25m) with elev -10/5/20 did.
  Recenter every release by the PRIOR release's median so camera jsons keep framing.
- **Hero-feature gaps (back of the spire) = orbit the FEATURE, not the focus.**
  `UE_SPIRE_ORBIT=1` rig (UE_SPIRE_CENTER_CM/RADIUS_CM/ELEV/NAZ) orbits a named point;
  dome orbits around the scene focus only see the focus-facing side of off-center features.
  Train the orbit standalone (192 poses → 5.3M gauss of fine detail), crop to the feature
  neighborhood (±25m of its own median), dedup vs base, and filter the ADDS only:
  blue-dominance (b>r+0.06 & b>g+0.06 — vegetation/rock is never blue-dominant) + glint
  (sat>0.6&val>0.6) + adds-only SOR (0.4m/k16). Per-layer filtering is safe where the
  island-wide glint filter would eat stylized terrain.
- **EV reminder (bit us AGAIN):** higher pinned `auto_exposure_*_brightness` = DARKER.
  EV=12 intending "brighter shadows" actually darkened the shadowed back ~20%.
- **Runpod ops:** pods are flaky — one died mid-train (H100 eviction), one sat 40min in
  pip (lemon). Detach with `setsid … < /dev/null & disown` (plain `nohup &` dies with the
  SSH session). Monitor the STAGE line + `ps | grep train.py` + checkpoint dir, not
  `tail -1` (pip notices mask the stage). Save checkpoints at 3000/7000/15000/30000 and
  opportunistically download EVERY new checkpoint mid-train so a pod death loses minutes,
  not the run. Upload to /root (local disk) NOT /workspace (MooseFS network mount — slow,
  corrupts big scp). JPEG-pack images (q92) before tar: 2.5GB PNG → 0.6GB, loader-compatible
  (update images.txt .png→.jpg; read-then-write, the chained one-liner truncates to 0 bytes).
  EVAL_FLAG uses `${EVAL_FLAG-default}` (NOT `:-`) so empty string means "skip eval".

## Reproducibility
Fixed seeds (rig, init, optim, densify RNG), deterministic ordering, committed
`results/baseline.json`. `make verify` flags regressions beyond per-metric
tolerance (`_TOL` in `verify.py`). Re-run on a clean checkout reproduces pass/fail
and metrics within margin.

## Web viewer: SOG on GitHub Pages
`scripts/make_sog_viewer.sh <ply> <site_dir>` -> `npx @playcanvas/splat-transform`
compresses the 3DGS .ply to a **SOG** ("WebP of splats", ~16x smaller, full SH kept)
and emits a self-contained **SuperSplat/PlayCanvas HTML viewer** (`-U` unbundled:
index.html + index.sog + index.js + settings.json). Host the folder on GitHub Pages
(static files). Live: https://phdev.github.io/electric-dreams-splat/ (from
out/brush_ed_final/ed_dense_15000.ply, 450MB -> 28MB SOG). Notes: the viewer needs
HTTP (not file://) and WebGPU/WebGL (renders on real devices, NOT in headless
Chromium -- `requestAdapter` returns nothing). True multi-chunk streamed LOD
(`lod-meta.json`) needs an LOD pyramid (decimated levels tagged `-l 0/1/2`, merged)
or SuperSplat's export dialog -- overkill below ~5M gaussians (a single SOG streams
fine). Publish steps are in the script header.

**Cache-busting (the deployed `out/site/index.html`):** browsers + the GitHub Pages CDN
cache the HTML and the multi-MB `.sog` aggressively — a new browser window can still
show a stale splat. Two mechanisms: (1) bump the SOG filename each deploy (`scene5`->
`scene6`...) and update the `fetch(...)` ref; (2) the viewer has a top-left **Reload**
button that reloads with a fresh `?cb=<timestamp>`, and the SOG + settings fetches append
it (`bust()` helper) so the new query string is a fresh cache key (browser + CDN miss).
The loaded filename is shown next to the button (define it once as `const sogName` so the
fetch + label can't drift). GOTCHA: the head `<script type="module">` is deferred, so any
body script that reads `window.__sogName` must ALSO be `type="module"` (a plain body
script runs first, before the head module sets it). Editing only index.html still needs
one hard-refresh to pick up the new file (can't fix a cached file from inside itself).

## Floater cleaning: `scripts/despike_ply.py`
brush splats of outdoor/edge scenes carry floater families that **survive** opacity,
box, sphere, and connected-cluster filters. `despike_ply.py <in.ply> <out.ply>` strips
them in one pass (all columns preserved, so downstream splat-transform still works).
**Critical gotchas:** in the raw .ply, **scales are LOG-space** (`exp()` to get meters)
and **opacity is a LOGIT** (`sigmoid()` to get [0,1]); splat-transform's `-V`/`-m` use
the activated values, Python must convert. Six filters, each targeting one artifact:
- **spikes** — long thin needles (longest axis big in abs terms AND >Nx the 2nd axis).
  Render as bright chromatic slivers. Per-axis scale caps miss them; the discriminator
  is **aspect ratio** `s2/s1` (median is already ~4.5, so gate on abs length too).
- **haze** — big AND faint blobs (milky fog). Big gaussians here are ~universally faint
  (s2>0.5m -> median opacity 0.029), so big+faint cleanly separates from real surfaces.
- **faint** — global opacity floor.
- **glint** — bright AND color-saturated chromatic confetti, plus a pure-hue clause
  (`sat>0.9`) for moderate-bright pure R/G/B blobs. NOTE: this is a *stylized* scene
  (~25% of gaussians have sat>0.7), so saturation-only filtering destroys real color —
  must gate on brightness.
- **SOR** — statistical outlier removal (mean dist to K-NN > thresh). Kills the diffuse
  confetti *shell* (sparse) while sparing the dense rock surface (~0.1m neighbor spacing).
  scipy cKDTree, ~1s for 800K pts. **This was the single biggest win.**
- **CC** — keep largest connected component (voxel-label at `cc_vox` m). The object is
  ONE big blob (~95%); detached bright clusters / dark specks / colored blobs are many
  tiny components. Use this **instead of** splat-transform `-D` (whose op=0.8 wrongly
  fragments the main mass and deletes real connected terrain).

Validated recipe (Electric Dreams hero rock, 2M -> 542K gaussians, 6.4MB SOG, deployed):
```
python3 scripts/despike_ply.py IN.ply /tmp/clean.ply 0.25 6 0.4 0.3 0.06 \
        -45,-45,-22,50,46,8 0.6 0.5 0.2 16 1.0
python3 scripts/set_viewer_camera.py /tmp/clean.ply SITE/settings.json   # fit cam to content
npx -y @playcanvas/splat-transform /tmp/clean.ply -N -G 0.15,0.15,0.02 -H 0 -r -90,0,0 SITE/sceneN.sog -w
```
Residual hard cases: moss-colored green blobs (moderate sat ~0.45, connected) read as
real moss — at the edge of separability, left in. `set_viewer_camera.py` refits the
SuperSplat camera to the cleaned content (the viewer auto-focuses on content bounds, so
the camera must track each crop). Bump the SOG filename (sceneN) to bust the Pages cache.

## Self-QA: render the viewer on the Mac GPU (headed browse)
The SOG viewer needs a real GPU — **headless** Chromium has no WebGPU adapter (blank
canvas). Use `/browse` **headed** so it renders on Metal, then screenshot + Read to
visually QA each candidate: `$B --headed goto URL && $B --headed wait --networkidle &&
$B --headed screenshot /tmp/x.png`. Pass `--headed` to EVERY command (mixed configs
error). Serve candidates locally (`python3 -m http.server`) and switch splats via
`?content=sceneX.sog` (index.html uses `fetch(contentUrl)`). QA from front AND opposite
side to catch floater farms hidden behind the hero angle.
