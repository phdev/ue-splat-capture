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
