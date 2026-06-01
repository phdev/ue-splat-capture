# ue-splat-capture -- capture -> train -> verify (Apple Silicon, no CUDA)
UV     ?= uv
SEED   ?= 0
ITERS  ?= 1500
NGAUSS ?= 6000
export PYTORCH_ENABLE_MPS_FALLBACK = 1

.PHONY: help setup fixtures test test-convert verify-poses verify-dataset \
        verify-recon verify capture baseline clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Create/sync the pinned uv venv
	$(UV) sync --extra dev

fixtures: ## Regenerate committed self-test fixtures (numpy stand-in for UE)
	$(UV) run python -m selftest.make_fixtures --ss 2

test-convert: ## T0  pure-math coordinate tests (round-trip, handedness, reproj)
	$(UV) run python -m splatkit.tier_t0

verify-poses: ## T1  reproject known fiducials, assert <1px mean error
	$(UV) run python -m splatkit.reproject

verify-dataset: ## T2  schema + frustum coverage + camera-in-geometry
	$(UV) run python -m splatkit.tier_t2

verify-recon: ## T3  train splat on train split, gate held-out PSNR/SSIM
	$(UV) run python -m splatkit.tier_t3 --iters $(ITERS) --n-gauss $(NGAUSS) --seed $(SEED)

test: ## Run the full pytest suite (all tiers' unit tests)
	$(UV) run pytest -q

verify: ## Run T0..T3, print summary table, exit 0 only if ALL gates pass
	$(UV) run python -m splatkit.verify --iters $(ITERS) --n-gauss $(NGAUSS) --seed $(SEED)

capture: ## Regenerate fixtures from UE if UnrealEditor-Cmd found, else numpy stand-in
	$(UV) run bash scripts/capture.sh

baseline: ## Freeze current results/*.json -> results/baseline.json
	$(UV) run python -m splatkit.make_baseline

clean: ## Remove scratch output and per-run tier results (keeps baseline + fixtures)
	rm -rf out/* results/t0.json results/t1.json results/t2.json results/t3.json
