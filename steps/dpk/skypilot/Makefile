# dpk (Data Prep Kit transform runner) step for the SkyPilot environment.
#
# dpk runs any DPK transform on the bare launcher node (or in a public container
# image), installing the transform's dependencies at run time with pip. It builds
# no custom image: there is no Dockerfile here, so common.mk auto-detects a
# non-image step and the image / publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/dpk/step.yaml + src/) into space/
#   make test     # render the Space, then run test/ (per-cluster tests self-skip)
#   make clean    # remove space/
#   make help     # list targets; points to the shared steps/README.md

STEP_NAME := dpk

# This step defines its own test-setup target (below), so tell common.mk to skip
# its no-op default (avoids an "overriding recipe" warning).
HAS_TEST_SETUP := true

include ../../common.mk

# `make test` imports src/validate_tokens.py, which leaves a src/__pycache__/
# behind, and common.mk's `space`/`publish-step` both `cp -R src/` verbatim — so
# stale bytecode would be rsynced to the cluster alongside the script. (It would
# NOT be committed: __pycache__/ is in the repo root .gitignore, which covers the
# published copy too.) Hygiene only, but dpk is the first step with a python
# src/, so keep the shipped tree to just the source.
space publish-step: drop-pycache

.PHONY: drop-pycache
drop-pycache:
	@find $(SRC_DIR) -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# Stand up the local infrastructure the slurm build test needs — a Docker SLURM
# cluster and a MinIO (S3) endpoint — by delegating to the repo-root Makefile
# targets (which source the repo-root .venv and run the setup scripts). Run once
# before `make test`; both scripts are idempotent, so re-running is safe.
# REPO_ROOT is provided by common.mk (the repo root relative to this step dir).
test-setup:
	$(MAKE) -C $(REPO_ROOT) slurm-setup minio-setup
