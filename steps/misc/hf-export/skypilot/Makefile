# hf-export step for the SkyPilot environment (SLURM and LSF).
#
# Packages one checkpoint-N of an HF Trainer / accelerate run into a self-contained,
# downloadable HuggingFace model directory. Builds its own small CPU image from the
# Dockerfile here; the script and its resume gate are baked into it.
#
# Usage:
#   make image publish-image   # build and push the image (needs registry login)
#   make space                 # render a Space (space.yaml + steps/<name>/step.yaml)
#   make test                  # render, then run test/
#   make publish-step          # write the published copy under configurations/
#   make lock                  # recompile requirements.lock from requirements.in
#   make help                  # list targets; see the shared steps/README.md

STEP_NAME := hf-export
REGISTRY ?= us.icr.io/cil15-shared-registry
IMAGE_NAME ?= gb-step-hf-export
IMAGE_VERSION ?= fd7abbfc
UV_LOCK ?= uvx --from uv==0.13.6 uv

include ../../../common.mk

.PHONY: lock
lock: ## Recompile requirements.lock (hashed) from requirements.in for linux x86_64
	$(UV_LOCK) pip compile requirements.in --python-version 3.12 \
	  --python-platform x86_64-manylinux_2_28 --generate-hashes -o requirements.lock
