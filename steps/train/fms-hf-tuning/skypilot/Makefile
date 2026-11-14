# fms-hf-tuning step for SkyPilot: full fine-tuning / LoRA / aLoRA with
# foundation-model-stack/fms-hf-tuning, ClearML tracking. Custom-image step: the
# Dockerfile bakes the hashed training stack and src/ into the image, pushed as
# $(REGISTRY)/steps/train/fms-hf-tuning/skypilot:<IMAGE_VERSION>.
STEP_NAME := fms-hf-tuning

# Public namespace; the step's repository there must be public (steps/README.md,
# "A custom image-based step's image repository").
REGISTRY ?= quay.io/granitebuild

# Stable tag, not the git SHA: BlueVela caches a .sqsh per tag. Bump the suffix
# when the Dockerfile, requirements.lock or src/ changes; the prefix with
# fms-hf-tuning.
IMAGE_VERSION ?= fms-v3.2.2-5

# Pinned uv for the lock, same version as the image's uv stage.
UV_LOCK ?= uvx --from uv==0.13.6 uv

include ../../../common.mk

.PHONY: lock
lock: ## Recompile requirements.lock (hashed) from requirements.in for linux x86_64
	$(UV_LOCK) pip compile requirements.in --python-version 3.12 \
	  --python-platform x86_64-manylinux_2_28 --generate-hashes -o requirements.lock
