# hf-peft-sft step for SkyPilot: LoRA / QLoRA / aLoRA SFT with HF PEFT + TRL,
# ClearML tracking. Custom-image step: the Dockerfile bakes the hashed training
# stack and src/ into the image, pushed as
# $(REGISTRY)/steps/train/hf-peft-sft/skypilot:<IMAGE_VERSION>.
STEP_NAME := hf-peft-sft

# Public namespace; the step's repository there must be public (steps/README.md,
# "A custom image-based step's image repository").
REGISTRY ?= quay.io/granitebuild

# Stable tag, not the git SHA: BlueVela caches a .sqsh per tag. Bump the suffix
# when the Dockerfile, requirements.lock or src/ changes; the prefix with peft.
IMAGE_VERSION ?= peft-v0.21.2-2

# Pinned uv for the lock, same version as the image's uv stage.
UV_LOCK ?= uvx --from uv==0.13.6 uv

include ../../../common.mk

.PHONY: lock
lock: ## Recompile requirements.lock (hashed) from requirements.in for linux x86_64
	$(UV_LOCK) pip compile requirements.in --python-version 3.12 \
	  --python-platform x86_64-manylinux_2_28 --generate-hashes -o requirements.lock
