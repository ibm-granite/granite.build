# openinstruct-sft step for SkyPilot: SFT with upstream Open-Instruct finetune.py,
# ClearML tracking. Custom-image step: the Dockerfile here bakes Open-Instruct and
# src/ into the image, pushed as
# $(REGISTRY)/steps/train/openinstruct-sft/skypilot:<IMAGE_VERSION>.
STEP_NAME := openinstruct-sft

# Public namespace. The step's repository there must be public; see
# steps/README.md, "Registry credentials" and "A custom image-based step's image repository".
REGISTRY ?= quay.io/granitebuild

# The image tag. A stable version instead of the per-commit git SHA: the image is large and
# BlueVela runs it from a .sqsh built from this exact tag, so it must not churn
# on unrelated commits. Bump the suffix when the Dockerfile or src/ changes; the
# prefix with the Open-Instruct pin.
IMAGE_VERSION ?= oi-v0.3.0-6

include ../../../common.mk
