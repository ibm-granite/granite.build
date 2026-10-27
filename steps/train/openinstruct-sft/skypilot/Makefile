# openinstruct-sft step for SkyPilot: SFT with upstream Open-Instruct finetune.py,
# ClearML tracking. Custom-image step: the Dockerfile here bakes Open-Instruct and
# src/ into the image, pushed as $(REGISTRY)/steps/train/openinstruct-sft/skypilot.
STEP_NAME := openinstruct-sft

# Public namespace (functional ID). The repo must exist there, be Public, and grant
# the granitebuild+gbpush robot Write before `make publish-image`.
REGISTRY ?= quay.io/granitebuild

# A stable tag instead of the per-commit git SHA: the image is large and BlueVela
# runs it from a .sqsh built from this exact tag, so it must not churn on unrelated
# commits. Bump the suffix when the Dockerfile or src/ changes; the prefix with the
# Open-Instruct pin.
IMAGE_TAG ?= oi-v0.3.0-2

include ../../../common.mk
