# openinstruct-sft step for SkyPilot: SFT with upstream Open-Instruct finetune.py,
# ClearML tracking. Custom-image step: the Dockerfile here bakes Open-Instruct and
# src/ into the image, pushed as
# $(REGISTRY)/gb-steps:train-openinstruct-sft-skypilot-<IMAGE_VERSION>.
STEP_NAME := openinstruct-sft

# Public namespace. Its shared gb-steps repo is pre-created, Public, and grants a
# push robot Write; log in with that robot before `make publish-image` (see
# steps/README.md, "Registry credentials").
REGISTRY ?= quay.io/granitebuild

# The version part of the image tag (common.mk derives the tag
# train-openinstruct-sft-skypilot-<IMAGE_VERSION> in the shared gb-steps repo).
# A stable version instead of the per-commit git SHA: the image is large and
# BlueVela runs it from a .sqsh built from this exact tag, so it must not churn
# on unrelated commits. Bump the suffix when the Dockerfile or src/ changes; the
# prefix with the Open-Instruct pin.
IMAGE_VERSION ?= oi-v0.3.0-5

include ../../../common.mk
