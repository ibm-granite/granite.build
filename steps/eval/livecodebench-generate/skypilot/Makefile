# eval/livecodebench-generate step for the SkyPilot environment.
#
# Generates LiveCodeBench completions for any chat model with vLLM (GPU): it
# installs vLLM and LiveCodeBench (pinned) into a venv and runs
# src/lcb_generate.py, which builds prompts with the model's own chat template.
# It builds no custom image: there is no Dockerfile here, so common.mk
# auto-detects a non-image step and the image / publish-image targets are no-ops.
#
# No local build test: vLLM needs a GPU and the local clusters have none, so this
# Makefile defines no test-setup.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/eval/livecodebench-generate/step.yaml) into space/
#   make test     # render the Space, then run test/
#   make clean    # remove space/
#   make help     # list targets; points to the shared steps/README.md

STEP_NAME := livecodebench-generate

include ../../../common.mk

# `make test` imports src/*.py, which leaves a src/__pycache__/ behind; keep it out
# of the shipped tree (space and publish-step both copy src/ verbatim).
space publish-step: drop-pycache

.PHONY: drop-pycache
drop-pycache:
	@find $(SRC_DIR) -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
