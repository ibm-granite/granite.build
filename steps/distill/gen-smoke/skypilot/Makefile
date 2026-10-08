# gen-smoke step for the SkyPilot environment (LSF / BlueVela).
#
# Greedy-decodes a handful of prompts from each exported checkpoint and measures how
# much of the output sits inside a run of repeated lines -- the failure df8512e0's
# distilled model showed, invisible to every training-time metric. Runs in the same
# prebuilt kd-sandbox image as the distill steps (torch + transformers are already in
# its venv), so there is no Dockerfile here: common.mk auto-detects a non-image step
# and the image / publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/distill/gen-smoke/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := gen-smoke

include ../../../common.mk
