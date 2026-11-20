# gen-smoke step for the SkyPilot environment (LSF / BlueVela).
#
# Greedy-decodes a handful of prompts from each exported checkpoint and measures how
# much of the output sits inside a run of repeated lines -- the failure df8512e0's
# distilled model showed, invisible to every training-time metric. Runs in the shared
# distillation GPU image (built by steps/train/distill/Makefile; torch + transformers
# are already in its venv). No Dockerfile here, so common.mk treats it as a non-image
# step and the image / publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/train/distill/repetition-check/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := repetition-check

include ../../../../common.mk
