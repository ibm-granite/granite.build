# preflight-check step for the SkyPilot environment (LSF / BlueVela).
#
# Three cheap probes that gate the granite4-350m distillation family: does the student
# load in this image, does the declared tokenizer_class change segmentation, and which
# segmentation the weights prefer. Runs in the shared distillation GPU image (built by
# steps/train/distill/Makefile; torch + transformers are already in its venv). No
# Dockerfile here, so common.mk treats it as a non-image step and the image /
# publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/train/distill/preflight-check/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := preflight-check

include ../../../../common.mk
