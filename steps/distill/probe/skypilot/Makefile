# distill-probe step for the SkyPilot environment (LSF / BlueVela).
#
# Three cheap probes that gate the granite4-350m distillation family: does the student
# load in this image, does the declared tokenizer_class change segmentation, and which
# segmentation the weights prefer. Runs in the same prebuilt kd-sandbox image as the
# distill steps (torch + transformers are already in its venv), so there is no
# Dockerfile here: common.mk auto-detects a non-image step and the image /
# publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/distill/probe/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := probe

include ../../../common.mk
