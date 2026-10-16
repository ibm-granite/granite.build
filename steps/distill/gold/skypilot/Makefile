# distill-gold step for the SkyPilot environment (LSF / BlueVela).
#
# GOLD knowledge distillation with the kd-sandbox trainer. Runs in a prebuilt
# public-registry image and takes the trainer from the live /proj checkout at
# runtime, so there is no Dockerfile here: common.mk auto-detects a non-image
# step and the image / publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/distill/gold/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := gold

include ../../../common.mk
