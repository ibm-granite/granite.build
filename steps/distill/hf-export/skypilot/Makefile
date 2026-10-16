# distill-hf-export step for the SkyPilot environment (LSF / BlueVela).
#
# Packages one checkpoint-N from a distillation run into a self-contained,
# downloadable HuggingFace model directory. Runs in a prebuilt public-registry
# image and takes the shared distillation package from the /proj checkout at run
# time, so there is no Dockerfile here: common.mk auto-detects a non-image step and
# the image / publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/<name>/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := hf-export

include ../../../common.mk
