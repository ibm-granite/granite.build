# distill-tokenizer-align step for the SkyPilot environment (LSF / BlueVela).
#
# Makes a raw Granite base student able to speak the teacher's ChatML, and emits
# the three tokenizer artifacts every later distillation step reads. Runs in the
# shared distillation CPU image (built by steps/train/distill/Makefile). No
# Dockerfile here, so common.mk treats it as a non-image step and the image /
# publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/<name>/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := tokenizer-align

include ../../../../common.mk
