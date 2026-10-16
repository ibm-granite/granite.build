# distill-tokenizer-align step for the SkyPilot environment (LSF / BlueVela).
#
# Makes a raw Granite base student able to speak the teacher's ChatML, and emits
# the three tokenizer artifacts every later distillation step reads. Runs in a
# prebuilt public-registry image and takes the shared distillation package from a
# git clone (or a filesystem checkout) at run time, so there is no Dockerfile
# here: common.mk auto-detects a non-image step and the image / publish-image
# targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/<name>/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := tokenizer-align

include ../../../common.mk
