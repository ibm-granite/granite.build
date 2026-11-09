# distill-corpus-prep step for the SkyPilot environment (LSF / BlueVela).
#
# Turns a conversation dataset into the GOLD training corpus: filter by rendered
# length, apply a think-tag policy, enforce the completion boundary, split
# deterministically, and record what was done in a manifest the trainer checks.
# Runs in the shared distillation CPU image (built by steps/train/distill/Makefile).
# No Dockerfile here, so common.mk treats it as a non-image step and the image /
# publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/<name>/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := data-prep

include ../../../../common.mk
