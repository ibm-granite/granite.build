# corpus-sources step for the SkyPilot environment (LSF / BlueVela).
#
# Builds ONE prep-ready corpus file from several raw SFT splits: renames
# `conversations` to `messages`, reservoir-samples each split in proportion to its
# size, and shuffles the result. Runs in the same prebuilt kd-sandbox image as the
# distill steps and needs only the standard library, so there is no Dockerfile here:
# common.mk auto-detects a non-image step and the image / publish-image targets are
# no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/distill/corpus-sources/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := corpus-sources

include ../../../common.mk
