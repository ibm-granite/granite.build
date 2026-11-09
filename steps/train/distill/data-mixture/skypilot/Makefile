# corpus-sources step for the SkyPilot environment (LSF / BlueVela).
#
# Builds ONE prep-ready corpus file from several raw SFT splits: renames
# `conversations` to `messages`, reservoir-samples each split in proportion to its
# size, and shuffles the result. Runs in the shared distillation CPU image (built by
# steps/train/distill/Makefile) and needs only the standard library. No Dockerfile here,
# so common.mk treats it as a non-image step and the image / publish-image targets are
# no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/train/distill/data-mixture/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := data-mixture

include ../../../../common.mk
