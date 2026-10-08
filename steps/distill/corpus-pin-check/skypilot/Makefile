# corpus-pin-check step for the SkyPilot environment (LSF / BlueVela).
#
# Refuses a pinned corpus whose manifest does not describe this build: a different
# tokenizer, different prep policies, or a missing split. CPU-only and stdlib-only,
# so there is no Dockerfile here: common.mk auto-detects a non-image step and the
# image / publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/distill/corpus-pin-check/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := corpus-pin-check

include ../../../common.mk
