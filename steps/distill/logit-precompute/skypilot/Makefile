# distill-logit-precompute step for the SkyPilot environment (LSF / BlueVela).
#
# Runs the teacher ONCE over the corpus and keeps the top-K logits per assistant
# token, so a later training run can do forward KL against a file instead of holding
# the teacher in memory. Runs in a prebuilt public-registry image and takes the
# shared distillation package from the /proj checkout at run time, so there is no
# Dockerfile here: common.mk auto-detects a non-image step and the image /
# publish-image targets are no-ops.
#
# WIRED INTO NO RECIPE, deliberately — see USAGE.md.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/<name>/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := logit-precompute

include ../../../common.mk
