# distill-eval step for the SkyPilot environment (LSF / BlueVela).
#
# Measures how far the student's next-token distribution is from the teacher's on a
# held-out corpus. Runs in the shared distillation GPU image (built by
# steps/train/distill/Makefile). No Dockerfile here, so common.mk treats it as a
# non-image step and the image / publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/<name>/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := teacher-divergence

include ../../../../common.mk
