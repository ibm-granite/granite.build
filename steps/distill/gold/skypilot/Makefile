# distill-gold step for the SkyPilot environment (LSF and SLURM subtypes).
#
# GOLD knowledge distillation. No Dockerfile here: common.mk auto-detects a
# non-image step and the image / publish-image targets are no-ops. Runs in
# the shared GPU image (steps/distill/gpu/, built and published separately --
# see that directory's README), which carries both the trainer source
# (steps/distill/src/, baked in -- no runtime clone) and its pinned venv at
# /stage/.venv. One image serves both LSF and SLURM subtypes, since nothing
# in this step or that image is subtype-specific.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/distill/gold/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := gold

include ../../../common.mk
