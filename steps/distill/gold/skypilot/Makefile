# distill-gold step for the SkyPilot environment (LSF and SLURM subtypes).
#
# GOLD knowledge distillation. Image is Dockerfile (next to this file) --
# common.mk auto-detects it, so image/publish-image are real targets and
# step-template.yaml's ${IMAGE_REF} token gets substituted at space/publish
# time. The image itself is deliberately slim (CUDA + uv + Python, no
# project deps): the actual trainer code AND its pinned dependencies both
# arrive at job launch time (git clone + `uv sync --locked`), not baked in
# here -- see step-template.yaml's "distill source delivery" and "runtime
# venv sync" blocks. One image serves both subtypes, since neither block
# bakes in anything LSF- or SLURM-specific.
#
# Usage:
#   make image          # build gb-step-gold:<sha> locally
#   make publish-image  # push it to REGISTRY
#   make space           # render a Space (space.yaml + steps/distill/gold/step.yaml)
#   make test             # render, build the image, run test/
#   make clean            # remove space/
#   make help             # list targets; see the shared steps/README.md

STEP_NAME := gold
REGISTRY ?= us.icr.io/cil15-shared-registry

include ../../../common.mk
