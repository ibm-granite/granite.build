# lm-engine-export step for the SkyPilot environment (kubernetes subtype: Vela).
#
# Nested step: steps/train/lm-engine/export/skypilot ->
# STEP_PATH=train/lm-engine/export, so common.mk is FOUR levels up.
#
# Like the sibling lm-engine-pretrain step, this step builds no image here: no
# Dockerfile sits next to this Makefile, so common.mk auto-detects a non-image step
# and the image / publish-image targets are no-ops. The lm-engine image is built one
# level up by steps/train/lm-engine/Makefile; a run picks it from config.image.
#
# Usage:
#   make space        # render a Space into space/
#   make test         # render the Space, then run test/
#   make publish-step # render into the committed assets tree

STEP_NAME := lm-engine-export

include ../../../../common.mk
