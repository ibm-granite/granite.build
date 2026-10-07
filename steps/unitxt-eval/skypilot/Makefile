# unitxt-eval step for the SkyPilot environment.
#
# unitxt-eval installs unitxt on the bare launcher node (or runs in a prebuilt image)
# and calls the unitxt-evaluate CLI. It builds no custom image: there is no Dockerfile
# here, so common.mk auto-detects a non-image step and the image / publish-image
# targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/unitxt-eval/step.yaml) into space/
#   make test     # render the Space, then run test/
#   make clean    # remove space/
#   make help     # list targets; points to the shared steps/README.md

STEP_NAME := unitxt-eval

include ../../common.mk
