# eval step for the SkyPilot environment.
#
# eval is a CUSTOM-IMAGE step: its evaluation code and dependencies are baked into
# an image built from ./Dockerfile, published to $(REGISTRY), and referenced from
# the generated step.yaml via image_id. common.mk detects the Dockerfile here and
# enables the image / publish-image targets automatically.
#
# Usage:
#   make image          # build the image for $(PLATFORM) (default linux/amd64)
#   make publish-image  # push to $(REGISTRY) (requires prior `podman login`)
#   make step           # render step-template.yaml -> step/step.yaml (offline)
#   make all            # image + publish-image + step (full pipeline)
#   make clean          # remove step/
#   make help           # show the shared steps/README.md
#
# Override the image coordinates as needed, e.g.:
#   make all REGISTRY=quay.io/myorg IMAGE_TAG=0.1.0

STEP_NAME := eval
# Required for image steps: where the built image is published (common.mk errors
# if unset). Override on the command line as needed, e.g.
# `make all REGISTRY=quay.io/myorg IMAGE_TAG=0.1.0`. IMAGE_TAG defaults to the git
# short SHA if not set.
REGISTRY ?= quay.io/your-org

include ../../common.mk
