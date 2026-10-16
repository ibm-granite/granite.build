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
#   make space          # render a Space (space.yaml + steps/eval/step.yaml) into space/ (offline)
#   make all            # image + publish-image + space (full pipeline)
#   make test           # run the step's Python tests in test/
#   make clean          # remove space/
#   make help           # list targets; points to the shared steps/README.md
#
# Override the image coordinates as needed, e.g.:
#   make all REGISTRY=quay.io/myorg IMAGE_TAG=0.1.0

STEP_NAME := eval
# PLACEHOLDER registry — replace `quay.io/your-org` with your own namespace
# before publishing, or override per invocation:
#   make all REGISTRY=quay.io/myorg IMAGE_TAG=0.1.0
# This `?=` default satisfies common.mk's image-step REGISTRY guard so offline
# targets (make space / make image) work out of the box; but `make publish-image`
# WILL push to this bogus registry and fail auth unless you set a real one.
# IMAGE_TAG defaults to the git short SHA if not set.
REGISTRY ?= quay.io/your-org

include ../../common.mk
