# granite-multi-challenge step for the SkyPilot environment.
#
# Like byoc, this step builds no image here: there is no Dockerfile, so common.mk
# auto-detects a non-image step and the image / publish-image targets are no-ops. The
# image is the granite-evals runtime (github.com/laminair/granite-evals, `make publish-image
# EXTRA=judged` there), chosen at run time from granite_config.image.
#
# Usage:
#   make space        # render a Space into space/
#   make publish-step # render into the committed assets tree
#   make unit-tests   # template contract tests, nothing installed

STEP_NAME := granite-multi-challenge

include ../../common.mk

.PHONY: unit-tests
unit-tests:
	uv run --no-project --with pytest --with pyyaml --with pydantic pytest test -q
