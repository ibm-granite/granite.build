# sage2-tau3-retail step for the SkyPilot environment.
#
# Like byoc, this step builds no image here: there is no Dockerfile, so common.mk
# auto-detects a non-image step and the image / publish-image targets are no-ops. The
# image is the sage2-evals runtime (github.com/laminair/sage2-evals, `make publish-image
# EXTRA=tau` there), chosen at run time from sage2_config.image.
#
# Usage:
#   make space        # render a Space into space/
#   make publish-step # render into the committed assets tree
#   make unit-tests   # template contract tests, nothing installed

STEP_NAME := sage2-tau3-retail

include ../../common.mk

.PHONY: unit-tests
unit-tests:
	uv run --no-project --with pytest --with pyyaml --with pydantic pytest test -q
