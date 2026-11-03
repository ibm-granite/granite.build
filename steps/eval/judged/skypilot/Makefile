# judged step for the SkyPilot environment.
#
# Like byoc, this step builds no image here: no Dockerfile sits next to this
# Makefile, so common.mk auto-detects a non-image step and the image /
# publish-image targets are no-ops. The family image is built one level up by
# steps/eval/Makefile (from ../Dockerfile.gpu and ../Dockerfile.cpu); a run
# picks it from config.image.
#
# Usage:
#   make space        # render a Space into space/
#   make publish-step # render into the committed assets tree
#   make unit-tests   # template contract tests, nothing installed

STEP_NAME := judged

include ../../../common.mk

.PHONY: unit-tests
unit-tests:
	uv run --no-project --with pytest --with pyyaml --with pydantic pytest test -q
