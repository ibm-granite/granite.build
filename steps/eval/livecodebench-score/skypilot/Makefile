# eval/livecodebench-score step for the SkyPilot environment.
#
# Scores LiveCodeBench completions written by eval/livecodebench-generate, on CPU:
# it installs LiveCodeBench (pinned) into a venv and runs LiveCodeBench's own
# scorer through src/lcb_score.py. It builds no custom image: there is no
# Dockerfile here, so common.mk auto-detects a non-image step and the image /
# publish-image targets are no-ops.
#
# Usage:
#   make space        # render a Space (space.yaml + steps/eval/livecodebench-score/step.yaml) into space/
#   make test-setup   # once: bring up the local Docker SLURM cluster the build test uses
#   make test         # render the Space, then run test/
#   make clean        # remove space/
#   make help         # list targets; points to the shared steps/README.md

STEP_NAME := livecodebench-score

# This step defines its own test-setup target (below), so tell common.mk to skip
# its no-op default.
HAS_TEST_SETUP := true

include ../../../common.mk

# `make test` imports src/*.py, which leaves a src/__pycache__/ behind; keep it out
# of the shipped tree (space and publish-step both copy src/ verbatim).
space publish-step: drop-pycache

.PHONY: drop-pycache
drop-pycache:
	@find $(SRC_DIR) -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# The slurm-cpu build test runs on the local Docker SLURM cluster; this brings it up
# through the repo-root target. Idempotent.
test-setup:
	$(MAKE) -C $(REPO_ROOT) slurm-setup
