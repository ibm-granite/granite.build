# byoc (bring-your-own-code) step for the SkyPilot environment.
#
# byoc runs in a PUBLIC container image and brings its code at runtime by cloning
# a public git repo in the launcher's setup phase, then running a user-defined
# command. It builds no custom image: there is no Dockerfile here, so common.mk
# auto-detects a non-image step and the image / publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/byoc/step.yaml) into space/
#   make clean    # remove space/
#   make help     # list targets; points to the shared steps/README.md

STEP_NAME := byoc

# This step defines its own test-setup target (below), so tell common.mk to skip
# its no-op default (avoids an "overriding recipe" warning).
HAS_TEST_SETUP := true

include ../../common.mk

# Stand up the local infrastructure the slurm build test needs — a Docker SLURM
# cluster — by delegating to the repo-root Makefile target (which sources the
# repo-root .venv and runs the setup script). Run once before `make test`; the
# script is idempotent, so re-running is safe. No S3 store: the test's outputs
# are env://. REPO_ROOT is provided by common.mk (repo root relative to here).
test-setup:
	$(MAKE) -C $(REPO_ROOT) slurm-setup
