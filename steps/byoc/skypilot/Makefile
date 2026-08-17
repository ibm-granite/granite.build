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
# cluster and a MinIO (S3) endpoint — by delegating to the repo-root Makefile
# targets (which source the repo-root .venv and run the setup scripts). Run once
# before `make test`; both scripts are idempotent, so re-running is safe.
# REPO_ROOT is provided by common.mk (the repo root relative to this step dir).
test-setup:
	$(MAKE) -C $(REPO_ROOT) slurm-setup minio-setup
