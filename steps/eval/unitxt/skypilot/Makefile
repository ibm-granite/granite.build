# eval/unitxt step for the SkyPilot environment.
#
# eval/unitxt installs unitxt on the bare launcher node (or runs in a prebuilt image)
# and calls the unitxt-evaluate CLI. It builds no custom image: there is no Dockerfile
# here, so common.mk auto-detects a non-image step and the image / publish-image
# targets are no-ops.
#
# Usage:
#   make space        # render a Space (space.yaml + steps/eval/unitxt/step.yaml) into space/
#   make test-setup   # once: bring up the local Docker SLURM cluster the build test uses
#   make test         # render the Space, then run test/
#   make clean        # remove space/
#   make help         # list targets; points to the shared steps/README.md

STEP_NAME := unitxt

# This step defines its own test-setup target (below), so tell common.mk to skip
# its no-op default.
HAS_TEST_SETUP := true

include ../../../common.mk

# The slurm-cpu build test runs on the local Docker SLURM cluster; this brings it up
# through the repo-root target. Idempotent.
test-setup:
	$(MAKE) -C $(REPO_ROOT) slurm-setup
