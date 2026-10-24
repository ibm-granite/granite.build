# fm-tune step for the SkyPilot environment (slurm subtype): runs one fm-tune
# (autotunex/src/fm-tune) tuning job on a bare node; the run script builds its
# own venv. Public image (no Dockerfile).
#
# Usage:
#   make test-setup   # once: bring up the local podman/Docker Slurm cluster
#   make test         # render the Space, then run test/ (slurm test self-skips)
#   make publish-step

STEP_NAME := fm-tune

# This step defines its own test-setup target (below).
HAS_TEST_SETUP := true

include ../../../common.mk

test-setup:
	$(MAKE) -C $(REPO_ROOT) slurm-setup
