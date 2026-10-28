# eval/lm-eval (EleutherAI lm-evaluation-harness) step for the SkyPilot environment.
#
# Runs `lm_eval run` against a bound model, in a public container image (default
# python:3.12-slim) or on the bare launcher node, installing lm-eval at setup time.
# It builds no custom image: there is no Dockerfile here, so common.mk auto-detects a
# non-image step and the image / publish-image targets are no-ops.
#
# Usage:
#   make space        # render a Space (space.yaml + steps/eval/lm-eval/step.yaml + src/) into space/
#   make test-setup   # bring up the local Docker SLURM cluster (once)
#   make test         # render the Space, then run test/ (per-cluster tests self-skip)
#   make help         # list targets; points to the shared steps/README.md

STEP_NAME := lm-eval

# This step defines its own test-setup target (below), so tell common.mk to skip
# its no-op default (avoids an "overriding recipe" warning).
HAS_TEST_SETUP := true

include ../../../common.mk

# The slurm-cpu build test needs the local Docker SLURM cluster; delegate to the
# repo-root target. Idempotent, so re-running is safe.
test-setup:
	$(MAKE) -C $(REPO_ROOT) slurm-setup
