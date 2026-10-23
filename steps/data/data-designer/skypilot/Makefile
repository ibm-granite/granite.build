# data-designer (NeMo Data Designer) step for the SkyPilot environment.
#
# data-designer runs one NeMo Data Designer generation on the bare launcher node (or
# in a public container image), installing the library at run time with uv. It
# builds no custom image: there is no Dockerfile here, so common.mk auto-detects a
# non-image step and the image / publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/data/data-designer/step.yaml + src/) into space/
#   make test     # render the Space, then run test/ (per-cluster tests self-skip)
#   make clean    # remove space/
#   make help     # list targets; points to the shared steps/README.md

STEP_NAME := data-designer

# This step defines its own test-setup target (below), so tell common.mk to skip
# its no-op default (avoids an "overriding recipe" warning).
HAS_TEST_SETUP := true

include ../../../common.mk

# `make test` imports src/dd_run.py, which leaves a src/__pycache__/ behind, and
# common.mk's `space`/`publish-step` both `cp -R src/` verbatim — so drop it first and
# keep the tree shipped to a cluster to just the source (same as the dpk step).
space publish-step: drop-pycache

.PHONY: drop-pycache
drop-pycache:
	@find $(SRC_DIR) -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# Stand up the local Docker SLURM cluster the slurm build test needs, by delegating
# to the repo-root Makefile target. Run once before `make test`; idempotent.
test-setup:
	$(MAKE) -C $(REPO_ROOT) slurm-setup

# The same for the kubernetes build test: a local kind cluster via `sky local up`, with
# the env's ReadWriteMany gb-shared volume. Separate from test-setup so neither cluster
# is brought up unless asked for. Makes kind-skypilot the current kube context.
.PHONY: test-setup-kube
test-setup-kube:
	$(MAKE) -C $(REPO_ROOT) kube-setup
