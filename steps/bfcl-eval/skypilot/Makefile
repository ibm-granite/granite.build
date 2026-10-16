STEP_NAME := bfcl-eval

# This is an image step (a Dockerfile sits beside this file), and common.mk makes REGISTRY
# mandatory with no default for those — `make image` errors out rather than guessing where
# to push. A placeholder here so the target names something, matching steps/eval; override
# it for a real build:
#
#   make image REGISTRY=quay.io/<your-org>
#
# NOTE FOR WHOEVER BUILDS THIS FIRST: no image has ever been built from this Dockerfile.
# It was authored on a host with no /etc/subuid entry and no registry credentials, so
# `make image`, `make publish-step` (which refuses until the image exists on the registry)
# and test/lsf/ have never run. See README.md for the rootless-podman invocation that got
# the layers to build there, and treat the first push as unproven.
REGISTRY ?= quay.io/your-org

include ../../common.mk

# ADDITIVE, not an override of common.mk's `test` (which depends on `space image`, i.e. a
# multi-GB torch+vLLM build plus a repo-root .venv, just to run pure-python unit tests).
# This is the loop to use while editing src/ or the template:
#
#   make unit-tests
#
# Runs in a throwaway uv environment so it needs nothing installed and nothing in the
# repo root. The tests that import the bfcl_eval harness itself are skipped by
# test/conftest.py when it is absent, which it is here — see README.md.
#
# pydantic is here because test_step_template.py asserts `type` against the real
# gbcommon.types.stepconfig.StepType enum (as every ported step's suite does), and
# importing it pulls pydantic. The repo-root pyproject.toml puts `src` on pytest's
# pythonpath, so gbcommon resolves from this subdirectory without an install.
#
# WHAT THIS TARGET DOES NOT RUN: test/lsf/. It subclasses libgbtest's build runner, whose
# dependency chain is the repo's whole [dev] extra — psutil, then filelock, then whatever is
# next (both measured, one per run). Naming them as `--with` flags here would be chasing a
# set one element at a time, and the set is already written down in pyproject.toml, so
# test/conftest.py skips that directory when the harness does not import. Run the per-cluster
# tests through the sanctioned route instead — `make -C ../../.. venv` then `make test`, or
# `.venv/bin/python -m pytest test` — where they collect and report their own four gates as
# skips (measured: 57 passed, 5 skipped here vs 57 passed, 7 skipped there).
.PHONY: unit-tests
unit-tests:
	uv run --no-project --with pytest --with pyyaml --with pydantic pytest test -q
