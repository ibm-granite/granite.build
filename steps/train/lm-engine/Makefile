# SPDX-License-Identifier: Apache-2.0
#
# Build the ONE lm-engine image shared by the steps in steps/train/lm-engine/<step>/<env>/
# (pretrain/skypilot and export/skypilot). Modelled on steps/eval/Makefile: the family image is
# built one level above its steps, and each step picks it up through its own
# config.image -- set by a recipe's LME_IMAGE parameter -- so no step Makefile builds
# an image.
#
# The recipe is Dockerfile.gpu; the build context is this directory (it needs UPSTREAM,
# patches/ and scripts/check-image.sh). Every external input is pinned there: the base
# by digest, uv and CPython by exact version, lm-engine by the UPSTREAM sha, and the
# flash-attn wheel by URL + sha256. See README.md to bump any of them.
#
#   make image                                  build $(IMAGE_REF)
#   make check-image                            run scripts/check-image.sh inside it
#   make publish-image                          push it (refuses to replace a tag)
#   make image IMAGE_VERSION=lme-a9c6e2b8-1     a release tag instead of the git SHA
#   make test                                   the pin + patch tests (tests/)
#   make print-image                            just print the ref
#
# publish-image needs a prior `podman login quay.io/granitebuild`, or REGISTRY_USER and
# REGISTRY_PASSWORD in the environment.

REGISTRY      ?= quay.io/granitebuild
IMAGE_NAME    := steps/train/lm-engine/images/gpu
PLATFORM      ?= linux/amd64
DOCKER        ?= podman
IMAGE_VERSION ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo latest)
IMAGE_REF      = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_VERSION)
# Extra flags for the image build, e.g. BUILD_ARGS='--no-cache'.
BUILD_ARGS    ?=

HERE      := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(HERE)../../..)
VENV_DIR  ?= $(REPO_ROOT)/.venv
# present / absent / unknown for an image ref (exit 0 / 1 / 2), shared with steps/common.mk.
TAG_STATUS := sh $(REPO_ROOT)/steps/scripts/registry-tag-status.sh

# A tag pins an image: SLURM's image cache keys its .sqsh by tag, so a re-pushed tag
# never reaches it. Bump IMAGE_VERSION instead, or set PUBLISH_OVERWRITE=true (exactly
# `true`) to replace the tag on purpose. A tag that cannot be checked is refused too,
# so a broken check never becomes a silent overwrite.
PUBLISH_OVERWRITE ?= false

PYTEST_ARGS ?=

.PHONY: help image check-image publish-image test check-patches print-image

help:
	@echo "Targets: image, check-image, publish-image, test, check-patches, print-image"
	@echo "IMAGE_REF = $(IMAGE_REF)"

print-image:
	@echo "$(IMAGE_REF)"

image:
	$(DOCKER) build --platform $(PLATFORM) $(BUILD_ARGS) -f $(HERE)Dockerfile.gpu $(HERE) -t $(IMAGE_REF)

# CPU-safe: imports, versions, and that the binaries the steps shell out to are there.
check-image:
	$(DOCKER) run --rm $(IMAGE_REF) bash /app/check-image.sh

publish-image:
	@if [ "$(PUBLISH_OVERWRITE)" = "true" ]; then \
		echo "[lm-engine] PUBLISH_OVERWRITE=true: pushing $(IMAGE_REF) without checking for an existing tag."; \
	else \
		status=$$($(TAG_STATUS) "$(IMAGE_REF)"); \
		case "$$status" in \
			absent) echo "[lm-engine] $(IMAGE_REF) is a new tag." ;; \
			present) echo "[lm-engine] ERROR: $(IMAGE_REF) already exists on the registry; a tag must keep pointing at one image. Bump IMAGE_VERSION, or pass PUBLISH_OVERWRITE=true to replace it on purpose." >&2; exit 1 ;; \
			*) echo "[lm-engine] ERROR: could not determine whether $(IMAGE_REF) already exists (registry unreachable, or it refused the check). Fix access and retry, or pass PUBLISH_OVERWRITE=true to push without the check." >&2; exit 1 ;; \
		esac; \
	fi
	$(DOCKER) push $(IMAGE_REF)

# The pin tests are offline; test_patches.py clones lm-engine and skips without network.
test:
	GB_ENVIRONMENT=$${GB_ENVIRONMENT:-STANDALONE} GB_HOME_DIR=$${GB_HOME_DIR:-$$(mktemp -d)} \
	  $(VENV_DIR)/bin/pytest $(HERE)tests $(PYTEST_ARGS)

check-patches:
	$(HERE)scripts/check-patches.sh
