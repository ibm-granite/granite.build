# Copyright 2026 Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Build the two images every distillation step runs on. The steps themselves build
# nothing (their Makefiles have no Dockerfile, so they build no image); their templates pin
# these images by tag. The build context is this directory: both Dockerfiles COPY the
# shared src/ package and each step's skypilot/src/ from here.
#
#   make images REGISTRY=us.icr.io/myorg                  build both
#   make image-gpu REGISTRY=...  /  make image-cpu ...    build one
#   make publish-images REGISTRY=...                      push both (needs a prior login)
#
# IMAGE_VERSION defaults to the git short SHA, so a tag names the commit it was built
# from. publish-* refuses to overwrite a tag already on the registry: a published tag
# must keep meaning one image. Pass PUBLISH_OVERWRITE=true to replace it on purpose.
# After publishing, repin the tag in every steps/train/distill/*/skypilot/step-template.yaml
# and run `make publish-step` in each step directory.

IMAGES := gpu cpu

REGISTRY          ?=
PLATFORM          ?= linux/amd64
DOCKER            ?= podman
IMAGE_VERSION     ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo latest)
PUBLISH_OVERWRITE ?= false
TAG_STATUS        := sh ../../scripts/registry-tag-status.sh

image_ref = $(REGISTRY)/gb-step-distillation-$(1):$(IMAGE_VERSION)

.PHONY: help images publish-images $(addprefix image-,$(IMAGES)) $(addprefix publish-image-,$(IMAGES))

help:
	@echo "Targets: images, publish-images, image-gpu, image-cpu, publish-image-gpu, publish-image-cpu"
	@echo "Set REGISTRY (e.g. us.icr.io/myorg); IMAGE_VERSION defaults to the git short SHA ($(IMAGE_VERSION))."

define require_registry
	@test -n "$(REGISTRY)" || { echo "REGISTRY is not set: make $@ REGISTRY=us.icr.io/myorg" >&2; exit 1; }
endef

images: $(addprefix image-,$(IMAGES))
publish-images: $(addprefix publish-image-,$(IMAGES))

$(addprefix image-,$(IMAGES)): image-%:
	$(require_registry)
	$(DOCKER) build --platform $(PLATFORM) -f images/$*/Dockerfile . -t $(call image_ref,$*)

$(addprefix publish-image-,$(IMAGES)): publish-image-%:
	$(require_registry)
	@if [ "$(PUBLISH_OVERWRITE)" != "true" ]; then \
		status=$$($(TAG_STATUS) "$(call image_ref,$*)"); \
		case "$$status" in \
			absent) ;; \
			present) echo "refusing: $(call image_ref,$*) is already published; bump IMAGE_VERSION or pass PUBLISH_OVERWRITE=true" >&2; exit 1 ;; \
			*) echo "refusing: could not check whether $(call image_ref,$*) is published (registry unreachable or not logged in)" >&2; exit 1 ;; \
		esac; \
	fi
	$(DOCKER) push $(call image_ref,$*)
