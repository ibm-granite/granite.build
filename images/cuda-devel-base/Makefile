# Shared CUDA-devel base image for GPU steps across granite.build.
# Not a step (no step.yaml) -- see images/README.md and this dir's README.md.
#
#   make image           build locally for $(PLATFORM)
#   make publish-image   push $(IMAGE_REF) to $(REGISTRY)
#
# Deliberately standalone rather than built on steps/common.mk: that file's
# image/publish-image targets are wrapped in step-publishing machinery (Space
# rendering, step.yaml, publish-step) that doesn't apply to a bare base image.
# This Makefile mirrors the same build/push conventions directly.

DOCKER     ?= podman
PLATFORM   ?= linux/amd64
DOCKERFILE ?= Dockerfile

REGISTRY   ?= us.icr.io/cil15-shared-registry
IMAGE_NAME ?= gb-cuda-devel-base
# Defaults to the git short SHA (else "latest") so iterative builds get a
# unique, traceable tag without ceremony. Override per release, e.g.
# IMAGE_TAG=12.9.1-1.
IMAGE_TAG  ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo latest)
IMAGE_REF  := $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
REGISTRY_HOST := $(firstword $(subst /, ,$(REGISTRY)))

.PHONY: image publish-image help

help:
	@echo "images/cuda-devel-base"
	@echo
	@echo "Targets:"
	@echo "  image          build $(IMAGE_REF) for $(PLATFORM)"
	@echo "  publish-image  push $(IMAGE_REF) to $(REGISTRY_HOST)"
	@echo "  help           show this message"
	@echo
	@echo "Common overrides: make image REGISTRY=... IMAGE_TAG=..."

image:
	$(DOCKER) build --platform $(PLATFORM) -f $(DOCKERFILE) . -t $(IMAGE_REF)

# By default this assumes you have already authenticated to the registry out
# of band (`podman login $(REGISTRY_HOST)`). For non-interactive use, export
# REGISTRY_USER and REGISTRY_PASSWORD in the environment (not as make
# variables) -- publish-image pipes the secret via --password-stdin so it
# never appears in `ps`, make's echo, or shell history.
publish-image:
	@if [ -n "$${REGISTRY_USER:-}" ] && [ -n "$${REGISTRY_PASSWORD:-}" ]; then \
		echo "logging in to $(REGISTRY_HOST) as $$REGISTRY_USER"; \
		printf '%s' "$$REGISTRY_PASSWORD" | $(DOCKER) login "$(REGISTRY_HOST)" -u "$$REGISTRY_USER" --password-stdin; \
	fi
	$(DOCKER) push $(IMAGE_REF)
