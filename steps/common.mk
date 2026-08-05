# steps/common.mk — shared conventions for generating Granite.build step directories.
#
# Each step's environment dir (e.g. steps/byoc/skypilot) has a thin Makefile that
# sets a few variables and then `include`s this file. It provides the canonical
# targets used across all steps:
#
#   image         build the custom image from ./Dockerfile for $(PLATFORM)
#                 (image steps only; cross-builds on an Apple Silicon host)
#   publish-image push the image to $(REGISTRY)                  (image steps only)
#   step          render step-template.yaml -> step/step.yaml and bundle src/
#   all           end-to-end: image + publish-image + step (image steps),
#                 or just step (non-image steps)
#   clean         remove the generated step/ bundle
#   help          display the shared steps/README.md
#   check-tools   verify required tooling (envsubst) is installed
#
# Variables an includer MUST/MAY set BEFORE the include:
#   STEP_NAME    (required) logical step name, e.g. "byoc"
#   REGISTRY     (required for image steps) image registry+namespace, e.g.
#                quay.io/my-org — no default; error if unset for an image step
#
# Whether a step builds a custom image is auto-detected: if a Dockerfile sits
# next to the including Makefile, `image`/`publish-image` are real and the
# generated step.yaml gets an image_id; otherwise they are no-ops (the byoc
# case, which runs in a public image). No flag to keep in sync.
#
# Variables a caller may override on the command line or in the environment:
#   DOCKER      container tool          (default: podman)
#   DOCKERFILE  Dockerfile to detect/build (default: Dockerfile)
#   PLATFORM    target build platform   (default: linux/amd64)
#   IMAGE_NAME  image repository name   (default: gb-step-$(STEP_NAME))
#   IMAGE_TAG   image tag               (default: git short SHA, else "latest")
#
# The rendered step/ directory is the deployable bundle; reference it from a
# build.yaml by an absolute file:// URI, e.g.
#   step_uri: file:///abs/path/steps/byoc/skypilot/step

# ---- Required / defaulted inputs -------------------------------------------

ifndef STEP_NAME
$(error STEP_NAME must be set by the including Makefile before 'include ../../common.mk')
endif

# A step builds a custom image iff a Dockerfile is present next to its Makefile.
# STEP_USES_IMAGE is derived from that presence, so an includer never sets it.
DOCKERFILE      ?= Dockerfile
STEP_USES_IMAGE := $(if $(wildcard $(DOCKERFILE)),true,false)

DOCKER     ?= podman

# Target platform for the image build. Defaults to linux/amd64 because SkyPilot
# provisions x86 nodes; on an Apple Silicon host this cross-builds. Both
# `podman build` and `docker build` (BuildKit) accept --platform and keep the
# result in the local image store.
PLATFORM   ?= linux/amd64

# Image registry + namespace (e.g. quay.io/my-org). This has NO default: an image
# step must declare where its image is published. The including Makefile is
# expected to set it (e.g. 'REGISTRY := quay.io/my-org'); it can still be
# overridden on the command line (make ... REGISTRY=quay.io/my-org). Enforced for
# image steps only — byoc-style steps build no image and need no registry.
ifeq ($(STEP_USES_IMAGE),true)
ifeq ($(strip $(REGISTRY)),)
$(error REGISTRY is not set. Define it in the including Makefile (e.g. 'REGISTRY := quay.io/my-org') or pass it on the command line: make ... REGISTRY=quay.io/my-org)
endif
endif

IMAGE_NAME ?= gb-step-$(STEP_NAME)
# Defaults to the git short SHA (else "latest") so iterative builds get a unique,
# traceable tag without ceremony. Override per release, e.g. IMAGE_TAG=0.1.0.
IMAGE_TAG  ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo latest)

# Registry host (e.g. quay.io) parsed from REGISTRY (quay.io/my-org) — this is
# what `docker/podman login` authenticates against. Used only by publish-image's
# optional non-interactive login below.
REGISTRY_HOST := $(firstword $(subst /, ,$(REGISTRY)))

# IMAGE_REF is the fully qualified image reference (e.g. quay.io/org/img:tag)
# that `make step` substitutes into the template's ${IMAGE_REF} token. The
# template wraps it as SkyPilot `docker:${IMAGE_REF}`. It is empty for non-image
# steps, whose templates carry no ${IMAGE_REF} token and select their image via
# runtime Jinja instead. Both the image/publish-image recipes and the render use
# this same value, so there is one source of truth.
ifeq ($(STEP_USES_IMAGE),true)
IMAGE_REF := $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
else
IMAGE_REF :=
endif

STEP_DIR = step
SRC_DIR  = src
TEMPLATE = step-template.yaml

# Location of this common.mk (and the shared README beside it), resolved from
# MAKEFILE_LIST at parse time. common.mk is the last-included file when a step
# Makefile pulls it in, so `lastword` names it regardless of the caller's cwd.
COMMON_MK_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
README        := $(COMMON_MK_DIR)README.md

.PHONY: all help image publish-image step clean check-tools

# ---- Default goal ----------------------------------------------------------
# `step` deliberately does NOT depend on image/publish-image so it stays a cheap,
# offline render during iteration. `all` runs the full pipeline for image steps.
ifeq ($(STEP_USES_IMAGE),true)
all: image publish-image step
else
all: step
endif

# ---- Help ------------------------------------------------------------------
# Render the shared README (steps/README.md, beside common.mk). Uses a markdown
# renderer if one is installed (glow/mdcat/bat), else falls back to plain cat.
help:
	@if command -v glow >/dev/null 2>&1; then glow "$(README)"; \
	elif command -v mdcat >/dev/null 2>&1; then mdcat "$(README)"; \
	elif command -v bat >/dev/null 2>&1; then bat --language=markdown --style=plain "$(README)"; \
	else cat "$(README)"; fi

# ---- Image build / publish (image steps only) ------------------------------

image:
ifeq ($(STEP_USES_IMAGE),true)
	$(DOCKER) build --platform $(PLATFORM) -f $(DOCKERFILE) . -t $(IMAGE_REF)
else
	@echo "[$(STEP_NAME)] no $(DOCKERFILE) found; no custom image to build."
endif

# By default this assumes you have already authenticated to the registry out of
# band (`podman login $(REGISTRY_HOST)` / `docker login`), so no credentials live
# in this Makefile. For non-interactive use (CI), export REGISTRY_USER and
# REGISTRY_PASSWORD (a robot-account token) in the ENVIRONMENT — not as make
# variables — and publish-image logs in first, piping the secret via
# --password-stdin so it never appears in `ps`, make's echo, or shell history.
publish-image:
ifeq ($(STEP_USES_IMAGE),true)
	@if [ -n "$${REGISTRY_USER:-}" ] && [ -n "$${REGISTRY_PASSWORD:-}" ]; then \
		echo "[$(STEP_NAME)] logging in to $(REGISTRY_HOST) as $$REGISTRY_USER"; \
		printf '%s' "$$REGISTRY_PASSWORD" | $(DOCKER) login "$(REGISTRY_HOST)" -u "$$REGISTRY_USER" --password-stdin; \
	fi
	$(DOCKER) push $(IMAGE_REF)
else
	@echo "[$(STEP_NAME)] no $(DOCKERFILE) found; nothing to publish."
endif

# ---- Render + bundle -------------------------------------------------------

# Render step-template.yaml into step/step.yaml, substituting ONLY $IMAGE_REF so
# runtime Jinja ({{ ... }}) and shell expansions (${VAR}, $(cmd)) in the run/
# setup blocks pass through untouched. Then copy src/ into the bundle if present
# and non-empty.
step: check-tools
	@mkdir -p $(STEP_DIR)
	IMAGE_REF='$(IMAGE_REF)' envsubst '$$IMAGE_REF' < $(TEMPLATE) > $(STEP_DIR)/step.yaml
	@if [ -d "$(SRC_DIR)" ] && [ -n "$$(ls -A $(SRC_DIR) 2>/dev/null)" ]; then \
		rm -rf "$(STEP_DIR)/$(SRC_DIR)"; \
		cp -R "$(SRC_DIR)" "$(STEP_DIR)/$(SRC_DIR)"; \
		echo "[$(STEP_NAME)] bundled $(SRC_DIR)/ into $(STEP_DIR)/"; \
	fi
	@echo "[$(STEP_NAME)] wrote $(STEP_DIR)/step.yaml (image_ref='$(IMAGE_REF)')"

clean:
	rm -rf $(STEP_DIR)

# ---- Tooling check ---------------------------------------------------------

check-tools:
	@command -v envsubst >/dev/null 2>&1 || { \
		echo "ERROR: 'envsubst' not found. Install it (macOS: brew install gettext; Debian/Ubuntu: apt-get install gettext-base)."; \
		exit 1; }
