# steps/common.mk — shared conventions for generating Granite.build step directories.
#
# Each step's environment dir (e.g. steps/byoc/skypilot) has a thin Makefile that
# sets a few variables and then `include`s this file. It provides the canonical
# targets used across all steps:
#
#   image         build the custom image from ./Dockerfile for $(PLATFORM)
#                 (custom image steps only; cross-builds on an Apple Silicon host)
#   publish-image push the image to $(REGISTRY)                  (custom image steps only)
#   space         render step-template.yaml + a generated space.yaml into a
#                 self-contained Space directory ($(SPACE_DIR)/) and bundle src/
#   publish       render the step into the committed assets tree
#                 (configurations/assets/environments/<env>/steps/<name>/) and copy
#                 its per-cluster build tests into the parallel top-level trees
#                 test/steps/<name>/<env>/<cluster>/ (tests) and
#                 test-data/steps/<name>/<env>/<cluster>/ (fixtures, with their
#                 space_uri repointed at the published step). NOT part of all.
#   all           end-to-end: image + publish-image + space (image steps),
#                 or just space (non-image steps)
#   test          render the Space (via `space`) and build the image locally
#                 (via `image`; no-op/no publish for non-image steps), then run
#                 the step's Python tests in $(TEST_DIR)/ (adjacent to src/,
#                 organised in per-cluster subdirs; not bundled)
#   clean         remove the generated Space directory
#   help          list the targets and point at the shared steps/README.md
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
#   STEP_ENV    step's environment segment      (default: this Makefile's dir name)
#   PUBLISH_STEP_DIR  where `publish` renders the step (default: the env-nested
#                     assets path configurations/assets/environments/<env>/steps/<name>)
#   PUBLISH_TEST_DIR  where `publish` copies build tests (default: test/steps/<name>/<env>)
#   PUBLISH_TESTDATA_DIR  where `publish` copies fixtures (default: test-data/steps/<name>/<env>)
#
# The rendered $(SPACE_DIR)/ directory (named "space" by default) is a
# self-contained Granite.build Space. Point a build test / build.yaml at it via
#   space_uri: <path to steps/byoc/skypilot/space>
# and reference the step by the stable URI  step_uri: space://steps/$(STEP_NAME)
# — everything else (environments, monitors, other steps) resolves through the
# generated space.yaml's base_uris chain to $(SPACE_BASE_URI).

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
# custom image steps only — byoc-style steps build no image and need no registry.
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
# that `make space` substitutes into the template's ${IMAGE_REF} token. The
# template wraps it as SkyPilot `docker:${IMAGE_REF}`. It is empty for non-image
# steps, whose templates carry no ${IMAGE_REF} token and select their image via
# runtime Jinja instead. Both the image/publish-image recipes and the render use
# this same value, so there is one source of truth.
ifeq ($(STEP_USES_IMAGE),true)
IMAGE_REF := $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
else
IMAGE_REF :=
endif

# Name of the generated Space directory, created as ./$(SPACE_DIR) next to the
# Makefile and referenced from a build test/build.yaml by `space_uri` plus
# `space://steps/$(STEP_NAME)`. It is a self-contained Space: a generated
# space.yaml (whose base_uris chain to $(SPACE_BASE_URI)) plus
# steps/$(STEP_NAME)/step.yaml and the bundled src/. Overridable on the command
# line or in the including Makefile.
SPACE_DIR  ?= space
# `name:` written into the generated space.yaml. Defaults to the step name.
SPACE_NAME ?= $(STEP_NAME)
# When non-empty, a variables.DEFAULT_ENVIRONMENT entry is written into the
# generated space.yaml (e.g. DEFAULT_ENVIRONMENT=skypilot/slurm). Empty by
# default — builds that set environment_uri explicitly do not need it.
DEFAULT_ENVIRONMENT ?=
SRC_DIR  = src
TEMPLATE = step-template.yaml

# Test directory (adjacent to src/, NOT bundled into the deployable step) and the
# interpreter used to run its Python tests via `make test`. Overridable.
TEST_DIR ?= test
PYTHON   ?= python3

# Location of this common.mk (and the shared README beside it), resolved from
# MAKEFILE_LIST at parse time. common.mk is the last-included file when a step
# Makefile pulls it in, so `lastword` names it regardless of the caller's cwd.
COMMON_MK_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
README        := $(COMMON_MK_DIR)README.md

# Relative file:// base_uri written into the generated space.yaml so space://
# URIs (environments, monitors, other steps) resolve into the shared assets
# directory. A relative base_uri resolves against the space.yaml's OWN directory
# (see build/space.py), which keeps the generated Space portable — it does not
# bake in an absolute checkout path.
#
# The space.yaml sits at $(SPACE_DIR)/space.yaml under the step env dir
# (steps/<step>/<env>), and configurations/assets is at the repo root. So the
# path back is: one `..` for $(SPACE_DIR) (a single-level dir by default), then
# STEP_TO_ROOT to climb from the step env dir to the repo root. common.mk is
# included as ../../common.mk, so $(COMMON_MK_DIR) is that ../../ and
# $(COMMON_MK_DIR).. is the repo root relative to the step env dir.
# For the default layout this yields file://../../../../configurations/assets.
# Override SPACE_BASE_URI if you nest SPACE_DIR or move the assets tree.
STEP_TO_ROOT := $(COMMON_MK_DIR)..
SPACE_BASE_URI ?= file://../$(STEP_TO_ROOT)/configurations/assets

# ---- Publish (render into the committed assets tree + copy build tests) ------
# `make publish` promotes a step from this authoring dir into the repo's shared,
# committed assets so builds can reference it by space://steps/$(STEP_NAME), and
# copies the step's per-cluster build tests into the top-level test/steps/ tree so
# they are discoverable/runnable from VSCode (Mode 2 — against the published step).
#
# STEP_ENV is the step's environment segment, taken from the Makefile's own dir
# name (steps/<step>/<env> -> <env>, e.g. skypilot). The step is published
# env-nested (the dominant assets convention) so its space://steps/<name> URI
# resolves for that environment family; ASSETS_DIR/PUBLISH_TEST_DIR are absolute
# repo paths derived from COMMON_MK_DIR (= steps/), so `..` is the repo root.
STEP_ENV         ?= $(notdir $(CURDIR))
ASSETS_DIR       ?= $(abspath $(COMMON_MK_DIR)..)/configurations/assets
PUBLISH_STEP_DIR ?= $(ASSETS_DIR)/environments/$(STEP_ENV)/steps/$(STEP_NAME)
# The copied build tests and their fixtures follow the repo's parallel
# test/ <-> test-data/ convention (resolved by libgbtest.get_test_data_dir_for):
# the per-cluster test dir lands under test/steps/<name>/<env>/<cluster>/ and its
# fixtures under the mirrored test-data/steps/<name>/<env>/<cluster>/. The step's
# own `test/<cluster>/` nesting is flattened away on copy so the published layout
# matches the top-level convention (test/<path> mirrors test-data/<path>).
PUBLISH_TEST_DIR     ?= $(abspath $(COMMON_MK_DIR)..)/test/steps/$(STEP_NAME)/$(STEP_ENV)
PUBLISH_TESTDATA_DIR ?= $(abspath $(COMMON_MK_DIR)..)/test-data/steps/$(STEP_NAME)/$(STEP_ENV)
# space_uri written into each copied buildtest.yaml so the Mode-2 (published) test
# resolves the step through the shared space configurations/spaces/local (which
# chains to configurations/assets). It is RELATIVE to the copied file's own dir,
# test-data/steps/<step>/<env>/<cluster>/ — always five single-name segments
# below the repo root — so it stays portable (no absolute checkout path is baked
# in) and from_yaml resolves it against the yaml's directory (see build/space.py).
MODE2_SPACE_URI  ?= ../../../../../configurations/spaces/local

.PHONY: all help image publish-image space publish test clean

# ---- Default goal ----------------------------------------------------------
# `space` deliberately does NOT depend on image/publish-image so it stays a cheap,
# offline render during iteration. `all` runs the full pipeline for image steps.
ifeq ($(STEP_USES_IMAGE),true)
all: image publish-image space
else
all: space
endif

# ---- Help ------------------------------------------------------------------
# List the available targets with a one-line description and point at the shared
# README for full documentation (rather than rendering it here).
help:
	@echo "Step '$(STEP_NAME)'"
	@echo
	@echo "Targets:"
	@echo "  space          render step.yaml + a space.yaml into the Space $(SPACE_DIR)/ (offline)"
	@echo "  publish        render the step into the assets tree + copy build tests into test/steps/ (not in all)"
	@echo "  image          build the image from $(DOCKERFILE) for $(PLATFORM)  (no-op when no Dockerfile is present)"
	@echo "  publish-image  push the image to the registry                 (no-op when no Dockerfile is present)"
	@echo "  all            $(if $(filter true,$(STEP_USES_IMAGE)),image + publish-image + space,space (this step builds no image))"
	@echo "  test           render the Space + build the image locally, then run the step's tests in $(TEST_DIR)/ (no-op when absent)"
	@echo "  clean          remove the generated Space $(SPACE_DIR)/"
	@echo "  help           show this message"
	@echo
	@echo "Common overrides: make all REGISTRY=quay.io/myorg IMAGE_TAG=0.1.0"
	@echo "Full documentation: $(README)"

# ---- Image build / publish (custom image steps only) ------------------------------

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

# ---- Rendering helper -------------------------------------------------------

# render-step-template — render step-template.yaml to the file named by $(1),
# substituting ONLY the literal ${IMAGE_REF} token (empty for non-image steps).
# Everything else passes through verbatim: runtime Jinja ({{ ... }}) and shell
# expansions (${VAR}, $(cmd)) in run:/setup: blocks. Uses sed (POSIX, always
# present) rather than envsubst, so no gettext install is required. The image
# ref is escaped for sed's replacement text (the # delimiter, & and \) so a
# value containing any of them cannot corrupt the substitution.
define render-step-template
	@ref=$$(printf '%s' '$(IMAGE_REF)' | sed 's/[#&\\]/\\&/g'); \
	sed "s#\$${IMAGE_REF}#$$ref#g" "$(TEMPLATE)" > "$(1)"
endef

# ---- Render Space ----------------------------------------------------------

# Render a self-contained Space into $(SPACE_DIR)/:
#   $(SPACE_DIR)/steps/$(STEP_NAME)/step.yaml   (+ bundled src/)
#   $(SPACE_DIR)/space.yaml                       (base_uris -> $(SPACE_BASE_URI))
# step-template.yaml is rendered substituting ONLY ${IMAGE_REF} so runtime Jinja
# ({{ ... }}) and shell expansions (${VAR}, $(cmd)) in the run/setup blocks pass
# through untouched (see render-step-template above). The generated space.yaml's
# own directory is the first base_uri, so `space://steps/$(STEP_NAME)` resolves
# here and everything else falls through to $(SPACE_BASE_URI).
space:
	@mkdir -p $(SPACE_DIR)/steps/$(STEP_NAME)
	$(call render-step-template,$(SPACE_DIR)/steps/$(STEP_NAME)/step.yaml)
	@if [ -d "$(SRC_DIR)" ] && [ -n "$$(ls -A $(SRC_DIR) 2>/dev/null)" ]; then \
		rm -rf "$(SPACE_DIR)/steps/$(STEP_NAME)/$(SRC_DIR)"; \
		cp -R "$(SRC_DIR)" "$(SPACE_DIR)/steps/$(STEP_NAME)/$(SRC_DIR)"; \
		echo "[$(STEP_NAME)] bundled $(SRC_DIR)/ into $(SPACE_DIR)/steps/$(STEP_NAME)/"; \
	fi
	@printf 'name: %s\nsecret_manager:\n  type: local\n  config: {}\nbase_uris:\n  - %s\n' \
		"$(SPACE_NAME)" "$(SPACE_BASE_URI)" > $(SPACE_DIR)/space.yaml
	@if [ -n "$(DEFAULT_ENVIRONMENT)" ]; then \
		printf 'variables:\n  DEFAULT_ENVIRONMENT: %s\n' "$(DEFAULT_ENVIRONMENT)" >> $(SPACE_DIR)/space.yaml; \
	fi
	@echo "[$(STEP_NAME)] wrote Space $(SPACE_DIR)/ (step space://steps/$(STEP_NAME); base_uri $(SPACE_BASE_URI); image_ref='$(IMAGE_REF)')"

# ---- Publish ---------------------------------------------------------------

# Promote the step into the committed assets tree and copy its build tests into
# the repo's parallel top-level test/ <-> test-data/ trees:
#   $(PUBLISH_STEP_DIR)/step.yaml               (+ bundled src/)   — the published step
#   $(PUBLISH_TEST_DIR)/<cluster>/              — per-cluster build tests (copied)
#   $(PUBLISH_TESTDATA_DIR)/<cluster>/          — their fixtures, with space_uri rewritten
#
# The step's own `test/<cluster>/` nesting is flattened on copy (the inner `test`
# segment is dropped) so the published test lands at test/steps/<name>/<env>/<cluster>/
# and its fixtures at the mirrored test-data/steps/<name>/<env>/<cluster>/. The test
# locates its fixtures with libgbtest.get_test_data_dir_for(__file__), which maps a
# `test/`-rooted path to the parallel `test-data/` one — so the SAME test file
# resolves in both homes (Mode 1 co-located beside the step, Mode 2 here).
#
# step.yaml is rendered exactly as `space` does (render-step-template substitutes
# ONLY ${IMAGE_REF}, so runtime Jinja/${VAR} pass through). Only the per-cluster
# SUBDIRS of $(TEST_DIR) are copied — the step's build/integration tests — so
# loose payload/unit tests in $(TEST_DIR)/ (which import from src/) and src/
# itself are intentionally NOT copied; those stay Mode-1 only (run via `make
# test`). The copied buildtest.yaml files have their space_uri rewritten to
# $(MODE2_SPACE_URI) so the Mode-2 tests resolve the PUBLISHED step via
# configurations/spaces/local instead of the local space/. Deliberately NOT part
# of `all`: publish writes the committed assets tree.
publish:
	@mkdir -p $(PUBLISH_STEP_DIR)
	$(call render-step-template,$(PUBLISH_STEP_DIR)/step.yaml)
	@if [ -d "$(SRC_DIR)" ] && [ -n "$$(ls -A $(SRC_DIR) 2>/dev/null)" ]; then \
		rm -rf "$(PUBLISH_STEP_DIR)/$(SRC_DIR)"; \
		cp -R "$(SRC_DIR)" "$(PUBLISH_STEP_DIR)/$(SRC_DIR)"; \
	fi
	@rm -rf "$(PUBLISH_TEST_DIR)" "$(PUBLISH_TESTDATA_DIR)"
	@mkdir -p "$(PUBLISH_TEST_DIR)" "$(PUBLISH_TESTDATA_DIR)"
	@copied=0; for d in $(TEST_DIR)/*/; do \
		[ -d "$$d" ] || continue; \
		cp -R "$${d%/}" "$(PUBLISH_TEST_DIR)/"; copied=1; \
	done; \
	if [ "$$copied" = 0 ]; then \
		echo "[$(STEP_NAME)] WARNING: no per-cluster build-test subdirs in $(TEST_DIR)/ to publish"; \
	fi
	@if [ -d test-data ]; then \
		for d in test-data/*/; do \
			[ -d "$$d" ] || continue; \
			cp -R "$${d%/}" "$(PUBLISH_TESTDATA_DIR)/"; \
		done; \
	fi
	@find "$(PUBLISH_STEP_DIR)" "$(PUBLISH_TEST_DIR)" "$(PUBLISH_TESTDATA_DIR)" \
		-name __pycache__ -type d -prune -exec rm -rf {} +
	@find "$(PUBLISH_TESTDATA_DIR)" -name buildtest.yaml -exec \
		sed -i.bak 's#^space_uri:.*#space_uri: $(MODE2_SPACE_URI)#' {} \; -exec rm -f {}.bak \;
	@echo "[$(STEP_NAME)] published step -> $(PUBLISH_STEP_DIR)"
	@echo "[$(STEP_NAME)] copied build tests -> $(PUBLISH_TEST_DIR)"
	@echo "[$(STEP_NAME)] copied fixtures   -> $(PUBLISH_TESTDATA_DIR) (space_uri -> $(MODE2_SPACE_URI))"

# ---- Tests -----------------------------------------------------------------

# Run the step's Python tests with pytest. Tests live in $(TEST_DIR)/ next to
# src/ and are NOT bundled into the deployable step. Organise them in per-cluster
# subdirs (e.g. $(TEST_DIR)/slurm, $(TEST_DIR)/docker) — pytest recurses, so the
# whole tree runs. $(SRC_DIR) is placed on PYTHONPATH so tests can import the
# step's modules directly. No-op (with a note) when $(TEST_DIR) is absent/empty.
#
# Prerequisites make `make test` the single entry point that guarantees a
# ready-to-run step:
#   * `space` renders the Space ($(SPACE_DIR)/) so a build test can reference it
#     (space_uri) instead of rendering it itself.
#   * `image` builds the custom image LOCALLY for image steps (no-op otherwise,
#     no publish) so a local-Docker build test finds it in the local image store
#     (the docker environment's pull_policy is if-not-present). Cross-builds for
#     $(PLATFORM); layer-cached, so repeat runs are cheap.
# Both are cheap/offline and harmless for steps whose tests don't touch them.
test: space image
	@if [ -d "$(TEST_DIR)" ] && [ -n "$$(ls -A $(TEST_DIR) 2>/dev/null)" ]; then \
		PYTHONPATH="$(SRC_DIR)$${PYTHONPATH:+:$$PYTHONPATH}" $(PYTHON) -m pytest $(TEST_DIR); \
	else \
		echo "[$(STEP_NAME)] no tests in $(TEST_DIR)/; nothing to run."; \
	fi

clean:
	rm -rf $(SPACE_DIR)
