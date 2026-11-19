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
#   make scan-image                             trivy the built image (run before publishing)
#   make publish-image                          scan, then push (refuses to replace a tag)
#   make image IMAGE_VERSION=lme-a9c6e2b8-1     a release tag instead of the git SHA
#   make test                                   the pin + patch tests (tests/)
#   make print-image                            just print the ref
#
# publish-image needs a prior `podman login quay.io/granitebuild`, or REGISTRY_USER and
# REGISTRY_PASSWORD in the environment.
#
# BUILD A PUBLISHED IMAGE WITH --no-cache: `make image BUILD_ARGS=--no-cache`. A cached
# build reuses an old `apt-get upgrade` layer and ships un-upgraded packages that the
# scan below would otherwise have caught.
#
# SCANNING. scan-image catches before the push most of what quay.io's own scan reports,
# so an image is not rejected at quay and then rebuilt. It runs trivy on the built image
# (OS packages from the base and apt, the Python venv, the uv binary), saved to a tar so
# no registry or daemon socket is needed. It prints the fixable HIGH and above findings
# as a report and FAILS on a fixable one at SCAN_SEVERITY (default CRITICAL). Unfixable
# findings never fail it. Accepted exceptions live in scan-ignore beside this file, one
# advisory id per line with the reason and the issue that removes it; a listed finding is
# printed as IGNORED. quay (Clair) can still flag a package trivy does not, so check
# quay's scan of the pushed tag too. trivy runs as a pinned container (no local install)
# with its database cached under SCAN_CACHE.
#
# publish-image runs scan-image first and pushes only if it passes, so an image is never
# pushed unscanned. SKIP_SCAN=true (exactly `true`) skips it, e.g. to push an image you
# scanned in this shell a minute ago.
#
# NO scan-lock/fix-locks HERE, unlike steps/train/distill. Those scan and constrain a
# REPO-CONTROLLED images/<x>/uv.lock; this image has none -- it runs `uv sync --locked`
# against upstream lm-engine's own lock inside the clone at /app/src, so there is nothing
# in this tree to scan before a build or to add a constraint to. trivy's venv scan of the
# built image covers the installed Python set; what it does not give is OSV (quay's source
# for Python advisories) before the push. Closing that needs a constraint mechanism over
# the upstream lock, not a port of distill's fix-locks.py.

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

# Scanning. Names and defaults match steps/train/distill/Makefile so that lifting both
# into steps/common.mk later is mechanical. The scanner is pinned by version, not
# `latest`: a floating scanner makes a clean scan unreproducible.
SCAN_SEVERITY ?= CRITICAL
SCAN_CACHE    ?= $(HOME)/.cache/gb-image-scan
TRIVY_IMAGE   ?= ghcr.io/aquasecurity/trivy:0.76.2
OSV_IMAGE     ?= ghcr.io/google/osv-scanner:v2.7.1
# `make relock` runs the pinned uv inside this, because `uv lock` needs an interpreter
# and the uv image is distroless. Same digest as the recipes' REPORT_IMAGE.
PYTHON_IMAGE  ?= docker.io/library/python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b
SKIP_SCAN     ?= false
# The uv that `scan-lock` exports the lock with, read out of Dockerfile.gpu so the two
# cannot drift: a different uv can resolve a lock differently, and then the set that is
# scanned is not the set that is installed.
UV_IMAGE       = ghcr.io/astral-sh/uv:$(shell awk -F= '/^ARG UV_VERSION=/{print $$2}' $(HERE)Dockerfile.gpu)@$(shell awk -F= '/^ARG UV_DIGEST=/{print $$2}' $(HERE)Dockerfile.gpu)
# Must match Dockerfile.gpu's `uv sync --extra cuda`: the export has to name the same
# extras, or the scan covers a different set than the build installs.
LOCK_EXTRAS   ?= cuda

PYTEST_ARGS ?=

.PHONY: help image check-image scan-lock scan-image relock publish-image test check-patches print-image

help:
	@echo "Targets: image, check-image, scan-lock, scan-image, relock, publish-image, test, check-patches, print-image"
	@echo "IMAGE_REF = $(IMAGE_REF)"

print-image:
	@echo "$(IMAGE_REF)"

image:
	$(DOCKER) build --platform $(PLATFORM) $(BUILD_ARGS) -f $(HERE)Dockerfile.gpu $(HERE) -t $(IMAGE_REF)

# CPU-safe: imports, versions, and that the binaries the steps shell out to are there.
check-image:
	$(DOCKER) run --rm $(IMAGE_REF) bash /app/check-image.sh

# scan-lock needs NO BUILD, which is the point: an image build is long, and a lock that
# will fail the gate should fail it first. It scans lock/, the VENDORED metadata the
# Dockerfile installs from, so the set scanned here is the set the image gets.
#
# `uv export` turns the lock into the exact set `uv sync --extra $(LOCK_EXTRAS)
# --no-install-project` installs. Exporting rather than scanning uv.lock directly is
# deliberate: the lock is universal and carries every extra, and findings behind extras
# this image never installs are noise in a gate -- which is how gates get switched off.
#
# --frozen, not --locked: --locked revalidates the lock against the project, which sends
# uv looking for an interpreter, and the pinned uv image is distroless (no /bin/sh, so
# libc detection fails). `make relock` is what revalidates, and tests/test_vendored_lock.py
# is what proves lock/pyproject.toml is still upstream's.
#
# osv-scanner exits 1 on ANY finding and has no severity filter, so its JSON goes through
# scripts/osv-gate.py, which prints the HIGH and above findings with their fix version
# and fails only on a fixable one at SCAN_SEVERITY, like trivy's gate below. Exit codes
# other than 0/1 (network, bad lockfile) fail as themselves.
scan-lock:
	@echo "[scan] lock/ as installed (--extra $(LOCK_EXTRAS)) (gate: fixable $(SCAN_SEVERITY))"
	@work=$$(mktemp -d) || exit 1; \
	trap 'rm -rf "$$work"' EXIT INT TERM; \
	cp $(HERE)lock/pyproject.toml $(HERE)lock/uv.lock "$$work/" || exit 1; \
	$(DOCKER) run --rm -v "$$work:/src:Z" -w /src $(UV_IMAGE) \
		export --frozen --extra $(LOCK_EXTRAS) --no-emit-project --no-hashes --no-dev \
		--no-python-downloads -o /src/requirements.txt >/dev/null || \
		{ echo "[scan] uv export failed; see the error above" >&2; exit 1; }; \
	$(DOCKER) run --rm -v "$$work:/src:ro,Z" $(OSV_IMAGE) \
		scan source --lockfile /src/requirements.txt --format json > "$$work/osv.json"; rc=$$?; \
	if [ $$rc -ne 0 ] && [ $$rc -ne 1 ]; then echo "[scan] osv-scanner failed (exit $$rc)" >&2; exit $$rc; fi; \
	ign=""; if [ -f $(HERE)scan-ignore ]; then ign="--ignore $(HERE)scan-ignore"; \
		echo "[scan] accepting the advisories listed in scan-ignore"; fi; \
	python3 $(HERE)scripts/osv-gate.py $(SCAN_SEVERITY) $$ign < "$$work/osv.json" || \
		{ echo "[scan] FAILED: lock/ has fixable $(SCAN_SEVERITY) vulnerabilities; see README.md 'Scanning'" >&2; exit 1; }

# Re-resolve lock/uv.lock from lock/pyproject.toml, after editing the security block or
# re-vendoring upstream's file for a new UPSTREAM sha. Needs network.
#
# The pinned uv is extracted and run inside a Python image rather than used directly:
# `uv lock` resolves, which needs an interpreter, and the uv image is distroless. Using
# the PINNED uv (not whatever is on PATH) matters -- a different uv can write a different
# lock format or resolve differently, and then the lock the build consumes is not the one
# that was scanned.
relock:
	@work=$$(mktemp -d) || exit 1; \
	trap 'rm -rf "$$work"' EXIT INT TERM; \
	cp $(HERE)lock/pyproject.toml $(HERE)lock/uv.lock "$$work/" || exit 1; \
	cid=$$($(DOCKER) create $(UV_IMAGE)) || exit 1; \
	$(DOCKER) cp "$$cid:/uv" "$$work/uv" >/dev/null && $(DOCKER) rm "$$cid" >/dev/null; \
	chmod +x "$$work/uv"; \
	echo "[relock] uv lock with the pinned uv, inside a python image"; \
	$(DOCKER) run --rm -v "$$work:/src:Z" -w /src -e UV_CACHE_DIR=/tmp/uvc $(PYTHON_IMAGE) \
		/src/uv lock --no-python-downloads || \
		{ echo "[relock] uv lock failed; see the error above" >&2; exit 1; }; \
	cp "$$work/uv.lock" $(HERE)lock/uv.lock; \
	echo "[relock] lock/uv.lock rewritten; review it with git diff, then run make scan-lock"

# The image is saved to a tar under SCAN_CACHE, scanned from there and the tar removed,
# so trivy needs neither the registry nor podman's socket. `image exists` is podman's;
# `image inspect` is the docker fallback, so this works under DOCKER=docker too.
#
# Two passes. The report pass lists the fixable HIGH and above findings and fails only if
# trivy ITSELF cannot run (pull, database download). The gate pass exits 3 on a fixable
# finding at SCAN_SEVERITY, which is what tells a finding apart from trivy failing -- a
# single --exit-code 1 pass would report both as the same thing.
scan-image: scan-lock
	@$(DOCKER) image exists $(IMAGE_REF) 2>/dev/null || $(DOCKER) image inspect $(IMAGE_REF) >/dev/null 2>&1 || \
		{ echo "[scan] $(IMAGE_REF) is not built: run make image first" >&2; exit 1; }
	@mkdir -p "$(SCAN_CACHE)/trivy"
	@tar="$(SCAN_CACHE)/lm-engine-gpu-$(IMAGE_VERSION).tar"; rm -f "$$tar"; \
	echo "[scan] saving $(IMAGE_REF)"; \
	$(DOCKER) save -o "$$tar" $(IMAGE_REF) || { rm -f "$$tar"; exit 1; }; \
	ign=""; if [ -f $(HERE)scan-ignore ]; then \
		ign="-v $(HERE)scan-ignore:/scan/ignore:ro,Z"; \
		echo "[scan] accepting the advisories listed in scan-ignore"; fi; \
	trivy() { $(DOCKER) run --rm -v "$$tar:/scan/image.tar:ro,Z" -v "$(SCAN_CACHE)/trivy:/root/.cache/trivy:Z" $$ign \
		$(TRIVY_IMAGE) image --input /scan/image.tar --scanners vuln --ignore-unfixed --quiet \
		$${ign:+--ignorefile /scan/ignore} "$$@"; }; \
	echo "[scan] trivy report (fixable HIGH,CRITICAL) for $(IMAGE_REF)"; \
	trivy --severity HIGH,CRITICAL --format table; rc=$$?; \
	if [ $$rc -ne 0 ]; then rm -f "$$tar"; echo "[scan] trivy did not run (exit $$rc); see the error above" >&2; exit $$rc; fi; \
	echo "[scan] trivy gate (fixable $(SCAN_SEVERITY)) for $(IMAGE_REF)"; \
	trivy --severity $(SCAN_SEVERITY) --exit-code 3 --format table --skip-db-update; rc=$$?; \
	rm -f "$$tar"; \
	case $$rc in \
		0) ;; \
		3) echo "[scan] FAILED: $(IMAGE_REF) has fixable $(SCAN_SEVERITY) vulnerabilities; fix them before publish-image" >&2; exit 1 ;; \
		*) echo "[scan] trivy did not run (exit $$rc); see the error above" >&2; exit $$rc ;; \
	esac; \
	echo "[scan] $(IMAGE_REF): no fixable $(SCAN_SEVERITY) vulnerabilities"

publish-image: $(if $(filter true,$(SKIP_SCAN)),,scan-image)
	@$(if $(filter true,$(SKIP_SCAN)),echo "[lm-engine] SKIP_SCAN=true: pushing $(IMAGE_REF) without scanning it",true)
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
