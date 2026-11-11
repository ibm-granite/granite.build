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
#   make images REGISTRY=quay.io/granitebuild             build both
#   make image-gpu REGISTRY=...  /  make image-cpu ...    build one
#   make fix-locks                                        raise vulnerable lock packages to their fix
#   make scan-images REGISTRY=...                         scan both built images (run before publishing)
#   make publish-images REGISTRY=...                      push both (needs a prior login)
#
# scan-images catches before the push most of what quay.io's own scan reports, so an
# image is not rejected at quay and then rebuilt. It runs, for each image:
#   - osv-scanner on images/<x>/uv.lock: the exact Python set `uv sync --locked`
#     installs, against OSV (quay's source for Python advisories);
#   - trivy on the built image (OS packages from the base and apt, the Python venv,
#     the uv binary), saved to a tar so no registry or daemon socket is needed.
# Each FAILS on a fixable vulnerability at SCAN_SEVERITY (default CRITICAL; CVSS >= 9
# for OSV) and prints the fixable HIGH and above findings as a report. Unfixable findings (e.g. linux-libc-dev in the CUDA devel base) never fail
# it. quay (Clair) can still flag a package neither tool does, so check quay's scan
# of the pushed tag too. Both scanners run as pinned containers (no local install);
# trivy's database is cached under SCAN_CACHE. scan-locks alone needs no build.
#
# fix-locks (images/fix-locks.py) automates the lock half: for each fixable finding at
# SCAN_SEVERITY it adds `<pkg>>=<fixed version>` to [tool.uv] constraint-dependencies in
# images/<x>/pyproject.toml, runs `uv lock` (which moves only what a constraint forces),
# rescans and repeats. It never edits [project] dependencies: a package pinned there
# EXACTLY below its fix (vllm==0.11.0 in the GPU image) is reported BLOCKED, and a fix
# a transitive pin rules out is reported UNSATISFIABLE and left out. It needs `uv` on
# PATH. Review `git diff images/`, then rebuild and test: a fix can be a major version.
#
# The images are $(REGISTRY)/$(IMAGE_PREFIX)/<gpu|cpu>:$(IMAGE_VERSION). IMAGE_PREFIX
# defaults to this directory's path from the repo root plus images/, matching the
# Dockerfiles' location (steps/train/distill/images/<gpu|cpu>/), the way every other
# quay.io/granitebuild repository is named after the folder it is built from. On
# quay.io the repository must exist and be PUBLIC before the first push (a push
# creates it private, and SkyPilot pulls anonymously).
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
IMAGE_PREFIX      ?= steps/train/distill/images
PUBLISH_OVERWRITE ?= false
TAG_STATUS        := sh ../../scripts/registry-tag-status.sh

SCAN_SEVERITY     ?= CRITICAL
SCAN_CACHE        ?= $(HOME)/.cache/gb-image-scan
TRIVY_IMAGE       ?= ghcr.io/aquasecurity/trivy:0.76.2
OSV_IMAGE         ?= ghcr.io/google/osv-scanner:v2.7.1

image_ref = $(REGISTRY)/$(IMAGE_PREFIX)/$(1):$(IMAGE_VERSION)

.PHONY: help images publish-images scan-images scan-locks fix-locks $(addprefix fix-lock-,$(IMAGES)) $(addprefix image-,$(IMAGES)) $(addprefix publish-image-,$(IMAGES)) $(addprefix scan-image-,$(IMAGES)) $(addprefix scan-lock-,$(IMAGES))

help:
	@echo "Targets: images, scan-locks, fix-locks, scan-images, publish-images, image-<x>, scan-lock-<x>, fix-lock-<x>, scan-image-<x>, publish-image-<x> (x: $(IMAGES))"
	@echo "Set REGISTRY (e.g. quay.io/granitebuild); IMAGE_VERSION defaults to the git short SHA ($(IMAGE_VERSION))."

define require_registry
	@test -n "$(REGISTRY)" || { echo "REGISTRY is not set: make $@ REGISTRY=quay.io/granitebuild" >&2; exit 1; }
endef

images: $(addprefix image-,$(IMAGES))
publish-images: $(addprefix publish-image-,$(IMAGES))
scan-images: $(addprefix scan-image-,$(IMAGES))
scan-locks: $(addprefix scan-lock-,$(IMAGES))
fix-locks: $(addprefix fix-lock-,$(IMAGES))

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

# osv-scanner exits 1 on ANY finding (0 clean, 128 nothing scanned) and has no
# severity filter, so its JSON goes through images/osv-gate.py, which prints the HIGH
# and above findings with their fix version and fails only on a fixable one at
# SCAN_SEVERITY, like trivy's gate below. Exit codes other than 0/1 (no lock, network)
# fail as themselves.
$(addprefix scan-lock-,$(IMAGES)): scan-lock-%:
	@echo "[scan] osv-scanner images/$*/uv.lock (gate: fixable $(SCAN_SEVERITY))"
	@json=$$(mktemp) || exit 1; \
	$(DOCKER) run --rm -v "$(CURDIR)/images/$*:/src:ro,Z" $(OSV_IMAGE) \
		scan source --lockfile /src/uv.lock --format json > "$$json"; rc=$$?; \
	if [ $$rc -ne 0 ] && [ $$rc -ne 1 ]; then rm -f "$$json"; echo "[scan] osv-scanner failed (exit $$rc) on images/$*/uv.lock" >&2; exit $$rc; fi; \
	python3 images/osv-gate.py $(SCAN_SEVERITY) < "$$json"; rc=$$?; rm -f "$$json"; \
	if [ $$rc -ne 0 ]; then echo "[scan] FAILED: images/$*/uv.lock has fixable $(SCAN_SEVERITY) vulnerabilities; raise those pins and run uv lock" >&2; exit $$rc; fi

# scan-image-<x> also runs scan-lock-<x>. The image is saved to a tar under SCAN_CACHE,
# scanned from there and the tar removed, so trivy needs neither the registry nor
# podman's socket. The report pass (HIGH and above, fixable) fails only if trivy
# itself cannot run; the gate pass exits 3 on a fixable finding at SCAN_SEVERITY, so
# a finding is told apart from trivy failing (pull, database download).
$(addprefix scan-image-,$(IMAGES)): scan-image-%: scan-lock-%
	$(require_registry)
	@$(DOCKER) image exists $(call image_ref,$*) 2>/dev/null || $(DOCKER) image inspect $(call image_ref,$*) >/dev/null 2>&1 || \
		{ echo "[scan] $(call image_ref,$*) is not built: run make image-$* REGISTRY=$(REGISTRY) first" >&2; exit 1; }
	@mkdir -p "$(SCAN_CACHE)/trivy"
	@tar="$(SCAN_CACHE)/$*-$(IMAGE_VERSION).tar"; rm -f "$$tar"; \
	echo "[scan] saving $(call image_ref,$*)"; \
	$(DOCKER) save -o "$$tar" $(call image_ref,$*) || { rm -f "$$tar"; exit 1; }; \
	trivy() { $(DOCKER) run --rm -v "$$tar:/scan/image.tar:ro,Z" -v "$(SCAN_CACHE)/trivy:/root/.cache/trivy:Z" \
		$(TRIVY_IMAGE) image --input /scan/image.tar --scanners vuln --ignore-unfixed --quiet "$$@"; }; \
	echo "[scan] trivy report (fixable HIGH,CRITICAL) for $(call image_ref,$*)"; \
	trivy --severity HIGH,CRITICAL --format table; rc=$$?; \
	if [ $$rc -ne 0 ]; then rm -f "$$tar"; echo "[scan] trivy did not run (exit $$rc); see the error above" >&2; exit $$rc; fi; \
	echo "[scan] trivy gate (fixable $(SCAN_SEVERITY)) for $(call image_ref,$*)"; \
	trivy --severity $(SCAN_SEVERITY) --exit-code 3 --format table --skip-db-update; rc=$$?; \
	rm -f "$$tar"; \
	case $$rc in \
		0) ;; \
		3) echo "[scan] FAILED: $(call image_ref,$*) has fixable $(SCAN_SEVERITY) vulnerabilities; fix them before publish-image-$*" >&2; exit 1 ;; \
		*) echo "[scan] trivy did not run (exit $$rc); see the error above" >&2; exit $$rc ;; \
	esac; \
	echo "[scan] $(call image_ref,$*): no fixable $(SCAN_SEVERITY) vulnerabilities"

$(addprefix fix-lock-,$(IMAGES)): fix-lock-%:
	@command -v uv >/dev/null 2>&1 || { echo "[fix-lock] uv is not on PATH (https://docs.astral.sh/uv/)" >&2; exit 1; }
	@echo "[fix-lock] images/$*/uv.lock (fixable $(SCAN_SEVERITY))"
	@python3 images/fix-locks.py images/$* $(SCAN_SEVERITY) -- \
		$(DOCKER) run --rm -v "$(CURDIR)/images/$*:/src:ro,Z" $(OSV_IMAGE) \
		scan source --lockfile /src/uv.lock --format json
