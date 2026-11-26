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
# Build the granite-evals images shared by the benchmark steps in steps/eval/<family>/.
# One image per benchmark family (the harnesses' dependencies conflict, so they
# cannot share an environment). The build context is this directory; each
# family's recipe is <family>/Dockerfile.gpu and, for the CPU_FAMILIES,
# <family>/Dockerfile.cpu, with any extra system setup in <family>/extras.sh
# and the family's benchmark modules in <family>/src/ (which the Dockerfiles
# overlay onto the core's granite_evals/benchmarks/ package directory).
#
#   make images REGISTRY=quay.io/myorg            build all 11 GPU + 8 CPU images
#   make image-nemoskills REGISTRY=...            build one (image-cpu-<family>: its CPU image)
#   make scan-image-nemoskills REGISTRY=...       the image gate on a built image (scan-image-cpu-<family>)
#   make publish-image-nemoskills REGISTRY=...    scan, then push one, refusing an existing tag
#   make publish-images REGISTRY=...              scan every image, check every tag, then push them all (needs `podman login`)
#   make scan-images REGISTRY=...                 the image gate on every built image (scan-images-cpu: the CPU ones)
#   make test-src                                 run the vendored unit tests (no GPU, no network)
#   make prefetch HF_HOME=/path/to/hf-cache       download the pinned HF datasets into a cache
#   make check-pins                               verify the dataset pins against the source
#
# Image references are the ones the recipes pin (recipes/evals/<family>/...):
# ${REGISTRY}/steps/eval/<family>/images/gpu:${IMAGE_VERSION} and .../images/cpu:...

FAMILIES := bfcl bird ifbench judged lmeval mcpmark nemoskills officeqa swebench tau tbench
# Families whose benchmarks split into GPU generate + CPU score (--phase); lmeval and tbench score inside generation.
CPU_FAMILIES := bfcl bird ifbench judged nemoskills officeqa swebench tau

REGISTRY  ?=
PLATFORM  ?= linux/amd64
DOCKER    ?= podman
IMAGE_VERSION ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo latest)
HF_HOME   ?=
PYTHON    ?= python

gpu_ref = $(REGISTRY)/steps/eval/$(1)/images/gpu:$(IMAGE_VERSION)
cpu_ref = $(REGISTRY)/steps/eval/$(1)/images/cpu:$(IMAGE_VERSION)

# ---- Image gate ---------------------------------------------------------------
# scan-image-<family> / scan-image-cpu-<family>: scan-deps (below), then trivy on the
# built image (saved to a tar under SCAN_CACHE), the fleet's rule: no fixable
# SCAN_SEVERITY finding, with scan-ignore as the documented exception list. trivy's
# --severity is a set, not a floor (`--severity HIGH` drops CRITICAL), so the gate is
# given SCAN_SEVERITY AND ABOVE, plus UNKNOWN so an unscored advisory is not skipped; a
# report pass prints HIGH and above (fixed and unfixed) first. publish-image-* runs the
# gate first, so an unscanned tag is never pushed; SKIP_SCAN=true (exactly `true`)
# skips it.
# HIGH, not CRITICAL: a fixable HIGH ships only if it is recorded in scan-ignore;
# silence is not acceptance. Same rule as steps/train/distill.
# trivy runs as the container TRIVY_IMAGE, pinned by digest (the same pin as
# steps/train/distill and steps/aux/hf-export), so a gate's verdict does not depend on
# whichever trivy a host has installed; its database is cached under SCAN_CACHE.
SCAN_SEVERITY ?= HIGH
SCAN_CACHE    ?= $(HOME)/.cache/gb-image-scan
SCAN_IGNORE   ?= scan-ignore
SKIP_SCAN     ?= false
TRIVY_IMAGE   ?= ghcr.io/aquasecurity/trivy:0.76.2@sha256:8f6bb947dbc528902e11f2f5be52f7b519f6a39d5d148f41166f2040499768fb
# Longer than distill's 30m: the eval GPU images are larger.
TRIVY_TIMEOUT ?= 60m
SEVERITY_UP_CRITICAL := CRITICAL
SEVERITY_UP_HIGH     := HIGH,CRITICAL
SEVERITY_UP_MEDIUM   := MEDIUM,HIGH,CRITICAL
SEVERITY_UP_LOW      := LOW,MEDIUM,HIGH,CRITICAL
TRIVY_GATE_SEVERITY   = UNKNOWN,$(SEVERITY_UP_$(SCAN_SEVERITY))
TRIVY_REPORT_SEVERITY = UNKNOWN,$(SEVERITY_UP_$(if $(filter CRITICAL,$(SCAN_SEVERITY)),HIGH,$(SCAN_SEVERITY)))
check-severity = $(if $(SEVERITY_UP_$(SCAN_SEVERITY)),,$(error SCAN_SEVERITY=$(SCAN_SEVERITY) is not one of CRITICAL HIGH MEDIUM LOW))
# Does a tag exist on its registry (present / absent / unknown)? Manifest only, no pull.
TAG_STATUS    := sh "$(CURDIR)/../scripts/registry-tag-status.sh"

# scan-ignore scopes each accepted id to the packages (purls) it names, and trivy's
# plain-text ignore format cannot say that: it would accept the id in every package.
# So trivy gets the YAML ignore file scripts/trivy-ignore.py writes from scan-ignore at
# gate time (its `purls` field), and scan-ignore stays the one hand-kept list.
PYTHON3          ?= python3
TRIVY_IGNORE_GEN := $(CURDIR)/scripts/trivy-ignore.py

# Shell, $(call gen-ignore,<label>): set ignsrc to scan-ignore's absolute path and ign to
# the trivy YAML generated from it under SCAN_CACHE, or both to empty when there is no
# scan-ignore. The recipes pass both quoted, so a path may hold spaces.
gen-ignore = ignsrc=""; ign=""; case "$(SCAN_IGNORE)" in /*) ignsrc="$(SCAN_IGNORE)";; *) ignsrc="$(CURDIR)/$(SCAN_IGNORE)";; esac; \
	if [ -f "$$ignsrc" ]; then ign="$(SCAN_CACHE)/trivy-ignore-$(1).yaml"; \
	$(PYTHON3) "$(TRIVY_IGNORE_GEN)" "$$ignsrc" "$$ign" || exit 1; \
	echo "[scan] accepting the advisories listed in $$ignsrc, each only in the packages it names"; else ignsrc=""; fi

# scan-image <image-ref> <label>
define scan-image
	$(check-severity)
	@$(DOCKER) image exists $(1) 2>/dev/null || $(DOCKER) image inspect $(1) >/dev/null 2>&1 || { echo "[scan] $(1) is not in local storage: build it first" >&2; exit 1; }
	@mkdir -p "$(SCAN_CACHE)/trivy"
	@tar="$(SCAN_CACHE)/$(2)-$(IMAGE_VERSION).tar"; rm -f "$$tar"; \
	echo "[scan] saving $(1)"; \
	$(DOCKER) save -o "$$tar" $(1) || { rm -f "$$tar"; exit 1; }; \
	$(call gen-ignore,$(2)); \
	t() { $(DOCKER) run --rm -v "$$tar:/scan/image.tar:ro,Z" -v "$(SCAN_CACHE)/trivy:/root/.cache/trivy:Z" \
		$${ign:+-v "$$ign:/scan/ignore.yaml:ro,Z"} $(TRIVY_IMAGE) image --quiet --input /scan/image.tar --scanners vuln --timeout $(TRIVY_TIMEOUT) \
		$${ign:+--ignorefile /scan/ignore.yaml} "$$@"; }; \
	echo "[scan] trivy report ($(TRIVY_REPORT_SEVERITY), fixed and unfixed) for $(1)"; \
	t --severity $(TRIVY_REPORT_SEVERITY) --format table; rc=$$?; \
	if [ $$rc -ne 0 ]; then rm -f "$$tar"; echo "[scan] trivy did not run (exit $$rc)" >&2; exit $$rc; fi; \
	echo "[scan] trivy gate (fixable $(TRIVY_GATE_SEVERITY)) for $(1)"; \
	t --severity $(TRIVY_GATE_SEVERITY) --ignore-unfixed --exit-code 3 --format table --skip-db-update; rc=$$?; \
	rm -f "$$tar"; \
	case $$rc in \
		0) echo "[scan] $(1): no fixable $(SCAN_SEVERITY) vulnerabilities" ;; \
		3) echo "[scan] FAILED: $(1) has fixable $(SCAN_SEVERITY) vulnerabilities" >&2; exit 1 ;; \
		*) echo "[scan] trivy did not run (exit $$rc)" >&2; exit $$rc ;; \
	esac
endef

# push-image <image-ref>: only a tag the registry reports absent; `unknown` is refused
# too, so a broken check never turns into an overwrite.
define push-image
	@$(if $(filter true,$(SKIP_SCAN)),echo "[publish] SKIP_SCAN=true: pushing $(1) without scanning it",true)
	@status=$$($(TAG_STATUS) "$(1)"); \
	case "$$status" in \
		absent) echo "[publish] $(1) is a new tag." ;; \
		present) echo "[publish] ERROR: $(1) already exists on the registry; a tag must keep pointing at one image. Commit, so IMAGE_VERSION moves on." >&2; exit 1 ;; \
		*) echo "[publish] ERROR: could not determine whether $(1) already exists (registry unreachable, or it refused the check)." >&2; exit 1 ;; \
	esac
	$(DOCKER) push $(1)
endef

.PHONY: help $(addprefix push-image-,$(FAMILIES)) $(addprefix push-image-cpu-,$(CPU_FAMILIES)) scan-deps scan-images scan-images-cpu images-cpu publish-images-cpu $(addprefix image-cpu-,$(CPU_FAMILIES)) $(addprefix publish-image-cpu-,$(CPU_FAMILIES)) images publish-images test-src prefetch check-pins $(addprefix image-,$(FAMILIES)) $(addprefix publish-image-,$(FAMILIES)) $(addprefix scan-image-,$(FAMILIES)) $(addprefix scan-image-cpu-,$(CPU_FAMILIES))

help:
	@echo "Targets: images, scan-images, publish-images (all scans, then all pushes), image-<family>, scan-image-<family>, publish-image-<family>, push-image-<family> (no scan) (and the -cpu- forms), test-src, prefetch, check-pins, scan-deps"
	@echo "Families: $(FAMILIES)"
	@echo "Set REGISTRY (e.g. quay.io/myorg); IMAGE_VERSION defaults to the git short SHA ($(IMAGE_VERSION))."

define require_registry
	@test -n "$(REGISTRY)" || { echo "REGISTRY is not set: make $@ REGISTRY=quay.io/myorg" >&2; exit 1; }
endef

images: $(addprefix image-,$(FAMILIES)) $(addprefix image-cpu-,$(CPU_FAMILIES))
images-cpu: $(addprefix image-cpu-,$(CPU_FAMILIES))
scan-images: $(addprefix scan-image-,$(FAMILIES)) $(addprefix scan-image-cpu-,$(CPU_FAMILIES))
scan-images-cpu: $(addprefix scan-image-cpu-,$(CPU_FAMILIES))

# publish-images / publish-images-cpu are all or nothing: every image is scanned, and
# every tag checked absent on the registry, before the first push. One failing scan
# or one existing tag stops them with nothing pushed. (Each push still re-checks its
# own tag, so a tag published meanwhile is refused too.)
ALL_REFS := $(foreach f,$(FAMILIES),$(call gpu_ref,$(f))) $(foreach f,$(CPU_FAMILIES),$(call cpu_ref,$(f)))
CPU_REFS := $(foreach f,$(CPU_FAMILIES),$(call cpu_ref,$(f)))
define check-all-absent
	@for ref in $(1); do \
		status=$$($(TAG_STATUS) "$$ref"); \
		[ "$$status" = absent ] || { echo "[publish] refusing: $$ref is $$status on the registry; nothing was pushed" >&2; exit 1; }; \
	done; echo "[publish] every tag is absent on the registry"
endef
publish-images: $(if $(filter true,$(SKIP_SCAN)),,scan-images)
	$(require_registry)
	$(call check-all-absent,$(ALL_REFS))
	@$(MAKE) --no-print-directory $(addprefix push-image-,$(FAMILIES)) $(addprefix push-image-cpu-,$(CPU_FAMILIES))
publish-images-cpu: $(if $(filter true,$(SKIP_SCAN)),,scan-images-cpu)
	$(require_registry)
	$(call check-all-absent,$(CPU_REFS))
	@$(MAKE) --no-print-directory $(addprefix push-image-cpu-,$(CPU_FAMILIES))

# push-image-<family> / push-image-cpu-<family>: the push alone, tag guard included,
# for publish-images after its scans. Run publish-image-* to scan and push one image.
$(addprefix push-image-,$(FAMILIES)): push-image-%:
	$(require_registry)
	$(call push-image,$(call gpu_ref,$*))
$(addprefix push-image-cpu-,$(CPU_FAMILIES)): push-image-cpu-%:
	$(require_registry)
	$(call push-image,$(call cpu_ref,$*))

$(addprefix image-cpu-,$(CPU_FAMILIES)): image-cpu-%:
	$(require_registry)
	$(DOCKER) build --platform $(PLATFORM) -f $*/Dockerfile.cpu . -t $(call cpu_ref,$*)

$(addprefix scan-image-cpu-,$(CPU_FAMILIES)): scan-image-cpu-%: scan-deps
	$(require_registry)
	$(call scan-image,$(call cpu_ref,$*),$*-cpu)

$(addprefix publish-image-cpu-,$(CPU_FAMILIES)): publish-image-cpu-%: $(if $(filter true,$(SKIP_SCAN)),,scan-image-cpu-%)
	$(require_registry)
	$(call push-image,$(call cpu_ref,$*))

$(addprefix image-,$(FAMILIES)): image-%:
	$(require_registry)
	$(DOCKER) build --platform $(PLATFORM) -f $*/Dockerfile.gpu . -t $(call gpu_ref,$*)

$(addprefix scan-image-,$(FAMILIES)): scan-image-%: scan-deps
	$(require_registry)
	$(call scan-image,$(call gpu_ref,$*),$*-gpu)

$(addprefix publish-image-,$(FAMILIES)): publish-image-%: $(if $(filter true,$(SKIP_SCAN)),,scan-image-%)
	$(require_registry)
	$(call push-image,$(call gpu_ref,$*))

# The vendored tests import granite_evals, which the images assemble from the shared
# core plus one family's modules. Do the same in a temp dir, one family at a time.
test-src:
	@set -e; for f in $(FAMILIES); do \
	  echo "== $$f"; \
	  d=$$(mktemp -d); \
	  mkdir -p $$d/src $$d/tests; \
	  cp -R src/core/granite_evals $$d/src/; cp -R $$f/src/. $$d/src/granite_evals/benchmarks/; \
	  cp -R src/core/tests/. $$d/tests/; cp -R $$f/tests/. $$d/tests/ 2>/dev/null || true; \
	  (cd $$d && PYTHONPATH=$$d/src $(PYTHON) -m pytest -q -p no:cacheprovider tests); rc=$$?; \
	  rm -rf $$d; [ $$rc -eq 0 ] || exit $$rc; \
	done

prefetch:
	@test -n "$(HF_HOME)" || { echo "HF_HOME is not set: make prefetch HF_HOME=/path/to/hf-cache" >&2; exit 1; }
	$(PYTHON) scripts/prefetch_datasets.py --hf-home "$(HF_HOME)"

check-pins:
	$(PYTHON) scripts/prefetch_datasets.py --check

# Pre-build dependency gate: trivy and osv-scanner over the resolved dependency sets -- the
# shared uv.lock (every family's forks resolve in it) and every family's nested
# requirement files (e.g. nemoskills' comet-requirements.txt). Catches the
# lock-level CVE class (a pinned dependency with a fix available) in seconds,
# before any image build; the image gate stays the backstop for what the lock
# cannot see: apt packages, vendored harnesses, and bits vendored inside tools
# (npm's own dependencies, setuptools' _vendor). Same rule as the image gate (fixable
# TRIVY_GATE_SEVERITY, scan-ignore the documented exception list), and scan-image-*
# runs it first. trivy recognises a pip requirements file only by the name
# requirements.txt (any other name is "no supported files", i.e. a silent pass), so
# each one is scanned from a temp dir under that name. The lock is scanned whole, so a
# package one fork pins below its fix fails here even when no image installs that fork
# (trivy reads no markers).
#
# Each file is copied alone into a temp dir under SCAN_CACHE (which a podman machine
# shares with its VM, unlike /tmp on macOS) and scanned there by the pinned TRIVY_IMAGE.
# --include-dev-deps: trivy's uv.lock parser marks every package it cannot reach from
# the project's default dependencies as dev and leaves it out, which skipped
# diskcache 5.6.3 (bird, ifbench and nemoskills install it through an extra) and
# pytest's set. With the flag, trivy reads all 460 packages, as osv-scanner does.
#
# Each file then goes through osv-scanner (OSV_IMAGE, pinned) and distill's
# osv-gate.py, as steps/train/distill's scan-lock-* does. trivy rates a Python
# advisory by its GHSA severity, and several that GHSA rates MODERATE, or not at all,
# carry a CVSS of 7 and above in their OSV record. osv-gate.py rates each advisory
# group by its highest CVSS, fails closed on an unscored fixable one, and reads
# scan-ignore with its package scopes, so a finding passes only if both gates pass.
OSV_IMAGE ?= ghcr.io/google/osv-scanner:v2.7.1@sha256:a70ffffc2bfd33482953e5d6b64c6fcdb30239717a07b0f0d3ab2859ce3e4cde
OSV_GATE  ?= $(CURDIR)/../train/distill/images/osv-gate.py
scan-deps:
	$(check-severity)
	@mkdir -p "$(SCAN_CACHE)/trivy"
	@rc=0; \
	$(call gen-ignore,deps); \
	for f in src/core/uv.lock $$(ls */*requirements*.txt 2>/dev/null); do \
	  echo "== $$f (fixable $(TRIVY_GATE_SEVERITY))"; \
	  d=$$(mktemp -d "$(SCAN_CACHE)/deps.XXXXXX") || exit 1; \
	  case "$$f" in *.txt) cp "$$f" "$$d/requirements.txt";; *) cp "$$f" "$$d/";; esac || { rm -rf "$$d"; exit 1; }; \
	  $(DOCKER) run --rm -v "$$d:/scan/src:ro,Z" -v "$(SCAN_CACHE)/trivy:/root/.cache/trivy:Z" \
	    $${ign:+-v "$$ign:/scan/ignore.yaml:ro,Z"} $(TRIVY_IMAGE) fs --quiet --scanners vuln --timeout $(TRIVY_TIMEOUT) \
	    --severity $(TRIVY_GATE_SEVERITY) --ignore-unfixed --include-dev-deps $${ign:+--ignorefile /scan/ignore.yaml} \
	    --exit-code 3 --format table /scan/src; frc=$$?; \
	  case $$frc in \
	    0) echo "OK: $$f has no fixable $(TRIVY_GATE_SEVERITY) vulnerabilities" ;; \
	    3) echo "SCAN-DEPS FAILED: $$f has fixable $(TRIVY_GATE_SEVERITY) vulnerabilities (above); raise them or record them in $(SCAN_IGNORE)" >&2; rc=1 ;; \
	    *) echo "SCAN-DEPS FAILED: trivy did not run on $$f (exit $$frc)" >&2; rc=1 ;; \
	  esac; \
	  case "$$f" in *.txt) lf="requirements.txt:/scan/src/requirements.txt";; *) lf="/scan/src/uv.lock";; esac; \
	  echo "== $$f (osv-scanner; gate: fixable $(SCAN_SEVERITY) by the advisory group's highest CVSS)"; \
	  $(DOCKER) run --rm -v "$$d:/scan/src:ro,Z" $(OSV_IMAGE) scan source --lockfile "$$lf" --format json > "$$d/osv.json"; orc=$$?; \
	  if [ $$orc -ne 0 ] && [ $$orc -ne 1 ]; then \
	    echo "SCAN-DEPS FAILED: osv-scanner did not run on $$f (exit $$orc)" >&2; rc=1; \
	  elif $(PYTHON3) "$(OSV_GATE)" $(SCAN_SEVERITY) $${ignsrc:+--ignore "$$ignsrc"} < "$$d/osv.json"; then \
	    echo "OK: $$f has no fixable $(SCAN_SEVERITY) vulnerabilities by CVSS"; \
	  else \
	    echo "SCAN-DEPS FAILED: $$f has fixable $(SCAN_SEVERITY) vulnerabilities by CVSS (above); raise them or record them in $(SCAN_IGNORE)" >&2; rc=1; \
	  fi; \
	  rm -rf "$$d"; \
	done; \
	[ $$rc -eq 0 ] && echo "OK: dependencies clean" || exit 1
