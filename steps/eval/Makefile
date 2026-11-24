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
#   make publish-images REGISTRY=...              the same for all (needs a prior `podman login`)
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
SCAN_SEVERITY ?= HIGH
SCAN_CACHE    ?= $(HOME)/.cache/gb-image-scan
SCAN_IGNORE   ?= scan-ignore
SKIP_SCAN     ?= false
TRIVY         ?= trivy
TRIVY_TIMEOUT ?= 60m
SEVERITY_UP_CRITICAL := CRITICAL
SEVERITY_UP_HIGH     := HIGH,CRITICAL
SEVERITY_UP_MEDIUM   := MEDIUM,HIGH,CRITICAL
SEVERITY_UP_LOW      := LOW,MEDIUM,HIGH,CRITICAL
TRIVY_GATE_SEVERITY   = UNKNOWN,$(SEVERITY_UP_$(SCAN_SEVERITY))
TRIVY_REPORT_SEVERITY = UNKNOWN,$(SEVERITY_UP_$(if $(filter CRITICAL,$(SCAN_SEVERITY)),HIGH,$(SCAN_SEVERITY)))
check-severity = $(if $(SEVERITY_UP_$(SCAN_SEVERITY)),,$(error SCAN_SEVERITY=$(SCAN_SEVERITY) is not one of CRITICAL HIGH MEDIUM LOW))
# Does a tag exist on its registry (present / absent / unknown)? Manifest only, no pull.
TAG_STATUS    := $(CURDIR)/../scripts/registry-tag-status.sh

# scan-image <image-ref> <label>
define scan-image
	$(check-severity)
	@$(DOCKER) image exists $(1) 2>/dev/null || { echo "[scan] $(1) is not in local storage: build it first" >&2; exit 1; }
	@mkdir -p "$(SCAN_CACHE)/trivy"
	@tar="$(SCAN_CACHE)/$(2)-$(IMAGE_VERSION).tar"; rm -f "$$tar"; \
	echo "[scan] saving $(1)"; \
	$(DOCKER) save -o "$$tar" $(1) || { rm -f "$$tar"; exit 1; }; \
	ign=""; [ -f $(SCAN_IGNORE) ] && ign="--ignorefile $(abspath $(SCAN_IGNORE))"; \
	t() { $(TRIVY) image --quiet --cache-dir "$(SCAN_CACHE)/trivy" --input "$$tar" --scanners vuln --timeout $(TRIVY_TIMEOUT) $$ign "$$@"; }; \
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
	@status=$$($(TAG_STATUS) "$(1)"); \
	case "$$status" in \
		absent) echo "[publish] $(1) is a new tag." ;; \
		present) echo "[publish] ERROR: $(1) already exists on the registry; a tag must keep pointing at one image. Commit, so IMAGE_VERSION moves on." >&2; exit 1 ;; \
		*) echo "[publish] ERROR: could not determine whether $(1) already exists (registry unreachable, or it refused the check)." >&2; exit 1 ;; \
	esac
	$(DOCKER) push $(1)
endef

.PHONY: help scan-deps images-cpu publish-images-cpu $(addprefix image-cpu-,$(CPU_FAMILIES)) $(addprefix publish-image-cpu-,$(CPU_FAMILIES)) images publish-images test-src prefetch check-pins $(addprefix image-,$(FAMILIES)) $(addprefix publish-image-,$(FAMILIES)) $(addprefix scan-image-,$(FAMILIES)) $(addprefix scan-image-cpu-,$(CPU_FAMILIES))

help:
	@echo "Targets: images, publish-images, image-<family>, scan-image-<family>, publish-image-<family> (and the -cpu- forms), test-src, prefetch, check-pins, scan-deps"
	@echo "Families: $(FAMILIES)"
	@echo "Set REGISTRY (e.g. quay.io/myorg); IMAGE_VERSION defaults to the git short SHA ($(IMAGE_VERSION))."

define require_registry
	@test -n "$(REGISTRY)" || { echo "REGISTRY is not set: make $@ REGISTRY=quay.io/myorg" >&2; exit 1; }
endef

images: $(addprefix image-,$(FAMILIES)) $(addprefix image-cpu-,$(CPU_FAMILIES))
images-cpu: $(addprefix image-cpu-,$(CPU_FAMILIES))
publish-images-cpu: $(addprefix publish-image-cpu-,$(CPU_FAMILIES))
publish-images: $(addprefix publish-image-,$(FAMILIES)) $(addprefix publish-image-cpu-,$(CPU_FAMILIES))

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

# Pre-build dependency gate: trivy over the resolved dependency sets -- the
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
scan-deps:
	$(check-severity)
	@command -v $(TRIVY) >/dev/null 2>&1 || { echo "$(TRIVY) is not on PATH; install it first" >&2; exit 1; }
	@mkdir -p "$(SCAN_CACHE)/trivy"
	@rc=0; \
	for f in src/core/uv.lock $$(ls */*requirements*.txt 2>/dev/null); do \
	  echo "== $$f (fixable $(TRIVY_GATE_SEVERITY))"; \
	  case "$$f" in *.txt) d=$$(mktemp -d) || exit 1; cp "$$f" "$$d/requirements.txt"; target=$$d;; *) d=; target=$$f;; esac; \
	  $(TRIVY) fs --quiet --cache-dir "$(SCAN_CACHE)/trivy" --scanners vuln --timeout $(TRIVY_TIMEOUT) \
	    --severity $(TRIVY_GATE_SEVERITY) --ignore-unfixed --ignorefile $(abspath $(SCAN_IGNORE)) \
	    --exit-code 3 --format table "$$target"; frc=$$?; \
	  [ -n "$$d" ] && rm -rf "$$d"; \
	  case $$frc in \
	    0) echo "OK: $$f has no fixable $(TRIVY_GATE_SEVERITY) vulnerabilities" ;; \
	    3) echo "SCAN-DEPS FAILED: $$f has fixable $(TRIVY_GATE_SEVERITY) vulnerabilities (above); raise them or record them in $(SCAN_IGNORE)" >&2; rc=1 ;; \
	    *) echo "SCAN-DEPS FAILED: trivy did not run on $$f (exit $$frc)" >&2; rc=1 ;; \
	  esac; \
	done; \
	[ $$rc -eq 0 ] && echo "OK: dependencies clean" || exit 1
