# hf-export step for the SkyPilot environment (SLURM and LSF).
#
# Packages one checkpoint-N of an HF Trainer / accelerate run into a self-contained,
# downloadable HuggingFace model directory. Builds its own small CPU image from the
# Dockerfile here; the script and its resume gate are baked into it.
#
# Usage:
#   make image publish-image   # build, scan, then push the image (needs registry login)
#   make scan-lock             # osv-scanner on requirements.lock (no build needed)
#   make scan-image            # scan-lock, then trivy on the image $(IMAGE_REF) in local storage
#   make space                 # render a Space (space.yaml + steps/<name>/step.yaml)
#   make test                  # render, then run test/
#   make publish-step          # write the published copy under configurations/
#   make lock                  # recompile requirements.lock from requirements.in
#   make help                  # list targets; see the shared steps/README.md

STEP_NAME := hf-export
REGISTRY ?= quay.io/granitebuild
IMAGE_NAME ?= steps/aux/hf-export
IMAGE_VERSION ?= 7fa2c207
UV_LOCK ?= uvx --from uv==0.13.6 uv

include ../../../common.mk

.PHONY: lock
lock: ## Recompile requirements.lock (hashed) from requirements.in for linux x86_64
	$(UV_LOCK) pip compile requirements.in --python-version 3.12 \
	  --python-platform x86_64-manylinux_2_28 --generate-hashes -o requirements.lock

# ---- Vulnerability scan gate ------------------------------------------------
#
# The same gate as steps/train/distill/Makefile's scan-lock-<x> / scan-image-<x>, for
# this step's one image, with the same semantics (one exception, below):
#   - scan-lock: osv-scanner on requirements.lock (the exact hashed Python set the
#     Dockerfile installs), against OSV. FAILS on a fixable finding at SCAN_SEVERITY
#     (default HIGH; CVSS >= 7) and, closed, on a fixable advisory with no score.
#   - scan-image: scan-lock, then trivy on the image (OS packages from the base and
#     apt, the Python venv, the uv binaries), saved to a tar so no registry or daemon
#     socket is needed. A report pass (HIGH and above, fixed and unfixed) fails only
#     if trivy cannot run; the gate pass (fixable SCAN_SEVERITY, --ignore-unfixed)
#     fails on a finding.
#   - trivy's --severity is a set, not a floor (`--severity HIGH` drops CRITICAL), so
#     both trivy passes are given SCAN_SEVERITY AND ABOVE, plus UNKNOWN so an unscored
#     fixable finding fails closed as it does in osv-gate.py. distill's gate does the
#     same; its report pass is fixed at HIGH,CRITICAL.
#   - scan-ignore (next to this Makefile) lists accepted advisory ids, one per line,
#     with the reason, the issue that removes it and an expiry; both passes read it.
#   - publish-image runs both first, so an unscanned tag is never pushed. SKIP_SCAN=true
#     (exactly `true`) skips them, e.g. for an image scanned in this shell a minute ago.
#
# scan-image scans whatever $(IMAGE_REF) is in local storage. IMAGE_VERSION is pinned
# here, so `make image` re-tags a local build under the published tag: to scan the
# PUBLISHED image, `$(DOCKER) pull $(IMAGE_REF)` first (or `podman rmi` a local build).
# quay (Clair) can still flag a package neither tool does; check quay's scan of the
# pushed tag too.
#
# Shared with distill, not copied: the osv JSON gate is distill's own osv-gate.py,
# called in place (it reads scan-ignore in the same format). Copied: the recipes and
# the scanner pins below. distill's are pattern rules over its two images under
# images/<x>/, so sharing them would mean rewriting that Makefile; keep TRIVY_IMAGE,
# OSV_IMAGE and the recipes in step with steps/train/distill/Makefile.

SCAN_SEVERITY ?= HIGH
SCAN_CACHE    ?= $(HOME)/.cache/gb-image-scan
# trivy's own default (5m) is too short for the multi-GB GPU images and fails the scan
# (closed) with "context deadline exceeded".
TRIVY_TIMEOUT ?= 30m
SCAN_IGNORE   ?= scan-ignore
SKIP_SCAN     ?= false
TRIVY_IMAGE   ?= ghcr.io/aquasecurity/trivy:0.76.2@sha256:8f6bb947dbc528902e11f2f5be52f7b519f6a39d5d148f41166f2040499768fb
OSV_IMAGE     ?= ghcr.io/google/osv-scanner:v2.7.1@sha256:a70ffffc2bfd33482953e5d6b64c6fcdb30239717a07b0f0d3ab2859ce3e4cde
OSV_GATE      ?= $(COMMON_MK_DIR)train/distill/images/osv-gate.py

# SCAN_SEVERITY and above, as trivy severity sets (see the note above). The report
# pass always covers at least HIGH.
SEVERITY_UP_CRITICAL := CRITICAL
SEVERITY_UP_HIGH     := HIGH,CRITICAL
SEVERITY_UP_MEDIUM   := MEDIUM,HIGH,CRITICAL
SEVERITY_UP_LOW      := LOW,MEDIUM,HIGH,CRITICAL
TRIVY_GATE_SEVERITY   = UNKNOWN,$(SEVERITY_UP_$(SCAN_SEVERITY))
TRIVY_REPORT_SEVERITY = UNKNOWN,$(SEVERITY_UP_$(if $(filter CRITICAL,$(SCAN_SEVERITY)),HIGH,$(SCAN_SEVERITY)))
check-severity = $(if $(SEVERITY_UP_$(SCAN_SEVERITY)),,$(error SCAN_SEVERITY=$(SCAN_SEVERITY) is not one of CRITICAL HIGH MEDIUM LOW))

.PHONY: scan-lock scan-image scan-skipped

# Adds prerequisites to common.mk's publish-image (its recipe is unchanged).
publish-image: $(if $(filter true,$(SKIP_SCAN)),scan-skipped,scan-lock scan-image)

scan-skipped:
	@echo "[publish] SKIP_SCAN=true: pushing $(IMAGE_REF) without scanning it"

# osv-scanner exits 1 on ANY finding (0 clean, 128 nothing scanned) and has no
# severity filter, so its JSON goes through osv-gate.py, which prints the HIGH and
# above findings with their fix version and fails only on a fixable one at
# SCAN_SEVERITY. Exit codes other than 0/1 (no lock, network) fail as themselves.
# requirements.lock is not a name osv-scanner recognises, so the parser is named.
scan-lock:
	$(check-severity)
	@echo "[scan] osv-scanner requirements.lock (gate: fixable $(SCAN_SEVERITY))"
	@json=$$(mktemp) || exit 1; \
	$(DOCKER) run --rm -v "$(CURDIR)/requirements.lock:/src/requirements.lock:ro,Z" $(OSV_IMAGE) \
		scan source --lockfile requirements.txt:/src/requirements.lock --format json > "$$json"; rc=$$?; \
	if [ $$rc -ne 0 ] && [ $$rc -ne 1 ]; then rm -f "$$json"; echo "[scan] osv-scanner failed (exit $$rc) on requirements.lock" >&2; exit $$rc; fi; \
	python3 $(OSV_GATE) $(SCAN_SEVERITY) --ignore $(SCAN_IGNORE) < "$$json"; rc=$$?; rm -f "$$json"; \
	if [ $$rc -ne 0 ]; then echo "[scan] FAILED: requirements.lock has fixable $(SCAN_SEVERITY) vulnerabilities; raise those pins and run make lock" >&2; exit $$rc; fi

# The image is saved to a tar under SCAN_CACHE, scanned from there and the tar
# removed. The gate pass exits 3 on a fixable finding at SCAN_SEVERITY, so a finding
# is told apart from trivy failing (pull, database download).
scan-image: scan-lock
	$(check-severity)
	@$(DOCKER) image exists $(IMAGE_REF) 2>/dev/null || $(DOCKER) image inspect $(IMAGE_REF) >/dev/null 2>&1 || \
		{ echo "[scan] $(IMAGE_REF) is not in local storage: run make image, or $(DOCKER) pull $(IMAGE_REF) to scan the published tag" >&2; exit 1; }
	@mkdir -p "$(SCAN_CACHE)/trivy"
	@tar="$(SCAN_CACHE)/$(STEP_NAME)-$(IMAGE_TAG).tar"; rm -f "$$tar"; \
	echo "[scan] saving $(IMAGE_REF)"; \
	$(DOCKER) save -o "$$tar" $(IMAGE_REF) || { rm -f "$$tar"; exit 1; }; \
	ign=""; if [ -f $(SCAN_IGNORE) ]; then \
		ign="-v $(abspath $(SCAN_IGNORE)):/scan/ignore:ro,Z"; \
		echo "[scan] accepting the advisories listed in $(SCAN_IGNORE)"; fi; \
	trivy() { $(DOCKER) run --rm -v "$$tar:/scan/image.tar:ro,Z" -v "$(SCAN_CACHE)/trivy:/root/.cache/trivy:Z" $$ign \
		$(TRIVY_IMAGE) image --input /scan/image.tar --timeout $(TRIVY_TIMEOUT) --scanners vuln --ignore-unfixed --quiet \
		$${ign:+--ignorefile /scan/ignore} "$$@"; }; \
	trivy_all() { $(DOCKER) run --rm -v "$$tar:/scan/image.tar:ro,Z" -v "$(SCAN_CACHE)/trivy:/root/.cache/trivy:Z" $$ign \
		$(TRIVY_IMAGE) image --input /scan/image.tar --timeout $(TRIVY_TIMEOUT) --scanners vuln --quiet \
		$${ign:+--ignorefile /scan/ignore} "$$@"; }; \
	echo "[scan] trivy report ($(TRIVY_REPORT_SEVERITY), fixed and unfixed) for $(IMAGE_REF)"; \
	trivy_all --severity $(TRIVY_REPORT_SEVERITY) --format table; rc=$$?; \
	if [ $$rc -ne 0 ]; then rm -f "$$tar"; echo "[scan] trivy did not run (exit $$rc); see the error above" >&2; exit $$rc; fi; \
	echo "[scan] trivy gate (fixable $(TRIVY_GATE_SEVERITY)) for $(IMAGE_REF)"; \
	trivy --severity $(TRIVY_GATE_SEVERITY) --exit-code 3 --format table --skip-db-update; rc=$$?; \
	rm -f "$$tar"; \
	case $$rc in \
		0) ;; \
		3) echo "[scan] FAILED: $(IMAGE_REF) has fixable $(SCAN_SEVERITY) vulnerabilities; fix them before publish-image" >&2; exit 1 ;; \
		*) echo "[scan] trivy did not run (exit $$rc); see the error above" >&2; exit $$rc ;; \
	esac; \
	echo "[scan] $(IMAGE_REF): no fixable $(SCAN_SEVERITY) vulnerabilities"
