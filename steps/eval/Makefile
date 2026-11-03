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
#   make images REGISTRY=quay.io/myorg            build all 9 GPU + 7 CPU images (`-cpu` suffix)
#   make image-nemoskills REGISTRY=...            build one
#   make publish-images REGISTRY=...              push all (needs a prior `podman login`)
#   make test-src                                 run the vendored unit tests (no GPU, no network)
#   make prefetch HF_HOME=/path/to/hf-cache       download the pinned HF datasets into a cache
#   make check-pins                               verify the dataset pins against the source
#
# The step images are referenced from each family's step-template.yaml as
# ${REGISTRY}/gb-step-eval-<family>:${IMAGE_TAG}.

FAMILIES := bfcl bird ifbench judged lmeval nemoskills swebench tau tbench
# Families whose benchmarks split into GPU generate + CPU score (--phase); lmeval and tbench score inside generation.
CPU_FAMILIES := bfcl bird ifbench judged nemoskills swebench tau

REGISTRY  ?=
PLATFORM  ?= linux/amd64
DOCKER    ?= podman
IMAGE_TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo latest)
HF_HOME   ?=
PYTHON    ?= python

.PHONY: help images-cpu publish-images-cpu $(addprefix image-cpu-,$(CPU_FAMILIES)) $(addprefix publish-image-cpu-,$(CPU_FAMILIES)) images publish-images test-src prefetch check-pins $(addprefix image-,$(FAMILIES)) $(addprefix publish-image-,$(FAMILIES))

help:
	@echo "Targets: images, publish-images, image-<family>, publish-image-<family>, test-src, prefetch, check-pins"
	@echo "Families: $(FAMILIES)"
	@echo "Set REGISTRY (e.g. quay.io/myorg); IMAGE_TAG defaults to the git short SHA ($(IMAGE_TAG))."

define require_registry
	@test -n "$(REGISTRY)" || { echo "REGISTRY is not set: make $@ REGISTRY=quay.io/myorg" >&2; exit 1; }
endef

images: $(addprefix image-,$(FAMILIES)) $(addprefix image-cpu-,$(CPU_FAMILIES))
images-cpu: $(addprefix image-cpu-,$(CPU_FAMILIES))
publish-images-cpu: $(addprefix publish-image-cpu-,$(CPU_FAMILIES))
publish-images: $(addprefix publish-image-,$(FAMILIES)) $(addprefix publish-image-cpu-,$(CPU_FAMILIES))

$(addprefix image-cpu-,$(CPU_FAMILIES)): image-cpu-%:
	$(require_registry)
	$(DOCKER) build --platform $(PLATFORM) -f $*/Dockerfile.cpu . -t $(REGISTRY)/gb-step-eval-$*-cpu:$(IMAGE_TAG)

$(addprefix publish-image-cpu-,$(CPU_FAMILIES)): publish-image-cpu-%:
	$(require_registry)
	$(DOCKER) push $(REGISTRY)/gb-step-eval-$*-cpu:$(IMAGE_TAG)

$(addprefix image-,$(FAMILIES)): image-%:
	$(require_registry)
	$(DOCKER) build --platform $(PLATFORM) -f $*/Dockerfile.gpu . -t $(REGISTRY)/gb-step-eval-$*:$(IMAGE_TAG)

$(addprefix publish-image-,$(FAMILIES)): publish-image-%:
	$(require_registry)
	$(DOCKER) push $(REGISTRY)/gb-step-eval-$*:$(IMAGE_TAG)

# The vendored tests import granite_evals, which the images assemble from the shared
# core plus one family's modules. Do the same in a temp dir, one family at a time.
test-src:
	@set -e; for f in $(FAMILIES); do \
	  echo "== $$f"; \
	  d=$$(mktemp -d); \
	  mkdir -p $$d/src $$d/tests; \
	  cp -R src/core/granite_evals $$d/src/; cp -R $$f/src/. $$d/src/granite_evals/benchmarks/; \
	  cp -R src/core/tests/. $$d/tests/; cp -R $$f/tests/. $$d/tests/ 2>/dev/null || true; \
	  (cd $$d && PYTHONPATH=src $(PYTHON) -m pytest -q -p no:cacheprovider tests); rc=$$?; \
	  rm -rf $$d; [ $$rc -eq 0 ] || exit $$rc; \
	done

prefetch:
	@test -n "$(HF_HOME)" || { echo "HF_HOME is not set: make prefetch HF_HOME=/path/to/hf-cache" >&2; exit 1; }
	$(PYTHON) scripts/prefetch_datasets.py --hf-home "$(HF_HOME)"

check-pins:
	$(PYTHON) scripts/prefetch_datasets.py --check
