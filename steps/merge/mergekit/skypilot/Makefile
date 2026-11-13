# Copyright Granite.Build Authors
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

# mergekit step for the skypilot environment: runs one MergeKit (mergekit-yaml) merge of
# the bound models and registers the merged model. Issue #238.
# Public image: no Dockerfile here, so common.mk treats it as a public-image step.
# Like data/data-designer, setup installs the library on the node with uv.

STEP_NAME := mergekit

# This file defines `test-setup` below (the slurm cluster), so tell common.mk to drop
# its own no-op default.
HAS_TEST_SETUP := true

include ../../../common.mk       # steps/<category>/<name>/<env>: three levels up

# Stand up the local Docker SLURM cluster the slurm build test needs. Once; idempotent.
test-setup:
	$(MAKE) -C $(REPO_ROOT) slurm-setup

# The same for the kube build test: a local kind cluster via `sky local up`.
.PHONY: test-setup-kube
test-setup-kube:
	$(MAKE) -C $(REPO_ROOT) kube-setup

# `make test` imports src/mergekit_prepare.py, which leaves a src/__pycache__/ behind, and
# space/publish-step copy src/ verbatim: drop it first (same as data-designer and dpk).
space publish-step: drop-pycache
.PHONY: drop-pycache
drop-pycache:
	@find $(SRC_DIR) -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

MERGEKIT_TEST_VERSION ?= 0.1.5
# The tests that run real mergekit-yaml (the real-merge tests, the 60-case tokenizer and
# template matrix, the MergeKit round-trips and sync checks). They skip under `make test`,
# whose repo venv must not get mergekit: its dependencies move click past SkyPilot's
# <8.2 pin. This target builds a separate, throwaway venv with the pinned mergekit (once,
# then reused; an interrupted build is redone from scratch) and runs them there, without
# the repo conftest (they need no gbserver).
# Takes about 5 minutes on a laptop CPU. The venv lives outside the repo (the user's cache
# dir, overridable), so the repo's `make lint` and `make format` never scan it.
MERGEKIT_TEST_VENV ?= $(or $(XDG_CACHE_HOME),$(HOME)/.cache)/granite-build/mergekit-test-venv-$(MERGEKIT_TEST_VERSION)
.PHONY: test-mergekit
test-mergekit:
	@command -v uv >/dev/null || { echo "[$(STEP_NAME)] test-mergekit needs uv (https://docs.astral.sh/uv/)"; exit 1; }
	@[ -x "$(MERGEKIT_TEST_VENV)/bin/mergekit-yaml" ] || { \
		rm -rf "$(MERGEKIT_TEST_VENV)" && uv venv -q --python 3.12 "$(MERGEKIT_TEST_VENV)" && \
		uv pip install -q --python "$(MERGEKIT_TEST_VENV)/bin/python" \
			"mergekit==$(MERGEKIT_TEST_VERSION)" pytest pyyaml; }
	PATH="$(MERGEKIT_TEST_VENV)/bin:$$PATH" PYTHONPATH=$(SRC_DIR) \
		"$(MERGEKIT_TEST_VENV)/bin/python" -m pytest --noconftest -p no:cacheprovider $(PYTEST_ARGS) \
		$(TEST_DIR)/test_mergekit_prepare.py $(TEST_DIR)/test_mergekit_prepare_checks.py \
		$(TEST_DIR)/test_mergekit_run_sh.py $(TEST_DIR)/test_mergekit_run_sh_retry.py \
		$(TEST_DIR)/test_mergekit_setup_sh.py
