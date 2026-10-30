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
#
# openinstruct-sft image: upstream allenai/open-instruct at a pinned release, its
# locked training stack, clearml, and this step's src/ scripts.
#
# SECURITY TRADE-OFF (documented exception to the UBI-minimal base rule):
# SkyPilot bootstraps itself by running apt-get INSIDE the container, so a UBI/RPM
# base fails provisioning. The base is therefore NVIDIA's official CUDA devel image
# on Ubuntu (devel: DeepSpeed JIT-compiles its ops), from nvcr.io (no Docker Hub
# rate limits), pinned by digest. Mitigations: digest pins for every base and a
# vulnerability scan before `make publish-image`. The container runs as root,
# like the other step images, because SkyPilot's docker runtime requires it (see
# the end of this file and README.md).
FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv

FROM nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97

# v0.3.0. finetune.py is deprecated upstream (in favour of olmo_core_finetune.py,
# which has no Granite support) but still maintained; the pin makes the
# deprecation an explicit decision. Bump with the Makefile's IMAGE_VERSION prefix.
ARG OPEN_INSTRUCT_REPO=https://github.com/allenai/open-instruct.git
ARG OPEN_INSTRUCT_COMMIT=26de99604002d6a7b25a2c5f2ee880e388a21dca

ARG DEBIAN_FRONTEND=noninteractive
# The Nsight Compute profiler (from the CUDA devel base) is not used at runtime and
# bundles a Go binary with CRITICAL CVEs, so it is purged.
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends git ca-certificates curl sudo \
 && apt-get purge -y cuda-nsight-compute-12-8 nsight-compute-2025.1.1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# Managed Python (open-instruct pins python-preference = only-managed) goes under
# /opt, not root's home: the venv's interpreter is a symlink into it, and the job
# runs as root (SkyPilot docker) or as the submitting user (Pyxis on SLURM).
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT=0.3.0

WORKDIR /opt/open-instruct
# Fetch exactly the pinned commit, then install the LOCKED environment. The
# default groups are kept: open_instruct.utils imports `beaker` unconditionally
# and beaker-py lives in the `dev` group at this release.
RUN git init -q . \
 && git remote add origin "${OPEN_INSTRUCT_REPO}" \
 && git fetch -q --depth 1 origin "${OPEN_INSTRUCT_COMMIT}" \
 && git checkout -q FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${OPEN_INSTRUCT_COMMIT}" \
 && rm -rf .git

# Upstream finetune.py calls snapshot_download(model_name_or_path) on rank 0 to
# pre-fetch Hub files, and snapshot_download() rejects a local directory
# (HFValidationError: "Repo id must be in the form ..."). The step always passes
# the staged base_model directory, so skip the pre-fetch for a local path. The
# patch fails the build if the upstream line ever changes.
COPY patches/finetune_local_model.py /tmp/finetune_local_model.py
RUN uv run --no-project --python 3.12 python /tmp/finetune_local_model.py open_instruct/finetune.py \
 && rm /tmp/finetune_local_model.py

# Security patches on top of the frozen lock (CRITICAL with a fix, from trivy or
# quay's scanner; see README.md "Scanning"). vllm is REMOVED, not upgraded: the fixed vllm needs
# torch 2.11, which would replace the locked torch 2.10 + flash-attn training
# stack, and nothing finetune.py imports needs vllm (it serves RL/generation).
# The rest are upgraded to the first fixed release. litellm leaves upstream's
# <1.75.2 pin; finetune.py imports it only through optional helpers.
# wandb is upgraded past upstream's ==0.23.1 pin for its bundled wandb-core Go
# binary; the step runs it with WANDB_MODE=disabled (see README.md).
RUN uv sync --frozen \
 && uv pip install --python .venv/bin/python "clearml==2.1.12" \
 && uv pip uninstall --python .venv/bin/python vllm \
 && uv pip install --python .venv/bin/python \
      "pillow==13.0.0" "nltk==3.10.3" "litellm==1.84.0" "PyJWT==2.14.0" \
      "Authlib==1.9.0" "fastmcp==3.2.0" "anyio==4.14.2" "GitPython==3.1.59" \
      "certifi==2026.10.19" "cryptography==46.0.7" "wandb==0.31.0" \
 && chmod -R a+rX /opt/open-instruct /opt/uv-python

# The step's scripts. step.yaml's setup:/run: call them at /opt/openinstruct-sft/.
COPY src/ /opt/openinstruct-sft/

# Readable by any uid (Pyxis runs as the job user); writable by nobody but root.
# The venv and managed Python were made readable in the install layer above, so
# this touches only the small scripts dir (no copy-up of the large venv layer).
RUN chmod -R a+rX /opt/openinstruct-sft

# OPEN_INSTRUCT_COMMIT, not GIT_COMMIT: finetune.py's wandb tagging calls the
# GitHub search API when GIT_COMMIT is set. The venv first on PATH also provides
# the `hf` CLI the aws environment's inline hf download uses.
ENV OPEN_INSTRUCT_HOME=/opt/open-instruct \
    OPEN_INSTRUCT_COMMIT=${OPEN_INSTRUCT_COMMIT} \
    PATH=/opt/open-instruct/.venv/bin:${PATH}

# Runs as root, like the repo's other step images. SkyPilot's docker runtime sets
# the container up as root and saves root's environment (HOME=/root) to
# /etc/profile.d, so a non-root default USER gets HOME=/root on login and
# provisioning fails ("mkdir: cannot create directory '/root'"). See README.md,
# "Security trade-off". Pyxis on SLURM still runs the container as the job user.
WORKDIR /root
