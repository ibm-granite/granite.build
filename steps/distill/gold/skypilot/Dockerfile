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
# GPU runtime base for the distill-gold step (LSF and SLURM subtypes alike),
# replacing the opaque kd-sandbox-distill:0.1.0-uv (no Dockerfile/build source
# for that image was ever found). Content is a deliberate copy of
# images/cuda-devel-base's Dockerfile -- per-step Dockerfiles, duplicated
# content, is the convention here (steps/common.mk auto-detects a Dockerfile
# next to a step's own Makefile and wires up image/publish-image/image_id;
# see that file and steps/bfcl-eval/skypilot/Dockerfile for the reference
# pattern). Keep this file in sync with images/cuda-devel-base/Dockerfile by
# hand when that one changes -- there is no shared include mechanism for
# Dockerfiles here.
#
# DELIBERATELY SLIM: no project dependencies baked in. This step's own
# torch/transformers/trl/vllm/deepspeed/etc. (pinned in gb-steps-distillation's
# steps/distill-gold-train/pyproject.toml + uv.lock) install at JOB LAUNCH
# TIME via `uv sync --locked`, cached on shared storage keyed by the lock
# file's hash -- see this step's step-template.yaml "runtime venv sync"
# block. This keeps one slim image/Dockerfile per step instead of one
# per-step-per-dependency-set image to rebuild whenever a pin changes, and it
# is also why the same image serves both the LSF and SLURM subtypes: neither
# bakes in anything subtype-specific.
#
# Pinned to CUDA 12.9.1 to match vLLM 0.27.1's prebuilt `+cu129` wheels (the
# version currently used by granite-evals) -- not NGC's latest tag (13.4.2
# as of writing). Bumping both together is a deliberate, separately-tested
# follow-up; see images/cuda-devel-base/README.md.
ARG CUDA_IMAGE_TAG=12.9.1-devel-ubuntu22.04
FROM nvcr.io/nvidia/cuda:${CUDA_IMAGE_TAG}

ARG DEBIAN_FRONTEND=noninteractive
# The default a bare `uv run`/`uv sync`/`uv venv` resolves to (no --python
# flag); SECONDARY is installed alongside it and reachable via
# `--python 3.13`. distill-gold-train itself pins requires-python==3.12.* --
# uv auto-downloads that interpreter transparently on first `uv sync`
# against this step's own pyproject.toml, so no third Python version needs
# installing here ahead of time (confirmed locally: `uv sync`/`uv lock`
# downloads a missing interpreter version with no `uv python install` step
# required).
ARG UV_PYTHON_DEFAULT=3.14
ARG UV_PYTHON_SECONDARY=3.13

# build-essential: for this step's own `uv sync` building a C extension from
# source if its lock ever needs one. No deadsnakes/apt Python here -- uv
# manages Python itself below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv is the default manager for both Python itself and dependencies in this
# image -- not apt/deadsnakes, not system pip. Installed interpreters live
# under UV_PYTHON_INSTALL_DIR, pinned here rather than left at uv's own
# default (~/.local/share/uv/python) so the path stays stable regardless of
# which user/HOME a later layer runs as.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv python install ${UV_PYTHON_DEFAULT} ${UV_PYTHON_SECONDARY} \
    && uv python pin ${UV_PYTHON_DEFAULT} --no-workspace

# Deliberately NOT symlinking a bare `python3`/`python` to either of these.
# The runtime venv-sync step (this step's own run: block) resolves its own
# interpreter via UV_PROJECT_ENVIRONMENT, not a system-level `python3`.

# CUDA toolkit env vars, restated explicitly (sage-style) so the intent is
# self-documenting even though the NGC devel base already sets these itself.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /
