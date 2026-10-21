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
# Shared GPU image for the distillation steps that load models and/or train
# (gold, sft, logit-precompute, eval, vllm-server, gen-smoke, probe).
#
# ONE image, not one per step: checked directly against the real uv.lock
# files in github.com/laminair/gb-steps-distillation at a5d59bc4 --
# distill-gold-train and distill-logit-precompute (the only two steps with
# resolved locks) pin IDENTICAL torch 2.8.0 / transformers 5.8.0 /
# accelerate 1.12.0 / tokenizers 0.22.2, and every other GPU step's declared
# range (e.g. transformers>=5.8.0,<6) is satisfied by that same set. The 11
# distillation steps already ran on one opaque shared image
# (kd-sandbox-distill:0.1.0-uv) with no reported incompatibility; splitting
# into one image per step (tried, then reverted) would only have turned
# into a full-time job keeping N sets of pins mutually compatible for no
# real isolation benefit. This image replaces that opaque one with a known,
# source-controlled build -- same one-shared-image shape, not a new one.
#
# Dependency manifest below (pyproject.toml + uv.lock, next to this
# Dockerfile) is distill-gold-train's own, vendored UNCHANGED from the
# pinned commit: it's the fullest GPU dependency set (trl, vllm, deepspeed,
# kernels) of any step, and every other GPU step's looser range already
# resolves inside it.
#
# Pinned to CUDA 12.8.2 (not 12.9.x) to match the vendored uv.lock's torch
# 2.8.0 cu128 wheels (nvidia-cublas-cu12==12.8.4.1 etc.) -- deepspeed is
# sdist-only and compiles native CUDA extensions against nvcc at install
# time below, so the devel toolkit's major.minor has to match what torch's
# own runtime libs expect.
ARG CUDA_IMAGE_TAG=12.8.2-devel-ubuntu22.04
FROM nvcr.io/nvidia/cuda:${CUDA_IMAGE_TAG}

ARG DEBIAN_FRONTEND=noninteractive
# 3.12 EXACTLY, matching the vendored pyproject.toml's own
# requires-python=">=3.12,<3.13" (itself a deliberate single-minor-version
# pin, not a preference -- see that file's own header comment).
ARG UV_PYTHON=3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv python install ${UV_PYTHON} \
    && uv python pin ${UV_PYTHON} --no-workspace

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Everything the GPU steps run lives under DISTILL_HOME, baked in at build time:
# nothing is cloned or mounted at job launch (site-specific files such as the NCCL
# tuning config, the HF cache and chat templates are supplied by the environment).
#   src/    the shared trainer library (gb_steps_post_training)
#   shell/  shell helpers sourced by the launchers
#   steps/distill-gold-train/configs  DeepSpeed / accelerate configs
#   steps/  each step's launcher scripts, one directory per step
ENV DISTILL_HOME=/opt/distill
COPY src/gb_steps_post_training ${DISTILL_HOME}/src/gb_steps_post_training
ENV PYTHONPATH=${DISTILL_HOME}/src

# Let uv own the venv's location and activation -- UV_PROJECT_ENVIRONMENT is
# uv's own supported mechanism for this, so `uv sync` itself places the venv
# exactly where every GPU step's code_config.python/PATH convention already
# expects it (/stage/.venv), with no separate manual PATH bookkeeping needed
# here beyond what `uv python install/pin` above already did.
COPY gpu/pyproject.toml gpu/uv.lock /opt/gold-train-src/
WORKDIR /opt/gold-train-src
# VIRTUAL_ENV makes `uv python find` (what the step launchers use) resolve to this
# venv's interpreter; UV_PROJECT_ENVIRONMENT makes `uv sync` build it here.
ENV UV_PROJECT_ENVIRONMENT=/stage/.venv
ENV VIRTUAL_ENV=/stage/.venv
RUN uv sync --locked
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Kept at its historical relative path: recipes pass ds_config relative to DISTILL_HOME.
COPY gpu/configs ${DISTILL_HOME}/steps/distill-gold-train/configs
COPY gpu/shell ${DISTILL_HOME}/shell
COPY gold/skypilot/src ${DISTILL_HOME}/steps/gold
COPY sft/skypilot/src ${DISTILL_HOME}/steps/sft
COPY logit-precompute/skypilot/src ${DISTILL_HOME}/steps/logit-precompute
COPY eval/skypilot/src ${DISTILL_HOME}/steps/eval
COPY gen-smoke/skypilot/src ${DISTILL_HOME}/steps/gen-smoke
COPY probe/skypilot/src ${DISTILL_HOME}/steps/probe

# The HF cache's fixed in-container mount point. The environment bind-mounts its
# cache here (or overrides HF_HOME); the steps never set it themselves.
ENV HF_HOME=/opt/hf-cache

WORKDIR /
