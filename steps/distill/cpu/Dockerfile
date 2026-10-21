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
# Shared CPU image for the distillation steps that do text/data processing
# only, no GPU (corpus-pin-check, corpus-prep, corpus-sources,
# tokenizer-align, hf-export). See steps/distill/gpu/Dockerfile for the GPU
# sibling and the shared rationale (one image per resource class, not one
# per step -- checked directly against each step's own pyproject.toml, not
# assumed).
#
# No CUDA base: a plain Ubuntu base is enough, and pulling in
# nvcr.io/nvidia/cuda for a CPU-only image would cost multiple GB for
# nothing these steps use. Dependency manifest below (pyproject.toml +
# uv.lock, next to this Dockerfile) is the UNION of corpus-prep's,
# tokenizer-align's, and hf-export's own declared ranges -- see
# pyproject.toml's header for exactly which step contributes which
# dependency. torch is pulled from PyPI's dedicated CPU-only wheel index
# (see pyproject.toml's [tool.uv.sources]), not the default index's cu12x
# wheels, since this image has no CUDA toolkit to match them against.
ARG UBUNTU_TAG=22.04
FROM ubuntu:${UBUNTU_TAG}

ARG DEBIAN_FRONTEND=noninteractive
# 3.12, matching steps/distill/gpu/'s pin for consistency -- none of this
# image's three contributing steps pin an exact minor version (all declare
# requires-python>=3.10), so there's no correctness requirement forcing a
# specific one.
ARG UV_PYTHON=3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv python install ${UV_PYTHON} \
    && uv python pin ${UV_PYTHON} --no-workspace

# The shared trainer library: only tokenizer-align among this image's steps
# imports from it today (confirmed: corpus-pin-check/corpus-sources are
# stdlib-only per their own step-template.yaml headers; corpus-prep/
# hf-export use their own standalone scripts). Copied in anyway so any of
# these steps CAN import it without a code change if one starts needing a
# shared helper.
COPY src/gb_steps_post_training /opt/gb-steps-distillation/src/gb_steps_post_training
ENV PYTHONPATH=/opt/gb-steps-distillation/src

# Let uv own the venv's location/activation -- same convention as the GPU
# image, /stage/.venv for compatibility with every CPU step's existing
# code_config.python/PATH convention (confirmed across corpus-pin-check,
# corpus-prep, corpus-sources, tokenizer-align, hf-export: all hardcode
# /stage/.venv/bin/python today against the opaque kd-sandbox-distill image).
COPY cpu/pyproject.toml cpu/uv.lock /opt/distill-cpu-src/
WORKDIR /opt/distill-cpu-src
ENV UV_PROJECT_ENVIRONMENT=/stage/.venv
RUN uv sync --locked
ENV PATH="/stage/.venv/bin:${PATH}"

WORKDIR /
