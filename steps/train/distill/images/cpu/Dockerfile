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
# only, no GPU (data-validate, data-prep, data-mixture, tokenizer-align).
# hf-export moved to steps/aux/hf-export with its own image.
# See steps/train/distill/images/gpu/Dockerfile for the GPU
# sibling and the shared rationale (one image per resource class, not one
# per step -- checked directly against each step's own pyproject.toml, not
# assumed).
#
# No CUDA base: a plain Ubuntu base is enough, and pulling in
# nvcr.io/nvidia/cuda for a CPU-only image would cost multiple GB for
# nothing these steps use. Dependency manifest below (pyproject.toml +
# uv.lock, next to this Dockerfile) is the UNION of data-prep's,
# tokenizer-align's, and (before it moved) hf-export's own declared ranges -- see
# pyproject.toml's header for exactly which step contributes which
# dependency. torch is pulled from PyPI's dedicated CPU-only wheel index
# (see pyproject.toml's [tool.uv.sources]), not the default index's cu12x
# wheels, since this image has no CUDA toolkit to match them against.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv
FROM ubuntu:22.04@sha256:076e7bf82319619f28dd0aa9c02ab34b055ad37100395b4f49a2ffba04fb5e11

ARG DEBIAN_FRONTEND=noninteractive
# 3.12, matching steps/train/distill/images/gpu/'s pin for consistency -- none of this
# image's three contributing steps pin an exact minor version (all declare
# requires-python>=3.10), so there's no correctness requirement forcing a
# specific one.
ARG UV_PYTHON=3.12

# Security updates from the Ubuntu archive: the base image is a snapshot, and its Ubuntu
# packages (kernel headers, openssl, apt) fall behind the archive's fixes. The upgrade
# reads /etc/apt/sources.list only (SourceParts=-), so a vendor repository in
# sources.list.d is never consulted and its packages keep the base image's versions.
# Not reproducible from the commit alone: a later rebuild picks up newer archive packages,
# so the published, trivy-scanned digest is the record of what shipped.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates \
    && apt-get upgrade -y --no-install-recommends -o Dir::Etc::SourceParts=- \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_NO_CACHE=1
COPY --from=uv /uv /usr/local/bin/uv
# pip is removed from the uv-managed interpreter: nothing runs it (uv builds and owns the
# venv), and its vendored copies (msgpack, setuptools, urllib3) carry advisories of their own.
RUN uv python install ${UV_PYTHON} \
    && uv python pin ${UV_PYTHON} --no-workspace \
    && "$(uv python find ${UV_PYTHON})" -m pip uninstall -y --break-system-packages --root-user-action=ignore pip

# Everything the CPU steps run lives under DISTILL_HOME, baked in at build time
# (see steps/train/distill/images/gpu/Dockerfile for the layout); nothing is cloned or mounted
# at job launch.
ENV DISTILL_HOME=/opt/distill
COPY src/gb_steps_post_training ${DISTILL_HOME}/src/gb_steps_post_training
ENV PYTHONPATH=${DISTILL_HOME}/src

# Let uv own the venv's location/activation -- same convention as the GPU
# image: /stage/.venv, the path the CPU steps' templates set as VIRTUAL_ENV
# (and that the opaque kd-sandbox-distill image these images replace used).
COPY images/cpu/pyproject.toml images/cpu/uv.lock /opt/distill-cpu-src/
WORKDIR /opt/distill-cpu-src
ENV UV_PROJECT_ENVIRONMENT=/stage/.venv
ENV VIRTUAL_ENV=/stage/.venv
RUN uv sync --locked
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY data-validate/skypilot/src ${DISTILL_HOME}/steps/data-validate
COPY data-mixture/skypilot/src ${DISTILL_HOME}/steps/data-mixture
COPY data-prep/skypilot/src ${DISTILL_HOME}/steps/data-prep
COPY tokenizer-align/skypilot/src ${DISTILL_HOME}/steps/tokenizer-align

# The HF cache's fixed in-container mount point (see the GPU image).
ENV HF_HOME=/opt/hf-cache

# Non-root is opt-in: the SLURM/enroot flow runs these images against
# group-shared filesystems and a shared HF cache whose group membership a fixed
# uid cannot assume, so the default (0) preserves the validated cluster
# behavior. Kubernetes and plain docker deployments, where running as root is
# the actual exposure, pass --build-arg DISTILL_UID=1000.
ARG DISTILL_UID=0
USER ${DISTILL_UID}

WORKDIR /