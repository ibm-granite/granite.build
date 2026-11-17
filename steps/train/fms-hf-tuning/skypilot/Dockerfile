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
# fms-hf-tuning image: foundation-model-stack/fms-hf-tuning 3.2.2 with its torch /
# transformers / trl / peft / accelerate / datasets closure, the prebuilt
# flash-attn wheel and clearml, from a fully hashed lock, plus this step's src/.
#
# SECURITY TRADE-OFFS (documented exceptions, README.md "Security trade-off"):
# - Base: SkyPilot bootstraps itself by running apt-get INSIDE the container, so
#   a UBI/RPM base fails provisioning. The base is NVIDIA's official CUDA *base*
#   image on Ubuntu from nvcr.io, pinned by digest. `base`, not `devel`: nothing
#   compiles at runtime (torch ships the CUDA 12.8 runtime, flash-attn is a
#   prebuilt wheel).
# - Pinned below latest: fms-hf-tuning 3.2.2 caps torch<2.9, transformers<=4.55.4,
#   trl/peft 0.19.x and datasets 4.x; only upstream can lift that.
# - Runs as root, like the other step images: SkyPilot's docker runtime saves
#   root's environment (HOME=/root) to /etc/profile.d, so a non-root default USER
#   fails provisioning (#566). Pyxis on SLURM remaps the job to root inside the
#   container (--container-remap-root); on the host it is the submitting user.
# Mitigations: digest pins, a hashed lock, apt-get upgrade, no pip in the image,
# and a vulnerability scan before `make publish-image`.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

# CUDA 12.8, matching torch 2.8.0's bundled runtime (nvidia-cuda-runtime-cu12
# 12.8.90). Its NVIDIA_REQUIRE_CUDA admits the 580 (aws) and 590 (BlueVela) drivers.
FROM nvcr.io/nvidia/cuda:12.8.2-base-ubuntu24.04@sha256:42366aad4eeef8e5a509c9f99db96d611ada727d0685e5b769c19a644dda484c

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends ca-certificates curl git sudo \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# Managed Python under /opt, not root's home: the venv's interpreter is a symlink
# into it, and /opt (a+rX) works whatever user and $HOME the container runtime
# gives the job. No bytecode at install time: quay's scanner stalls on it (see
# README, Scanning). SkyPilot runs the job as root in a writable container (Pyxis:
# --container-remap-root --container-writable), so Python writes it on first import.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0

# Every package, transitive ones included, comes from the hashed lock (the
# flash-attn wheel by URL + sha256). uv never needs pip, so pip and ensurepip are
# removed from the managed CPython: their vendored packages were the only
# fixable HIGH findings in hf-peft-sft's first scan.
COPY requirements.lock /tmp/requirements.lock
RUN uv venv --python 3.12 /opt/fms-hf-tuning/.venv \
 && uv pip install --python /opt/fms-hf-tuning/.venv/bin/python \
      --require-hashes --no-deps -r /tmp/requirements.lock \
 && rm /tmp/requirements.lock \
 && rm -rf /opt/uv-python/*/lib/python3.12/site-packages/pip /opt/uv-python/*/lib/python3.12/site-packages/pip-*.dist-info /opt/uv-python/*/lib/python3.12/ensurepip \
 && chmod -R a+rX /opt/fms-hf-tuning /opt/uv-python

# Security patch on the locked transformers 4.55.4 (fms-hf-tuning caps it there):
# backport of the CVE-2026-5241 fix (LightGlue nested trust_remote_code; quay rates
# it Critical). The patch fails the build if its target lines ever change. See
# README.md "Scanning".
COPY patches/lightglue_no_remote_code.py /tmp/lightglue_no_remote_code.py
# -B: this build-time run writes no bytecode into the venv, not even for
# setuptools' _distutils_hack, which a .pth imports at startup.
RUN /opt/fms-hf-tuning/.venv/bin/python -B /tmp/lightglue_no_remote_code.py \
      /opt/fms-hf-tuning/.venv/lib/python3.12/site-packages/transformers/models/lightglue \
 && rm /tmp/lightglue_no_remote_code.py

# The step's scripts and accelerate configs, called by step.yaml at /opt/fms-hf-tuning/.
COPY src/ /opt/fms-hf-tuning/
RUN find /opt/fms-hf-tuning -maxdepth 2 -not -path '/opt/fms-hf-tuning/.venv*' -exec chmod a+rX {} +

# The venv first on PATH also provides `accelerate` and the `hf` CLI the aws
# inline download uses.
ENV FMS_HF_TUNING_HOME=/opt/fms-hf-tuning \
    PATH=/opt/fms-hf-tuning/.venv/bin:${PATH} \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /root
