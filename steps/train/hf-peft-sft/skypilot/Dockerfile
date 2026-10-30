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
# hf-peft-sft image: torch + HF PEFT/TRL/transformers/accelerate + bitsandbytes,
# DeepSpeed and clearml from a fully hashed lock, plus this step's src/.
#
# SECURITY TRADE-OFF (documented exception to the UBI-minimal base rule): SkyPilot
# bootstraps itself by running apt-get INSIDE the container, so a UBI/RPM base
# fails provisioning. The base is NVIDIA's official CUDA devel image on Ubuntu
# (devel: DeepSpeed JIT-compiles its ops), from nvcr.io, pinned by digest.
# Mitigations: digest pins, a hashed lock, apt-get upgrade, and a vulnerability
# scan before `make publish-image`. Runs as root, like the other step images,
# because SkyPilot's docker runtime requires it (README.md, "Security trade-off").
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

# CUDA 13.2, matching torch 2.15.0's bundled runtime (DeepSpeed JIT uses this
# nvcc). Its NVIDIA_REQUIRE_CUDA admits the 580 (aws) and 590 (BlueVela) drivers.
FROM nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:44a9504c6dfb50b1241464241b02a93871928f373de6f5a644cf5fe9f080aa63

ARG DEBIAN_FRONTEND=noninteractive
# Nsight Compute (from the CUDA devel base) is unused at runtime and bundles a Go
# binary with CRITICAL CVEs, so it is purged. The names are versioned per CUDA
# release (`dpkg-query -W | grep -i nsight` in the base lists them).
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends ca-certificates curl git sudo \
 && apt-get purge -y cuda-nsight-compute-13-2 nsight-compute-2026.1.1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# Managed Python under /opt, not root's home: the venv's interpreter is a symlink
# into it, and Pyxis on SLURM runs the job as the submitting user.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0

# Every package, transitive ones included, comes from the hashed lock. DeepSpeed
# ships only an sdist; DS_BUILD_OPS=0 builds it without pre-compiling ops (they
# JIT-compile on first use, hence the devel base).
COPY requirements.lock /tmp/requirements.lock
RUN uv venv --python 3.12 /opt/hf-peft-sft/.venv \
 && DS_BUILD_OPS=0 uv pip install --python /opt/hf-peft-sft/.venv/bin/python \
      --require-hashes --no-deps -r /tmp/requirements.lock \
 && rm /tmp/requirements.lock \
 && chmod -R a+rX /opt/hf-peft-sft /opt/uv-python

# The step's scripts and accelerate configs, called by step.yaml at /opt/hf-peft-sft/.
COPY src/ /opt/hf-peft-sft/
RUN find /opt/hf-peft-sft -maxdepth 2 -not -path '/opt/hf-peft-sft/.venv*' -exec chmod a+rX {} +

# The venv first on PATH also provides the `hf` CLI the aws inline download uses.
ENV HF_PEFT_SFT_HOME=/opt/hf-peft-sft \
    PATH=/opt/hf-peft-sft/.venv/bin:${PATH} \
    HF_HUB_DISABLE_TELEMETRY=1

# Root, like the repo's other step images: SkyPilot's docker runtime saves root's
# environment (HOME=/root) to /etc/profile.d, so a non-root default USER fails
# provisioning (#566). Pyxis on SLURM still runs the container as the job user.
WORKDIR /root
