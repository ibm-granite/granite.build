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
# Shared GPU base image for granite.build steps. See README.md for what this
# is, when to use it, and the two residual caveats (pip can shadow the
# toolkit; BlueVela's pyxis/enroot runtime still needs an `ldconfig` refresh
# at container launch) that this image does NOT solve by itself.
#
# Pinned to CUDA 12.9.1 to match vLLM 0.27.1's prebuilt `+cu129` wheels (the
# version currently used by granite-evals) -- not NGC's latest tag (13.4.2
# as of writing). Bumping both together is a deliberate, separately-tested
# follow-up; see README.md.
ARG CUDA_IMAGE_TAG=12.9.1-devel-ubuntu22.04
FROM nvcr.io/nvidia/cuda:${CUDA_IMAGE_TAG}

ARG PYTHON_VERSION=3.12
ARG DEBIAN_FRONTEND=noninteractive

# Minimal OS + Python toolchain (mirrors the legacy sage Dockerfile's
# deadsnakes pattern -- Ubuntu 22.04's own repos only ship Python 3.10).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget ca-certificates software-properties-common gnupg build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && rm -rf /var/lib/apt/lists/*

# uv, via the same pinned-binary COPY pattern already used by bfcl-eval's Dockerfile.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# CUDA toolkit env vars, restated explicitly (sage-style) so the intent is
# self-documenting even though the NGC devel base already sets these itself.
# See README.md's "pip can shadow the toolkit" caveat: a consuming step's own
# dependency install can still land its own CUDA runtime wheels ahead of
# these for anything invoked via `python -m` rather than bare `nvcc`.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

WORKDIR /
