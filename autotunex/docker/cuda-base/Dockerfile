# syntax=docker/dockerfile:1

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

# ---------------------------------------------------------------------------
# CUDA kernel base for the root Dockerfile's DEVICE=cuda build: the CUDA devel
# toolchain, python3.12 in /opt/venv, torch, and flash-attn / mamba-ssm /
# causal-conv1d compiled from source. The projects ship no wheel for torch 2.11 +
# CUDA 13, and compiling them is ~35 minutes on 29 cores, so they live here,
# where they are rebuilt only when one of these pins changes, instead of in
# every AutoTuneX image build.
#
# Build from autotunex/ (the build context), not from this directory:
#
#   docker build --platform linux/amd64 -f docker/cuda-base/Dockerfile -t <ref> .
#
# `make publish-base-image` does that and pushes it under a tag that is never
# reused. The root Dockerfile pins the published image by digest.
#
# The pins below must equal torch in fm-tune's [core] and the three kernels in
# its [full] (tests/test_dockerfile_variants.py checks this): the app build
# installs those extras on top and refuses to compile the kernels, so a stale
# base fails that build instead of silently recompiling.
# ---------------------------------------------------------------------------

# Pinned by digest as well as tag, so the tag cannot move under a published image.
# The digest is the multi-arch index, identical on Docker Hub and nvcr.io; bump
# both together. It must stay a CUDA 13 `-devel` tag: torch 2.11's PyPI wheels are
# built for CUDA 13.0, the kernels compile with its nvcc, and deepspeed JIT-compiles
# its ops at runtime in the image built on top.
ARG CUDA_BASE=docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d
FROM ${CUDA_BASE}

# python3.12 specifically, to match the CPU stage of the root Dockerfile. Ubuntu
# 24.04's system python already IS 3.12, so no third-party PPA is needed.
# python3.12-dev provides Python.h, which the kernels' C++ sources include through
# torch's headers.
#
# build-essential is named explicitly even though the default `-devel` base
# already ships gcc (verified: apt reports it "already the newest version").
# deepspeed and verl fall back to an sdist build that needs a host compiler, and
# CUDA_BASE is overridable — a leaner `-runtime` or `-base` tag would not have
# one. A no-op on the default base, so it costs nothing to keep.
#
# The venv is not cosmetic either: Ubuntu marks its system python PEP 668
# "externally-managed" and pip refuses to install into it. Putting the venv's
# bin on PATH also makes bare `python` and `pip` resolve, which is what lets the
# root Dockerfile's shared runtime stage (pip calls, HEALTHCHECK, CMD) work
# unchanged on this base and on the CPU one.
#
# userdel: Ubuntu 24.04 ships a stock `ubuntu` account at UID 1000, which the
# runtime stage's `useradd --uid 1000 appuser` would collide with. `-r` exits
# non-zero when there is no home directory to remove, hence the guard.
#
# DEBIAN_FRONTEND is exported rather than set with ENV so it does not persist
# into the image: without it, tzdata (pulled in by python3.12) opens its
# interactive geographic-area prompt and only falls through because the build has
# no tty.
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && (userdel -r ubuntu 2>/dev/null || true)
ENV PATH=/opt/venv/bin:$PATH

# The kernels' build recipe (torch matched to the runtime env, ninja, CUDA arch
# list, MAX_JOBS, ...) is fm-tune's: src/fm-tune/pyproject.toml, under
# `[tool.uv.extra-build-dependencies]` / `[tool.uv.extra-build-variables]`. uv
# reads those tables from the pyproject.toml in its working directory, so only
# that file is copied, and the install runs from beside it. pip does NOT read
# them: an isolated pip build has no torch, so flash-attn's setup.py would fail
# to import it.
#
# torch first, with its dependencies (the nvidia-* CUDA runtime wheels). Then the
# kernels with --no-deps, so this layer depends on nothing but the pins below;
# their runtime dependencies come in the next layer. torch is named again in
# that second install, already satisfied, because match-runtime resolves the
# build-time torch from the packages in the install.
COPY src/fm-tune/pyproject.toml /opt/kernel-build/pyproject.toml
RUN pip install --no-cache-dir uv==0.13.7 \
    && cd /opt/kernel-build \
    && uv pip install --no-cache --python "$(command -v python)" "torch==2.11.0" \
    && uv pip install --no-cache --python "$(command -v python)" --no-deps \
        "torch==2.11.0" \
        "flash-attn==2.8.3" \
        "mamba-ssm==2.3.1" \
        "causal-conv1d==1.7.0" \
    && pip uninstall -y uv \
    && rm -rf /opt/kernel-build

# The kernels' runtime dependencies, the ones their packages declare: without
# them `import flash_attn` and `import mamba_ssm` fail (einops, transformers),
# and this base is the image that is published and smoke-tested on its own.
# Constrained to fm-tune's uv.lock, so they are the versions the app build's
# [full] install would choose anyway, and that install leaves them alone.
# `--only-binary` keeps a resolution surprise from recompiling a kernel here.
# A separate layer, so a lock change does not invalidate the compile above.
COPY src/fm-tune/pyproject.toml src/fm-tune/uv.lock /opt/kernel-deps/
RUN pip install --no-cache-dir uv==0.13.7 \
    && cd /opt/kernel-deps \
    && uv export --frozen --no-hashes --no-emit-project --extra full -o constraints.txt \
    && uv pip install --no-cache --python "$(command -v python)" -c constraints.txt \
        --only-binary flash-attn --only-binary mamba-ssm --only-binary causal-conv1d \
        einops packaging ninja triton transformers \
    && pip uninstall -y uv \
    && rm -rf /opt/kernel-deps

# What the kernels were compiled against, for the app build to check: it refuses a
# base whose torch differs from fm-tune's pin (an ABI mismatch would only show up
# as an ImportError at runtime). The image is linux/amd64 only.
ENV AUTOTUNEX_KERNEL_BASE_TORCH=2.11.0
