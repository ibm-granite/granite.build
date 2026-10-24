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
# Public replacement for the private us.icr.io/cil15-shared-registry/bfcl-py311:0.02.
#
# Deliberately NOT based on nvcr.io/nvidia/vllm: the env below pulls its own
# pip CUDA/PyTorch wheels regardless of base image (confirmed via a real
# build: an NGC base's own torch/triton/vllm got uninstalled and replaced by
# these). Pulling a multi-GB proprietary NGC container just to have it
# overwritten is wasted bandwidth, so this uses a plain, fully public
# python:3.14-slim base instead -- pip's own nvidia-*-cu12 wheel packages
# provide the CUDA runtime; only an NVIDIA driver on the host is needed, same
# as vLLM's official wheels.
ARG PYTHON_BASE_TAG=3.14-slim
FROM python:${PYTHON_BASE_TAG}

# Base image ships stale util-linux packages with known CVEs; patch them so
# the pushed image is clean (same fix applied to granite42-data-blend).
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# tree-sitter==0.21.3 (pinned by bfcl-eval itself -- see pyproject.toml's
# override-dependencies comment for why this can't be bumped) has no cp314
# wheel, so uv must build it from source on this base image. build-essential
# provides the C compiler that requires; removed from the image isn't worth
# the layer complexity since this is a slim eval-runtime image, not a
# size-optimized production one.
#
# APT::Sandbox::User=root: apt-get normally drops privileges to the _apt user
# for the download phase via setgroups(), which needs a second UID inside the
# build container's user namespace. Rootless podman on a host with no
# /etc/subuid entry for the building user (confirmed on BlueVela -- no
# subuid/subgid ranges configured for this account) falls back to a
# single-UID mapping with no second UID available, so that setgroups() call
# fails with "operation not permitted" and the build never reaches
# build-essential's actual install. This only disables apt's own internal
# build-time sandboxing -- it has no effect on the resulting image's runtime
# contents, so it's safe to always set rather than gate behind a build arg.
RUN apt-get update -o APT::Sandbox::User=root \
    && apt-get install -y --no-install-recommends -o APT::Sandbox::User=root build-essential \
    && rm -rf /var/lib/apt/lists/*

# One uv-managed venv for everything this step runs: bfcl-eval, soundfile,
# and vllm all together (see pyproject.toml/uv.lock). This used to need two
# separate venvs -- bfcl-eval's own `oss_eval_vllm` extra hard-pinned
# vllm==0.8.5, which would have collided with a current vllm in the same
# site-packages -- but that extra is gone: bfcl_eval's own vllm usage is a
# `subprocess.Popen(["vllm", "serve", ...])` call, gated entirely behind the
# `skip_server_setup` flag (see model_handler/local_inference/base_oss_handler.py
# in the installed package -- confirmed by reading it, bfcl_eval never
# `import vllm`s). This step always runs `bfcl generate --skip-server-setup`
# and points bfcl_eval at the standalone server run-bfcl.sh launches from
# this same venv, via REMOTE_OPENAI_BASE_URL/REMOTE_OPENAI_TOKENIZER_PATH, so
# there's nothing left pulling in the old pin (confirmed by a real `uv lock`:
# vllm, current transformers, and bfcl-eval's own deps all resolve together
# with no conflict).
#
# soundfile: not declared by any of bfcl-eval's own dependencies, but
# bfcl_eval imports it unconditionally (multimodal category support) even
# when running text-only categories -- confirmed via a real run, not from
# reading bfcl-eval's own dependency list. It's in pyproject.toml alongside
# bfcl-eval and vllm for the same reason they are: one locked, reproducible
# venv, not a floating `pip install` resolved fresh at every build.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /opt/bfcl-eval
COPY pyproject.toml uv.lock /opt/bfcl-eval/
RUN uv sync --locked --no-dev --no-cache
ENV PATH="/opt/bfcl-eval/.venv/bin:/opt/bfcl-eval:${PATH}"

COPY src/ /opt/bfcl-eval/
RUN chmod +x /opt/bfcl-eval/run-bfcl.sh
