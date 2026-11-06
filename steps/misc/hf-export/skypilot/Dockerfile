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
# Image for the generic hf-export step. CPU only and no torch: the step copies files
# and loads config and tokenizer, never the weights. Built from this directory:
#
#   make image publish-image IMAGE_VERSION=<version>
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

COPY requirements.lock /tmp/requirements.lock
RUN uv venv --python 3.12 /opt/hf-export/.venv \
 && uv pip install --python /opt/hf-export/.venv/bin/python \
      --require-hashes --no-deps -r /tmp/requirements.lock \
 && rm /tmp/requirements.lock \
 && chmod -R a+rX /opt/hf-export /opt/uv-python

COPY src/ /opt/hf-export/
RUN find /opt/hf-export -not -path '/opt/hf-export/.venv*' -exec chmod a+rX {} +

# PYTHONPATH and VIRTUAL_ENV are also set in the step template: the SLURM executor
# drops both at run time. HF_HOME is the fixed in-container HF cache mount point.
ENV HF_EXPORT_HOME=/opt/hf-export \
    PYTHONPATH=/opt/hf-export \
    VIRTUAL_ENV=/opt/hf-export/.venv \
    PATH=/opt/hf-export/.venv/bin:${PATH} \
    HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /root
