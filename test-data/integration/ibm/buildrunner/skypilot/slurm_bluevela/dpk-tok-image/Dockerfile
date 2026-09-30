# Image for the granite.build `dpk` step's image mode (dpk_config.dpk_image).
# Contract (steps/dpk/skypilot/USAGE.md): the image must already provide PUBLIC
# DPK, because the step runs `python -m dpk_<transform>.runtime` and skips the
# pip install when dpk_image is set.
FROM python:3.10-slim-bookworm

# SkyPilot's SLURM container-init runs `apt-get install ca-certificates rsync
# curl git wget fuse`; pre-installing them (plus openssh-client) means the
# bootstrap has nothing to fetch and is immune to mirror churn.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates rsync curl git wget fuse openssh-client \
 && rm -rf /var/lib/apt/lists/*

ARG DPK_VERSION=1.1.8
RUN pip install --no-cache-dir \
      "data-prep-toolkit-transforms[tokenization2arrow]==${DPK_VERSION}"

# Fail the build, not a cluster run, if the step's derived module is missing.
RUN python -c "import dpk_tokenization2arrow.runtime"
