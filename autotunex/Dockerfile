# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build-time device and job-backend selection. Declared before the first FROM
# because only a global ARG can be used in a FROM line.
#
# Build from the REPOSITORY ROOT, not autotunex/: the llmb backend installs
# granite.build from the same checkout. Always name this file — without -f the
# root Dockerfile (the gbserver image) is built instead. What enters the context
# is set by autotunex/Dockerfile.dockerignore, not the root .dockerignore.
#
#   docker build -f autotunex/Dockerfile -t autotunex:local .                           # CPU, local backend (default)
#   docker build -f autotunex/Dockerfile -t autotunex:cuda --build-arg DEVICE=cuda .    # CUDA + [full]
#   docker build -f autotunex/Dockerfile -t autotunex:gb --build-arg JOB_BACKEND=llmb . # llmb backend
#
# An llmb build from a git worktree or a shallow clone has no reachable v* tag;
# pass the version instead: --build-arg GB_VERSION=0.3.10 (see the gb-wheel stage).
#
# DEVICE names one of the two base stages below, not an image reference —
# override CUDA_KERNEL_BASE for that. The value vocabulary (cpu|cuda) matches
# fm-tune's own runtime override, FMTUNE_DEVICE=cuda|mps|cpu, on purpose.
#
# JOB_BACKEND picks one of the two backend stages below (`backend-local`,
# `backend-llmb`): `local` trains in
# process; `llmb` submits builds to a granite.build server, trains nothing here,
# and is therefore CPU-only.
# ---------------------------------------------------------------------------
ARG DEVICE=cpu
ARG JOB_BACKEND=local
# The CUDA kernel base: the CUDA 13 devel toolchain, python3.12 in /opt/venv, torch,
# and flash-attn / mamba-ssm / causal-conv1d compiled from source. It is built from
# docker/cuda-base/Dockerfile (`make publish-base-image`) and pinned by digest, so
# the tag cannot move under this build. Compiling the kernels takes ~35 minutes;
# keeping them in that base means an AutoTuneX build does not repeat it.
# linux/amd64 only, like the base it starts from.
ARG CUDA_KERNEL_BASE=quay.io/granitebuild/autotunex/cuda-base:cu130-torch2.11-1@sha256:fd916fffb24066b9003829d691b58c9a450e9098a158de285f848f8148089ff4

# ---------------------------------------------------------------------------
# Stage 1 — build the SvelteKit SPA
# ---------------------------------------------------------------------------
FROM node:20-slim AS ui-builder

WORKDIR /app/ux

# Same-origin by default: the SPA client appends /api/v1, /auth and /health to
# this host root itself, so an empty value makes it call the API on the same
# origin the container serves. Override only if the API is fronted elsewhere:
#   --build-arg PUBLIC_AUTOTUNEX_API_URL=https://api.example.com
ARG PUBLIC_AUTOTUNEX_API_URL=""

# Install deps first for layer caching (re-runs only when the lockfile changes).
COPY autotunex/src/ux/package.json autotunex/src/ux/package-lock.json ./
RUN npm ci

# Build the static SPA (adapter-static; base path /autotune; index.html fallback).
COPY autotunex/src/ux/ ./
RUN echo "PUBLIC_AUTOTUNEX_API_URL=$PUBLIC_AUTOTUNEX_API_URL" >> .env \
    && npm run build

# ---------------------------------------------------------------------------
# Stage 1b — a granite.build wheel from THIS checkout, for the llmb backend.
# Only the `backend-llmb` stage below uses it, so a `local` build never runs this stage
# and needs neither .git nor its tags.
#
# granite.build takes its version from git tags (setuptools_scm). With no
# reachable v* tag it is stamped below the llmb CLI's supported floor, and every
# `llmb build start` from AutoTuneX then exits 1 — so fail the build here. A git
# worktree fails too: its .git is a file pointing outside the build context.
#
# GB_VERSION skips that check and stamps the version given, for checkouts that
# cannot reach a tag (worktrees, shallow CI clones). An empty value is ignored by
# setuptools_scm, so the ENV is safe to set unconditionally.
#
# .git is bind-mounted into the one RUN that reads it rather than copied: a copy
# would put it in a layer, and make every commit, fetch or rebase invalidate the
# layers after it.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS gb-wheel
ARG GB_VERSION=""
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_GRANITE_BUILD=${GB_VERSION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /gb
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY test/ ./test/
COPY configurations/ ./configurations/
RUN --mount=type=bind,source=.git,target=/gb/.git \
    { [ -n "$GB_VERSION" ] || git describe --tags --match 'v[0-9]*' >/dev/null 2>&1; } \
      || { echo "FATAL: no v* version tag is reachable in this checkout (or it is a git worktree)." >&2; \
           echo "       granite.build would be stamped below the llmb CLI's version floor, and" >&2; \
           echo "       every job launch would exit 1. Build from a full clone with tags, or" >&2; \
           echo "       pass the version: --build-arg GB_VERSION=X.Y.Z." >&2; \
           exit 1; } \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ---------------------------------------------------------------------------
# Stage 2a — base: CPU. python:3.12-slim already has python and pip, so the only
# job here is to declare what the shared runtime stage installs. Both ENVs below
# are inherited by the stage that does `FROM cpu`.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS cpu
ENV FM_TUNE_EXTRAS=core

# ---------------------------------------------------------------------------
# Stage 2b — base: CUDA, for GPU training with the `local` job backend.
#
# Everything this stage needs is already in the kernel base (see
# docker/cuda-base/Dockerfile): the CUDA 13 devel toolchain (deepspeed JIT-compiles
# its ops at runtime), python3.12 in /opt/venv on PATH, so the runtime stage's
# pip calls, HEALTHCHECK and CMD work unchanged on both bases, the stock `ubuntu`
# account removed, so `useradd --uid 1000 appuser` does not collide, and torch
# plus the three compiled kernels.
#
# Nothing here sets NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES on
# purpose: NVIDIA's own images already set them and the base inherits them, so
# re-declaring would only invite drift.
# ---------------------------------------------------------------------------
FROM ${CUDA_KERNEL_BASE} AS cuda
ENV FM_TUNE_EXTRAS=full

# ---------------------------------------------------------------------------
# Stage 2c — backend: local. Trains in process with fm-tune's extras from the
# device stage. These defaults are the ones this image has always baked.
# ---------------------------------------------------------------------------
FROM ${DEVICE} AS backend-local
ENV AUTOTUNEX_JOB_BACKEND=local \
    AUTOTUNEX_AUTO_CREATE_SCHEMA=true \
    AUTOTUNEX_EXTRAS="" \
    GB_WHEEL=""

# ---------------------------------------------------------------------------
# Stage 2d — backend: llmb. Submits builds to an external granite.build server
# and trains nothing here, so it brings in the granite.build wheel from this
# checkout (the llmb CLI, and the gbcli package live build logs read) for the
# runtime stage to install together with AutoTuneX's mysql + mcp extras, and
# fm-tune at its torch-free base (FM_TUNE_EXTRAS=none).
#
# AUTO_CREATE_SCHEMA stays false and TRAINER_REF oss-main, as Dockerfile.gb had
# them: this is the image deployments run, against a migrated database.
# ---------------------------------------------------------------------------
FROM ${DEVICE} AS backend-llmb
RUN [ "$FM_TUNE_EXTRAS" = "core" ] \
    || { echo "FATAL: JOB_BACKEND=llmb trains on the granite.build server, not in this image;" >&2; \
         echo "       build it with DEVICE=cpu." >&2; \
         exit 1; }
# git, as Dockerfile.gb had it: granite.build depends on GitPython, which shells out to it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=gb-wheel /wheels/ /tmp/gb-wheels/
ENV AUTOTUNEX_JOB_BACKEND=llmb \
    AUTOTUNEX_AUTO_CREATE_SCHEMA=false \
    AUTOTUNEX_JOB_TRAINER_REF=oss-main \
    AUTOTUNEX_EXTRAS="[mysql,mcp]" \
    FM_TUNE_EXTRAS=none \
    GB_WHEEL="/tmp/gb-wheels/*"

# ---------------------------------------------------------------------------
# Stage 3 — runtime: the AutoTuneX API, serving the SPA in-process. `FROM
# backend-${JOB_BACKEND}` picks one of the two backend stages above, each sitting on the
# `${DEVICE}` base; everything below is shared, which is why the extras and the
# backend defaults arrive as inherited ENV.
# ---------------------------------------------------------------------------
FROM backend-${JOB_BACKEND} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install AutoTuneX + its exactly-pinned deps, plus the backend's extras
# (AUTOTUNEX_EXTRAS: none for `local`, mysql + mcp for `llmb`). Every dependency
# ships a cp312 manylinux wheel, so no compiler/build-essential layer is needed.
# For `llmb`, GB_WHEEL names the granite.build wheel the backend stage copied in,
# so pip resolves granite.build and AutoTuneX in one call; for `local` it is
# empty. Left unquoted on purpose so the shell expands its glob.
#
# The wheel's version changes with every commit, so this layer reruns often; the
# pip cache mount keeps the reruns from downloading the dependency tree again.
# It is a mount, so the cache never lands in the image.
COPY autotunex/pyproject.toml autotunex/README.md ./
COPY autotunex/src/ ./src/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install ".${AUTOTUNEX_EXTRAS}" $GB_WHEEL

# Install the vendored autotune core (src/fm-tune). WHICH extras arrive is set by
# the base stage: `core` on CPU — the lean training stack the default `local` job
# backend runs an in-process trial with (ray, torch, transformers, trl, peft,
# accelerate, datasets, tokenizers) — and `full` on CUDA, which adds the online-RL
# stack (verl/vLLM, deepspeed, flash-attn). Either way `autotune.catalog` becomes
# importable, which the config-template / dataset-type wizard endpoints need.
# Quoted so the shell does not glob `[...]`.
#
# The two-step order is load-bearing:
#
#   1. `[core]` first with pip, so torch==2.11.0 is installed before anything else
#      can express an opinion about it. NOTE: on linux/amd64 torch==2.11.0
#      resolves to the default CUDA wheel — multi-GB, and it simply runs on CPU
#      when there is no GPU; pulling CPU torch from download.pytorch.org/whl/cpu
#      to slim the amd64 CPU image is a known, still-deferred optimization. On
#      linux/arm64 PyPI serves a CPU-only build instead, so that image is smaller
#      already and the CUDA base does not apply there at all.
#   2. `[$FM_TUNE_EXTRAS]` when it is not `core`, installed with uv from inside
#      src/fm-tune. On the CUDA base, flash-attn, mamba-ssm and causal-conv1d are
#      already installed (compiled in docker/cuda-base/Dockerfile at the same
#      pins), so uv only adds verl/vLLM, deepspeed and the rest. uv reads
#      fm-tune's [tool.uv] tables only from the project directory, hence the `cd`.
#
# Two guards keep the base and fm-tune's pins in step. `--only-binary` forbids
# compiling the kernels here, so a moved kernel pin fails at resolution instead
# of spending ~35 minutes compiling. And torch must still be the one the base
# compiled the kernels against (AUTOTUNEX_KERNEL_BASE_TORCH): step 1 would
# otherwise quietly replace it, and kernels built for another torch only fail at
# import, at runtime. Neither guard sees a changed CUDA arch list or MAX_JOBS in
# [tool.uv.extra-build-variables]; rebuild the base for those too.
#
# On the CPU base the guard is false and this collapses to exactly
# `pip install "./src/fm-tune[core]"` — the same install this image has today.
#
# The llmb backend sets FM_TUNE_EXTRAS=none: it trains nothing here, so it
# installs fm-tune at its torch-free base (pyyaml only, so the catalog imports)
# and stops.
RUN if [ "$FM_TUNE_EXTRAS" = "none" ]; then pip install --no-cache-dir ./src/fm-tune; exit 0; fi \
    && pip install --no-cache-dir "./src/fm-tune[core]" \
    && if [ "$FM_TUNE_EXTRAS" != "core" ]; then \
         python -c "import os, sys, torch; base = os.environ['AUTOTUNEX_KERNEL_BASE_TORCH']; sys.exit(0 if torch.__version__.split('+')[0] == base else f'torch {torch.__version__} differs from the {base} the kernel base was compiled against; rebuild it (make publish-base-image)')" \
         && pip install --no-cache-dir uv==0.13.7 \
         && cd src/fm-tune \
         && uv pip install --no-cache --python "$(command -v python)" \
              --only-binary flash-attn --only-binary mamba-ssm --only-binary causal-conv1d \
              ".[$FM_TUNE_EXTRAS]" \
         && pip uninstall -y uv; \
       fi

# The prebuilt SPA from stage 1.
COPY --from=ui-builder /app/ux/build /app/ux-build

# Self-sufficient standalone defaults. Every one is overridable at run time with
# `podman run -e AUTOTUNEX_...=...`. All writable state lives under /data (a
# volume) so the SQLite DB and artifacts survive container restarts. The job
# backend, schema creation and trainer ref come from the backend stage.
ENV AUTOTUNEX_ENVIRONMENT=dev \
    AUTOTUNEX_DATABASE_URL=sqlite+aiosqlite:////data/autotunex.db \
    AUTOTUNEX_AUTH_PROVIDERS=[\"disabled\"] \
    AUTOTUNEX_STANDALONE_ROLE=admin \
    AUTOTUNEX_ARTIFACT_DIR=/data/artifacts \
    AUTOTUNEX_DATASET_STORAGE_DIR=/data/artifacts/datasets \
    AUTOTUNEX_LOCAL_OUTPUT_DIR=/data/artifacts/local \
    AUTOTUNEX_FRONTEND_DIR=/app/ux-build \
    AUTOTUNEX_FRONTEND_BASE_PATH=/autotune

# Run as non-root; own /data and /app so the app can write the DB and artifacts.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app
USER appuser

VOLUME /data
EXPOSE 8000

# slim has no curl; probe /health with the Python stdlib instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"]

CMD ["uvicorn", "autotunex.main:app", "--host", "0.0.0.0", "--port", "8000"]
