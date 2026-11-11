# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build-time device selection. Declared before the first FROM because only a
# global ARG can be used in a FROM line.
#
#   docker build -t autotunex:local .                        # CPU (default)
#   docker build -t autotunex:cuda --build-arg DEVICE=cuda . # CUDA + [full]
#
# DEVICE names one of the two base stages below, not an image reference —
# override CUDA_KERNEL_BASE for that. The value vocabulary (cpu|cuda) matches
# fm-tune's own runtime override, FMTUNE_DEVICE=cuda|mps|cpu, on purpose.
# ---------------------------------------------------------------------------
ARG DEVICE=cpu
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
COPY src/ux/package.json src/ux/package-lock.json ./
RUN npm ci

# Build the static SPA (adapter-static; base path /autotune; index.html fallback).
COPY src/ux/ ./
RUN echo "PUBLIC_AUTOTUNEX_API_URL=$PUBLIC_AUTOTUNEX_API_URL" >> .env \
    && npm run build

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
# Stage 3 — runtime: the AutoTuneX API, serving the SPA in-process. `FROM
# ${DEVICE}` picks one of the two base stages above; everything below is
# identical for both, which is why the extras arrive as inherited ENV rather
# than as a conditional here.
# ---------------------------------------------------------------------------
FROM ${DEVICE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install AutoTuneX + its exactly-pinned BASE deps only. Every base dependency
# ships a cp312 manylinux wheel, so no compiler/build-essential layer is needed.
# No postgres/mysql extras.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

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
RUN pip install --no-cache-dir "./src/fm-tune[core]" \
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
# volume) so the SQLite DB and artifacts survive container restarts.
ENV AUTOTUNEX_ENVIRONMENT=dev \
    AUTOTUNEX_DATABASE_URL=sqlite+aiosqlite:////data/autotunex.db \
    AUTOTUNEX_AUTO_CREATE_SCHEMA=true \
    AUTOTUNEX_AUTH_PROVIDERS=[\"disabled\"] \
    AUTOTUNEX_STANDALONE_ROLE=admin \
    AUTOTUNEX_JOB_BACKEND=local \
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
