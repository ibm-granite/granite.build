# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build-time device selection. Declared before the first FROM because only a
# global ARG can be used in a FROM line.
#
#   docker build -t autotunex:local .                        # CPU (default)
#   docker build -t autotunex:cuda --build-arg DEVICE=cuda . # CUDA + [full]
#
# DEVICE names one of the two base stages below, not an image reference —
# override CUDA_BASE for that. The value vocabulary (cpu|cuda) matches
# fm-tune's own runtime override, FMTUNE_DEVICE=cuda|mps|cpu, on purpose.
# ---------------------------------------------------------------------------
ARG DEVICE=cpu
ARG CUDA_BASE=nvidia/cuda:12.8.0-devel-ubuntu24.04

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
ENV FM_TUNE_EXTRAS=core \
    FLASH_ATTN_WHEEL=

# ---------------------------------------------------------------------------
# Stage 2b — base: CUDA, for GPU training with the `local` job backend.
#
# Nothing here sets NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES on
# purpose: NVIDIA's own images already set them and the runtime stage inherits
# them, so re-declaring would only invite drift.
# ---------------------------------------------------------------------------
FROM ${CUDA_BASE} AS cuda
ENV FM_TUNE_EXTRAS=full \
    FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# python3.12 specifically: the flash-attn wheel above is cp312-only. Ubuntu
# 24.04's system python already IS 3.12, so no third-party PPA is needed.
#
# build-essential is named explicitly even though the default `-devel` base
# already ships gcc (verified: apt reports it "already the newest version").
# deepspeed and verl fall back to an sdist build that needs a host compiler, and
# CUDA_BASE is overridable — a leaner `-runtime` or `-base` tag would not have
# one. A no-op on the default base, so it costs nothing to keep.
# docker/runtime/Dockerfile names it on the same base for the same reason.
#
# The venv is not cosmetic either: Ubuntu marks its system python PEP 668
# "externally-managed" and pip refuses to install into it. Putting the venv's
# bin on PATH also makes bare `python` and `pip` resolve, which is what lets the
# shared runtime stage's pip calls, HEALTHCHECK and CMD work unchanged on both
# bases.
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
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && (userdel -r ubuntu 2>/dev/null || true)
ENV PATH=/opt/venv/bin:$PATH

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
COPY pyproject.toml README.md LICENSE ./
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
# The three-step order is load-bearing:
#
#   1. `[core]` first, so torch==2.8.0 is already pinned before anything else can
#      express an opinion about it. NOTE: on linux/amd64 torch==2.8.0 resolves to
#      the default cu128 CUDA wheel — multi-GB, and it simply runs on CPU when
#      there is no GPU; pulling CPU torch from download.pytorch.org/whl/cpu to
#      slim the amd64 CPU image is a known, still-deferred optimization. On
#      linux/arm64 PyPI serves a CPU-only build (2.8.0+cpu) instead, so that
#      image is smaller already and the CUDA base does not apply there at all.
#   2. The prebuilt flash-attn wheel, because pip does NOT read
#      `[tool.uv.sources]`, where src/fm-tune/pyproject.toml pins its URL — so an
#      unguarded `[full]` install would source-build flash-attn against nvcc
#      instead (an hour-plus build). It runs AFTER step 1 so the wheel's own
#      unpinned `torch` requirement is already satisfied; first, it would drag in
#      a newer torch that step 3 then downgrades again.
#   3. `[$FM_TUNE_EXTRAS]`. PEP 440 `==2.8.1` matches the wheel's
#      `2.8.1+cu12torch2.8cxx11abiFALSE` local version, so `[full]` treats
#      flash-attn as satisfied and adds only verl/vLLM and deepspeed.
#
# On the CPU base both guards are false and this collapses to exactly
# `pip install "./src/fm-tune[core]"` — the same install this image has today.
RUN pip install --no-cache-dir "./src/fm-tune[core]" \
    && if [ -n "$FLASH_ATTN_WHEEL" ]; then \
         pip install --no-cache-dir "$FLASH_ATTN_WHEEL"; \
       fi \
    && if [ "$FM_TUNE_EXTRAS" != "core" ]; then \
         pip install --no-cache-dir "./src/fm-tune[$FM_TUNE_EXTRAS]"; \
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
