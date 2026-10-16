# syntax=docker/dockerfile:1

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
# Stage 2 — runtime: the AutoTuneX API, serving the SPA in-process
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install AutoTuneX + its exactly-pinned BASE deps only. Every base dependency
# ships a cp312 manylinux wheel, so no compiler/build-essential layer is needed.
# No postgres/mysql extras.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Install the vendored autotune core (src/fm-tune) with the `[core]` extra: the
# lean training stack (ray, torch, transformers, trl, peft, accelerate, datasets,
# tokenizers, ...) the default `local` job backend runs an in-process trial with.
# It also makes `autotune.catalog` importable (the config-template / dataset-type
# wizard endpoints need it). The GPU / online-RL extras ([full] -> verl/vLLM/
# flash-attn/deepspeed) are left out: they need nvcc + a GPU and don't build on
# this CPU image.
#
# All `[core]` deps ship cp312 manylinux wheels, so still no compiler layer is
# needed. NOTE: torch==2.8.0 resolves to the default (CUDA) wheel, which is
# multi-GB and runs on CPU anyway; pulling CPU torch from
# download.pytorch.org/whl/cpu to slim the image is a known, still-deferred
# optimization. Quoted so the shell does not glob `[core]`.
RUN pip install --no-cache-dir "./src/fm-tune[core]"

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
