# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# api-bridge — the synchronous MySQL write-path/logging service the tuning
# pipeline records jobs/trials/results/logs through. Independent artifact from
# the main service (distribution `autotunex-api-bridge`, package `api_bridge`);
# build context is this directory (src/api-bridge).
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# 3.12 matches the main service image and the AIO api-bridge venv. Every runtime
# dependency (pydantic-core, psycopg[binary], …) ships a cp312 manylinux wheel,
# so no compiler/build-essential layer is needed — slim already carries
# ca-certificates.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Opt-in: also install the synchronous PostgreSQL driver (psycopg 3). Off by
# default so the base image carries no extra driver; the bridge coerces any
# Postgres URL to postgresql+psycopg at run time.
#   docker build --build-arg INSTALL_POSTGRES=1 .
ARG INSTALL_POSTGRES=0

WORKDIR /app
COPY . .
RUN case "$INSTALL_POSTGRES" in \
      1|true|TRUE|yes|on) EXTRAS="[postgres]" ;; \
      *)                  EXTRAS="" ;; \
    esac \
    && pip install ".${EXTRAS}"

# Run as non-root. The package is installed into system site-packages (readable
# by all), and the service writes nothing to disk — it is a network write-path —
# so no owned/writable directory is required.
RUN useradd --create-home --uid 1000 appuser
USER appuser

# Code default is 8000 (server.py: API_BRIDGE_SERVER_PORT, fallback 8000);
# deployments such as the AIO override it to 8001 via that env var.
EXPOSE 8000

# No /health route exists — routes live under /fmtune/*. Probe /openapi.json,
# which FastAPI always serves and which touches no database. Shell form so the
# probe tracks an API_BRIDGE_SERVER_PORT override rather than a baked-in port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,sys,urllib.request; p=os.getenv('API_BRIDGE_SERVER_PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/openapi.json').status==200 else 1)"

# `python -m` runs server.py's __main__ block, preserving log_config=None and
# the API_BRIDGE_SERVER_PORT / DEV_MODE env-driven startup (a bare
# `uvicorn api_bridge.server:app` would drop both).
# AUTOTUNEX_DATABASE_URL is required at run time (Database() raises without it)
# and is deliberately NOT baked — supply it via -e / --env-file.
CMD ["python", "-m", "api_bridge.server"]
