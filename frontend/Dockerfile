# ── Stage 1: Build Next.js ────────────────────────────────────────────────────
FROM node:22-slim AS nextjs-builder

# No NEXT_PUBLIC_* build-time vars needed — auth config is fetched at runtime
# from /api/config which reads APP_* env vars injected via K8s ConfigMap.

WORKDIR /app

COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

COPY . .
RUN yarn build && mkdir -p public

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Node.js for `next start`
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y curl \
    && rm -rf /var/lib/apt/lists/*

# Python server
COPY server/ ./server/
RUN pip install --no-cache-dir "./server[k8s,cos]"

# Next.js build artifacts and production deps
COPY --from=nextjs-builder /app/.next ./.next
COPY --from=nextjs-builder /app/node_modules ./node_modules
COPY --from=nextjs-builder /app/public ./public
COPY package.json next.config.ts tsconfig.json /app/

# Startup script: run Python sidecar + Next.js together
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

CMD ["./docker-entrypoint.sh"]
