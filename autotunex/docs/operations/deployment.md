# Production deployment

The defaults are tuned for local development — SQLite, no authentication,
schema auto-creation. A production deployment must change several of them
deliberately. This is a hardening checklist: work top to bottom, then read the
short notes under each item.

Many of these settings are enforced at **startup**, not at request time. The app
is built when the module is imported (`app = create_app()`), so a bad production
configuration fails immediately when the server (or `pytest`) starts, naming the
offending setting — a feature, not an inconvenience.

## Checklist

- [ ] `AUTOTUNEX_ENVIRONMENT=prod`
- [ ] A real auth provider configured (not `["disabled"]`)
- [ ] A real database (MySQL/PostgreSQL) with `AUTOTUNEX_AUTO_CREATE_SCHEMA=false` and Alembic applied
- [ ] `AUTOTUNEX_DEBUG=false` (set it explicitly)
- [ ] CORS/cookies locked down — only if a browser UI/BFF is used
- [ ] A job backend chosen deliberately
- [ ] Secrets injected from the environment or a secret store, never committed
- [ ] Served under a production ASGI server, behind TLS

---

## 1. Set the environment to `prod`

```
AUTOTUNEX_ENVIRONMENT=prod
```

This is the switch that turns on the production guards below (chiefly the
refusal to run without authentication). Leaving it at the default `dev` disables
those guards.

## 2. Configure a real auth provider

Production **refuses to start** with authentication disabled:

```
# This combination fails startup:
AUTOTUNEX_ENVIRONMENT=prod
AUTOTUNEX_AUTH_PROVIDERS=["disabled"]
```

Configure at least one real provider instead — any combination of `api_key`,
`oidc`, and `session`:

```
AUTOTUNEX_AUTH_PROVIDERS=["oidc"]
AUTOTUNEX_OIDC_ISSUER=https://issuer.example.com
AUTOTUNEX_OIDC_JWKS_URI=https://issuer.example.com/jwks
AUTOTUNEX_OIDC_AUDIENCE=<your-client-audience>
```

If a provider is only half-configured, startup names the exact missing setting
(for example, enabling `"oidc"` without `AUTOTUNEX_OIDC_AUDIENCE`, or `"session"`
without `AUTOTUNEX_SESSION_SECRET`). Fix what it names.

**Escape hatch (use with care).** A deliberate single-tenant deployment that
genuinely wants no authentication can override the refusal:

```
AUTOTUNEX_ALLOW_INSECURE_NO_AUTH=true
```

This logs a loud warning at startup and makes **every caller the one system
owner**, which sees its own rows by default and can widen to every row with
`?scope=all` — in a single-tenant deployment where every write is attributed to
that one owner, "its own rows" is in practice all of them. Only set it if that is
exactly what you want.

See [authentication](../api/authentication.md) for provider details, key
minting, and token handling.

## 3. Use a real database with Alembic migrations

Point at MySQL or PostgreSQL — **not** SQLite — and manage the schema with
Alembic rather than startup auto-creation:

```
AUTOTUNEX_AUTO_CREATE_SCHEMA=false
AUTOTUNEX_DATABASE_URL=mysql+asyncmy://user:pass@db.example.com/autotunex
# or: postgresql+asyncpg://user:pass@db.example.com/autotunex
```

`AUTOTUNEX_AUTO_CREATE_SCHEMA` defaults to `true`, a development convenience that
creates tables on startup. In production, turn it **off** and run migrations
explicitly — otherwise the app starts happily while Alembic has no version
recorded, hiding the fact that migrations were never applied.

See [database operations](database.md) for the migration commands, including the
special case of deploying against a database that already has the schema.

## 4. Turn off debug explicitly

```
AUTOTUNEX_DEBUG=false
```

`false` is the default, and a startup validator **refuses to boot** when
`AUTOTUNEX_DEBUG=true` is combined with `AUTOTUNEX_ENVIRONMENT=prod` — debug mode
leaks tracebacks, so it is never permitted in production. A stray `true` in a
`.env` or the environment therefore fails fast at startup, naming the offending
setting, rather than being silently tolerated. Set it explicitly to `false`
regardless, so the intent is on the record.

## 5. Lock down CORS and cookies (browser UI / BFF only)

Skip this section if you serve the API only (no browser front-end or
backend-for-frontend session flow). If you do use one:

- **`AUTOTUNEX_CORS_ALLOW_ORIGINS` must be an explicit allowlist and must never
  contain `"*"`.** Startup rejects a wildcard outright: combined with the
  credentialed CORS the BFF requires, a wildcard would let the middleware echo
  back any request's `Origin` with the session cookie attached, defeating the
  allowlist entirely.

  ```
  AUTOTUNEX_CORS_ALLOW_ORIGINS=["https://app.example.com"]
  ```

- **`AUTOTUNEX_SESSION_COOKIE_SAME_SITE=none` requires a non-empty allowlist.**
  Cross-site cookies are only legal alongside an explicit
  `AUTOTUNEX_CORS_ALLOW_ORIGINS`; startup rejects `none` with an empty list.

- **Session cookies are always `Secure`, so serve behind TLS.** Over plain HTTP
  the browser will not send the cookie and sessions will not work.

- **`AUTOTUNEX_SESSION_SECRET` must be at least 32 random characters.** It signs
  the session JWT and is the entire strength of the session cookie; a short
  secret makes forging one tractable. Startup rejects a value shorter than 32
  characters. Generate one from a cryptographic source:

  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(48))"
  ```

## 6. Choose a job backend deliberately

The default backend, `none`, **never executes anything** — accepted jobs stay
`pending` forever. Decide what you actually want:

```
AUTOTUNEX_JOB_BACKEND=none    # or: local | llmb
```

See [job backends](job-backends.md) for what each does and the settings it
requires. Note that some backends fail startup if their required settings are
missing — configure them alongside this choice.

## 7. Keep secrets out of the repository

Inject secrets — database credentials, `AUTOTUNEX_SESSION_SECRET`, OIDC client
secrets, `GB_TOKEN`, `HF_TOKEN`, API keys — from the environment or a secret
store, never from a committed file. `.env` is git-ignored; the only committed
placeholders are `.env.example`, `.env.aio.example` and the per-runner
`envs/*.env.example` files, plus the two subprojects' own
`src/api-bridge/.env.example` and `src/ux/.env.example`. Register only the
**SHA-256 digest** of an API key, never the raw key.

## 8. Run under a production ASGI server

Run the app with `uvicorn` (or another ASGI server), not the `--reload` dev
command:

```bash
uvicorn autotunex.main:app --host 0.0.0.0 --port 8000
```

Because the app is constructed at import time (`app = create_app()`), a bad
production configuration fails **immediately at startup** rather than on the
first request — and, for the same reason, will break a local `pytest` run if your
`.env` sets `AUTOTUNEX_ENVIRONMENT=prod` without a configured provider. Serve
behind a TLS-terminating reverse proxy (required for session cookies; good
practice regardless).

### Container image

The repository ships a multi-stage `Dockerfile` and a `compose.yaml` at its root
that run the whole service — the API plus the built SvelteKit UI — from a single
image. Stage one builds the SPA; the runtime stage installs AutoTuneX's base
dependencies (no PostgreSQL/MySQL extras) plus the vendored `src/fm-tune[core]`
training stack (`torch` + `ray[tune,default]`) and serves the SPA in-process
alongside the API. With `AUTOTUNEX_JOB_BACKEND=local` baked in (see the table
below), tuning runs in-container; only the GPU / online-RL extras (`[full]`:
verl/vLLM, flash-attn, deepspeed) and the macOS-arm64-only `[mlx]` extra are left
out. That describes the default `DEVICE=cpu` build: `--build-arg DEVICE=cuda`
selects a CUDA stage in the same `Dockerfile` that starts from the
published kernel base (`docker/cuda-base/Dockerfile`, overridable with
`--build-arg CUDA_KERNEL_BASE=...`) and installs `[full]` instead, so the `local`
backend can train on an NVIDIA GPU.
`compose.yaml` carries the same build behind a `cuda` profile —
`docker compose --profile cuda up --build` builds `autotunex:cuda` with its own
`autotunex-cuda-data` volume and an NVIDIA device reservation. See
the README's *GPU (CUDA)* section for the run flags and the image-size caveat.
The image runs as a non-root user, exposes port 8000, keeps all writable state —
the SQLite DB and artifacts — under a `/data` volume so it survives restarts, and
defines a `HEALTHCHECK` that probes `/health`.

The image builds from the **repository root**, not `autotunex/` — the `llmb`
variant below installs granite.build from the same checkout — so always name the
Dockerfile: without `-f`, the root `Dockerfile` (the gbserver image) is built
instead. What enters the context is set by the allowlist
`autotunex/Dockerfile.dockerignore`, which BuildKit reads in place of the root
`.dockerignore`:

```bash
docker build -f autotunex/Dockerfile -t autotunex:local .   # from the repo root
```

```bash
docker compose up --build      # Docker only; with Podman see the README (--ignorefile)
```

The image bakes in **standalone** defaults, every one overridable at run time
(`podman run -e AUTOTUNEX_...=...`, or the compose `environment:` block):

| Setting | Baked default |
| --- | --- |
| `AUTOTUNEX_ENVIRONMENT` | `dev` |
| `AUTOTUNEX_DATABASE_URL` | `sqlite+aiosqlite:////data/autotunex.db` |
| `AUTOTUNEX_AUTO_CREATE_SCHEMA` | `true` |
| `AUTOTUNEX_AUTH_PROVIDERS` | `["disabled"]` |
| `AUTOTUNEX_STANDALONE_ROLE` | `admin` |
| `AUTOTUNEX_JOB_BACKEND` | `local` |
| `AUTOTUNEX_ARTIFACT_DIR` | `/data/artifacts` |
| `AUTOTUNEX_DATASET_STORAGE_DIR` | `/data/artifacts/datasets` |
| `AUTOTUNEX_LOCAL_OUTPUT_DIR` | `/data/artifacts/local` |
| `AUTOTUNEX_FRONTEND_DIR` | `/app/ux-build` |
| `AUTOTUNEX_FRONTEND_BASE_PATH` | `/autotune` |

These are a self-contained dev/standalone deployment, **not** the hardened
production posture this checklist describes: the environment is `dev`, schema
auto-creation is on, and no real auth provider is configured. To run the image in
production, override them accordingly — at minimum `AUTOTUNEX_ENVIRONMENT=prod`, a
real database with `AUTOTUNEX_AUTO_CREATE_SCHEMA=false`, and a real auth provider.

The database is the one override that an environment variable alone cannot deliver.
The runtime stage installs the base dependencies only (`pip install .`), so neither
`asyncmy` nor `asyncpg` is in the image — they live in the `mysql` and `postgres`
extras — and pointing this image at `mysql+asyncmy://...` or
`postgresql+asyncpg://...` fails at engine creation, not at some later query.
Rebuild it with the matching extra (`pip install ".[mysql]"` or `".[postgres]"` in
place of `pip install .`), or use the AIO image below, which installs both.

### All-in-one (AIO) image

`Dockerfile.aio` is a second, separate image that runs the **whole stack** in one
container under `supervisord`. It is the only deployment path that also runs the
api-bridge and a granite.build `gbserver` beside the API, which is what makes it
self-sufficient:

- **8000** (published) — the AutoTuneX API and the built SvelteKit UI.
- **8080** (published) — the granite.build `gbserver` and its dashboard, fronted
  by a co-located Caddy proxy that dials gbserver over loopback. gbserver's
  standalone auth grants access to loopback peers only, so behind any NAT (the
  Podman bridge, an OpenShift Route) a browser would otherwise get a `401`.
- **8001** (loopback only, deliberately **not** published) — the api-bridge, the
  synchronous MySQL/PostgreSQL write path the tuning pipeline logs through.

Reach for it when you want job execution to work end to end — a demo, a dev box,
an evaluation — without standing up gbserver and the api-bridge separately. Its
baked defaults are `AUTOTUNEX_ENVIRONMENT=dev`, `AUTOTUNEX_JOB_BACKEND=llmb` and
`GB_ENVIRONMENT=standalone`, with every writable path under `/data`: the same
dev/standalone posture as the image above, not the hardened one this checklist
describes.

Unlike the single-service image it needs an external MySQL or PostgreSQL database
and a real token, so its compose service sits behind a **profile** — a bare
`docker compose up` never builds or starts it:

```bash
cp .env.aio.example .env.aio                 # fill in real values first
docker compose --profile aio up --build      # Docker only
```

`.env.aio` is git-ignored and carries only what cannot be baked in: the external
MySQL or PostgreSQL URL (**shared** — one connection string configures both AutoTuneX
and the api-bridge), `GB_TOKEN` (**required**: the `llmb` backend refuses to start without
it, and the entrypoint also uses it for runtime `git` clones), a session-signing
secret for the api-bridge, and optionally the private git host those clones
authenticate against. If your MySQL requires verified TLS, mount the CA cert into
the container and point `AUTOTUNEX_DATABASE_SSL_CA` and
`AUTOTUNEX_DATABASE_SSL_MODE` at it — both services read the same two settings
(`compose.yaml` carries the mount, commented out).

It builds from the repository root like the image above (`docker build -f
autotunex/Dockerfile.aio .`, context set by `autotunex/Dockerfile.aio.dockerignore`),
and its gbserver, dashboard and `llmb` CLI come from **that same checkout**: the
dashboard is built with the repo's own `make build-frontend`, and granite.build is
installed from the copied tree, so an image built from a commit runs the gbserver
in that commit. The checkout must carry granite.build's `v*` version tags:
setuptools_scm stamps the version from them, and without one the build stops rather
than ship an `llmb` CLI whose version check fails every job launch. The same check
fails a build from a git worktree, whose `.git` is a file. For a checkout that
cannot reach a tag (a worktree, a shallow clone), pass the version instead:
`--build-arg GB_VERSION=X.Y.Z` stamps it and skips the check. `.git` is only
bind-mounted into the steps that stamp the version, so a commit that does not touch
`frontend/` reuses the cached dashboard build, and the image carries no `.git`.
The image also ships granite.build's `docs/`, which the dashboard chat's
`search_docs` tool reads.
`PUBLIC_AUTOTUNEX_API_URL` (empty by default, meaning same-origin) matters only if
the API is fronted elsewhere. `INSTALL_AUTOTUNE_CORE=1` additionally installs the
`autotune` training core that the `local` backend needs, from the vendored
`src/fm-tune[core,mlx]`, and installs granite.build from the same tree into the
AutoTuneX virtualenv, so the image still carries one granite.build — just a much
larger image (torch/Ray). `GB_TOKEN` and `GITHUB_HOST` play no part in the build:
they are run-time settings (the token the `llmb` backend requires, plus the
entrypoint's git credentials, which matter only if you point
`AUTOTUNEX_BASH_FM_TUNE_ROOT` at a repo URL instead of the vendored trainer at
`/opt/fm-tune`).

On Kubernetes/OpenShift, mount a PVC at `/data`: it is both the volume and `HOME`,
so artifacts, datasets and gbserver's own SQLite state all land there. The image
also trusts every git repository directory at the system level
(`safe.directory '*'`), because a tuning build runs `git` over a checkout on
`/data` whose tree is owned by the image's build-time uid `1000` while the pod may
run under a different one (`runAsUser: 0`) — which git would otherwise reject as
"dubious ownership".

[`docker/aio/README.md`](../../docker/aio/README.md) goes deeper: the full service
table, the plain `docker run` invocation, the optional training-core build, why
the baked gbserver URL is plain HTTP, and a runtime smoke test.

### `llmb` image (`JOB_BACKEND=llmb`)

`--build-arg JOB_BACKEND=llmb` builds the same `Dockerfile` for the **`llmb`**
backend instead of `local`, with the same single-container shape — API plus the
built SPA, non-root, port 8000, a `/data` volume, a `/health` `HEALTHCHECK`. It
installs granite.build from the checkout being built (the `llmb` CLI, and the
`gbcli` package live build logs read), AutoTuneX's `mysql` and `mcp` extras, and
the vendored `src/fm-tune` at its torch-free base (the catalog imports; no
Ray/torch, so no in-process training). It is CPU-only — `DEVICE=cuda` with
`JOB_BACKEND=llmb` stops the build, since training happens on the granite.build
side — and like the AIO image it needs granite.build's `v*` tags in the checkout,
or `--build-arg GB_VERSION=X.Y.Z`. granite.build and AutoTuneX are resolved in
one `pip install`.

```bash
docker build -f autotunex/Dockerfile --build-arg JOB_BACKEND=llmb -t autotunex:gb .
```

Its baked defaults differ from the table above in three places:
`AUTOTUNEX_JOB_BACKEND=llmb`, `AUTOTUNEX_AUTO_CREATE_SCHEMA=false`, and
`AUTOTUNEX_JOB_TRAINER_REF=oss-main`. It bundles neither gbserver nor the
api-bridge, so unlike the AIO image it needs an **external** granite.build
server — and it does not bake `GB_ENVIRONMENT`, so the `llmb` backend starts in
its **custom_code** shape, whose startup check requires four settings the image
leaves unset: `AUTOTUNEX_JOB_RUNTIME_IMAGE`, `AUTOTUNEX_JOB_TRAINER_REPO`,
`AUTOTUNEX_JOB_OUTPUT_URI_ROOT` and `AUTOTUNEX_GB_SERVER_URL`. Supply all four at
run time, plus the token environment variable `AUTOTUNEX_GB_TOKEN_ENV` names
(`GB_TOKEN` by default), or the `llmb` backend refuses to start. Setting
`GB_ENVIRONMENT=standalone` (unprefixed — granite.build's own variable) instead
selects the local-`bash` spec, which drops the three custom_code-only inputs and
leaves `AUTOTUNEX_GB_SERVER_URL` plus the token variable as the only run-time
requirements. It has no `make` target; `compose.yaml` shows the build arg, commented out, on the
`autotunex` service.

### Building in a CI toolchain

An IBM Cloud Classic pipeline image job that runs `buildctl` from the `autotunex`
working directory needs these edits, because the context is now the repository
root:

- every job: `--local context=.` becomes `--local context=..`. Keep
  `--local dockerfile=.`, so BuildKit still finds the Dockerfile and its
  `.dockerignore` in `autotunex/`;
- the `llmb` job: remove `--opt filename=Dockerfile.gb` (`Dockerfile.gb` is gone;
  `Dockerfile` is the default filename) and add
  `--opt build-arg:JOB_BACKEND=llmb`;
- the AIO job keeps `--opt filename=Dockerfile.aio`, but remove any
  `--opt build-arg:GB_REF=...` / `GB_REPO=...`. BuildKit ignores them silently
  now, so a job that pinned `GB_REF` quietly switches to building from the
  checkout.

Edit the jobs when this change merges: until then all three fail.

The pipeline's checkout must include tags: a tag-less or shallow clone fails the
first build at the version check, by design. A pipeline that cannot fetch tags
passes the version instead, `--opt build-arg:GB_VERSION=X.Y.Z`, which also makes
the stamped version deterministic.

## Security

For how to report a vulnerability and the project's security posture, see
[SECURITY.md](../../../SECURITY.md).
