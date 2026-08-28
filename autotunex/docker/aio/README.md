# All-in-one image (`Dockerfile.aio`)

A single container that runs the whole local AutoTuneX stack for demo / dev /
evaluation. Design: [`docs/superpowers/specs/2026-08-18-unified-aio-docker-image-design.md`](../../docs/superpowers/specs/2026-08-18-unified-aio-docker-image-design.md).

| Service | Port | Published | UI |
| --- | --- | --- | --- |
| AutoTuneX API + SvelteKit SPA | 8000 | yes | `http://localhost:8000/autotune` |
| granite.build `gbserver` (standalone) | 8080 | yes | `http://localhost:8080` |
| api-bridge (MySQL write-path) | 8001 | **no** (loopback only) | — |

The three services run in one container under `supervisord`, each in its own
virtualenv. granite.build is built from the fork branch
`feat/autotunex-endpoint-migration` via its public **standalone** path (no
internal Artifactory). AutoTuneX submits builds to the co-located gbserver
through the `llmb` CLI (`AUTOTUNEX_JOB_BACKEND=llmb`, `GB_ENVIRONMENT=standalone`).

## Configuration

Non-secret defaults (ports, backend, the gbserver/callback URLs, frontend and
artifact dirs) are **baked into the image**. Secrets and hostnames are supplied
at run time — nothing sensitive lives in the tracked Dockerfile.

```bash
cp .env.aio.example .env.aio     # then fill in real values
```

`.env.aio` must provide the external MySQL (`AUTOTUNEX_DATABASE_URL`, shared by
both AutoTuneX and the api-bridge — the same database), `GB_TOKEN` (**required** —
AutoTuneX will not start without it when `job_backend=llmb`), and `SESSION_SECRET`
for api-bridge. `gbserver` keeps its own local SQLite under `/data`.

`GITHUB_HOST` (your private git host) is **optional**. The trainer is vendored at
`/opt/fm-tune` and used by default, so a standard build clones nothing and needs
no git credentials — leave `GITHUB_HOST` unset. Set it only when a build performs
a runtime `git` clone against a private host — e.g. you override
`AUTOTUNEX_BASH_FM_TUNE_ROOT` with a private repo URL for a remote trainer. When
both `GB_TOKEN` and `GITHUB_HOST` are set, the entrypoint provisions git
credentials from them so those clones authenticate; the token is written to an
ephemeral file under `/tmp`, never the `/data` volume, and nothing is baked into
the image.

> **Note on the gbserver URL.** The baked `AUTOTUNEX_GB_SERVER_URL` is
> `http://localhost:8080` (plain HTTP). AutoTuneX's reconcile client verifies TLS
> with no skip-verify hook, so a self-signed `https://localhost:8080` would fail;
> HTTP is used on the co-located loopback hop instead.

## Run

With Docker Compose (recommended):

```bash
cp .env.aio.example .env.aio     # edit first
docker compose --profile aio up --build
```

Or plain Docker:

```bash
docker build -f Dockerfile.aio -t autotunex-aio:local .
docker run --rm \
  --env-file .env.aio \
  -p 8000:8000 -p 8080:8080 \
  -v autotunex-aio-data:/data \
  autotunex-aio:local
```

Then open `http://localhost:8000/autotune` (AutoTuneX) and
`http://localhost:8080` (granite.build).

## Optional: the autotune training core (`local` backend)

By default AutoTuneX installs the `mysql` extra only and reaches gbserver through
the `llmb` CLI — a lean image. To also install the autotune training core, used by
the in-process `local` job backend, enable the opt-in build arg. The core is
vendored in-tree at `src/fm-tune` (already COPYed into the build context) and is
installed from `src/fm-tune[core,mlx]` — no credentials and no private repo fetch
are needed.

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.aio \
  --build-arg INSTALL_AUTOTUNE_CORE=1 \
  -t autotunex-aio:local .
```

Then, to actually run on it, set `AUTOTUNEX_JOB_BACKEND=local` in `.env.aio`.

Caveats: `fm-tune[core, mlx]` pulls torch/ray, so the image is large. `core` is
the lean training stack — single-device SFT + offline RL, without the GPU-only
`verl`/`flash-attn`/`deepspeed` that `full` adds. `mlx` is marker-guarded to
Apple silicon (`sys_platform == 'darwin' and platform_machine == 'arm64'`), so on
`linux/amd64` it resolves to nothing and leaves the build unaffected.

## Verify at runtime (smoke test)

These are the integration points that cannot be checked at build time:

1. `curl -f http://localhost:8000/health` → AutoTuneX up and DB reachable.
2. Open `http://localhost:8080` → gbserver dashboard renders.
3. `docker exec <container> supervisorctl status` → all three `RUNNING`.
4. Submit a job through AutoTuneX and confirm the `llmb` CLI reaches the local
   gbserver (see the design doc's "assumptions to validate").

## Debugging SQLite state inside a running container

The image ships the `sqlite3` CLI for this. Two things in the stack can be
SQLite-backed: gbserver's standalone metadata (`GBSERVER_METADATA_STORAGE=sqlite`,
written under `$HOME` = `/data`, the persistent volume) and AutoTuneX's own
database whenever `AUTOTUNEX_DATABASE_URL` is left at its SQLite default instead
of pointing at MySQL. Neither path is fixed by this image, so locate the file
first:

```bash
# Podman/Docker: podman exec -it <container> bash    OpenShift: oc rsh <pod>
find /data /app \( -name '*.db' -o -name '*.sqlite*' \) 2>/dev/null

sqlite3 /data/<file>.db '.tables'
sqlite3 /data/<file>.db '.schema <table>'
sqlite3 -header -column /data/<file>.db 'select * from <table> order by rowid desc limit 5;'
```

Pass `-readonly` when a service is live — a second writer on the same file can
block the process that owns it:

```bash
sqlite3 -readonly /data/<file>.db 'select count(*) from <table>;'
```

AutoTuneX's MySQL database is not reachable this way; the image carries no
`mysql` client. Query it from the API, or from the AutoTuneX venv's driver.
