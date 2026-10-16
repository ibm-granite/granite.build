# Configuration

> **Audience:** operators deploying and configuring gbserver (the REST server, the build/PR watchers,
> the build runner, or the all-in-one standalone server).

gbserver's runtime is configured almost entirely through **environment variables** (`GBSERVER_*` /
`GB_*`), with two optional config files and a per-environment selector on top. This section covers the
settings that matter most, then links to the full reference.

## How configuration works

Settings are resolved with this precedence (highest wins):

1. **CLI options** — flags on the `gbserver` command (see the [`gbserver` CLI reference](../cli/gbserver-cli-reference.md)).
2. **Environment variables** — `GBSERVER_*` / `GB_*`; the primary knob (see [environment-variables.md](environment-variables.md)).
3. **`--server-runtime-config`** — per-environment defaults loaded from a YAML file (see [config-files.md](config-files.md)).
4. **`GB_ENVIRONMENT` built-ins** — DEV/STAGING/PROD/STANDALONE defaults (see [gb-environment.md](gb-environment.md)).
5. **Hardcoded defaults** — in [`src/gbserver/types/constants.py`](../../src/gbserver/types/constants.py), the source of truth.

## Most consequential settings

The variables an operator most often sets. (Full grouped list in [environment-variables.md](environment-variables.md).)

| Variable | Purpose |
|----------|---------|
| `GB_ENVIRONMENT` | `DEV` / `STAGING` / `PROD` / `STANDALONE` — selects cluster, namespace, SQL schema, and (for `STANDALONE`) a set of local defaults. See [gb-environment.md](gb-environment.md). |
| `GBSERVER_METADATA_STORAGE` | `sql` (PostgreSQL, default) or `sqlite` (standalone/local). |
| `GBSERVER_DEFAULT_BUILDRUNNER_TYPE` | How builds run: `job` (Kubernetes), `process`, or `thread` (use `thread` for local). |
| `GBSERVER_DEFAULT_LOG_LEVEL` | `debug` / `info` (default) / `warning` / `error` / `critical`. |
| `GBSERVER_AUTH_MODE` | `github` (default), `apikey`, `ibmid`, or `multi`. See [authentication](../rest-api/multi-provider-authentication.md). |
| `GBSERVER_API_KEY` | Shared secret for `apikey` mode (standalone/remote). |
| `GBSERVER_GITHUB_TOKEN` | GitHub Enterprise token used by the watchers and runner. |
| `GBSERVER_SQL_HOST` / `_PORT` / `_DBNAME` / `_SCHEMA` / `_USER` / `_PASSWD` | PostgreSQL connection (when `GBSERVER_METADATA_STORAGE=sql`). |
| `GBSERVER_EVENT_PUBLISHING_ENABLED` | Publish build events (default `false`; standalone `true`). |
| `GBSERVER_NATS_EMBEDDED` | Start an embedded NATS server (standalone default `true`); otherwise set `RABBITMQ_HOST` + `GBSERVER_RABBITMQ_MGMT_*` for RabbitMQ. |
| `GBSERVER_IMAGE_TAG` / `GBSERVER_SIDECAR_MONITORING_IMAGE_TAG` | Build-runner and monitoring-sidecar image tags (IBM/Kubernetes deployments). |

Credentials themselves are resolved per space by a [secret manager](../secrets/README.md), not set here.

## By component

All components read the same env vars; a few settings are component-specific (full options in the
[`gbserver` CLI reference](../cli/gbserver-cli-reference.md)):

- **REST server** (`gbserver rest-server`) — `GBSERVER_REST_SERVER_WORKERS`,
  `GBSERVER_REST_SERVER_TIMEOUT_KEEP_ALIVE`, and the auth vars above.
- **Build watcher** (`gbserver build-watch`) — a `--config` YAML file
  (spaces to watch, `monitoring_interval`, `buildrunner_type`, …) and `--asset_stores_dir`;
  see [config-files.md](config-files.md).
- **Build runner** (`gbserver build-runner`) — `GBSERVER_DEFAULT_BUILDRUNNER_TYPE` and, for `job`,
  the `GBSERVER_BUILDRUNNERJOB_*` (k8s) vars and workspace dirs.
- **Standalone** (`gbserver standalone`) — forces `GB_ENVIRONMENT=STANDALONE` and applies the standalone
  defaults (sqlite, thread runner, apikey auth, event publishing); see [gb-environment.md](gb-environment.md).

## Standalone mode

`gbserver standalone` sets `GB_ENVIRONMENT=STANDALONE`, which applies a set of local defaults (SQLite
storage, thread build runner, API-key auth, event publishing) and swaps in standalone storage/space
managers. See **[Deployment environments → Standalone mode](gb-environment.md#standalone-mode)** for how
that config is applied and the full list of defaults.

## In this section

- [Environment variables](environment-variables.md) — the full grouped reference.
- [Config files](config-files.md) — `--server-runtime-config` and the watcher `--config`.
- [Deployment environments (`GB_ENVIRONMENT`)](gb-environment.md) — DEV/STAGING/PROD/STANDALONE and standalone defaults.

## See also

- [`gbserver` CLI reference](../cli/gbserver-cli-reference.md) · [Secrets](../secrets/README.md) · [Authentication](../rest-api/multi-provider-authentication.md)
