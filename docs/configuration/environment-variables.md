# Environment variables

> **Audience:** operators who need the full list. For the handful that matter most, start with the
> [configuration overview](README.md#most-consequential-settings).

gbserver reads its configuration from `GBSERVER_*` / `GB_*` environment variables. The authoritative,
exhaustive list — with exact defaults — is the central registry
[`src/gbserver/types/constants.py`](../../src/gbserver/types/constants.py); the tables below group the
important ones by concern. Defaults shown are the built-in values (before any `GB_ENVIRONMENT` or
`--server-runtime-config` overrides).

## Core

| Variable | Default | Purpose |
|----------|---------|---------|
| `GB_ENVIRONMENT` | `PROD` | Deployment environment: `DEV` / `STAGING` / `PROD` / `STANDALONE`. See [gb-environment.md](gb-environment.md). |
| `GBSERVER_METADATA_STORAGE` | `sql` | Metadata backend: `sql` (PostgreSQL) or `sqlite`. |
| `GBSERVER_DEFAULT_BUILDRUNNER_TYPE` | `job` | `job` (k8s), `process`, or `thread`. |
| `GBSERVER_DEFAULT_LOG_LEVEL` | `info` | `debug`/`info`/`warning`/`error`/`critical`. |
| `GBSERVER_DEBUG_MODE` | — | Optional debug flag. |
| `GBSERVER_PROCEED_WITHOUT_SECRETS` | `false` | Skip the secret manager (standalone default `true`). |

## REST server

| Variable | Default | Purpose |
|----------|---------|---------|
| `GBSERVER_REST_SERVER_WORKERS` | `1` | uvicorn worker processes. |
| `GBSERVER_REST_SERVER_TIMEOUT_KEEP_ALIVE` | `120` | Keep-alive timeout (seconds). |
| `GBSERVER_AUTH_MODE` | `github` | `github` / `apikey` / `ibmid` / `multi`. |
| `GBSERVER_API_KEY` | `` | Shared secret for `apikey` mode. |
| `GBSERVER_API_USER` | `standalone` | Default user for API-key auth. |

## Authentication (IBMid OIDC)

Needed when `GBSERVER_AUTH_MODE` is `ibmid` or `multi` (see [authentication](../rest-api/multi-provider-authentication.md)).

| Variable | Default | Purpose |
|----------|---------|---------|
| `GBSERVER_IBMID_CLIENT_ID` / `GBSERVER_IBMID_CLIENT_SECRET` | `` | OIDC client credentials (server is the confidential client). |
| `GBSERVER_IBMID_CALLBACK_URL` | `` | Redirect URI registered with IBMid. |
| `GBSERVER_IBMID_ISSUER` / `GBSERVER_IBMID_JWKS_URI` | `login.ibm.com/…` | Issuer + JWKS for token validation. |
| `GBSERVER_IBMID_AUTHORIZE_URL` / `_TOKEN_URL` / `_USERINFO_URL` | `login.ibm.com/…` | OIDC endpoints. |

## SQL storage

Used when `GBSERVER_METADATA_STORAGE=sql`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GBSERVER_SQL_SCHEME` | `postgresql` | DB scheme. |
| `GBSERVER_SQL_HOST` / `GBSERVER_SQL_PORT` | (IBM Cloud) / `31842` | Host and port. |
| `GBSERVER_SQL_DBNAME` | `ibmclouddb` | Database name. |
| `GBSERVER_SQL_SCHEMA` | per-environment | Schema (e.g. `granite_dot_build_prod`). |
| `GBSERVER_SQL_USER` / `GBSERVER_SQL_PASSWD` | (IBM Cloud) / `` | Credentials. |
| `GBSERVER_SQL_SSLROOT_CERT_FILE` / `_SSLROOT_CERT_BASE64` | — | SSL root cert (path or base64). |
| `GBSERVER_SQL_ECHO` | `false` | Log SQL statements. |

## Messaging & events

| Variable | Default | Purpose |
|----------|---------|---------|
| `GBSERVER_EVENT_PUBLISHING_ENABLED` | `false` | Publish build events (standalone `true`). |
| `GBSERVER_NATS_EMBEDDED` | `true` | Start an embedded NATS/JetStream server (standalone). |
| `GBSERVER_NATS_URL` | `nats://localhost:4222` | NATS server address. |
| `GBSERVER_NATS_STREAM_MAX_AGE` / `_MAX_DELIVER` / `_ACK_WAIT` | `604800` / `5` / `30` | JetStream tuning. |
| `RABBITMQ_HOST` | — | Presence selects RabbitMQ (cloud) over NATS. |
| `GBSERVER_RABBITMQ_MGMT_URL` / `_USER` / `_PASSWORD` | `localhost:15672` / `guest` / `guest` | RabbitMQ management API. |
| `GBSERVER_BUILD_EVENTS_EXCHANGE` | `build-events` | Topic exchange name. |
| `GBSERVER_EVENT_SUBSCRIBE_TTL` | `60` | Subscription credential TTL (seconds). |

## Build runner (Kubernetes job mode)

Used when `GBSERVER_DEFAULT_BUILDRUNNER_TYPE=job`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GBSERVER_IMAGE_TAG` | `latest` | Build-runner image tag. |
| `GBSERVER_SIDECAR_MONITORING_IMAGE_TAG` | `latest` | Monitoring-sidecar image tag. |
| `GBSERVER_BUILDRUNNERJOB_NAMESPACE` | per-environment | Namespace for build-runner jobs. |
| `GBSERVER_BUILDRUNNERJOB_IMAGE_OVERRIDE` | (gbserver image) | Override the runner image. |
| `GBSERVER_BUILDRUNNERJOB_SECRET_NAME` | (svc-acct secret) | Pull/service-account secret. |
| `GBSERVER_BUILDRUNNERJOB_BUILD_WORKSPACE_PVC_NAME` | `gb-buildws-pvc` | Workspace PVC. |
| `GBSERVER_BUILDRUNNERJOB_CONFIGMAP_NAME` | `granite-dot-build-configmap` | ConfigMap. |
| `GBSERVER_DEFAULT_ROOT_WORKSPACE_DIR` | `gbserverworkspace` | Root workspace dir (watchers derive subdirs). |

## GitHub & git

| Variable | Default | Purpose |
|----------|---------|---------|
| `GBSERVER_GITHUB_TOKEN` (aka `GBSERVER_DEFAULT_GITHUB_TOKEN`) | `` | GHE token (falls back to `GITHUB_TOKEN`). |
| `GBSERVER_DEFAULT_GH_REQUEST_TIMEOUT` | `60` | GitHub request timeout (seconds). |
| `GITHUB_API_MAX_RETRIES` / `_RETRY_BASE_DELAY` / `_RETRY_MAX_DELAY` | `10` / `1.0` / `60.0` | GitHub API retry policy. |
| `GIT_CLONE_MAX_RETRIES` / `_RETRY_MIN_WAIT` / `_RETRY_MAX_WAIT` | `5` / `1.0` / `30.0` | git clone retry policy. |

## Monitoring & metrics

| Variable | Default | Purpose |
|----------|---------|---------|
| `GBSERVER_MONITORING_GRACE_PERIOD` | `30` | Grace period for event consumption (seconds). |
| `GBSERVER_API_FAILURE_TIMEOUT` | `300` | Max sustained API-failure duration (seconds). |
| `GBSERVER_METRICS_ENDPOINT` / `_AUTH_TOKEN` | `` | Metrics push endpoint + token. |
| `GBSERVER_PUSH_METRICS_TIMEOUT` | `10` | Metrics push timeout (seconds). |

## Secrets & lineage

| Variable | Default | Purpose |
|----------|---------|---------|
| `GBSERVER_USER_SECRET_MANAGER` | `ibmcloud` (standalone `local`) | Per-user secret backend. See [secrets](../secrets/README.md). |
| `GBSERVER_USER_SECRET_DIR` / `_MANAGER_CONFIG` | — | Local backend dir / JSON config. |
| `GBSERVER_IBM_SEC_MAN_ENDPOINT` / `_API_KEY` | — | IBM Secrets Manager endpoint + key. |
| `GBSERVER_LINEAGE_PROVIDER` | `wandb` (standalone `none`) | Lineage backend. See [lineage](../builds/lineage.md). |
| `GBSERVER_WANDB_API_KEY` / `_PROJECT` / `_ENTITY` / `_BASE_URL` | (W&B) | W&B lineage settings. |

## SkyPilot / LSF tuning

| Variable | Default | Purpose |
|----------|---------|---------|
| `GBSERVER_SKYPILOT_LAUNCH_CONCURRENCY` | `4` | Concurrent `sky.launch` operations. |
| `GBSERVER_SKYPILOT_PROVISION_MAX_ATTEMPTS` / `_BACKOFF_MAX` | `4` / `30` | Provision retry policy. |
| `GBSERVER_LSF_TRANSIENT_ERROR_MAX_RETRIES` / `_RETRY_DELAY` | `3` / `30` | LSF transient-error retry policy. |
| `GBSERVER_K8S_USE_ASPERA` / `GBSERVER_LSF_USE_ASPERA` | `true` / `false` | Use Aspera for asset transfer. |
| `GBSERVER_ENABLE_STEP_RETRY` | `true` | Master switch for step-level retry. See [step retry](../builds/step-retry-configuration.md). |
| `GBSERVER_ENABLE_SSH_HOST_KEY_VERIFICATION` | `false` | Verify SSH host keys. |

## Build-files API caps

Limits on the build-files REST endpoints: `GBSERVER_BUILD_FILES_DOWNLOAD_MAX_BYTES` (1 GiB),
`_LIST_MAX_ENTRIES` (10000), `_GREP_MAX_HITS` (5000), `_GREP_LINE_MAX_BYTES` (512),
`_GREP_MAX_CONTEXT` (50), `_PEEK_MAX_LINES` (10000), `_PEEK_MAX_BYTES` (256 KiB),
`_STAT_BATCH_MAX` (500).

## See also

- [Configuration overview](README.md) — the most consequential settings and precedence
- [`constants.py`](../../src/gbserver/types/constants.py) — the exhaustive source of truth
