# Configuration reference

All runtime settings for AutoTuneX come from environment variables prefixed with
`AUTOTUNEX_`, or from a local `.env` file. This page documents every setting: its
environment-variable name, what it means, and its default.

A few rules hold everywhere:

- **Naming.** Each setting's environment variable is `AUTOTUNEX_` + the upper-snake-case
  field name — for example, the `log_level` field is `AUTOTUNEX_LOG_LEVEL`. The one
  exception is [`GB_ENVIRONMENT`](#execution-job-launch), which is read **without** the
  `AUTOTUNEX_` prefix (see that row).
- **Where settings come from.** They are read only through the settings singleton
  (`get_settings()`), never from `os.environ` directly, and a local `.env` file in the
  working directory is loaded automatically at startup. Values in the real environment
  override values in `.env`. Copy `.env.example` to `.env` to start — its values are the
  defaults documented here, and they work as-is for local development.
- **List and map settings are JSON, not comma-separated.** Fields such as
  `AUTOTUNEX_AUTH_PROVIDERS`, `AUTOTUNEX_API_KEYS`, `AUTOTUNEX_OIDC_ALGORITHMS`, and
  `AUTOTUNEX_CORS_ALLOW_ORIGINS` are parsed as JSON. Write `["disabled"]`, not
  `disabled` — a bare value crashes at startup with a parse error. Keep the brackets and
  quotes.
- **Secrets belong in the environment or a secret store, never in a committed file.**
  Tokens and secrets — the value of the variable named by `AUTOTUNEX_GB_TOKEN_ENV`
  (`GB_TOKEN` by default), the one named by `AUTOTUNEX_HF_TOKEN_ENV` (`HF_TOKEN`), your
  API keys, `AUTOTUNEX_OIDC_CLIENT_SECRET`, `AUTOTUNEX_LLM_API_KEY`, and
  `AUTOTUNEX_SESSION_SECRET` — should be injected at deploy time. Never commit `.env`.

## Fail-fast startup validation

Misconfiguration is caught at startup, not at request time. The service refuses to start
(with a `ValidationError` naming the offending setting) in these cases:

- **An empty provider list.** `AUTOTUNEX_AUTH_PROVIDERS` must name at least one credential
  kind; `[]` is refused.
- **Authentication disabled in production.** `AUTOTUNEX_ENVIRONMENT=prod` combined with
  `AUTOTUNEX_AUTH_PROVIDERS=["disabled"]` is refused unless
  `AUTOTUNEX_ALLOW_INSECURE_NO_AUTH=true` is also set (which logs a loud warning).
- **`"disabled"` combined with another provider.** Standalone mode cannot be mixed with a
  real provider in the same list.
- **`"api_key"` enabled with no keys**, or with a key that is not a 64-character lowercase
  SHA-256 hex digest.
- **`"oidc"` enabled** without all of `AUTOTUNEX_OIDC_ISSUER`, `AUTOTUNEX_OIDC_JWKS_URI`,
  and `AUTOTUNEX_OIDC_AUDIENCE`.
- **`"session"` enabled** without the full backend-for-frontend set (issuer, JWKS URI,
  audience, client id, client secret, authorization endpoint, token endpoint, public base
  URL, and session secret).
- **A wildcard CORS origin.** `AUTOTUNEX_CORS_ALLOW_ORIGINS` must never contain `"*"`, in
  any configuration.
- **`same_site=none` without an allowlist.** `AUTOTUNEX_SESSION_COOKIE_SAME_SITE=none`
  requires a non-empty `AUTOTUNEX_CORS_ALLOW_ORIGINS`.
- **A partial LLM configuration.** `AUTOTUNEX_LLM_BASE_URL`, `AUTOTUNEX_LLM_API_KEY`, and
  `AUTOTUNEX_LLM_MODEL` must all be set together or all be unset.
- **`job_backend=llmb` missing a required input** (see the
  [Execution](#execution-job-launch) section for the exact per-mode set).
- **`dataset_storage_backend=huggingface` forced in bash standalone.**
  `GB_ENVIRONMENT=standalone` with `AUTOTUNEX_LSF_CLUSTER` unset is refused regardless of
  the tokens — the HuggingFace push runs `llmb artifact push`, which granite.build disables
  there. The LSF/SkyPilot variant (`AUTOTUNEX_LSF_CLUSTER` set) is exempt.
- **`dataset_storage_backend=huggingface` forced** without the GB and HF token environment
  variables present.
- **`debug=true` in production.** `AUTOTUNEX_DEBUG=true` combined with
  `AUTOTUNEX_ENVIRONMENT=prod` is refused — debug mode leaks tracebacks, so it is never
  permitted in production.
- **An unknown standalone role.** `AUTOTUNEX_STANDALONE_ROLE` must be one of `admin` or
  `user`.

---

## App

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_APP_NAME` | Service name shown in the OpenAPI docs and logs. | `AutoTuneX API` |
| `AUTOTUNEX_ENVIRONMENT` | Deployment environment: `dev`, `test`, or `prod`. `prod` enables the fail-fast auth checks above. | `dev` |
| `AUTOTUNEX_DEBUG` | Verbose error output. Keep `false` in production. | `false` |
| `AUTOTUNEX_LOG_LEVEL` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, or `ERROR`. | `INFO` |
| `AUTOTUNEX_API_PREFIX` | Path prefix the versioned API is mounted under. `/health` sits outside it. | `/api/v1` |

## Frontend (optional)

Set these only to serve the built SvelteKit single-page app from this service. Leaving
`AUTOTUNEX_FRONTEND_DIR` unset (the default) runs the API only, with the UI hosted
separately.

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_FRONTEND_DIR` | Directory of the built SPA to serve. When set to an existing directory, the app mounts it with SPA-fallback routing. A local checkout builds to `src/ux/build`. | *(unset — API only)* |
| `AUTOTUNEX_FRONTEND_BASE_PATH` | URL prefix the SPA is mounted under. Must match the value baked into the UI's asset URLs at build time; changing it here alone is not enough — rebuild the UI with a matching base. | `/autotune` |

## Persistence

See [database.md](database.md) for supported databases, connection-string examples, and
migrations.

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_DATABASE_URL` | Async SQLAlchemy connection URL. SQLite needs no server; PostgreSQL and MySQL need the matching install extra. | `sqlite+aiosqlite:///./autotunex.db` |
| `AUTOTUNEX_DATABASE_ECHO` | Log every SQL statement. Noisy; useful when debugging queries. | `false` |
| `AUTOTUNEX_AUTO_CREATE_SCHEMA` | Create missing tables on startup. A development convenience — set `false` in production and run Alembic migrations instead. | `true` |
| `AUTOTUNEX_DATABASE_SSL_CA` | Path to a CA-certificate PEM enabling verified TLS to the database; required for managed MySQL such as IBM Cloud Databases. MySQL URLs only. | *(unset)* |
| `AUTOTUNEX_DATABASE_SSL_MODE` | How to negotiate TLS: `disable`, `require`, or `verify`. When unset it derives from `AUTOTUNEX_DATABASE_SSL_CA` (`verify` when a CA is set, else `disable`). MySQL URLs only. | *(unset — derived from SSL_CA)* |
| `AUTOTUNEX_DATABASE_POOL_SIZE` | Warm connections kept open per worker and reused; size to cover request handlers plus the reconcile loop. Ignored for SQLite. | `10` |
| `AUTOTUNEX_DATABASE_MAX_OVERFLOW` | Extra connections opened on demand beyond the pool size; the hard ceiling is size + overflow. Ignored for SQLite. | `5` |
| `AUTOTUNEX_DATABASE_POOL_TIMEOUT_SECONDS` | Seconds a request waits for a free connection before failing. Ignored for SQLite. | `30.0` |
| `AUTOTUNEX_DATABASE_POOL_RECYCLE_SECONDS` | Recycle a pooled connection older than this many seconds (`-1` disables); keep it under the database's `wait_timeout`. Server databases only. | `1800` |
| `AUTOTUNEX_DATABASE_POOL_PRE_PING` | Liveness-check a pooled connection on checkout, reconnecting if it is dead. Applies to every pool, SQLite included. | `true` |
| `AUTOTUNEX_DATABASE_POOL_USE_LIFO` | Hand out the most-recently-used connection first, keeping a small subset hot under bursty traffic. Ignored for SQLite. | `true` |

## Artifacts

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_ARTIFACT_DIR` | Directory where trained adapters and checkpoints are written. Object storage is not implemented yet. | `./artifacts` |

## Datasets & storage

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_DATASET_STORAGE_DIR` | Root directory for locally-stored dataset files. In-flight uploads stage under a `.staging` subdirectory of this path. | `artifacts/datasets` |
| `AUTOTUNEX_DATASET_UPLOAD_MAX_BYTES` | Hard cap on a single uploaded file, enforced while streaming (returns 413 when exceeded). Must be ≥ 1. | `5368709120` (5 GiB) |
| `AUTOTUNEX_DATASET_UPLOAD_MAX_CONCURRENT` | Max dataset uploads processed concurrently in-process; over-limit uploads queue. | `2` |
| `AUTOTUNEX_DATASET_PROCESSING_TIMEOUT_SECONDS` | Backstop timeout for a dataset's off-request processing; on expiry the dataset is marked `error`. | `3600` |
| `AUTOTUNEX_DATASET_PUSH_TIMEOUT_SECONDS` | Timeout for each `llmb` auth/push subprocess in the HuggingFace backend. | `1800` |
| `AUTOTUNEX_DATASET_CLIENT_GZIP_ENABLED` | Whether the frontend gzip-compresses compressible (jsonl/json/csv) dataset uploads. Surfaced via `GET /api/v1/app-config`. | `true` |
| `AUTOTUNEX_DATASET_CLIENT_GZIP_MIN_BYTES` | Skip client-side gzip below this file size. Surfaced via `GET /api/v1/app-config`. | `1048576` (1 MiB) |
| `AUTOTUNEX_DATASET_CLIENT_PARQUET_PREVIEW_MAX_BYTES` | Byte threshold above which the frontend skips a local Parquet preview. Surfaced via `GET /api/v1/app-config`. | `104857600` (100 MiB) |
| `AUTOTUNEX_DATASET_STORAGE_BACKEND` | Where datasets are stored: `auto`, `local`, or `huggingface`. `auto` resolves to `local` whenever `GB_ENVIRONMENT` is `standalone` (where `llmb artifact push` is unavailable), regardless of tokens; otherwise it resolves to `huggingface` when the `llmb` tooling and **both** token env vars are present, and `local` if not. Forcing `huggingface` without both tokens is refused at startup. | `auto` |
| `AUTOTUNEX_HF_TOKEN_ENV` | **Name** of the environment variable holding the HuggingFace token used for the HuggingFace storage backend. Only the variable's presence is checked; its value is never loaded into settings. | `HF_TOKEN` |
| `AUTOTUNEX_HF_NAMESPACE` | Optional HuggingFace org/namespace prefix for the derived dataset-repo name. | *(unset)* |
| `AUTOTUNEX_HF_PREVIEW_ENABLED` | Enable dataset preview via the HuggingFace dataset viewer. When off, HuggingFace-stored datasets return an empty preview; locally-stored datasets read rows from disk regardless. | `true` |
| `AUTOTUNEX_HF_VIEWER_BASE_URL` | Base URL of the HuggingFace dataset-viewer service; the client appends `/rows`. Override to point at a mirror. | `https://datasets-server.huggingface.co` |
| `AUTOTUNEX_HF_VIEWER_TIMEOUT_SECONDS` | Per-call HTTP timeout for one viewer fetch. Must be > 0. | `2.5` |
| `AUTOTUNEX_HF_HUB_BASE_URL` | HuggingFace Hub API base URL used to list/download a job's output-model files (result-report endpoint). | `https://huggingface.co` |

## LLM intelligence (optional)

These power the dataset-intelligence endpoints (suggest parsing strategy, suggest column
mapping). They target **any OpenAI-compatible gateway** — the adapter appends
`/chat/completions` to the base URL — so keep the values vendor-neutral.

`AUTOTUNEX_LLM_BASE_URL`, `AUTOTUNEX_LLM_API_KEY`, and `AUTOTUNEX_LLM_MODEL` must be set
**together or all unset**; a partial configuration fails at startup. When all three are
unset the feature is disabled and the dataset-intelligence endpoints return `503`.

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_LLM_BASE_URL` | OpenAI-compatible gateway base URL, e.g. `https://gateway.example.com/v1`. | *(unset)* |
| `AUTOTUNEX_LLM_API_KEY` | Bearer token for the gateway. A secret — inject from the environment, never commit it. | *(unset)* |
| `AUTOTUNEX_LLM_MODEL` | Model name passed through to the gateway. No provider-model default. | *(unset)* |
| `AUTOTUNEX_LLM_TIMEOUT_SECONDS` | Per-call HTTP timeout for a single chat completion. Must be > 0. | `30.0` |
| `AUTOTUNEX_LLM_MAX_RETRIES` | Bounded parse self-correction retries (total attempts = this + 1). Must be ≥ 0. | `2` |
| `AUTOTUNEX_LLM_MAX_SAMPLE_BYTES` | Cap on sample bytes sent to the LLM (cost and injection-surface bound). Must be ≥ 1. | `8000` |
| `AUTOTUNEX_LLM_STRUCTURED_OUTPUT` | How to request JSON from the gateway: `json_schema` (strict OpenAI-style schema), `json_object` (portable JSON mode), or `none` (no `response_format`, prompt only). Gateways diverge on `response_format`; `json_object` is the most widely accepted. | `json_object` |

## Reward-function sandbox (online-RL reward step)

These bound one sandboxed reward-function run, used by the online-RL reward step. Both are
enforced on Linux (production/Docker); on macOS the address-space rlimit may be a no-op, but
the wall-clock kill always applies.

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_REWARD_TIMEOUT_SECONDS` | Hard wall-clock (and CPU rlimit) budget for one sandboxed reward-function run. Must be ≥ 1. | `5` |
| `AUTOTUNEX_REWARD_MEMORY_BYTES` | Address-space rlimit for the reward-sandbox child. Must be ≥ 1. | `536870912` (512 MiB) |

## Chat assistant + MCP server

The in-app chat assistant reuses the LLM gateway configured above
(`AUTOTUNEX_LLM_*`) — set `AUTOTUNEX_LLM_MODEL` to a model that is reliable at
tool use, since chat tool-calling quality depends on it. The MCP server is a
separate, opt-in surface: mounting it does **not** affect the in-app chat.

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_ENABLE_MCP` | Mount the standalone MCP server at `/mcp` for external MCP clients (Claude Desktop, Cursor). Requires the `[mcp]` extra (`fastmcp`) and, for external callers, `"api_key"` in `AUTOTUNEX_AUTH_PROVIDERS`. The in-app chat does not depend on it. | `false` |
| `AUTOTUNEX_CHAT_MAX_TOOL_ITERATIONS` | Upper bound on ReAct tool-call rounds per chat turn — a runaway guard. Must be ≥ 1. | `8` |
| `AUTOTUNEX_CHAT_THREAD_TTL_SECONDS` | Idle lifetime, in seconds, of a `thread_id` conversation held in in-process memory. Must be > 0. | `3600` |
| `AUTOTUNEX_CHAT_MAX_THREADS` | LRU cap on stored conversations, bounding chat memory growth. Must be ≥ 1. | `500` |

## Execution: job launch

`AUTOTUNEX_JOB_BACKEND` selects which job runner is built. See
[job-backends.md](job-backends.md) for the full description of each runner and its
behavior — the tables below only list the settings each one reads.

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_QUEUE_URL` | **Reserved** broker URL for a future task-queue runner. Unused today — setting it has no effect, and submitted jobs still stay `pending`. | *(unset)* |
| `AUTOTUNEX_JOB_BACKEND` | Which runner to build: `none` (accept and do nothing — jobs stay `pending`), `llmb` (submit builds via the `llmb` CLI), or `local` (run the HPO in-process). | `none` |

### `llmb` backend (remote `custom_code`)

Used when `AUTOTUNEX_JOB_BACKEND=llmb` and `GB_ENVIRONMENT` is **not** `standalone`.
`AUTOTUNEX_JOB_RUNTIME_IMAGE`, `AUTOTUNEX_JOB_TRAINER_REPO`,
`AUTOTUNEX_JOB_OUTPUT_URI_ROOT`, and `AUTOTUNEX_GB_SERVER_URL` are **all required** in this
mode, and the environment variable named by `AUTOTUNEX_GB_TOKEN_ENV` must be present.

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_JOB_RUNTIME_IMAGE` | Container image the cluster runs the tuning build in. **Required** in this mode. | *(unset)* |
| `AUTOTUNEX_JOB_TRAINER_REPO` | Trainer source repository the build checks out. **Required** in this mode. | *(unset)* |
| `AUTOTUNEX_JOB_TRAINER_REF` | Branch, tag, or commit of the trainer repo to check out. | `main` |
| `AUTOTUNEX_JOB_OUTPUT_URI_ROOT` | Root URI for run artifacts; each run's output is written under a subpath. **Required** in this mode. | *(unset)* |
| `AUTOTUNEX_JOB_CALLBACK_URL` | The api-bridge / callback base URL the build reports to. The `custom_code` and LSF start commands emit it as `--autotunex_server_url` **only when it is set**; the local-`bash` variant always emits it, with a default (see that table below). | *(unset)* |
| `AUTOTUNEX_GB_SERVER_URL` | Base URL of the build server the reconcile loop polls for job status, e.g. `https://gb.example.com`. **Required** whenever `job_backend=llmb` (all three variants) — without it, accepted jobs sit `pending` forever. | *(unset)* |
| `AUTOTUNEX_GB_TOKEN_ENV` | **Name** of the environment variable holding the build-server auth token. The token **value** is read at the subprocess/request boundary and never loaded into settings. Set the named variable (`GB_TOKEN` by default) in the environment. | `GB_TOKEN` |
| `AUTOTUNEX_JOB_SPEC_DIR` | Directory the generated `build.yaml` is written to, one per job at `<dir>/<job_id>/build.yaml`. Relative paths resolve against the working directory. The spec is kept after submission (including on failure) so it can be inspected or replayed. | `tmp` |
| `AUTOTUNEX_JOB_RECONCILE_INTERVAL_SECONDS` | How often the reconcile loop sweeps non-terminal jobs for status changes. Must be ≥ 1. | `30` |
| `AUTOTUNEX_JOB_RECONCILE_CONCURRENCY` | Upper bound on simultaneous status reads against the build server per sweep. Must be ≥ 1. | `5` |
| `AUTOTUNEX_LLMB_COMMAND` | The `llmb` CLI binary name or path used to submit builds — and to push datasets when the HuggingFace storage backend is active. | `llmb` |
| `AUTOTUNEX_GB_TAGS` | Tag(s) passed to `llmb build start` and `llmb artifact push` via `--tag`. A single tag or a comma-separated list, forwarded verbatim (`llmb` splits it); set empty to omit the flag from both commands. | `autotunex` |

### `llmb` backend (local `bash` variant)

When `GB_ENVIRONMENT=standalone` **and** `AUTOTUNEX_JOB_BACKEND=llmb`, the launcher emits
a local-`bash` build spec instead of `custom_code` — an example of running the build
tooling and AutoTuneX together on a single machine. In this mode
`AUTOTUNEX_JOB_RUNTIME_IMAGE`, `AUTOTUNEX_JOB_TRAINER_REPO`, and
`AUTOTUNEX_JOB_OUTPUT_URI_ROOT` are **not** required; only `AUTOTUNEX_GB_SERVER_URL`
(plus the `AUTOTUNEX_GB_TOKEN_ENV` variable) is. In place of an output-URI root, the bash
spec anchors each run's output under `AUTOTUNEX_ARTIFACT_DIR`, resolved to an absolute
`file://` URI — the bash environment only writes `file://`.

| Variable | Meaning | Default |
| --- | --- | --- |
| `GB_ENVIRONMENT` | **Read without the `AUTOTUNEX_` prefix** — it reuses the build tooling's own `GB_ENVIRONMENT` variable, and the prefixed `AUTOTUNEX_GB_ENVIRONMENT` is deliberately ignored. Set to `standalone` to select the local-`bash` build spec. | *(unset)* |
| `AUTOTUNEX_BASH_FM_TUNE_ROOT` | Trainer checkout/repo injected into the bash spec's environment. | *(unset)* |
| `AUTOTUNEX_BASH_FM_TUNE_REF` | Branch/tag/commit of the trainer to check out; unset uses the repo's default branch. | *(unset)* |
| `AUTOTUNEX_BASH_FM_TUNE_EXTRA` | The extras to install in the bash spec. | `full,mlx` |
| `AUTOTUNEX_BASH_BACKEND` | Compute backend for the bash spec: `mlx` (Apple Silicon) or `torch`. | `mlx` |
| `AUTOTUNEX_JOB_CALLBACK_URL` | The api-bridge / callback base URL the build reports to, injected into the bash spec's environment as `AUTOTUNEX_SERVER_URL`. Unlike the `custom_code` and LSF start commands — which emit `--autotunex_server_url` only when it is set — the bash spec **always** emits it, falling back to `http://localhost:8001` (the api-bridge's default port) when unset. | *(unset — the bash spec then emits `http://localhost:8001`)* |

### `llmb` backend (LSF / SkyPilot variant)

When `GB_ENVIRONMENT=standalone` **and** `AUTOTUNEX_LSF_CLUSTER` is set (with
`AUTOTUNEX_JOB_BACKEND=llmb`), the launcher emits the LSF/SkyPilot build spec instead of
the local-`bash` spec. `AUTOTUNEX_LSF_CLUSTER` is the discriminator: unset keeps the bash
default. In this mode `AUTOTUNEX_LSF_ENVIRONMENT_URI`, `AUTOTUNEX_LSF_IMAGE`, and
`AUTOTUNEX_JOB_TRAINER_REPO` become **required** (alongside `AUTOTUNEX_GB_SERVER_URL` and
the `AUTOTUNEX_GB_TOKEN_ENV` variable). `AUTOTUNEX_JOB_TRAINER_REF` and
`AUTOTUNEX_JOB_CALLBACK_URL`, documented in the `custom_code` table above, apply here too —
the LSF spec reads both.

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_LSF_CLUSTER` | SkyPilot/LSF cluster name. **The discriminator** — set together with `AUTOTUNEX_JOB_BACKEND=llmb` and `GB_ENVIRONMENT=standalone`, it selects the LSF/SkyPilot spec over the local-`bash` spec. | *(unset)* |
| `AUTOTUNEX_LSF_ENVIRONMENT_URI` | granite.build space environment for the LSF build, e.g. `space://environments/skypilot/lsf/<cluster>`. **Required** in this mode. | *(unset)* |
| `AUTOTUNEX_LSF_IMAGE` | Runtime container image the LSF step runs in. **Required** in this mode. | *(unset)* |
| `AUTOTUNEX_LSF_ACCELERATORS` | SkyPilot accelerators string, e.g. `H100:2`. Unset omits the key entirely, producing a 0-GPU build. | *(unset)* |
| `AUTOTUNEX_LSF_QUEUE` | LSF queue, mapped to the build's `resources.zone`. Omitted when unset. | *(unset)* |
| `AUTOTUNEX_LSF_MEMORY` | Requested memory for the build's `resources.memory`. Omitted when unset. | *(unset)* |
| `AUTOTUNEX_LSF_VENV_PATH` | Virtual-env path on the runtime image (`skypilot.venv_path`) where torch/ray are installed. | `/step_venv` |
| `AUTOTUNEX_LSF_CUDA_HOME` | CUDA toolkit path exported in the LSF start command. | `/opt/share/cuda-12.9` |
| `AUTOTUNEX_LSF_NUM_CPUS_PER_NODE` | `compute_config.num_cpus_per_node` for the LSF build. Must be ≥ 1. | `32` |
| `AUTOTUNEX_LSF_TOTAL_MEMORY_PER_NODE` | `compute_config.total_memory_per_node` for the LSF build. | `256Gi` |
| `AUTOTUNEX_LSF_POLL_INTERVAL_SECONDS` | Poll and log-retrieval interval on the LSF step. Must be ≥ 1. | `30` |

### `local` backend (in-process HPO)

Used when `AUTOTUNEX_JOB_BACKEND=local`. This runner requires nothing at startup; whether
the optional in-process HPO package is installed is a runtime concern.

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_LOCAL_RAY_ADDRESS` | Ray cluster address. Unset means a local Ray (`ray.init()`). | *(unset — local Ray)* |
| `AUTOTUNEX_LOCAL_OUTPUT_DIR` | Root for a local run's output; the per-job subdir is `<dir>/<job_id>/`. Defaults to a subdirectory of the artifact directory, so overriding `AUTOTUNEX_ARTIFACT_DIR` moves it too unless set explicitly. | `<ARTIFACT_DIR>/local` (i.e. `artifacts/local`) |
| `AUTOTUNEX_LOCAL_CANCEL_TIMEOUT_SECONDS` | Seconds `LocalJobRunner.cancel` waits for an in-process run to stop before returning `JobCancellationInProgressError`. The cancel is latched regardless; this only bounds the wait. | `30.0` |

## Limits

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_MAX_TRIALS_LIMIT` | **Reserved / not currently enforced.** A server-side ceiling on a job's trial count that no code path checks today. Must be ≥ 1. Do not rely on it to bound trials. | `100` |

## Authentication

The settings below configure how callers are authenticated and authorized. For how the
providers fit together — standalone mode, per-caller scoping, API keys, OIDC bearer
tokens, and browser sessions — see [../api/authentication.md](../api/authentication.md).

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_AUTH_PROVIDERS` | JSON list of accepted credential kinds: `disabled`, `api_key`, `oidc`, `session`. `["disabled"]` is standalone mode and cannot combine with any other provider. | `["disabled"]` |
| `AUTOTUNEX_STANDALONE_EMAIL` | Email the standalone principal's writes are attributed to. When unset, writes go to a default system owner, provisioned lazily on first request. | *(unset — default system owner)* |
| `AUTOTUNEX_STANDALONE_ROLE` | Role the standalone principal carries: `admin` or `user`. Wins over any `users.role` column value. | `admin` |
| `AUTOTUNEX_API_KEYS` | JSON map of a key's SHA-256 hex digest → the owner's email. Never store raw keys. Required and non-empty when `"api_key"` is enabled. | `{}` |
| `AUTOTUNEX_AUTO_PROVISION_USERS` | Just-in-time provision a `users` row (always `role=user`) on a caller's first request when they have a resolvable, verified email but no row yet. Off by default; makes even a `GET` write on first request. | `false` |
| `AUTOTUNEX_ALLOW_INSECURE_NO_AUTH` | Permit `auth_providers=["disabled"]` while `environment=prod`. Off by default; setting it logs a loud startup warning. Only for a deliberate single-tenant, no-auth deployment. | `false` |

### OIDC bearer tokens

Required together when `"oidc"` is in `AUTOTUNEX_AUTH_PROVIDERS`. There is no default
issuer, deliberately.

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_OIDC_ISSUER` | The issuer (`iss`) a bearer token must carry. **Required** when `"oidc"` is enabled. | *(unset)* |
| `AUTOTUNEX_OIDC_JWKS_URI` | Where to fetch the issuer's signing keys. **Required** when `"oidc"` is enabled. | *(unset)* |
| `AUTOTUNEX_OIDC_AUDIENCE` | The audience (`aud`) a bearer token must carry, checked unconditionally. **Required** when `"oidc"` is enabled. | *(unset)* |
| `AUTOTUNEX_OIDC_ALGORITHMS` | JSON list of signature algorithms accepted when verifying a token. | `["RS256"]` |
| `AUTOTUNEX_OIDC_EMAIL_CLAIMS` | JSON list of claim names checked, in order, to resolve a token to an email. | `["email","emailAddress"]` |
| `AUTOTUNEX_OIDC_LEEWAY_SECONDS` | Clock-skew tolerance for `exp`/`nbf` checks. Bounded 0–300. | `30` |

### Browser sessions (backend-for-frontend)

Required when `"session"` is in `AUTOTUNEX_AUTH_PROVIDERS`, in addition to the three OIDC
settings above (the audience here is the ID token's, which is always the client id).

| Variable | Meaning | Default |
| --- | --- | --- |
| `AUTOTUNEX_OIDC_CLIENT_ID` | The BFF's own OIDC client id. **Required** when `"session"` is enabled. | *(unset)* |
| `AUTOTUNEX_OIDC_CLIENT_SECRET` | The BFF's OIDC client secret, exchanged at the token endpoint. A secret — inject it. **Required** when `"session"` is enabled. | *(unset)* |
| `AUTOTUNEX_OIDC_AUTHORIZATION_ENDPOINT` | Where `/auth/login` redirects the browser, e.g. `https://idp.example.com/authorize`. **Required** when `"session"` is enabled. | *(unset)* |
| `AUTOTUNEX_OIDC_TOKEN_ENDPOINT` | Where `/auth/callback` exchanges the authorization code. **Required** when `"session"` is enabled. | *(unset)* |
| `AUTOTUNEX_OIDC_END_SESSION_ENDPOINT` | Where `/auth/logout` redirects to end the upstream session, if the issuer supports RP-initiated logout. Optional. | *(unset)* |
| `AUTOTUNEX_PUBLIC_BASE_URL` | Public base URL used to build `redirect_uri` as `<base>/auth/callback`. **No trailing slash.** Never taken from a request header. **Required** when `"session"` is enabled. | *(unset)* |
| `AUTOTUNEX_SESSION_SECRET` | Secret signing the session JWT (HS256). Must be **≥ 32 characters** (e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"`). No random fallback. A secret — inject it. **Required** when `"session"` is enabled. | *(unset)* |
| `AUTOTUNEX_SESSION_TTL_HOURS` | How long a minted session cookie remains valid. Bounded 1–24. | `8` |
| `AUTOTUNEX_SESSION_COOKIE_SAME_SITE` | Cookie `SameSite` policy: `lax` (same-origin UI) or `none` (cross-origin UI — then a non-empty CORS allowlist is mandatory). | `lax` |
| `AUTOTUNEX_CORS_ALLOW_ORIGINS` | JSON list of origins CORS allows, e.g. `["https://ui.example.com"]`. Must **never** contain `"*"`. | `[]` |

---

## Related

- [database.md](database.md) — supported databases, connection strings, and migrations.
- [job-backends.md](job-backends.md) — what each `AUTOTUNEX_JOB_BACKEND` runner does.
- [../api/authentication.md](../api/authentication.md) — how the auth providers fit together.
- [deployment.md](deployment.md) — deploying the service.
