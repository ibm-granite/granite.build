# Deployment environments (`GB_ENVIRONMENT`)

> **Audience:** operators choosing which deployment a gbserver process targets. For the env-var list see
> [environment-variables.md](environment-variables.md).

`GB_ENVIRONMENT` selects a **built-in per-environment config** that supplies defaults for the cluster,
namespace, SQL schema, and space git branches. The four values are `DEV`, `STAGING`,
`PROD`, and `STANDALONE`; the config objects live in
[`src/gbcommon/types/gbenvconfig.py`](../../src/gbcommon/types/gbenvconfig.py) (default: `PROD`). These
are defaults — individual environment variables still override them.

| Aspect | `PROD` | `STAGING` | `DEV` | `STANDALONE` |
|--------|--------|-----------|-------|--------------|
| K8s namespace | `llm-build-prod` | `llm-build-staging` | `llm-build-dev` | default |
| SQL schema | `granite_dot_build_prod` | `granite_dot_build_staging` | `granite_dot_build_dev` | `standalone` |
| Space-config branch | `gbspace-config` | `gbspace-config` | `gbspace-config` | `main` |

You can define or override an environment's defaults with a
[`--server-runtime-config`](config-files.md#server-runtime-config---server-runtime-config) YAML file.

## Standalone mode

`GB_ENVIRONMENT=STANDALONE` (which `gbserver standalone` sets for you) is the local/offline profile: no
Kubernetes or IBM services required. On startup `check_and_init_for_standalone()`
([`src/gbserver/commands/utils.py`](../../src/gbserver/commands/utils.py)) applies these defaults **only
where you haven't set them** (`STANDALONE_ENV_DEFAULTS` in
[`constants.py`](../../src/gbserver/types/constants.py)):

| Variable | Standalone default | Effect |
|----------|--------------------|--------|
| `GBSERVER_METADATA_STORAGE` | `sqlite` | Local SQLite instead of PostgreSQL. |
| `GBSERVER_DEFAULT_BUILDRUNNER_TYPE` | `thread` | Run builds in-process instead of as k8s jobs. |
| `GBSERVER_AUTH_MODE` | `apikey` | Simple shared-key auth (localhost allowed unauthenticated). |
| `GBSERVER_EVENT_PUBLISHING_ENABLED` | `true` | Publish build events (embedded NATS). |
| `GBSERVER_PROCEED_WITHOUT_SECRETS` | `true` | Don't require a remote secret manager. |

The defaults are applied with `os.environ.setdefault(...)`, i.e. **only where you haven't already set
them**, so any variable you export before starting the server wins (e.g. export `GBSERVER_METADATA_STORAGE=sql`
to use PostgreSQL even in standalone). Two settings are resolved dynamically rather than written to the
environment: the per-user secret backend defaults to `local`, and the lineage provider defaults to `none`.

Beyond applying those defaults, `check_and_init_for_standalone()` performs the rest of the one-time
standalone setup: it reloads the `constants` module so import-time values pick up the defaults, installs
the **SQLite** storage factory (migrating any legacy database first), installs the standalone **space
access manager** (which bypasses remote authorization), and — for the standalone *server* — registers
the `--space-dir` space under `public` (and the legacy aliases `standalone` / `local`). Outside standalone
the function is a no-op.

## See also

- [Configuration overview](README.md) · [Environment variables](environment-variables.md) · [Config files](config-files.md)
- [`gbserver` CLI reference](../cli/gbserver-cli-reference.md) — the `standalone` command
