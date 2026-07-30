# `gbserver` CLI reference

Quick reference for the `gbserver` command — the server-side CLI that runs the REST API, the build and
PR watchers, the build runner, the all-in-one standalone server, and admin tasks.

> **Audience:** operators running gbserver. For the client CLI that talks to a running server, see the
> [`gb` CLI reference](gb-cli-reference.md).

> Run `gbserver --help` or `gbserver <command> --help` for the exhaustive list of options. This page is
> the cheat sheet — it lists what's there and points at the source for the details.

## Top-level options

Global options on the `gbserver` group (before the subcommand):

| Option | Notes |
|--------|-------|
| `--log-level` | `debug`, `info` (default), `warning`, `error`, `critical`. |
| `--log-file <path>` | Also write logs to this file. |
| `--gb-admin-table-prefix <prefix>` | Prefix for the admin metadata table names (mainly for testing). |
| `--server-runtime-config <path>` | Path to a server runtime config file. |

Behaviour is driven largely by environment variables (see [below](#key-environment-variables)).

## Running the server

| Command | Purpose |
|---------|---------|
| `gbserver standalone [--host 0.0.0.0] [--port 8080] [--space-dir <dir>]` | All-in-one local server: REST API + BuildWatcher in one process. Forces `GB_ENVIRONMENT=STANDALONE` and applies standalone-friendly defaults (SQLite storage, thread build runner, API-key auth, embedded NATS). `--space-dir` defaults to the in-repo `configurations/spaces/local`. |
| `gbserver rest-server [--port 8080]` | Start just the REST API server (`/api/v1`). |
| `gbserver build-watch [--gh-token <t>] [--config <f>] [--watch/--no-watch]` | Watch for pending builds (from PRs or a config) and dispatch build runners. |
| `gbserver build-runner (--build-id <id> \| --build-dir <dir>) [...]` | Execute a single build — either a `PENDING` build from storage (`--build-id`) or one loaded from a directory (`--build-dir`). `--build-id` and `--build-dir` are mutually exclusive. |

`build-runner` extras (directory mode): `--space-name`, `--space-config-uri`, `--space-dir`, `--target/-t` (repeatable),
`--username`, `--workspace-dir`, `--monitoring-interval`, `--create-pr`, `--enable-resume`, `--dry-run`.
`--space-name`, `--space-config-uri`, and `--space-dir` are mutually exclusive — pick one to select the build's space.
`--space-dir <dir>` is a convenience form of `--space-config-uri` for a local space: point it at a directory containing a
`space.yaml` (e.g. `configurations/spaces/local`) and it is resolved to a `file://` URI.
The runner backend is chosen by `GBSERVER_DEFAULT_BUILDRUNNER_TYPE` (`job` / `process` / `thread`).

Signals: pressing **Ctrl+C** (SIGINT) cancels a running build (marked `CANCELLED`); **SIGTERM** fails it (marked `FAILED`).
In both cases the runner finalizes the build (and its targets/steps/artifacts) before exiting rather than leaving it `RUNNING`.
Exit code: `0` for `SUCCESS` and `CANCELLED` (a deliberate Ctrl+C cancel); non-zero for `FAILED` (including SIGTERM) and `INVALID`, so callers can key off the exit code.

## Local builds

Run a build directly from a directory, without the watcher:

| Command | Purpose |
|---------|---------|
| `gbserver build run [BUILD_DIR] [--space-name <n> \| --space-config-uri <uri>] [-t TARGET]... [--dry-run]` | Run a build from `BUILD_DIR` (defaults to the current directory). |
| `gbserver build run-and-monitor [BUILD_DIR] [...]` | Same as `build run`, but streams all emitted events. Does not exit on its own — Ctrl+C to stop. |

Both accept `--cancel_on_error` and `--user-name`.

## Administration

| Command | Purpose |
|---------|---------|
| `gbserver add-users <users_file>` | Add users to spaces from a YAML file (`spaces: <name>: [{username, role}]`). |
| `gbserver create-spaces [--spaces-path <f>] [--clear] [--replace] [--force]` | Create the predefined spaces (registers them in the `gb_spaces` table). Each space in the YAML accepts `name`, `git_repo_uri`, `lakehouse_namespace`, and an optional `hf_default_resource_group_id` (the HF Enterprise resource group id for the space's default group, cached for HF pushes; no HF API lookup is done here). |
| `gbserver admin-tables --operation <op> [--dry-run]` | Repair admin metadata: `fail-zombie-builds`, `fix-zombie-targets`, `fix-zombie-steps`, `fail-pending-without-pr`. Prompts for confirmation. |

`rest-server-worker` is an internal pseudo-command used when running the REST server with multiple
workers; it is not invoked directly.

## Key environment variables

`gbserver`'s behaviour is driven mostly by `GBSERVER_*` / `GB_*` env vars. The full grouped reference,
the config files, and the `GB_ENVIRONMENT` / standalone mechanics live in
**[Configuration](../configuration/README.md)** (source of truth:
[`constants.py`](../../src/gbserver/types/constants.py)). A few that most directly affect these commands:

| Variable | Purpose |
|----------|---------|
| `GB_ENVIRONMENT` | `DEV` / `STAGING` / `PROD` / `STANDALONE` — the profile every command runs under (`standalone` forces `STANDALONE`). |
| `GBSERVER_DEFAULT_BUILDRUNNER_TYPE` | `job` (k8s), `process`, or `thread` — how `build-watch` / `build-runner` execute builds. |
| `GBSERVER_GITHUB_TOKEN` | Default GitHub token for the watchers/runner (overridable per-command with `--gh-token`). |

For everything else — storage, [authentication](../rest-api/README.md#authentication), messaging, image
tags, etc. — see [Configuration](../configuration/README.md).

## Where commands live

`gbserver`'s root ([`src/gbserver/cli.py`](../../src/gbserver/cli.py)) discovers subcommands dynamically
from [`src/gbserver/commands/`](../../src/gbserver/commands/): a file `command_<name>.py` becomes the
`gbserver <name>` command (underscores → hyphens), exporting a `cli` Click command.

## See also

- [CLIs overview](README.md) — how `gb` and `gbserver` relate
- [`gb` CLI reference](gb-cli-reference.md) — the client CLI
- [Environment setup](../environments/setup/) · [Troubleshooting](../help/troubleshooting.md)
