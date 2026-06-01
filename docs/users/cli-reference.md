# `gb` CLI reference

Quick reference for the `gb` command. The CLI is a thin client over
gbserver's REST API at `/api/v1`.

> Run `gb <command> --help` or `gb <command> <subcommand> --help` for the
> exhaustive list of options. This page is the cheat sheet — it lists what's
> there and points at the source for the details.

## Top-level options

| Option       | Notes |
|--------------|-------|
| `--loglevel` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `--format`   | `simple` (default), `json`, or `plain` — varies by command. |
| `--quiet`    | Suppress informational output. |
| `--skip-version-check` | Skip the client/server version check at startup. |

Server target and credentials are picked up from environment variables
configured during `gb auth login`. See
[Authentication](#auth) and
[`operators/multi-provider-authentication.md`](../operators/multi-provider-authentication.md)
for the full credential model.

## Command groups

Each top-level group is implemented in a single file under
[`src/gbcli/commands/`](../../src/gbcli/commands/) — `command_build.py`
becomes `gb build`, etc.

### `build` — work with builds

Most-used group. Submit, monitor, and inspect builds.

| Subcommand | Purpose |
|------------|---------|
| `gb build start [-f <build.yaml>] [TARGETS...]` | Submit a build. Optionally name specific targets. |
| `gb build validate [TARGETS...]` | Validate without submitting. |
| `gb build list` | List builds. |
| `gb build status <build-id>` | Show status and per-step state. |
| `gb build log <build-id>` | Stream or fetch build logs. |
| `gb build cancel <build-id>` | Cancel a running build. |
| `gb build describe [<build-id> \| -f <build.yaml>]` | Describe targets and steps. |
| `gb build diff <build-id-1> [<build-id-2>]` | Diff two builds (or one against its `build.yaml`). |
| `gb build lineage <build-id>` | Show artifact lineage for a build. |
| `gb build monitor <build-id>` | Live progress view. |
| `gb build init [NAME \| -f <file>]` | Scaffold a new build definition. |
| `gb build update <build-id>` | Update description or tags. |
| `gb build notification [on\|off]` | Toggle build notifications for the active space. |

Common flags: `--param key=value` (repeatable), `--tag <name>`,
`--skip-validation`, `--format json`.

### `artifact` — work with artifacts

Upload, download, register, and tag artifacts produced or consumed by builds.

| Subcommand | Purpose |
|------------|---------|
| `gb artifact push --from-local <path> --artifact-name <name> --type <type>` | Upload and register a local artifact. `--type` is one of `model`, `table`, `fileset`, `dataset`, `bucket`. |
| `gb artifact register --artifact-name <name>` | Register an existing artifact at Lakehouse or HuggingFace without uploading. |
| `gb artifact list` | List artifacts; filter by build, space, user, tag, or checksum. |
| `gb artifact download <artifact-id>` | Download by UUID or URI. |
| `gb artifact describe <artifact-id>` | Show description and tags. |
| `gb artifact checksum <artifact-id>` | Print the checksum. |
| `gb artifact lineage <artifact-id>` | Show artifact lineage. |
| `gb artifact update <artifact-id>` | Update description or tags. |
| `gb artifact archive <artifact-id>` / `unarchive` | Toggle archived state. |
| `gb artifact copy <artifact-id> --space-to <space>` | Copy a model to another space. |

### `auth` — authentication

| Subcommand | Purpose |
|------------|---------|
| `gb auth login [--token \| --sso \| --gbserver]` | Authenticate. Default is GitHub; `--sso` uses IBMid; `--gbserver` uses an API-key gbserver. |
| `gb auth provider [--set <provider>]` | Show or change the default auth provider. |

### `space` — work with spaces

A space is the gbserver-side container for environments, steps, builds, and
artifacts.

| Subcommand | Purpose |
|------------|---------|
| `gb space list [--all] [--refresh]` | List spaces visible to the user. |
| `gb space set <space-name>` | Set the active space for subsequent commands. |

### `secret` — manage secrets

Per-space or per-user secrets pulled by builds at runtime. See also
[`operators/local-secrets-manager.md`](../operators/local-secrets-manager.md).

| Subcommand | Purpose |
|------------|---------|
| `gb secret list [--space \| --personal]` | List secrets. |
| `gb secret get <name>` | Show a secret. |
| `gb secret create <name> --value <text>` (or `--from-file <path>`) | Create. |
| `gb secret update <name>` | Update. |
| `gb secret delete <name>` | Delete. |

### `model` — work with deployed models (RITS)

| Subcommand | Purpose |
|------------|---------|
| `gb model list [--byom] [--uri]` | List standard and BYOM checkpoints deployed in RITS. |
| `gb model prompt <msg> --model <name>` | Single-prompt completion. |
| `gb model chat --model <name>` | Interactive chat. |

Tuning flags: `--temp`, `--max`, `--top_p`, `--system`, `--chat_template`.

### `step` — work with step definitions

| Subcommand | Purpose |
|------------|---------|
| `gb step list [--space \| --step-repo]` | List available steps. |
| `gb step describe <name>` | Show step contents and config. |

### `template` — work with build templates

| Subcommand | Purpose |
|------------|---------|
| `gb template list [--space \| --template-repo]` | List templates. |
| `gb template describe <name>` | Show template contents. Supports `--format simple\|full\|json`. |

### `tag` — list tags

| Subcommand | Purpose |
|------------|---------|
| `gb tag list [--builds \| --artifacts]` | List tags. Filter with `--space` and `--username`. |

### `cleanup` — clean local state

| Subcommand | Purpose |
|------------|---------|
| `gb cleanup --all` | Clean everything below. |
| `gb cleanup --config` | Remove `~/.gbcli/config`. |
| `gb cleanup --credentials` | Remove `~/.gbcli/credentials`. |
| `gb cleanup --local-cache` | Clear `~/.gbcli/workdir`. |
| `gb cleanup --space-repo-fork` | Print instructions to delete a fork repo. |

### `admin` — admin-only operations

| Subcommand | Purpose |
|------------|---------|
| `gb admin log <module>` | Fetch server logs for `gbserver-rest-server`, `gbserver-pr-watch`, `gbserver-build-watch`, or `gbserver-build-runner`. |
| `gb admin space-membership` | List/add/remove/update space members. |

### `dataset` — *placeholder*

Subcommands (`search`, `diff`, `import`, `list`, `move`, `download`) currently
print "not yet available". Use `gb artifact` for dataset operations today.

### `version` — client version

| Subcommand | Purpose |
|------------|---------|
| `gb version` | Show client version. |
| `gb version --check-updates` | Check for a newer release. |
| `gb version --client` | Print only the client version. |

## Where commands live

Each command's source is the place to look when `--help` isn't enough:

```
src/gbcli/
├── cli.py                    # top-level group, global options
├── commands/
│   ├── command_admin.py
│   ├── command_artifact.py
│   ├── command_auth.py
│   ├── command_build.py
│   ├── command_cleanup.py
│   ├── command_dataset.py
│   ├── command_model.py
│   ├── command_secret.py
│   ├── command_space.py
│   ├── command_step.py
│   ├── command_tag.py
│   ├── command_template.py
│   ├── command_version.py
│   └── common_options.py     # shared flags
├── client/                   # REST client
└── services/                 # higher-level operations
```
