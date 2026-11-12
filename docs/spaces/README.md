# Spaces and `space.yaml`

> **Audience:** operators setting up a space, and anyone who needs to understand how `space://` URIs,
> secrets, and template variables are configured. For how a build references a space see the
> [build.yaml reference](../builds/build-yaml-reference.md); for how `space://steps/<name>` resolves to
> an implementation see [step-resolution.md](../environments/step-resolution.md).

## What is a space?

A **space** is the top-level context a build runs in. It bundles together everything a build needs but
doesn't declare inline:

- **Environments** — the compute backends a target's `environment_uri` resolves to (see the
  [environments overview](../environments/README.md)).
- **Steps** — the step implementations `space://steps/<name>` resolves to.
- **Asset stores** — the stores `space://assetstores/<name>` resolves to.
- **Secrets** — resolved through the space's configured secret manager.
- **Variables** — space-level template values available to every `build.yaml` run in the space.

A space is defined by a `space.yaml` file at its root. Everything else (environments, steps,
assetstores) is discovered by resolving `space://…` URIs against the space's `base_uris` chain. The
implementation is [`Space`](../../src/gbserver/build/space.py); the config model is
[`SpaceConfig`](../../src/gbserver/types/spaceconfig.py).

## `space.yaml` schema

```yaml
name: public                      # Space identifier (e.g. "public", "standalone").
secret_manager:                   # Required. How secrets are resolved for this space.
  type: local                     # local | env | hybrid | ibmcloud
  config: {}                      # Type-specific config (see the secret-manager section).
base_uris:                        # Optional. Where space:// URIs resolve. Relative paths resolve
  - file://../../assets           # against this space.yaml's directory; absolute URIs pass through.
variables:                        # Optional. Template values exposed to build.yaml as
  DEFAULT_ENVIRONMENT: skypilot/kubernetes   # ${variables.<key>} / space.variables.<key>.
```

| Field | Type | Required | Purpose |
|-------|------|----------|---------|
| `name` | string | — (default `""`) | Space identifier, used in secret grouping and logs. |
| `secret_manager` | `{type, config}` | **yes** | Backend that resolves the build's secrets. `type` is one of `local`, `env`, `hybrid`, `ibmcloud`; `config` is type-specific. |
| `base_uris` | list of URIs | no | Base locations searched to resolve `space://…` URIs. Relative `file://`/bare paths resolve against the space directory; absolute URIs (any scheme, e.g. `git://`) pass through unchanged. |
| `variables` | map of string→string | no | Space-level template variables, referenced in `build.yaml` as `${variables.<key>}` (or `space.variables.<key>`). |

### `base_uris` and `space://` resolution

`base_uris` is the search path for `space://` URIs. When a target uses
`environment_uri: space://environments/skypilot/kubernetes` or a step is `space://steps/hfpull`, the
resolver searches the `base_uris` chain to find the matching
asset. For steps, only copies that match the active environment count, and the earliest base with
one wins: the space's own directory, then `base_uris` in the order listed, then the builtins. The full
algorithm, including how copies inside one base are ranked, is in
[step-resolution.md](../environments/step-resolution.md).

#### Worked example: how a space references environments and asset stores

A space doesn't inline environments or asset stores — it points `base_uris` at a directory that holds
them, and refers to them by `space://` URI. The in-repo `local` space
([`configurations/spaces/local/space.yaml`](../../configurations/spaces/local/space.yaml)) sets
`base_uris: [file://../../assets]`, which resolves to `configurations/assets/`:

```
configurations/spaces/local/space.yaml         # base_uris: [file://../../assets]
configurations/assets/                          # ← what base_uris points at
├── environments/   bash · docker · runpod · skypilot · skypilot-managed   → space://environments/<name>
├── assetstores/    env-local · hf · local · mem · s3                      → space://assetstores/<name>
└── steps/          per-environment step impls (e.g. k8s/)                 → space://steps/<name>
```

So a build running in this space resolves, for example:

- `environment_uri: space://environments/docker` → [`configurations/assets/environments/docker/environment.yaml`](../../configurations/assets/environments/) — a complete [environment](../environments/README.md) definition;
- `environment_uri: space://environments/skypilot/kubernetes` → `configurations/assets/environments/skypilot/kubernetes/` (the space's `DEFAULT_ENVIRONMENT` variable);
- an [asset store](../asset-stores/README.md) reference like `space://assetstores/hf` → [`configurations/assets/assetstores/hf/`](../../configurations/assets/assetstores/).

**The space's own directory wins.** The directory containing `space.yaml` is always searched first,
ahead of any `base_uris` — it is `base_uris[0]` in the resolver. This holds for **all three** asset
kinds: an `environments/<name>`, `assetstores/<name>`, or `steps/<name>` the space itself ships
**overrides** any copy inherited through the rest of the `base_uris` chain (e.g. a shared
`configurations/assets` tree). So a space can develop and test its own environment, asset store, or
step locally and have it take effect before — or instead of — the published/inherited version. The
same order runs through the rest of the chain: an earlier base's copy outranks a later one's (for
steps, among copies that match the environment; see
[step-resolution.md](../environments/step-resolution.md#resolution-order)).

**`base_uris` is optional.** Because the space's own directory is always searched, a **self-contained**
space can drop the `environments/`, `assetstores/`, and `steps/` directories right next to its
`space.yaml` and omit `base_uris` entirely:

```
my-space/
├── space.yaml            # no base_uris needed
├── environments/         → space://environments/<name>
├── assetstores/          → space://assetstores/<name>
└── steps/                → space://steps/<name>
```

Use `base_uris` when you'd rather pull assets in from elsewhere — a shared sibling directory (as the
`local` space does) or a `git://` URL — instead of, or in addition to, colocating them.

### `secret_manager`

The secret manager backend resolves every secret the build references (environment credentials, HF
tokens, SSH keys, …). `type` is one of `local`, `env`, `hybrid`, or `ibmcloud`, and `config` is
backend-specific. See **[Secrets](../secrets/README.md)** for what each backend does and how secrets are
consumed.

### `variables`

`variables` are space-scoped template values available to every build in the space. A `build.yaml`
references them as `${variables.<key>}` / `space.variables.<key>` (see the
[build.yaml reference](../builds/build-yaml-reference.md)). A common use is `DEFAULT_ENVIRONMENT` and
deployment-specific bucket URIs, so builds stay portable across spaces.

## Creating a space

A space is just a directory containing a `space.yaml`. Point the server or a build at it with a space
URI (a local path or a `git://` URI). Minimal `env`-secrets space:

```yaml
name: standalone
secret_manager:
  type: env
  config: {}
```

A local-development space that pulls its environments/steps/assetstores from the shared assets
directory and defines template variables:

```yaml
name: public
secret_manager:
  type: local
  config: {}
base_uris:
  - file://../../assets     # relative to this space.yaml's directory
variables:
  DEFAULT_ENVIRONMENT: skypilot/kubernetes
```

Working examples in the repo:

- [`configurations/spaces/local/space.yaml`](../../configurations/spaces/local/space.yaml) — local dev space with `base_uris` + `variables`.
- [`samples/spaces/env-secrets-example/space.yaml`](../../samples/spaces/env-secrets-example/space.yaml) — env-var secrets.
- [`samples/spaces/hybrid-secrets-example/space.yaml`](../../samples/spaces/hybrid-secrets-example/space.yaml) — chained/fallback secret managers.

## Registering and referencing a space by name

A space is **registered in gbserver under its `name`** (the `name` from `space.yaml`). The server keeps
a registry of spaces (the `gb_spaces` table) recording each space's name and the git repo URI its assets
live in. Register the predefined spaces with
[`gbserver create-spaces`](../cli/gbserver-cli-reference.md#administration); a standalone server registers
the space passed via [`gbserver standalone --space-dir`](../cli/gbserver-cli-reference.md#running-the-server)
(its name overridable with `--space-name`).

Once registered, builds and the CLI reference a space **by name** rather than by path:

- **CLI** — `gb space list` shows registered spaces; `gb space set <space-name>` sets the active space
  for subsequent commands; most commands also take a `--space <name>` filter (e.g.
  `gb build list --space public`). See the [CLI reference](../cli/gb-cli-reference.md#space--work-with-spaces).
- **Builds** — a submitted build runs under a named space; the BuildRunner resolves the build's
  `space://` URIs against that space's registered git repo. In [`gbtest`](../cli/gbtest-cli-reference.md), the
  space is named with `space_name:` plus a required `space_uri:` override (a git URI or a local
  directory) that replaces the registered repo.

The name is also what the IBM Cloud secret manager matches against and what appears in lineage
(`{space_name}/{build_name}`).

## Storing a space in the database (no git access)

A registered space can carry a snapshot of its space repo and of the assets repo its `base_uris`
names, so builds and the REST server never fetch either from git. Take or refresh the snapshot with
[`gbserver space-snapshot`](../cli/gbserver-cli-reference.md#administration):

```shell
# From the remote branch (any of https://, git+ssh://, ssh://, git@host:org/repo):
gbserver space-snapshot --space-name public --space-repo https://github.example.com/org/gbspace-public
# From local checkouts (e.g. while the remote is unreachable):
gbserver space-snapshot --space-name public --space-repo ../gbspace-public --assets-repo ../assets
```

The command stores only the files at the head of the space config branch (default: the
deployment's config branch, e.g. `gbspace-config`) and of the assets branch named in `space.yaml`
(its `@<ref>`). No git history, `.git` directory, or untracked files are kept. It then:

- sets the row's `git_repo_uri` to `db://spaces/<name>` (pass `--no-switch` to store the snapshot
  but keep using git);
- rewrites the stored `space.yaml` so that the `base_uris` entry for the assets repo becomes
  `db://spaces/<name>/assets` (a `#subdirectory=` suffix is kept);
- records each snapshot's source repo (normalized to `https://host/org/repo`), branch, and commit.

The snapshots are kept in the row's `json` column as `space` and `assets`, each holding `archive` (a
base64-encoded zip, like a build's `build_archive`), `source_uri`, `source_ref`, `source_commit`,
and `snapshot_time`. At use, gbserver unpacks each snapshot into a local cache directory named by
its content hash, and serves it from there.

The space is still identified by its source repo, not by the `db://` URI. Secret groups and
build/space consistency checks match on the snapshot's `source_uri`, so moving a space to the
database does not change which secrets it resolves. When the space uses the IBM Cloud secret
manager, `space-snapshot` compares the secret groups matched before and after the switch, and
refuses the switch if they differ (`--skip-secret-group-check` overrides this).

Re-running `gbserver create-spaces` (including `--replace`/`--clear`) for a db-backed space keeps
its snapshot when the spec names the same repo (e.g. the predefined public space), so the space is
not moved back to git. Likewise, the environment-files API reads the public space's environments
from its snapshot.

`db://spaces/<other>` makes a row an alias that is served from another row's snapshot. `GET
/api/v1/spaces/` omits the archives (but keeps the source metadata); `GET
/api/v1/spaces/<name>` returns them.

## See also

- [Secrets](../secrets/README.md) — the `secret_manager` backends and how secrets are resolved
- [Step resolution](../environments/step-resolution.md) — how `space://` URIs route via `base_uris`
- [Environments overview](../environments/README.md) — the environments a space provides
- [build.yaml reference](../builds/build-yaml-reference.md) — referencing a space and its variables
- [Glossary](../glossary.md)
