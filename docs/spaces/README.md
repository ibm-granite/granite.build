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
resolver walks the `base_uris` chain (plus the active environment's own directory) to find the matching
asset. The full three-tier resolution algorithm — env-co-located lookup, env-class matching, and
env-agnostic fallback — is documented in
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

**`base_uris` is optional.** The space's own directory (the one containing `space.yaml`) is always
searched first, ahead of any `base_uris`. So a **self-contained** space can drop the `environments/`,
`assetstores/`, and `steps/` directories right next to its `space.yaml` and omit `base_uris` entirely:

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
  space is named with `space_name:` (with an optional `space_uri:` override that points at a local
  directory instead of the registered repo).

The name is also what the IBM Cloud secret manager matches against and what appears in lineage
(`{space_name}/{build_name}`).

## See also

- [Secrets](../secrets/README.md) — the `secret_manager` backends and how secrets are resolved
- [Step resolution](../environments/step-resolution.md) — how `space://` URIs route via `base_uris`
- [Environments overview](../environments/README.md) — the environments a space provides
- [build.yaml reference](../builds/build-yaml-reference.md) — referencing a space and its variables
- [Glossary](../glossary.md)
