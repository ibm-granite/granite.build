# Asset stores

> **Audience:** operators configuring where a build's inputs and outputs live, and anyone tracing how an
> artifact URI resolves. Asset stores are declared per environment — see the
> [environments overview](../environments/README.md#asset-stores) for the `environment.yaml` config.

An **asset store** describes *where* a build's artifacts live and *how to reach them* — it does not move
the data itself. Given an artifact **URI** (chosen by scheme — `hf://`, `cos://`, `file://`, …) a store
maps it to a concrete location (`get_relpath`), classifies the artifact (model, dataset, …), and supplies
the credentials to access it. The actual transfer — pulling inputs before a step runs and pushing outputs
after — is performed by the **environment**'s `pullasset_*` / `pushasset_*` methods, which typically
inject a built-in transfer step (`hfpull`, `cosrclone`, `lhpull` / `lhpush`) or mount a volume.

Each environment declares the stores it can reach in its `environment.yaml` `assetstores` block, and
builds refer to them via `space://assetstores/<name>` URIs resolved through the space's `base_uris`.

> **`env://` and `mem://` are always available.** The env-local (`env://`) and in-memory (`mem://`)
> stores are registered implicitly for **every** environment class — neither needs an `assetstores`
> entry in `environment.yaml`. Their push/pull transfer nothing (env:// is a shared-filesystem no-op;
> mem:// passes a value through the build's shared memory), so any backend can consume/produce
> `env://` / `mem://` artifacts. Each store is resolved in the same order: an explicit
> `environment.yaml` entry for that scheme wins; otherwise a space-provided
> `space://assetstores/env-local` / `space://assetstores/mem-local` store *if one exists*; otherwise a
> bundled
> [`builtins/assetstores/env-local`](../../src/gbserver/builtins/assetstores/env-local/store.yaml) /
> [`builtins/assetstores/mem-local`](../../src/gbserver/builtins/assetstores/mem-local/store.yaml)
> default. **No shipped space defines an `env-local`/`mem-local` store**, so out of the box the bundled
> default is used; adding one to a space is an optional customization. See
> [Env-local / In-memory](#store-types-and-uri-schemes) below and
> [`Environment._register_default_envstore`](../../src/gbserver/environment/environment.py) /
> [`_register_default_memstore`](../../src/gbserver/environment/environment.py).

> **The File store (`file:`) is a declared store, not auto-registered.** Unlike `env://`/`mem://`, an
> environment that supports `file:` declares it in its `environment.yaml` as
> `space://assetstores/file/` — the bundled
> [`builtins/assetstores/file`](../../src/gbserver/builtins/assetstores/file/store.yaml) store, resolved
> via the builtins base_uri. Declare only the modes the backend actually implements (its
> `pullasset_filestore` / `pushasset_filestore` methods): `bash` implements both `load` and `push`;
> `docker` implements `push` only (no `pullasset_filestore`), so it declares `push` and not `load`.

## Store types and URI schemes

| Store | URI scheme(s) | Maps a URI to… | Credentials (default secret name) |
|-------|---------------|----------------|-----------------------------------|
| File | `file://` | a local filesystem path | none |
| Git | `git+https://`, `git+ssh://`, `git+git://` | a cloned repo (optional `#subdirectory=`, `@ref`) | `GITHUB_PAT_TUNING` (https) or `GIT_SSH_KEY` (ssh) |
| COS / S3 | `cos://`, `s3://` | a bucket + object path | `COS_ACCESS_KEY_ID`, `COS_SECRET_ACCESS_KEY` |
| HuggingFace | `hf://` | `owner/repo[/revision][/path]` | `HF_TOKEN` |
| Lakehouse | `lh://` | a Lakehouse asset → its backing COS path | `LAKEHOUSE_TOKEN` |
| Env-local | `env://` | a path on the environment's own filesystem | none |
| In-memory | `mem://` | an opaque key in the build's shared memory | none |

Notes:

- **Env-local (`env://`)** models an artifact that already lives on a filesystem the worker can see, so
  the store resolves the path directly and transfers nothing. Common on bare-metal HPC backends (e.g.
  LSF/SLURM with shared GPFS), but supported by **all** environment classes and registered
  automatically — no `environment.yaml` `assetstores` entry is required (see the note above).
  Because it transfers nothing, an `env://` path must be **absolute**: relative `env:` URIs (e.g.
  `env:outputs/foo`) have no resolution root and are **rejected at build-config load**. Use an absolute
  `env:///…` (a templated `env://{{ binding.path }}` is fine — it resolves to an absolute path).
- **In-memory (`mem://`)** passes a producer's binding value (e.g. a service URL) verbatim to downstream
  consumers without touching a filesystem. Like `env://`, it is supported by **all** environment classes
  and registered automatically — no `environment.yaml` `assetstores` entry is required (see the note
  above). Because a `mem://` URI is an **opaque key** rather than a path, the value is passed through
  unchanged — unlike `env://`, it applies no path normalisation, so a value such as `http://host:8000`
  survives intact instead of being mangled into `/http:/host:8000`.
  - **Consuming** a `mem://` input works on every environment (the transport is host-side and
    environment-agnostic). **Producing** a `mem://` output additionally requires the step's **monitor**
    to recognize the `LLMB_ARTIFACT_STATE` marker the workload prints — the shipped `bash`, `skypilot`,
    and `docker` library monitors carry that rule; other environments' monitors need it added (see
    [Value outputs (`mem://`)](../steps/monitoring-and-artifact-events.md#value-outputs-mem)).

The store implementations live in [`src/gbserver/asset/`](../../src/gbserver/asset/); the matching URI
parsers in [`src/gbcommon/uri/`](../../src/gbcommon/uri/).

## Secrets

A store reads the credentials it needs **by name** from the space's secret manager — nothing sensitive is
inlined in the store config. Defaults are shown above (`HF_TOKEN`, `COS_ACCESS_KEY_ID` /
`COS_SECRET_ACCESS_KEY`, `LAKEHOUSE_TOKEN`, `GITHUB_PAT_TUNING` / `GIT_SSH_KEY`); each name is
**configurable** in the store's `store.yaml` (e.g. `token_secretname`, `cos_access_key_id_secret_name`).
If a secret isn't found in the space, stores fall back to the same-named environment variable. See
[Secrets](../secrets/README.md) for the backends that resolve these.

## Store configuration (`store.yaml`)

A store is defined by a `store.yaml` ([`AssetStoreConfig`](../../src/gbserver/types/assetstoreconfig.py))
that declares which URIs it handles and any store-specific settings:

- `base_uri` or `uri_regex` — the URIs this store handles (used to route a URI to the right store).
- `config` — store-specific settings: the secret names above, plus e.g. COS `cos_endpoint` / `cos_region`
  and Lakehouse `env`.

Stores are referenced from a build/environment as `space://assetstores/<name>`, resolved against the
space's `base_uris` (the same mechanism as steps and environments — see
[Spaces](../spaces/README.md) and [step resolution](../environments/step-resolution.md)).

## Load and push modes

An `environment.yaml` `assetstores` entry maps a store URI to **load** (input) and **push** (output)
behaviour. Dispatch to a `pullasset_*` / `pushasset_*` handler is by store **type** (derived from the
URI scheme), not by `mode`. Outside k8s the `mode` field is therefore not meaningful, and the enforced
invariant is that it must be **unset or `default`**.

Only **k8s** genuinely branches on `mode` — it selects mounting vs. copying vs. queuing a step:

| `mode` (k8s only) | Direction | Effect |
|-------------------|-----------|--------|
| `afm_mount` / `cos_mount` | load | Mount the asset into the pod instead of copying it. |
| `hf_pull` | load | Download from a HuggingFace repo. |
| `cos_pull` / `dmf_pull` | load | Queue a COS/DMF transfer step. |

For every **other** environment (skypilot, lsf, bash, docker, runpod, and the base mem/env handlers),
`mode` is ignored (dispatch is by store *type*). Declare `mode: default` (or omit it); a non-`default`
value is still accepted for backwards compatibility but logs a deprecation warning at pull/push time
(via `Environment._warn_non_default_mode`).

Each store type is implemented by a `pullasset_*` / `pushasset_*` method on the environment class — some
pull/push inline, others inject a built-in step (e.g. `hfpull`, `cosrclone`, `lhpull`). Which methods an
environment provides, and whether they mount volumes or queue steps, is environment-specific: see
[environments](../environments/README.md#asset-stores) for the `environment.yaml` config and
[environment classes](../architecture/environment-classes.md) for the per-environment implementations.

## How a store is selected

Stores are auto-discovered and registered by the **URI class** they handle
([`src/gbserver/asset/assetstore.py`](../../src/gbserver/asset/assetstore.py)): a filename like
`hfstore.py` → `Hfstore`, which declares the URI classes it supports. At runtime the artifact's URI scheme
picks the URI class, which selects the store; when several stores could match, the longest `base_uri` /
`uri_regex` match wins.

## See also

- [Environments](../environments/README.md#asset-stores) — declaring `assetstores` in `environment.yaml`
- [Secrets](../secrets/README.md) — how store credentials are resolved
- [Builds](../builds/README.md#artifacts-inputs-and-outputs) — artifacts as target inputs/outputs
- [Environment classes](../architecture/environment-classes.md) — per-environment `pullasset_*`/`pushasset_*`
