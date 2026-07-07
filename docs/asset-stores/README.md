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

- **Env-local (`env://`)** is used by bare-metal HPC backends (e.g. LSF/SLURM with shared GPFS): the
  artifact is already on a filesystem the worker can see, so the store resolves the path directly and
  transfers nothing.
- **In-memory (`mem://`)** passes a producer's binding value (e.g. a service URL) verbatim to downstream
  consumers without touching a filesystem.

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
behaviour via a `mode`:

| `mode` | Direction | Effect |
|--------|-----------|--------|
| `hf_pull` / `hf_push` | load / push | Download from / upload to a HuggingFace repo. |
| `cos_rclone` / `cos_pull` / `cos_push` | load / push | COS / S3 transfer (rclone). |
| `env_local` | load / push | No-op: the artifact already lives on a shared filesystem reachable by the worker. |
| `default` | load / push | The environment's built-in handling for that store. |

Each mode is implemented by a `pullasset_*` / `pushasset_*` method on the environment class — some pull/push
inline, others inject a built-in step (e.g. `hfpull`, `cosrclone`, `lhpull`). Which methods an environment
provides, and whether they mount volumes or queue steps, is environment-specific: see
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
