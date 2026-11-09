# Step resolution: routing steps to environments

When a target runs, every `space://steps/<name>` URI is resolved against the active environment.  The rule in one line: **only step copies that can serve the active environment are candidates, and the first base in `base_uris` order that has one wins** — the space's own directory first, then the space's `base_uris` in the order listed, then the builtins.  The sections below say which copies are candidates and how one base's copies are ranked; [Resolution order](#resolution-order) puts it together.

## Space-root steps (highest priority)

The **space's own directory** — the first entry in `base_uris`, i.e. the space's own `uristr` — is the most authoritative step source.  A step the space ships at `<space>/steps/<name>/step.yaml` **overrides** any copy in a later base, including one in the environment's own folder or inherited from a published `configurations/assets` tree.

This is what lets a step be **developed and tested inside its own space before it is published** into an inherited assets tree.  A step's `space/` dir is re-rendered on every `make test`, so the co-located test exercises that freshly-rendered local copy — not the already-published copy reachable via `base_uris` — and publishing (`make publish-step`) is only needed once the test passes.

The override honors the same environment match (class and `subtypes`, below) as every other copy: a space-root step that doesn't match the active env is skipped, and resolution moves on to the next base.

> Practically this only changes resolution when the space itself ships a `steps/<name>`.  A consuming space (e.g. `configurations/spaces/local`) that ships no steps is unaffected — its steps still resolve against the inherited assets tree.

> **Environments and asset stores follow the same order.**  `space://environments/<name>` and `space://assetstores/<name>` resolve directly against the `base_uris` chain in order, and the space's own directory is `base_uris[0]`, so a space's own copy overrides any inherited one.  See [Spaces and `space.yaml`](../spaces/README.md#base_uris-and-space-resolution).

## Where a step lives: environment folders and shared `steps/`

What decides which environments a `steps/` folder serves is the folder that holds it — specifically, whether that folder holds an `environment.yaml`.  Folder *names* play no part.

- **An environment's own `steps/`.**  A `steps/` folder next to an `environment.yaml` holds that environment's own steps, and serves **only that environment**: the one whose directory it is.  `configurations/assets/environments/skypilot/aws/steps/` serves `skypilot/aws` and no other environment, wherever it is listed and even if another environment has the same class and sub-type.
- **A shared `steps/`.**  Any other `steps/` folder — a base's top-level `steps/` (the builtins, a generated step space, a shared step repo), or one in a folder that isn't an environment's, such as `configurations/assets/environments/skypilot/steps/` — is shared.  It serves every environment whose class (and sub-type, below) the step's own `environment_configs` admit.

Every step needs `environment_configs` listing the active environment's class to be a candidate, wherever it lives.  A step with no `environment_configs` has no launcher for any environment: the build picks the launcher from the step's own `environment_configs`, so the basestep's `k8s` and `Lsf` entries don't supply one.  Such a step is never selected and the URI is reported as unresolvable.

So to share one step among several environments, put it in a shared folder and let its `environment_configs` (class, and `subtypes` if needed) say which environments it serves; to give one environment its own variant, put it in that environment's `steps/`:

```
configurations/assets/environments/skypilot/      # no environment.yaml: its steps/ are shared
├── steps/
│   └── digit/step.yaml          # shared: serves every env its environment_configs admit
├── kubernetes/
│   ├── environment.yaml
│   └── steps/
│       └── digit/step.yaml      # kubernetes's own copy: wins over ../steps/digit for kubernetes
└── slurm/
    └── local/
        └── environment.yaml     # no steps/ of its own: gets the shared ../../steps/digit
```

> **A misconfigured own step is an error.**  A step in an environment's own `steps/` whose `environment_configs` don't admit that environment (wrong class, or `subtypes` that exclude it) raises `Invalid step for its environment` when it would be the copy selected.  It is not skipped in favour of a shared copy, since a step placed in an environment's folder is meant for that environment.  If an earlier base supplies the step anyway, the misconfigured copy is never reached and nothing is reported.

## Restricting a shared step by sub-type (`subtype` / `subtypes`)

A shared step serves **every** environment of a class its `environment_configs` list.  When environments differ only in compute endpoint they often share one class (all skypilot endpoints are class `Skypilot`), so the class alone can't restrict a shared step to some of them.  An optional **sub-type** provides that discriminator:

- An `environment.yaml` may declare an optional `subtype` — any free-form string (no predefined set; the skypilot endpoints use `kubernetes`/`slurm`/`aws`/`lsf` by convention, mirroring `config.default_cloud`).
- A step's `environment_configs.<Class>` may declare an optional `subtypes` list.  **Empty (the default) = universal**, matching any env of that class (so builtins and general shared steps keep resolving everywhere).  A **non-empty** list restricts the step to environments whose `subtype` is one of the listed values (exact string match); an env with no `subtype` does not match such a step.  This applies wherever the step lives, including an environment's own `steps/` (where a mismatch is an error, above).  A step in one environment's own `steps/` doesn't need `subtypes` at all: its folder already limits it to that environment, which is how the steps under `skypilot/aws/steps/` and `skypilot/lsf/ibm-bluevela/steps/` are scoped.

```yaml
# environments/skypilot/steps/digit/step.yaml — shared by kubernetes + slurm only
environment_configs:
  Skypilot:
    subtypes: [kubernetes, slurm]   # aws / lsf endpoints are excluded
    ...
```

So a single shared `skypilot/steps/digit` is resolved by `skypilot/kubernetes` and `skypilot/slurm/local` but is unresolvable for `skypilot/aws` and `skypilot/lsf/...`.  The filter is applied wherever a copy is found (space root, the ordered search, and the env-agnostic fallback), so a copy excluded by sub-type is never a candidate, whichever base it is in.

## Env-class matching against existing `environment_configs`

The resolver picks a step variant by the env's class name (`Bash`, `Docker`, `K8s`, `Lsf`, `Runpod`, `Skypilot`, ...), read from each candidate `step.yaml`'s existing `environment_configs` keys.  No new field on `step.yaml` is required.  The class-name match is **case-insensitive** (env class `K8s` matches an `environment_configs` key `k8s`).

The builtins keep one variant per env class, in an env-class folder `steps/<env-class-lowercase>/<name>/` ([Step names](#step-names-the-env-class-folder-may-be-left-out)):

```
src/gbserver/builtins/steps/
├── k8s/s3push/step.yaml      # only environment_configs.K8s
├── lsf/s3push/step.yaml      # only environment_configs.Lsf
├── skypilot/s3push/step.yaml # only environment_configs.Skypilot
└── ...
```

When the active env class is `K8s`, `space://steps/s3push` resolves to `steps/k8s/s3push/step.yaml`; when `Lsf`, to `steps/lsf/s3push/step.yaml`.

**Within one base, the most specific copy wins.**  When a base has more than one shared copy of a name that admits the active environment, the one whose `environment_configs` declare the **fewest** classes wins, so a per-class copy (`steps/k8s/s3push`, declaring only `K8s`) beats a catch-all beside it that also lists `K8s`.  Two copies in one base that are equally specific raise `Ambiguous space uri` — they are never picked between silently by path.  This applies to every base, not only the builtins.

## Step names: the env-class folder may be left out

A step's name is its folder path below `steps/`: `<base>/steps/train/openinstruct-sft/` is `space://steps/train/openinstruct-sft`.  The one folder a URI may leave out is an **env-class folder** directly under `steps/`, one of `bash`, `docker`, `k8s`, `lsf`, `runpod`, `skypilot` or `skypilot_managed`.  Any other folder, such as a category (`train/`, `data/`, `distill/`), is part of the name and must be in the URI.

| Step's folder | URIs that match it |
|---|---|
| `src/gbserver/builtins/steps/skypilot/command/` | `space://steps/skypilot/command`, `space://steps/command` |
| `<base>/steps/skypilot/dpk/` | `space://steps/skypilot/dpk`, `space://steps/dpk` |
| `…/environments/skypilot/steps/train/openinstruct-sft/` | `space://steps/train/openinstruct-sft` only |
| `…/environments/skypilot/steps/data/dpk/` | `space://steps/data/dpk` only |

The env-class folder can be left out because the builtins keep one copy of a step per env class (`steps/skypilot/command`, `steps/bash/command`, …): builds write `space://steps/command`, and the active env's class picks the copy (see [Env-class matching](#env-class-matching-against-existing-environment_configs)).  No other folder can be left out, so a URI always names the step it means: `space://steps/openinstruct-sft` never reaches the unrelated `train/openinstruct-sft`.  On an environment with no copy named exactly `openinstruct-sft` (e.g. `skypilot/slurm`), that URI is unresolvable.

The env-class folder names are fixed in the resolver (`_ENV_PARTITION_DIRS` in `src/gbcommon/uri/space.py`), so a new environment class that keeps per-class step copies needs adding there.

## Resolution order

For `space://steps/<name>` and an active environment loaded from `<env-dir>`:

1. **The space's own root `steps/<name>`**, if it admits the environment ([Space-root steps](#space-root-steps-highest-priority)).
2. **Otherwise, each base in order** — the space's own dir, then `base_uris` as listed, then the builtins (`src/gbserver/builtins`, always last).  The first base with a candidate wins, so a base listed earlier always outranks a later one, and a base listed in `space.yaml` can replace a builtin.  Within that base:
   1. the copy in `<env-dir>/steps/` (the active environment's own folder);
   2. otherwise a shared copy; among several, the most specific ([above](#env-class-matching-against-existing-environment_configs)).

   Copies in any other environment's own `steps/` are never candidates.  If `<env-dir>` is outside every base (an environment loaded from an absolute `file://` path, or from a git repo not listed in `base_uris`), it is searched as an extra base right after the space's own dir, so its own `steps/` are still found.  With no `<env-dir>` known (the validator couldn't load the environment, or it has no `environment_asset`), no folder is the active environment's, so only shared copies and builtins can match.
3. **If no base has a candidate,** an env-agnostic fallback tries `<base>/steps/<name>/step.yaml` in base order.  Existence is checked via the resolved URI's own scheme-aware `exists()`, and the resolved (git or file) URI is returned as-is; the environment match, `<rest>` containment, and the exclusion of other environments' folders are still enforced against the base's local materialization.
4. Otherwise the URI is unresolvable → `ValueError`.

**The longest name wins.**  For `space://steps/a/b/c`, the longest prefix that names a step (`a/b` if `<base>/steps/a/b/step.yaml` exists) is used, and the rest (`c`) is a sub-file inside it.  If that longest name exists but no copy of it serves the environment, the URI is unresolvable: it is never served as a shorter step plus a sub-file.

**Sub-files come from the chosen copy.** `space://steps/<name>/<rest>` resolves `<rest>` inside the copy `space://steps/<name>` picks.  If that copy lacks it, the URI is unresolvable; it is never taken from a different copy.

**Testing a step against a space that has its own copy.** To make sure a locally rendered step wins over a published copy, list the space holding it ahead of the published tree in `base_uris`.

### Git-backed spaces

`base_uris` may be git URIs (`git+ssh://…`), not just local `file://` dirs.  All step lookups operate off a repo's **already-cloned** local copy (the thread-local git clone cache, reused — no re-clone), via a git-aware local-path resolver:

- The ordered search and the env-agnostic fallback scan/inspect any git base's clone.
- A git-backed env dir is resolved to its clone too, so its own `steps/` is recognised beside its `environment.yaml`.
- The space-root check resolves `base_uris[0]` to its clone as well, so a git-backed space's own `steps/<name>` still takes priority.

A base that can't be materialized locally (unsupported scheme, or a clone failure) is treated as "can't inspect": the ordered search (step 2) skips it and the fallback (step 3) admits by path existence only, matching pre-`subtypes` behavior.

## Manual override via `base_uris`

A `space.yaml` can also list a folder of shared steps as a base, so its steps are available to every environment they admit:

```yaml
name: my-space
base_uris:
  - file://./shared-steps          # a folder of shared steps
  - file://./../assets             # plus the shared assets
```

Listing an environment's own directory as a base doesn't widen it: its `steps/` still serve only that environment.

## See also

- [Spaces and `space.yaml`](../spaces/README.md) — the `base_uris` chain these lookups search.
- [Environments overview](README.md) — `environment.yaml` / `step.yaml` reference and the per-type pages.
- [`src/gbcommon/uri/space.py`](../../src/gbcommon/uri/space.py) — the `SpaceURI` resolver (`_resolve_step_by_base_order`, `_choose`, and the fallback).
- [`src/gbserver/build/targetstep.py`](../../src/gbserver/build/targetstep.py) — scopes the active env on the resolver thread-local during step assimilation.
