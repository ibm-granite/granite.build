# Step resolution: routing steps to environments

When a target runs, every `space://steps/<name>` URI is resolved against the active environment.  The rule in one line: **only step copies that match the active environment are candidates, and `base_uris` order decides between them** — the space's own directory first, then the space's `base_uris` in the order listed, then the builtins as the fallback.  The sections below describe where a step can live and how copies inside one base are ranked; [Resolution order](#resolution-order) puts it together.

## Space-root steps (highest priority)

The **space's own directory** — the first entry in `base_uris`, i.e. the space's own `uristr` — is the most authoritative step source.  A step the space ships at `<space>/steps/<name>/step.yaml` **overrides** any copy in a later base, including one in the environment's own folder or inherited from a published `configurations/assets` tree.

This is what lets a step be **developed and tested inside its own space before it is published** into an inherited assets tree.  A step's `space/` dir is re-rendered on every `make test`, so the co-located test exercises that freshly-rendered local copy — not the already-published copy reachable via `base_uris` — and publishing (`make publish-step`) is only needed once the test passes.

The override honors the same environment match (class and `subtypes`, below) as every other copy: a space-root step that doesn't match the active env is skipped, and resolution moves on to the next base.

> Practically this only changes resolution when the space itself ships a `steps/<name>`.  A consuming space (e.g. `configurations/spaces/local`) that ships no steps is unaffected — its steps still resolve against the inherited assets tree.

> **Environments and asset stores follow the same order.**  `space://environments/<name>` and `space://assetstores/<name>` resolve directly against the `base_uris` chain in order, and the space's own directory is `base_uris[0]`, so a space's own copy overrides any inherited one.  See [Spaces and `space.yaml`](../spaces/README.md#base_uris-and-space-resolution).

## Where a step lives: environment folders and top-level `steps/`

Every `steps/` folder belongs to the folder that holds it, and that decides which environments it serves:

- A base's **top-level** `steps/` (e.g. `src/gbserver/builtins/steps/`, a generated step space, a shared step repo) serves **any** environment.
- An `environments/<…>/steps/` folder serves only the environments **at or below** its owner.  `environments/skypilot/aws/steps/` serves `skypilot/aws` and nothing else; `environments/skypilot/steps/` serves every skypilot environment.

Owners and the environment are compared from their last `environments` folder on, so repos with different prefixes still match: a family's shared `environments/skypilot/steps/` in one repo (or `configurations/assets/environments/skypilot/steps/` in this one) serves `environments/skypilot/slurm/bluevela` loaded from another.  A sibling environment's copy never serves the active one, whichever base it's in.

> **Only the path from `environments` on is compared, not where that folder sits in the base.**  Any `…/environments/skypilot/steps/` serves Skypilot environments, wherever it is: a leftover `test-data/old/environments/skypilot/steps/x` in a space repo serves them too, and since the space is searched first, it would win.  Keep `environments/` folders that hold steps only where they are meant to be used, and keep copies (fixtures, old trees) out of folders named `environments`.  Within one base, the **deepest** owner wins: the environment's own folder beats its family's, which beats the base's top-level `steps/`.

Every step needs `environment_configs` listing the active environment's class to be a candidate, wherever it lives.  A step with no `environment_configs` has no launcher for any environment: the build picks the launcher from the step's own `environment_configs`, so the basestep's `k8s` and `Lsf` entries don't supply one.  Such a step is never selected and the URI is reported as unresolvable.

This lets sibling environments under a common family directory share one step implementation without duplicating it into each env dir:

```
configurations/assets/environments/skypilot/
├── steps/
│   └── digit/step.yaml          # shared by ALL skypilot envs below
├── kubernetes/
│   ├── environment.yaml         # resolves ../steps/digit
│   └── steps/
│       └── digit/step.yaml      # OPTIONAL override — wins over ../steps/digit
└── slurm/
    └── environment.yaml         # also resolves ../steps/digit
```

Place a step at the level of the family whose members should share it (`skypilot/steps/` for all skypilot envs, `skypilot/lsf/steps/` for just the lsf endpoints).  Keep a per-endpoint override in the env's own `steps/` dir when one endpoint genuinely diverges.

## Restricting a shared step by sub-type (`subtype` / `subtypes`)

A family `steps/` folder shares a step with **every** environment under it, and env-class-match (below) matches **every** environment of the same class.  When environments differ only in compute endpoint they often share one class (all skypilot endpoints are class `Skypilot`), so neither mechanism can, on its own, restrict a shared step to a subset of them.  An optional **sub-type** provides that discriminator:

- An `environment.yaml` may declare an optional `subtype` — any free-form string (no predefined set; the skypilot endpoints use `kubernetes`/`slurm`/`aws`/`lsf` by convention, mirroring `config.default_cloud`).
- A step's `environment_configs.<Class>` may declare an optional `subtypes` list.  **Empty (the default) = universal**, matching any env of that class (so builtins and general shared steps keep resolving everywhere).  A **non-empty** list restricts the step to environments whose `subtype` is one of the listed values (exact string match); an env with no `subtype` does not match such a step.

```yaml
# environments/skypilot/steps/digit/step.yaml — shared by kubernetes + slurm only
environment_configs:
  Skypilot:
    subtypes: [kubernetes, slurm]   # aws / lsf endpoints are excluded
    ...
```

So a single `skypilot/steps/digit` is resolved by `skypilot/kubernetes` and `skypilot/slurm` but is unresolvable for `skypilot/aws` and `skypilot/lsf/...`.  The filter is applied wherever a copy is found (space root, env-class match, and the env-agnostic fallback), so a copy excluded by sub-type is never a candidate, whichever base it is in.

## Env-class matching against existing `environment_configs`

The resolver can also pick a step variant based on the env's class name (`Bash`, `Docker`, `K8s`, `Lsf`, `Runpod`, `Skypilot`, ...) by reading each candidate `step.yaml`'s existing `environment_configs` keys.  No new field on `step.yaml` is required.

The resolver scans recursively under each base for any file at `<base>/steps/**/<name>/step.yaml`, parses each one, and keeps the candidates whose `environment_configs` contain the active env's class name.  The class-name match is **case-insensitive** (env class `K8s` matches an `environment_configs` key `k8s`).  Subdirectory naming is **conventional only** — the match is by file content, so step variants can live anywhere (the builtins use `steps/<env-class-lowercase>/<name>/`):

```
src/gbserver/builtins/steps/
├── k8s/s3push/step.yaml      # only environment_configs.K8s
├── lsf/s3push/step.yaml      # only environment_configs.Lsf
├── skypilot/s3push/step.yaml # only environment_configs.Skypilot
└── ...
```

When the active env class is `K8s`, the resolver picks `steps/k8s/s3push/step.yaml`; when `Lsf`, `steps/lsf/s3push/step.yaml`.  Bases are scanned in order and the first base with a match wins.  Within that base, the deepest owning folder wins (above); then the candidate with FEWER `environment_configs` keys (the most env-specific file beats a multi-env file that also lists the env); then lexicographic path.

## Resolution order

For `space://steps/<name>` and an env of class `K8s` loaded from `<env-dir>`:

The bases are, in order: the space's own dir (`base_uris[0]`), the space's `base_uris` as listed, then the builtins (`src/gbserver/builtins`).  If `<env-dir>` is outside every base (an environment loaded from an absolute `file://` path, or from a git repo not listed in `base_uris`), it's searched as an extra base right after the space's own dir, ahead of the listed bases; only `<env-dir>` itself is added, not its parent folders.  A copy is a candidate only when it serves the active env: class `K8s` (case-insensitively), the `subtypes` restriction, and a `steps/` folder that is top-level or owned by `<env-dir>` or one of its parents.  The resolver tries each base in that order and returns the first candidate it finds there:

1. In the space's own dir: its root `steps/<name>/step.yaml` (when an env is active).
2. In any base: the recursive glob `<base>/steps/**/<name>/step.yaml` — deepest owning folder first, then most env-specific, then by path.

So a base listed earlier always outranks a later one, and a base listed in `space.yaml` can replace a builtin step for the environments it serves.  If no base has a candidate:

3. `<base>/steps/<name>/step.yaml` — env-agnostic fallback, bases in order.  Existence is checked via the resolved URI's own scheme-aware `exists()`, and the resolved (git or file) URI is returned as-is; the environment match and `<rest>` containment are still enforced against the base's local materialization.
4. unresolvable → `ValueError`.

**Sub-files come from the chosen copy.** `space://steps/<name>/<rest>` resolves `<rest>` inside the copy `space://steps/<name>` picks.  If that copy lacks it, the URI is unresolvable; it is never taken from a different copy.

**Testing a step against a space that has its own copy.** To make sure a locally rendered step wins over a published copy, list the space holding it ahead of the published tree in `base_uris`.

### Git-backed spaces

`base_uris` may be git URIs (`git+ssh://…`), not just local `file://` dirs.  All step lookups operate off a repo's **already-cloned** local copy (the thread-local git clone cache, reused — no re-clone), via a git-aware local-path resolver:

- The env-class-match glob and the env-agnostic fallback scan/inspect any git base's clone.
- A git-backed env dir is resolved to its clone too, so its path inside the repo decides which environment folders serve it.
- The space-root check resolves `base_uris[0]` to its clone as well, so a git-backed space's own `steps/<name>` still takes priority.

A base that can't be materialized locally (unsupported scheme, or a clone failure) is treated as "can't inspect": the ordered search (steps 1–2) skips it and the fallback (step 3) admits by path existence only, matching pre-`subtypes` behavior.

Use a family `steps/` folder for impls shared by environments under a common directory; add a `subtypes` list to restrict a shared step to specific endpoints of the same class; use env-class-match for splitting a multi-env step.yaml into per-env files (the builtins approach).

## Manual override via `base_uris`

A `space.yaml` can also explicitly `base_uri` into a specific env directory if you want its steps available regardless of which target runs:

```yaml
name: my-space
base_uris:
  - file://./environments/skypilot/kubernetes   # always check this env's steps
  - file://./../assets                          # plus the shared assets
```

Auto-discovery of the active env's dir still happens on top of this — listing it manually is rarely necessary.

## See also

- [Spaces and `space.yaml`](../spaces/README.md) — the `base_uris` chain these lookups search.
- [Environments overview](README.md) — `environment.yaml` / `step.yaml` reference and the per-type pages.
- [`src/gbcommon/uri/space.py`](../../src/gbcommon/uri/space.py) — the `SpaceURI` resolver (`_resolve_step_by_base_order` and the fallback).
- [`src/gbserver/build/targetstep.py`](../../src/gbserver/build/targetstep.py) — scopes the active env on the resolver thread-local during step assimilation.
