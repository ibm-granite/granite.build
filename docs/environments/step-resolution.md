# Step resolution: routing steps to environments

When a target runs, every `space://steps/<name>` URI is resolved against the active environment.  Two mechanisms are available — pick whichever fits the step:

## Co-located steps and the ancestor-walk

A step impl placed inside the env's own directory is auto-discovered whenever that env is the active target's env.  No `base_uris` change is needed.  The resolver then **walks up** parent directories, checking `<dir>/steps/<name>/step.yaml` at each level, and stops at (and including) the deepest `base_uri` that encloses the env dir — it never escapes above the base_uri subtree.  The match is **nearest-wins**: the env's own dir beats a parent, which beats a grandparent.

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

The ancestor-walk shares a step with **every** environment under its directory, and env-class-match (below) matches **every** environment of the same class.  When environments differ only in compute endpoint they often share one class (all skypilot endpoints are class `Skypilot`), so neither mechanism can, on its own, restrict a shared step to a subset of them.  An optional **sub-type** provides that discriminator:

- An `environment.yaml` may declare an optional `subtype` — any free-form string (no predefined set; the skypilot endpoints use `kubernetes`/`slurm`/`aws`/`lsf` by convention, mirroring `config.default_cloud`).
- A step's `environment_configs.<Class>` may declare an optional `subtypes` list.  **Empty (the default) = universal**, matching any env of that class (so builtins and general shared steps keep resolving everywhere).  A **non-empty** list restricts the step to environments whose `subtype` is one of the listed values (exact string match); an env with no `subtype` does not match such a step.

```yaml
# environments/skypilot/steps/digit/step.yaml — shared by kubernetes + slurm only
environment_configs:
  Skypilot:
    subtypes: [kubernetes, slurm]   # aws / lsf endpoints are excluded
    ...
```

So a single `skypilot/steps/digit` is resolved by `skypilot/kubernetes` and `skypilot/slurm` but is unresolvable for `skypilot/aws` and `skypilot/lsf/...`.  The filter is applied by **all three** tiers (ancestor-walk, env-class-match, and the env-agnostic fallback), so the restriction holds regardless of which tier finds the file: a candidate excluded by sub-type is skipped (the walk keeps ascending; class-match ignores it; the fallback rejects it).

## Env-class matching against existing `environment_configs`

The resolver can also pick a step variant based on the env's class name (`Bash`, `Docker`, `K8s`, `Lsf`, `Runpod`, `Skypilot`, ...) by reading each candidate `step.yaml`'s existing `environment_configs` keys.  No new field on `step.yaml` is required.

The resolver scans recursively under each base_uri for any file at `<...>/<name>/step.yaml`, parses each one, and selects the candidate whose `environment_configs` contains the active env's class name.  The class-name match is **case-insensitive** (env class `K8s` matches an `environment_configs` key `k8s`).  Subdirectory naming is **conventional only** — the match is by file content, so step variants can live anywhere (the convention is `<base>/steps/<env-class-lowercase>/<name>/`):

```
src/gbserver/builtins/steps/
├── s3push/step.yaml          # multi-env catch-all (environment_configs: K8s, Lsf, Skypilot, ...)
├── k8s/s3push/step.yaml      # only environment_configs.K8s
├── lsf/s3push/step.yaml      # only environment_configs.Lsf
├── skypilot/s3push/step.yaml # only environment_configs.Skypilot
└── ...
```

When the active env class is `K8s`, the resolver picks `steps/k8s/s3push/step.yaml`.  When `Lsf`, it picks `steps/lsf/s3push/step.yaml`.  When the env class is one not represented by a single-env split file, it falls back to the multi-env catch-all.  Among multiple matches, the candidate with FEWER `environment_configs` keys wins — i.e. the most env-specific file beats a multi-env file that happens to list the same env.  Lexicographic path is the secondary tie-break.

## Resolution order

For `space://steps/<name>` and an env of class `K8s` loaded from `<env-dir>`:

1. Walk `<env-dir>` → parents up to the enclosing `base_uri`, first `steps/<name>/step.yaml` hit wins (nearest overrides); a candidate whose `subtypes` restriction excludes the active env is skipped and the walk continues.
2. Recursive glob `<base>/**/<name>/step.yaml` across `base_uris` — first candidate (by specificity, then lex) whose `environment_configs` contains `K8s` (case-insensitively) **and** whose `subtypes` restriction admits the active env's sub-type.
3. `<base>/steps/<name>/step.yaml` — env-agnostic fallback.  Existence is checked via the resolved URI's own scheme-aware `exists()`, and the resolved (git or file) URI is returned as-is; the `subtypes` restriction and `<rest>` containment are still enforced against the base's local materialization, so a candidate the active env's sub-type excludes is skipped.
4. unresolvable → `ValueError`.

The `subtypes` restriction is honored by **all** step tiers, so it is never bypassed by falling through from one tier to the next.

### Git-backed spaces

`base_uris` may be git URIs (`git+ssh://…`), not just local `file://` dirs.  All three tiers operate off a repo's **already-cloned** local copy (the thread-local git clone cache, reused — no re-clone), via a git-aware local-path resolver:

- Tier 2 (glob) and Tier 3 (fallback) scan/inspect any git base's clone.
- Tier 1 (ancestor-walk) resolves the env dir to its clone and walks within it, so co-located steps in the **same repo** (base_uri) as the environment resolve.  Steps in a *different* repo from the env live in a separate clone and are not reachable by the walk — use env-class-match or the fallback for those.

A base that can't be materialized locally (unsupported scheme, or a clone failure) is treated as "can't inspect": Tier 1/2 skip it and Tier 3 admits by path existence only, matching pre-`subtypes` behavior.

Use the ancestor-walk (a shared `steps/` dir at the family level) for impls shared by environments under a common directory; add a `subtypes` list to restrict a shared step to specific endpoints of the same class; use env-class-match for splitting a multi-env step.yaml into per-env files (the builtins approach).

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

- [Spaces and `space.yaml`](../spaces/README.md) — the `base_uris` chain these lookups walk.
- [Environments overview](README.md) — `environment.yaml` / `step.yaml` reference and the per-type pages.
- [`src/gbcommon/uri/space.py`](../../src/gbcommon/uri/space.py) — the `SpaceURI` resolver implementing the three-tier lookup.
- [`src/gbserver/build/targetstep.py`](../../src/gbserver/build/targetstep.py) — scopes the active env on the resolver thread-local during step assimilation.
