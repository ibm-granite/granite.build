# Step resolution: routing steps to environments

When a target runs, every `space://steps/<name>` URI is resolved against the active environment.  Two mechanisms are available — pick whichever fits the step:

## Co-located steps in the env's own directory

A step impl placed inside the env's own directory is auto-discovered whenever that env is the active target's env.  No `base_uris` change is needed.  Example layout:

```
configurations/assets/environments/skypilot/kubernetes/
├── environment.yaml
└── steps/
    └── digit/step.yaml          # used when this env runs the target
```

Co-located steps are ideal for impls that are tightly coupled to a specific environment instance and don't need to be shared with other envs.

## Env-class matching against existing `environment_configs`

The resolver can also pick a step variant based on the env's class name (`Bash`, `Docker`, `K8s`, `Lsf`, `Runpod`, `Skypilot`, ...) by reading each candidate `step.yaml`'s existing `environment_configs` keys.  No new field on `step.yaml` is required.

The resolver scans recursively under each base_uri for any file at `<...>/<name>/step.yaml`, parses each one, and selects the candidate whose `environment_configs` contains the active env's class name.  Subdirectory naming is **conventional only** — the match is by file content, so step variants can live anywhere (the convention is `<base>/steps/<env-class-lowercase>/<name>/`):

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

1. `<env-dir>/steps/<name>/step.yaml` — env-co-located impl (highest priority).
2. Recursive glob `<base>/**/<name>/step.yaml` across `base_uris` — first candidate (by specificity, then lex) whose `environment_configs` contains `K8s`.
3. `<base>/steps/<name>/step.yaml` — env-agnostic fallback.
4. unresolvable → `ValueError`.

Use co-located steps for impls tightly coupled to a specific environment instance; use env-class-match for splitting a multi-env step.yaml into per-env files (the builtins approach).

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

- [`environment-yaml-config.md`](environment-yaml-config.md) — full `environment.yaml` and `step.yaml` reference.
- [`src/gbcommon/uri/space.py`](../../src/gbcommon/uri/space.py) — the `SpaceURI` resolver implementing the three-tier lookup.
- [`src/gbserver/build/targetstep.py`](../../src/gbserver/build/targetstep.py) — scopes the active env on the resolver thread-local during step assimilation.
