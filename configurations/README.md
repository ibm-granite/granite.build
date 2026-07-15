# `configurations/`

This directory holds the space, environment, step, and assetstore
configurations consumed by builds.
Primarily this is designed to support the local compute environment
in which: 

1. the gbserver components run locally in standalone mode
(independent of space definitions),
2. the compute environments also run on the local machine where the 
build is started.

## Layout

```
configurations/
├── README.md
├── assets/                              # reusable primitives — referenced via base_uris
│   ├── assetstores/
│   │   └── <name>/store.yaml
│   ├── environments/
│   │   ├── bash/environment.yaml
│   │   ├── docker/environment.yaml
│   │   ├── skypilot/
│   │   │   ├── steps/                   # shared by ALL skypilot envs below
│   │   │   │   └── <name>/step.yaml     # (via the ancestor-walk, nearest-wins)
│   │   │   ├── kubernetes/
│   │   │   │   ├── environment.yaml     # inherits ../steps; may add its own
│   │   │   │   └── steps/               # optional: env-co-located override
│   │   │   │       └── <name>/step.yaml # wins over the shared ../steps copy
│   │   │   └── slurm/environment.yaml   # inherits ../steps
│   │   └── skypilot-managed/
│   │       └── kubernetes/environment.yaml
│   ├── steps/
│   │   ├── <name>/step.yaml             # env-agnostic step (matches any env)
│   │   └── <env-class>/<name>/step.yaml # env-class-keyed split (matches envs whose
│   │                                    # class is `<env-class>`, e.g. `bash/`,
│   │                                    # `docker/`, `skypilot/`, `k8s/`)
│   └── templates/                       # reusable build.yaml templates
│       └── <name>/
│           ├── README.md
│           └── build.yaml
└── spaces/
    └── local/                       # user-facing local/public/standalone space
        └── space.yaml               # name: public, base_uris: [file://../../assets]
```

## How it composes

- **`assets/`** is the leaf primitives directory. It holds no `space.yaml`; it's a target for `base_uris`, not a space in its own right.
- **`configurations/spaces/local/space.yaml`** is the canonical user-facing space. Its `base_uris` chains into `configurations/assets/` so `space://environments/...`, `space://assetstores/...`, and `space://steps/...` resolve through to the assets tree.
- **Step resolution tiers**: when a target runs, `space://steps/<name>` is resolved in order:
  1. Walk up from `<env-dir>` to the enclosing base_uri, checking `<dir>/steps/<name>/` at each level — nearest-wins, so an env's own dir overrides a shared ancestor (e.g. `skypilot/steps/<name>` is shared by `skypilot/kubernetes` and `skypilot/slurm`).
  2. Recursive glob `<base>/**/<name>/step.yaml` — first candidate whose `environment_configs` keys contain the active env's class name (e.g. `Bash`, `Docker`, `K8s`, `Skypilot`).
  3. `<base>/steps/<name>/` — env-agnostic fallback.

  Both step-scoped tiers honour an optional per-step `environment_configs.<Class>.subtypes` list, which restricts a shared step to environments declaring one of those `subtype` values (e.g. `digit`/`sage` are shared by `skypilot/steps/` but limited to `subtypes: [kubernetes, slurm]`). See [docs/environments/step-resolution.md](../docs/environments/step-resolution.md) for the full reference.

## Consumers

- **`gbserver standalone`** — `--space-dir` defaults to `configurations/spaces/local` ([command_standalone.py:259-267](../src/gbserver/commands/command_standalone.py#L259-L267)).
- **Build tests** under `test/integration/standalone/buildrunner/` — each `buildtest.yaml` sets `space_uri` to a relative path resolving to `configurations/spaces/local`.
- **Templates** — `gbserver build run-and-monitor configurations/assets/templates/<name> --space-name public` runs the reusable build templates under `configurations/assets/templates/`.

## See also

- [docs/environments/README.md](../docs/environments/README.md) — environment.yaml and step.yaml reference.
- [docs/environments/step-resolution.md](../docs/environments/step-resolution.md) — full step resolution rules (ancestor-walk, shared-steps pool, env-class match, env-agnostic).
- [src/gbcommon/uri/space.py](../src/gbcommon/uri/space.py) — `SpaceURI` resolver implementing the tiered lookup.
- [src/gbserver/build/targetstep.py](../src/gbserver/build/targetstep.py) — scopes the active env on the resolver thread-local during step assimilation.
