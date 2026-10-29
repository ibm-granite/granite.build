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
│   │   │   │   └── <name>/step.yaml     # (deepest owning folder wins)
│   │   │   ├── kubernetes/
│   │   │   │   ├── environment.yaml     # inherits ../steps; may add its own
│   │   │   │   └── steps/               # optional: env's own override
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
- **Step resolution**: when a target runs, `space://steps/<name>` resolves to one copy of the step:
  1. **Candidates.** A copy counts only if its `environment_configs` list the active environment's class (e.g. `Bash`, `Docker`, `K8s`, `Skypilot`), its optional `subtypes` admit the environment, and its `steps/` folder serves the environment. A base's top-level `steps/` serves every environment. An `environments/<…>/steps/` folder serves only environments at or below it, so `skypilot/steps/<name>` is shared by `skypilot/kubernetes` and `skypilot/slurm`, while `skypilot/aws/steps/<name>` serves only `skypilot/aws`. A step with no `environment_configs` is never a candidate.
  2. **Base order.** Bases are searched in order: the space's own directory, the environment's own folder if no base contains it, the space's `base_uris` as listed, then the builtins. The first base holding a candidate wins.
  3. **Within one base.** The space's root `steps/<name>` first, then the deepest owning folder (an environment's own `steps/` overrides its family's), then fewest `environment_configs` keys, then path.
  4. **Fallback.** If no base has a candidate, `<base>/steps/<name>/` is tried in base order, still subject to the environment check.

  `subtypes` restricts a shared step to environments declaring one of those `subtype` values; for example, `digit`/`sage` are shared by `skypilot/steps/` but limited to `subtypes: [kubernetes, slurm]`. See [docs/environments/step-resolution.md](../docs/environments/step-resolution.md) for the full reference.

## Consumers

- **`gbserver standalone`** — `--space-dir` defaults to `configurations/spaces/local` ([command_standalone.py:259-267](../src/gbserver/commands/command_standalone.py#L259-L267)).
- **Build tests** under `test/integration/standalone/buildrunner/` — each `buildtest.yaml` sets `space_uri` to a relative path resolving to `configurations/spaces/local`.
- **Templates** — `gbserver build run-and-monitor configurations/assets/templates/<name> --space-name public` runs the reusable build templates under `configurations/assets/templates/`.

## See also

- [docs/environments/README.md](../docs/environments/README.md) — environment.yaml and step.yaml reference.
- [docs/environments/step-resolution.md](../docs/environments/step-resolution.md) — full step resolution rules (base order, env-folder ownership, env-class match, env-agnostic fallback).
- [src/gbcommon/uri/space.py](../src/gbcommon/uri/space.py) — `SpaceURI` resolver implementing the lookup.
- [src/gbserver/build/targetstep.py](../src/gbserver/build/targetstep.py) — scopes the active env on the resolver thread-local during step assimilation.
