# `configurations/`

This directory holds the space, environment, step, and assetstore configurations consumed by builds. It is the canonical location after the `spaces/` reorganization.

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
│   │   │   ├── kubernetes/
│   │   │   │   ├── environment.yaml
│   │   │   │   └── steps/               # optional: env-co-located steps
│   │   │   │       └── <name>/step.yaml # picked when this env runs the target
│   │   │   └── slurm/environment.yaml
│   │   └── skypilot-managed/
│   │       └── kubernetes/environment.yaml
│   └── steps/
│       ├── <name>/step.yaml             # env-agnostic step (matches any env)
│       └── <env-class>/<name>/step.yaml # env-class-keyed split (matches envs whose
│                                        # class is `<env-class>`, e.g. `bash/`,
│                                        # `docker/`, `skypilot/`, `k8s/`)
└── spaces/
    └── standalone/
        └── public/                      # user-facing STANDALONE space
            ├── space.yaml               # name: public, base_uris: [file://../../../assets]
            └── templates/               # build templates shipped with this space
                └── <name>/
                    ├── README.md
                    └── build.yaml
```

## How it composes

- **`assets/`** is the leaf primitives directory. It holds no `space.yaml`; it's a target for `base_uris`, not a space in its own right.
- **`configurations/spaces/standalone/public/space.yaml`** is the canonical user-facing space. Its `base_uris` chains into `configurations/assets/` so `space://environments/...`, `space://assetstores/...`, and `space://steps/...` resolve through to the assets tree.
- **Step resolution tiers**: when a target runs, `space://steps/<name>` is resolved in order:
  1. `<env-dir>/steps/<name>/` — steps co-located in the active env's own directory (auto-discovered).
  2. Recursive glob `<base>/**/<name>/step.yaml` — first candidate whose `environment_configs` keys contain the active env's class name (e.g. `Bash`, `Docker`, `K8s`, `Skypilot`).
  3. `<base>/steps/<name>/` — env-agnostic fallback.

  See [docs/operators/step-resolution.md](../docs/operators/step-resolution.md) for the full reference.

## Consumers

- **`gbserver standalone`** — `--space-dir` defaults to `configurations/spaces/standalone/public` ([command_standalone.py:259-267](../src/gbserver/commands/command_standalone.py#L259-L267)).
- **Build tests** under `test/integration/standalone/buildrunner/` — each `buildtest.yaml` sets `space_uri` to a relative path resolving to `configurations/spaces/standalone/public`.
- **Templates** — `gbserver build run-and-monitor configurations/spaces/standalone/public/templates/<name>` runs the templates that ship with the public space.

## See also

- [docs/operators/environment-yaml-config.md](../docs/operators/environment-yaml-config.md) — environment.yaml and step.yaml reference.
- [docs/operators/step-resolution.md](../docs/operators/step-resolution.md) — full step resolution rules (env-co-located, env-class match, env-agnostic).
- [src/gbcommon/uri/space.py](../src/gbcommon/uri/space.py) — `SpaceURI` resolver implementing the three-tier lookup.
- [src/gbserver/build/targetstep.py](../src/gbserver/build/targetstep.py) — scopes the active env on the resolver thread-local during step assimilation.
