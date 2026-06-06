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
│       └── <step_type>/<name>/step.yaml # step_type-specific step (matches envs whose
│                                        # `step_type` chain contains this dir name)
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
  2. `assets/steps/<step_type>/<name>/` — for each `step_type` in the env's chain (cross-env-class pools).
  3. `assets/steps/<name>/` — env-agnostic fallback.

  See [Step type routing](../docs/operators/environment-yaml-config.md#step_type-routing-steps-to-environments) for the full reference.

## Consumers

- **`gbserver standalone`** — `--space-dir` defaults to `configurations/spaces/standalone/public` ([command_standalone.py:259-267](../src/gbserver/commands/command_standalone.py#L259-L267)).
- **Build tests** under `test/integration/standalone/buildrunner/` — each `buildtest.yaml` sets `space_uri` to a relative path resolving to `configurations/spaces/standalone/public`.
- **Templates** — `gbserver build run-and-monitor configurations/spaces/standalone/public/templates/<name>` runs the templates that ship with the public space.

## See also

- [docs/operators/environment-yaml-config.md](../docs/operators/environment-yaml-config.md) — environment.yaml and step.yaml reference, including the `step_type` matching rules.
- [src/gbcommon/uri/space.py](../src/gbcommon/uri/space.py) — `SpaceURI` resolver that walks the `step_type` chain.
- [src/gbserver/build/targetstep.py](../src/gbserver/build/targetstep.py) — sets the active env's `step_type_chain` on the resolver thread-local during step assimilation.
