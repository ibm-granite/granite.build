# Step-Authoring Framework (Makefile → image → quay.io → step.yaml → configurations/)

## Context

Today, container-backed SkyPilot steps (e.g. `configurations/assets/environments/skypilot/steps/sage/`)
carry a `Dockerfile` next to their `step.yaml`, but **nothing builds or publishes those images** — the
image tag inside `step.yaml` (`image_id: docker:us.icr.io/.../sage:cpu-0.0.4`) is hand-bumped. There is no
scaffolding, no publish path, and no convention for wiring a freshly-built image into a step. This is the gap
this framework fills.

Goal: a repeatable, per-step convention that (1) builds a Docker image, (2) publishes it to **quay.io**,
(3) renders a `step.yaml` from a template with the published image reference inserted, and (4) installs the
rendered `step.yaml` into the correct environment location under `configurations/assets/environments/`.
Steps launch via `launch_skypilot` initially, whose launcher `config` takes `run`, `image_id`, and `setup`
(see `src/gbserver/environment/skypilot.py:699` image_id, `:762` setup, `:793` run).

## Decisions (confirmed with user)

- **Each step lives in its own directory with its own Makefile.** Source is separate from the installed
  config tree; `make install` copies the rendered `step.yaml` into `configurations/`.
- **Image ref inserted via a plain-text sed token** (`@IMAGE_REF@`), not Jinja. sed replaces only the literal
  token, leaving the step's runtime `{{ config.* }}` Jinja (rendered later by gbserver) untouched.
- **Publish auth is robot-token only** (CI-oriented): require `QUAY_ROBOT_USER` / `QUAY_ROBOT_TOKEN`.
- First cut targets **SkyPilot** (`type: skypilot` launcher); layout is env-extensible.

## Layout (new top-level `steps/`)

```
steps/
├── README.md                 # source-vs-installed distinction + usage
├── Makefile                  # scaffolding: `make new-step NAME=<name>`
├── common.mk                 # shared targets/vars/guards, included by each step Makefile
├── _template/                # skeleton copied by new-step
│   ├── Makefile
│   ├── Dockerfile
│   ├── step.yaml.tmpl
│   └── requirements.txt
└── hello-skypilot/           # worked example that proves the framework end-to-end
    ├── Makefile
    ├── Dockerfile
    ├── step.yaml.tmpl
    └── requirements.txt
```

Distinct from the **installed** tree `configurations/assets/environments/<env-path>/steps/<name>/step.yaml`,
which gbserver resolves for `space://steps/<name>` (ancestor-walk, see `configurations/README.md`).

## Per-step `Makefile` (thin; delegates to `common.mk`)

Each step's Makefile sets a few variables then `include ../common.mk`:

```make
STEP_NAME  := hello-skypilot          # MUST equal dir name AND step.yaml `name`/launcher key
IMAGE_NAME := $(STEP_NAME)
ENV_PATH   := skypilot                # install dest under configurations/assets/environments/
include ../common.mk                  # e.g. skypilot, skypilot/aws, docker
```

## `steps/common.mk` (shared logic — avoids duplication per CLAUDE.md)

Reuses the root `Makefile`'s proven conventions (`Makefile:5,16-19,133-135,601-613,625-641`):
`DOCKER ?= podman`, `IMAGE_TAG ?= $(GIT_DIRTY)commit-$(GIT_COMMIT)`, `buildx --platform linux/x86_64 --load`,
tag→push, git-clean guard.

Key variables:
- `QUAY_ORG ?= granite-build`; `QUAY_REGISTRY := quay.io/$(QUAY_ORG)`
- `IMAGE_REF := $(QUAY_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)`
- `CONFIG_DEST := <repo-root>/configurations/assets/environments/$(ENV_PATH)/steps/$(STEP_NAME)`

Targets (each independently runnable; every recipe kept short):
- `image` / `imagex` — build `$(IMAGE_NAME):$(IMAGE_TAG)` (imagex = cross-platform for Mac→cluster).
- `check-quay-env` — fail clearly unless `QUAY_ROBOT_USER`+`QUAY_ROBOT_TOKEN` set.
- `quay-login` (dep: check-quay-env) — `echo "$QUAY_ROBOT_TOKEN" | $(DOCKER) login quay.io -u "$QUAY_ROBOT_USER" --password-stdin`.
- `push` (dep: imagex quay-login) — tag to `$(IMAGE_REF)`, push.
- `stepyaml` — `sed 's|@IMAGE_REF@|$(IMAGE_REF)|g' step.yaml.tmpl > .gen/step.yaml` (standalone-runnable for inspection).
- `install` (dep: stepyaml) — `mkdir -p $(CONFIG_DEST)`; copy `.gen/step.yaml` → `$(CONFIG_DEST)/step.yaml`; copy optional `bash_scripts/` if present.
- `all` — orders `imagex push install` (build → publish → render → copy).
- `check-git-clean` — refuse dirty tree unless `ALLOW_DIRTY=1`.

## `step.yaml.tmpl` (skypilot skeleton)

Models `configurations/assets/environments/skypilot/steps/sage/step.yaml` and the builtin
`src/gbserver/builtins/steps/skypilot/command/step.yaml`. Only `@IMAGE_REF@` is sed-substituted; the
`run:`/`setup:` bodies keep live `{{ config.* }}` Jinja and `ref: space://monitors/skypilot`:

```yaml
name: hello-skypilot            # == dir name / launcher key
version: 1.0.0
type: exec
config:
  hello_config: { message: "hello from a quay-published step" }
environment_configs:
  Skypilot:
    default_launcher: hello-skypilot
    launchers:
      hello-skypilot:
        type: skypilot                       # -> launch_skypilot (run/image_id/setup)
        monitors: [skypilot_monitor]
        config:
          image_id: docker:@IMAGE_REF@       # sed target
          resources: { cpus: 2+, memory: 8+ }
          setup: |
            echo "optional setup"
          run: |
            set -e
            echo "{{ config.hello_config.message }}"   # preserved for gbserver
    monitors:
      skypilot_monitor:
        ref: space://monitors/skypilot
```

## `steps/Makefile` scaffolding

`new-step`: copy `_template/` → `steps/$(NAME)/`, then `sed` the placeholder name into `Makefile` and
`step.yaml.tmpl`. Keep the recipe under 40 lines; error if `NAME` unset or target dir exists.

## Files to create

- `steps/common.mk`, `steps/Makefile`, `steps/README.md`
- `steps/_template/{Makefile,Dockerfile,step.yaml.tmpl,requirements.txt}`
- `steps/hello-skypilot/{Makefile,Dockerfile,step.yaml.tmpl,requirements.txt}` (worked example)
- Doc touch-up: note the framework in `steps/README.md`; optionally add a short design note under
  `docs/superpowers/specs/` (repo convention, e.g. today's skypilot-aws design doc).

## Verification (end-to-end)

1. **Scaffold**: `make -C steps new-step NAME=hello-skypilot` → confirm skeleton with name substituted.
2. **Build**: `make -C steps/hello-skypilot image` (podman) → image present in `podman images`.
3. **Render (no creds needed)**: `make -C steps/hello-skypilot stepyaml` → assert `.gen/step.yaml` has
   `image_id: docker:quay.io/granite-build/hello-skypilot:<tag>` AND still contains the literal
   `{{ config.hello_config.message }}` (runtime Jinja preserved). Validate YAML parses.
4. **Auth guard**: run `push` with no robot vars → confirm it fails with the clear message; then with
   `QUAY_ROBOT_USER`/`QUAY_ROBOT_TOKEN` set, `make push` tags+pushes to quay.io.
5. **Install**: `make -C steps/hello-skypilot install` → `configurations/assets/environments/skypilot/steps/hello-skypilot/step.yaml` exists and matches `.gen/step.yaml`.
6. **Resolves + runs**: with gbserver up (`gbserver_status`/`gbserver_start`, run-gbserver skill), submit a
   build referencing `space://steps/hello-skypilot` against a skypilot space via `build_start(file_content=...)`;
   monitor to SUCCESS and read `build_job_log(build_id)` to confirm the `run:` executed inside the quay image.
   (Full VM run needs a real cluster; without one, steps 3+5 verify render/install correctness and the
   installed `step.yaml` resolves.)
```

