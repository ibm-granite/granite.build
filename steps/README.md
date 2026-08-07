# Step Implementation Framework

This is the root of a directory that contains content to generate step
implementations (`step.yaml` plus assets) suitable for inclusion in a
Granite.build space.

Subdirectories contain step implementations (`eval`, `byoc`, etc.), each with
per-compute-environment subdirectories (`skypilot`, `lsf`, `k8s`, ...). For
example, `steps/eval/skypilot` holds the eval step's SkyPilot implementation.

## Layout of a step/environment directory

Each step/environment directory (e.g. `steps/byoc/skypilot`) contains:

* **`step-template.yaml`** — the template for the generated `step.yaml`, into which
  an optional image reference and other substitutions are made.
* **`Dockerfile`** — optional; provided only when the step requires a custom image
  to execute (see the two step types below).
* **`src/`** — optional; a directory of code referenced by the step. For image
  steps it is baked into the image; for public-image steps it can be made
  available to the running step (e.g. via SkyPilot `file_mounts`).
* **`test/`** — optional; the step's own tests, run with `make test` (`src/` is
  put on `PYTHONPATH`; pytest recurses the whole tree). These are developed and
  run **independently** of the repository's central suite (they are not in its
  `testpaths`). Cluster-specific build tests live in a **per-cluster subdir**
  (`test/<cluster>/`, e.g. `test/slurm/`, `test/docker/`) so each cluster's test
  is self-contained; cluster-agnostic unit tests for `src/` can stay at `test/`'s
  root (see `eval/skypilot/test/test_eval.py`). It sits beside `src/` but is
  **not** bundled into the deployable step.
* **`test-data/`** — optional; fixtures for the tests in `test/`, mirrored into
  the **same per-cluster subdir** (`test-data/<cluster>/`, e.g. a build test's
  `build.yaml`/`buildtest.yaml` in `test-data/slurm/` or `test-data/docker/`).
  Kept beside the step rather than in the repo's parallel `test/` ↔ `test-data/`
  tree, so the step is self-contained.
* **`Makefile`** — a thin file that sets a couple of variables and includes the
  shared [`common.mk`](common.mk). It exposes the conventional targets below.
* **`README.md`** — documents that step's function, its `config` contract, and its
  inputs/outputs (see [`byoc/skypilot`](byoc/skypilot/README.md) and
  [`eval/skypilot`](eval/skypilot/README.md) for the two step-type examples).
* **`space/`** — *generated* by `make space`; a self-contained Granite.build
  **Space** — a `space.yaml` (whose `base_uris` chain to `configurations/assets`)
  plus `steps/<step-name>/step.yaml` and any bundled `src/`. The dir name defaults
  to `space` (overridable via `SPACE_DIR`). This directory is git-ignored.

## Two step types

Which type a step is is **auto-detected from the presence of a `Dockerfile`**
next to the Makefile — there is no flag to set.

1. **Custom-image step** (has a `Dockerfile`, exemplar: **`eval`**) — the
   step's code/deps are baked into an image built from `Dockerfile`, published to a
   registry, and referenced from the generated `step.yaml` via `image_id`.
2. **Public-image step** (no `Dockerfile`, exemplar: **`byoc`**) — the
   step runs in a public container image and brings its code at runtime. `byoc`
   clones a public git repo in the launcher's `setup` phase and runs a
   user-defined `command`; it builds no custom image.

## Makefile target conventions

Defined once in [`common.mk`](common.mk) and shared by every step:

* **`image`** — build the image from `./Dockerfile` for `$(PLATFORM)` (default
  `linux/amd64`, so it cross-builds on an Apple Silicon host for the x86 clusters
  SkyPilot provisions). Image steps only; no-op otherwise.
* **`publish-image`** — push the image to `$(REGISTRY)` (no-op for non-image
  steps). Requires authentication — see [Registry credentials](#registry-credentials).
* **`space`** — render a self-contained Space into `$(SPACE_DIR)/`: a generated
  `space.yaml` plus `steps/<step-name>/step.yaml` (from `step-template.yaml`) and
  bundled `src/`. Cheap and offline; it does *not* rebuild/push.
* **`publish`** — promote the step into the repo's committed assets tree
  (`configurations/assets/environments/<env>/steps/<step-name>/`, rendered exactly as
  `space` renders `step.yaml` + bundled `src/`) **and** copy the step's per-cluster
  build tests into the top-level `test/steps/<step-name>/<env>/<cluster>/` tree, with
  their fixtures in the parallel `test-data/steps/<step-name>/<env>/<cluster>/` tree
  (mirroring the repo's `test/` ↔ `test-data/` convention) and each copied
  `buildtest.yaml`'s `space_uri` repointed at the published step. See
  [Two test modes](#two-test-modes). Deliberately **not** part of `all` — it writes
  tracked files you then commit.
* **`all`** — the full pipeline: `image` + `publish-image` + `space` for image steps,
  or just `space` for public-image steps.
* **`test`** — render the Space (depends on `space`) **and build the image
  locally** (depends on `image`; a no-op with no publish for non-image steps),
  then run the step's tests in `test/` with `pytest` (`src/` is put on
  `PYTHONPATH`; pytest recurses per-cluster subdirs). Depending on `space` lets a
  build test just reference the rendered `$(SPACE_DIR)/`; depending on `image`
  means a local-Docker build test finds the freshly built image in the local
  store (the `docker` environment's `pull_policy` is `if-not-present`, so **no
  registry publish** is needed). No-op when `test/` is absent or empty.
* **`clean`** — remove the generated `$(SPACE_DIR)/`.
* **`help`** — list the targets with a one-line description and point to this
  README for full documentation.

### Variables (override on the command line or via the environment)

| Variable          | Default                                   | Meaning                          |
|-------------------|-------------------------------------------|----------------------------------|
| `STEP_NAME`       | *(set by each step's Makefile)*           | logical step name                |
| `DOCKER`          | `podman`                                  | container tool                   |
| `DOCKERFILE`      | `Dockerfile`                              | its presence enables image build/push |
| `REGISTRY`        | *(required for image steps; set by the step's Makefile)* | image registry + namespace |
| `IMAGE_NAME`      | `gb-step-$(STEP_NAME)`                     | image repository name            |
| `IMAGE_TAG`       | git short SHA, else `latest`              | image tag                        |
| `IMAGE_REF`       | `$(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)`   | full image reference (derived)   |
| `SPACE_DIR`       | `space`                                   | generated Space directory name   |
| `SPACE_NAME`      | `$(STEP_NAME)`                            | `name:` in the generated `space.yaml` |
| `SPACE_BASE_URI`  | relative `file://` to `configurations/assets` (resolved against the space.yaml's dir) | base_uri chained by the generated Space |
| `DEFAULT_ENVIRONMENT` | *(empty)*                             | if set, written as `variables.DEFAULT_ENVIRONMENT` in `space.yaml` |
| `TEST_DIR`        | `test`                                    | dir of Python tests run by `make test` |
| `PYTHON`          | `python3`                                 | interpreter used to run tests    |
| `STEP_ENV`        | the Makefile's own dir name (e.g. `skypilot`) | step's environment segment, used by `publish` |
| `PUBLISH_STEP_DIR`| `configurations/assets/environments/$(STEP_ENV)/steps/$(STEP_NAME)` | where `publish` renders the step |
| `PUBLISH_TEST_DIR`| `test/steps/$(STEP_NAME)/$(STEP_ENV)`     | where `publish` copies the per-cluster build tests |
| `PUBLISH_TESTDATA_DIR` | `test-data/steps/$(STEP_NAME)/$(STEP_ENV)` | where `publish` copies the tests' fixtures |
| `MODE2_SPACE_URI` | relative `file://`-less path to `configurations/spaces/local` (5 levels up from a copied fixture's dir) | `space_uri` written into copied `buildtest.yaml`s |

Example: `make all REGISTRY=quay.io/myorg IMAGE_TAG=0.1.0`.

### Registry credentials

`make publish-image` must be authenticated to `$(REGISTRY)`. Two ways:

* **Interactive / local (default):** `podman login <registry-host>` (or
  `docker login`) once; the push reuses the stored token. Nothing else to
  configure.
* **CI / non-interactive:** export `REGISTRY_USER` and `REGISTRY_PASSWORD` (a
  robot-account token) **in the environment** — do *not* pass them as `make`
  variables. `publish-image` logs in first, piping the token via
  `--password-stdin`, so the secret never appears in `ps`, make's output, or
  shell history:

  ```sh
  export REGISTRY_USER='my-org+ci'
  export REGISTRY_PASSWORD="$QUAY_TOKEN"   # from your CI secret store
  make publish-image REGISTRY=quay.io/my-org IMAGE_TAG=0.1.0
  ```

## Rendering: how the image reference is inserted

`make space` (and `make publish`) render the template by substituting **only** the
literal `${IMAGE_REF}` token, with a single `sed` replacement — no `envsubst`/gettext
dependency, just the POSIX `sed` every system already has:

```sh
ref=$(printf '%s' '<full image ref>' | sed 's/[#&\]/\\&/g')   # escape sed's replacement metachars
sed "s#\${IMAGE_REF}#$ref#g" step-template.yaml > $(SPACE_DIR)/steps/<step-name>/step.yaml
```

Because only the literal `${IMAGE_REF}` is replaced, everything else passes through
untouched — importantly, the **runtime Jinja** `{{ ... }}` (resolved later by the
build, e.g. `{{ config.eval_config.model_path }}`) and shell expansions like
`${GB_BUILD_WORKDIR}` / `$(hostname)` inside `run:`/`setup:` blocks. (The image ref
is escaped first so a value containing `#`, `&`, or `\` can't corrupt the
substitution.) This is factored into the `render-step-template` helper in
[`common.mk`](common.mk), shared by both targets.

* Image steps put `image_id: "docker:${IMAGE_REF}"` in the template; `${IMAGE_REF}`
  becomes the published image reference (`$(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)`)
  at render time.
* Public-image steps carry no `${IMAGE_REF}` token — they select their image via
  runtime Jinja from `config.*`, so rendering is effectively a copy plus bundling.

## Referencing a generated step from a build.yaml

After `make space`, the step lives in the generated Space at
`steps/byoc/skypilot/space/`. Point the build's Space at that directory (its
`space_uri`) and reference the step by the stable `space://steps/<step-name>` URI —
the generated `space.yaml`'s `base_uris` resolve everything else (environments,
monitors, other steps) from `configurations/assets`:

```yaml
steps:
  - step_uri: space://steps/byoc
    config:
      byoc_config:
        image: "python:3.12-slim"
        repo: "https://github.com/org/repo"
        ref: "main"
        command: "python main.py"
```

Two worked examples, one per step type, each in a per-cluster subdir:

* **Public-image step on SkyPilot/slurm** — the `byoc` build test at
  [`byoc/skypilot/test/slurm/test_skypilot_byoc.py`](byoc/skypilot/test/slurm/test_skypilot_byoc.py),
  driven by its fixtures in the sibling
  [`byoc/skypilot/test-data/slurm/`](byoc/skypilot/test-data/slurm/). Run it with
  `make -C steps/byoc/skypilot test`, which renders the Space (`make space`) first
  so the test's `space_uri` resolves. End-to-end execution needs a real SkyPilot
  cluster (see the local `make slurm-setup`).
* **Custom-image step on the local Docker environment** — the `eval` build test at
  [`eval/skypilot/test/docker/test_docker_eval.py`](eval/skypilot/test/docker/test_docker_eval.py),
  driven by its fixtures in
  [`eval/skypilot/test-data/docker/`](eval/skypilot/test-data/docker/). Run it with
  `make -C steps/eval/skypilot test`, which renders the Space **and builds the
  image locally** (`make image`) first, then runs the container against the local
  Docker daemon — the image is used from the local store with **no publish**. This
  is the way to build and exercise a custom-image step end to end without a
  registry or a container-capable cluster.

Both submit their `build.yaml` through the buildtest framework; for ad-hoc runs,
submit a `build.yaml` via the `gbserver` MCP tools (see the `run-gbserver` and
`create-step` skills).

## Two test modes

A step's build tests run in **two modes** against the **same test files** — the
`build.yaml`/`buildtest.yaml` and the test code are identical; only which Space
resolves the step differs (the `space_uri`):

* **Mode 1 — pre-publish (in the step dir).** `steps/<step>/<env>/test/<cluster>/`
  run by `make test`, resolving the step from the locally rendered
  `steps/<step>/<env>/space/` (the `test` target's `space` prerequisite renders it
  first). This is the fast authoring loop and needs nothing published. The
  fixtures' `space_uri: ../../space` points at that local Space.
* **Mode 2 — post-publish (in `test/steps/`).** `make publish` copies the
  per-cluster build tests to `test/steps/<step>/<env>/<cluster>/` — the step dir's
  inner `test/` segment is **flattened away**, so the published tree parallels the
  step's own layout one level shallower — with their fixtures in the matching
  `test-data/steps/<step>/<env>/<cluster>/` (the repo's `test/` ↔ `test-data/`
  convention, so a test resolves its fixtures via
  [`get_test_data_dir_for`](../test/libgbtest/buildrunner/buildtest.py) in **both**
  modes with no code change). It rewrites each copied `buildtest.yaml`'s `space_uri`
  to the shared space [`configurations/spaces/local`](../configurations/spaces/local)
  (`../../../../../configurations/spaces/local`, five levels up from the fixture's
  dir), which resolves the **published** step from `configurations/assets`. These
  live under the repo's top-level `test/` tree, so they are **discoverable and
  runnable from VSCode's Test Explorer**, yet stay **out of every whole-tree Makefile
  suite**. They are not in `pyproject.toml`'s `testpaths` and are not listed by the
  `test-pr` target; and [`test/steps/conftest.py`](../test/steps/conftest.py)
  auto-applies the `step_build_test` marker to the whole tree, which `quick-tests`,
  `extended-tests`, and the shared `DEFAULT_PYTEST_MARKERS` base all deselect
  (`-m "… and not step_build_test"`). Run them **from VSCode** or **explicitly**
  (`pytest test/steps/…`). Being real-infra tests (also `@extended`-marked, and
  `docker_required` / `skypilot_integration` per cluster), they still require the
  relevant infra — a Docker daemon / a reachable cluster — to actually execute
  rather than skip.

`make publish` creates the assets step and the `test/steps/` + `test-data/steps/`
copies **together**, so a committed `test/steps/<step>/<env>/` always has a matching
published step to resolve against. Only per-cluster build tests (`test/<cluster>/`
subdirs) are copied — `src/` and cluster-agnostic unit tests at `test/`'s root (e.g.
[`eval/skypilot/test/test_eval.py`](eval/skypilot/test/test_eval.py)) stay Mode-1
only.
