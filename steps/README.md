# Step Implementation Framework

This is the root of a directory that contains content to generate step
implementations (`step.yaml` plus assets) suitable for inclusion in a
Granite.build space.

Subdirectories contain step implementations (`eval`, `byoc`, etc.), each with
per-compute-environment subdirectories (`skypilot`, `lsf`, `k8s`, ...). For
example, `steps/eval/skypilot` holds the eval step's SkyPilot implementation.

## Source of truth: `steps/` vs. `configurations/`

Think of the two trees as **source vs. release**: `steps/` is a source-like
**development tree** where a step is authored and iterated (a `step-template.yaml` +
`Makefile`, plus an optional `Dockerfile`/`src/`/`test/`, with `make test` for the
fast authoring loop), and `configurations/` is the **fully vetted set of steps** a
Granite.build space actually consumes — analogous to a code release. `make
publish-step` is the promotion step between them: it **generates** a step's released
form under `configurations/assets/.../steps/<name>/` (together with its build-test
trees under `test/steps/` and `test-data/steps/`). That `configurations/` copy is a
*release artifact*, not something to edit by hand.

**Going-forward rule — all new steps live in `steps/`.** Every new step, whether
custom-image or public-image, is authored under `steps/`; writing a `step.yaml`
directly under `configurations/assets/.../steps/*` with no `steps/` source is
**legacy** and should not be done for new work. The rule does not key off step type:
`byoc` is a public-image step and still lives in `steps/`, because the framework
(templated rendering, the publish pipeline, and the synced Mode-1/Mode-2 test trees —
see [Two test modes](#two-test-modes)) applies to it just as much as to a
custom-image step like `eval`.

**The two tiers today.** The repo currently contains both:

* **Sourced steps** — authored under `steps/`, with a *generated* `configurations/`
  counterpart (today: `byoc`, `eval`).
* **Hand-authored asset steps** — ~30+ step definitions that exist **only** under
  `configurations/assets/.../steps/*` with no `steps/` source (e.g. `digit`, `sage`,
  the `sage-eval-*` / `openinstruct-*` / `bfcl-*` families, `hello`). These predate
  the framework — effectively a release with no corresponding source in the tree.

This split is **transitional, not a design goal.** The target end-state is that every
step has a `steps/` source and the `configurations/` step assets are entirely
generated; the hand-authored asset steps are legacy debt to be migrated under `steps/`
over time (opportunistically as each is next substantially changed, and eventually all
of them).

**Recommended default for a new author:** author the step in `steps/`. And if you set
out to change a hand-authored step under `configurations/assets/.../steps/*`, prefer
migrating it into `steps/` first (add a `step-template.yaml` + `Makefile`, move any
code into `src/`, then let `make publish-step` regenerate the asset) rather than
editing the released asset in place.

### What promotion does — and what the framework is *for*

Promotion is deliberately narrow. `make publish-step` (and the `make space` render it
builds on) does four things and no more:

1. **Renders one token.** `step-template.yaml` → `step.yaml` substituting only the
   literal `${IMAGE_REF}` (empty for public-image steps); all runtime Jinja `{{ … }}`
   and shell `${VAR}` pass through verbatim.
2. **Copies `src/`** into the released step.
3. **Copies the per-cluster build tests** into `test/steps/<name>/<env>/<cluster>/`
   and their fixtures into `test-data/steps/…`, **rewriting each `buildtest.yaml`'s
   `space_uri`** so the *same* test file resolves in both modes (see
   [Two test modes](#two-test-modes)).
4. **Freezes the image tag** for custom-image steps — the per-commit `IMAGE_REF`
   (default tag = git short SHA) that you can't reasonably hand-maintain.

It is **not** a one-to-many generator and **not** parameterized templating: one source
directory produces exactly one released step per environment, so for a public-image
step like `byoc` the promotion is a byte-for-byte copy. Per-environment variation is
expressed by **authoring** a separate source dir per environment
(`steps/<name>/skypilot`, `.../lsf`, …) and by putting multiple
`environment_configs`/launchers in a single template (e.g. eval's `Skypilot` +
`Docker`) — it is authored, not fanned out from one definition. There is no roadmap to
generate multiple asset variants from a shared source.

So the value is **not** "less YAML to maintain." It is:

* **Release discipline / provenance.** `configurations/` is a vetted release; you
  promote into it from a reviewed, tested source rather than hand-editing the release.
  Even when the copy is literal, the definition gains a development home where it is
  changed and tested before it ships.
* **Tests travel with the step, and the three trees stay in sync.** The step's build
  tests live beside it and run via `make test` (Mode 1) before promotion, then land in
  `test/steps/` (Mode 2) — assets, tests, and fixtures are published together and
  can't silently drift (`make check-published` re-renders and diffs them).
* **Image-ref rendering** (custom-image only) — the one genuinely non-copy piece, and
  the strongest single reason for the mechanism.

**"Why not just edit `configurations/` directly?"** Because that edits the release in
place, bypassing the source, its tests, and the sync — the same reason you don't patch
a build artifact instead of its source. Change the step in `steps/` and re-run
`make publish-step`.

## Layout of a step/environment directory

Each step/environment directory (e.g. `steps/byoc/skypilot`) mixes files you
author with a directory `make` generates. They fall into three groups.

### Required

The minimum a step/environment directory needs for `make` to render it:

* **`step-template.yaml`** — the template for the generated `step.yaml`, into which
  an optional image reference and other substitutions are made.
* **`Makefile`** — a thin file that sets a couple of variables (notably `STEP_NAME`)
  and includes the shared [`common.mk`](common.mk). It exposes the conventional
  targets below.

### Optional

Present or absent depending on what the step needs:

* **`Dockerfile`** — provided only when the step requires a custom image
  to execute; its presence is what marks the step as a custom-image step (see the
  two step types below).
* **`src/`** — a directory of code referenced by the step. For image
  steps it is baked into the image; for public-image steps it can be made
  available to the running step (e.g. via SkyPilot `file_mounts`).
* **`test/`** — the step's own tests, run with `make test` (`src/` is
  put on `PYTHONPATH`; pytest recurses the whole tree). These are developed and
  run **independently** of the repository's central suite (they are not in its
  `testpaths`). Cluster-specific build tests live in a **per-cluster subdir**
  (`test/<cluster>/`, e.g. `test/slurm/`, `test/docker/`) so each cluster's test
  is self-contained; cluster-agnostic unit tests for `src/` can stay at `test/`'s
  root (see `eval/skypilot/test/test_eval.py`). It sits beside `src/` but is
  **not** bundled into the deployable step.
* **`test-data/`** — fixtures for the tests in `test/`, mirrored into
  the **same per-cluster subdir** (`test-data/<cluster>/`, e.g. a build test's
  `build.yaml`/`buildtest.yaml` in `test-data/slurm/` or `test-data/docker/`).
  Kept beside the step rather than in the repo's parallel `test/` ↔ `test-data/`
  tree, so the step is self-contained.
* **`README.md`** — documents that step's function, its `config` contract, and its
  inputs/outputs (see [`byoc/skypilot`](byoc/skypilot/README.md) and
  [`eval/skypilot`](eval/skypilot/README.md) for the two step-type examples).
  Conventional for every step, though not needed to build or render one.

### Generated

Produced by `make`, not authored by hand:

* **`space/`** — produced by `make space`; a self-contained Granite.build
  **Space** — a `space.yaml` (whose `base_uris` chain to `configurations/assets`)
  plus `steps/<step-name>/step.yaml` and any bundled `src/`. The dir name defaults
  to `space` (overridable via `SPACE_DIR`). This directory is git-ignored.

`make publish-step` generates further trees, but **outside** this directory (the
committed step under `configurations/assets/`, its build tests under `test/steps/`,
and fixtures under `test-data/steps/`) — see the publish sections below.

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
* **`publish-step`** — promote the step into the repo's committed assets tree
  (`configurations/assets/environments/<env>/steps/<step-name>/`, rendered exactly as
  `space` renders `step.yaml` + bundled `src/`) **and** copy the step's per-cluster
  build tests into the top-level `test/steps/<step-name>/<env>/<cluster>/` tree, with
  their fixtures in the parallel `test-data/steps/<step-name>/<env>/<cluster>/` tree
  (mirroring the repo's `test/` ↔ `test-data/` convention) and each copied
  `buildtest.yaml`'s `space_uri` repointed at the published step. See
  [Two test modes](#two-test-modes). Deliberately **not** part of `all` — it writes
  tracked files you then commit. For a **custom-image step** it first verifies the
  step's image is actually published to its registry (a published `step.yaml`/test that
  references an unpushed image is broken — the Mode-2 skypilot test's remote node can't
  pull it), so run `make image publish-image` **before** `make publish-step`. The check
  queries the registry manifest only (no pull; `skopeo` preferred, else `$(DOCKER)
  manifest inspect`) and needs no credentials for a public image. Bypass it with
  `make publish-step PUBLISH_REQUIRE_IMAGE=false` (offline/local publishing).
  Non-image (public-image) steps have no `IMAGE_REF`, so the check is skipped.
* **`check-published`** — verify the committed generated artifacts still match a
  fresh render of the current source (see
  [Generated artifacts and drift](#generated-artifacts-and-drift)). Re-renders
  `publish-step` into a temp dir and `diff`s it against the three committed trees,
  exiting non-zero on drift. It re-renders image steps with the **already-committed**
  image reference (extracted from the published `step.yaml`), so it is immune to the
  per-commit `IMAGE_TAG` churn and flags only genuine content drift. Exit 0 when in
  sync (or nothing is published yet).
* **`all`** — the full pipeline: `image` + `publish-image` + `space` for image steps,
  or just `space` for public-image steps.
* **`test`** — render the Space (depends on `space`) **and build the image
  locally** (depends on `image`; a no-op with no publish for non-image steps),
  then run the step's tests in `test/` with `pytest` (`src/` is put on
  `PYTHONPATH`; pytest recurses per-cluster subdirs). Depending on `space` lets a
  build test just reference the rendered `$(SPACE_DIR)/`; depending on `image`
  means a local-Docker build test finds the freshly built image in the local
  store (the `docker` environment's `pull_policy` is `if-not-present`, so **no
  registry publish** is needed). No-op when `test/` is absent or empty. The tests
  import `gbserver`/`libgbtest`, so they run under the **repo-root virtualenv**
  (`$(VENV_DIR)`, default `.venv` at the repo root): `make test` activates it
  automatically and fails fast with a pointer to `make venv` if it is missing.
  Override `VENV_DIR` if your virtualenv lives elsewhere.
* **`test-setup`** — optional per-step hook that stands up the infrastructure a
  step's tests need (e.g. a local SLURM + MinIO cluster). It is a **separate
  target, deliberately not a prerequisite of `test`**, so the (often slow) infra
  bring-up runs only when you ask: run `make test-setup` **once**, then iterate
  with `make test`. `common.mk` supplies a no-op default; a step opts in by
  setting `HAS_TEST_SETUP := true` **before** the `include` and defining its own
  `test-setup` target after it — typically delegating to the repo-root Makefile,
  e.g. `test-setup: ; $(MAKE) -C $(REPO_ROOT) slurm-setup minio-setup` (the
  `HAS_TEST_SETUP` flag makes `common.mk` skip its default, avoiding an
  "overriding recipe" warning). `REPO_ROOT` is provided by `common.mk`.
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
| `VENV_DIR`        | `.venv` at the repo root                  | virtualenv `make test` activates before running pytest |
| `HAS_TEST_SETUP`  | *(unset)*                                 | set to `true` (before the include) when the step defines its own `test-setup` target |
| `STEP_ENV`        | the Makefile's own dir name (e.g. `skypilot`) | step's environment segment, used by `publish-step` |
| `PUBLISH_STEP_DIR`| `configurations/assets/environments/$(STEP_ENV)/steps/$(STEP_NAME)` | where `publish-step` renders the step |
| `PUBLISH_TEST_DIR`| `test/steps/$(STEP_NAME)/$(STEP_ENV)`     | where `publish-step` copies the per-cluster build tests |
| `PUBLISH_TESTDATA_DIR` | `test-data/steps/$(STEP_NAME)/$(STEP_ENV)` | where `publish-step` copies the tests' fixtures |
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

`make space` (and `make publish-step`) render the template by substituting **only** the
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
* **Custom-image step on SkyPilot/aws** — the `eval` build test at
  [`eval/skypilot/test/aws/test_skypilot_aws_eval.py`](eval/skypilot/test/aws/test_skypilot_aws_eval.py),
  driven by its fixtures in
  [`eval/skypilot/test-data/aws/`](eval/skypilot/test-data/aws/). Run it with
  `make -C steps/eval/skypilot test`, which renders the Space (`make space`) first
  so the test's `space_uri` resolves. End-to-end execution provisions a real EC2
  instance via SkyPilot, which **pulls the eval image from its registry** by the
  reference frozen into the step's `image_id` — so the image must be **published
  to a pullable registry first** (`make image publish-image REGISTRY=...`) and AWS
  credentials exported. (A custom-image step cannot run on the local Docker slurm
  cluster, which has no Pyxis — hence the cloud backend.)

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
* **Mode 2 — post-publish (in `test/steps/`).** `make publish-step` copies the
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
  runnable from VSCode's Test Explorer** and run in the **extended** suite, yet stay
  **out of the quick / PR selections**. They are not in `pyproject.toml`'s
  `testpaths` and are not listed by the `test-pr` target; and
  [`test/steps/conftest.py`](../test/steps/conftest.py) auto-applies the `extended`
  marker to the whole tree, so `quick-tests` (and any `not extended` selection)
  deselects them while `make extended-tests` runs them — no per-file
  `@extended_testing_only` needed on the copies. Run them **from VSCode**,
  via `make extended-tests`, or **explicitly** (`pytest test/steps/…`). Being
  real-infra tests (`skypilot_integration`, and `docker_required` per cluster), they
  still require the relevant infra — a Docker daemon / a reachable cluster — to
  actually execute rather than skip.

`make publish-step` creates the assets step and the `test/steps/` + `test-data/steps/`
copies **together**, so a committed `test/steps/<step>/<env>/` always has a matching
published step to resolve against. Only per-cluster build tests (`test/<cluster>/`
subdirs) are copied — `src/` and cluster-agnostic unit tests at `test/`'s root (e.g.
[`eval/skypilot/test/test_eval.py`](eval/skypilot/test/test_eval.py)) stay Mode-1
only.

> **Image steps: keep the published image tag and your local build in sync.** For a
> custom-image step, `make publish-step` freezes a concrete image reference into the
> committed `step.yaml` — with the default `IMAGE_TAG` (git short SHA) that tag is
> the commit `publish-step` ran at, and it goes stale on the next commit. A Mode-2 test
> whose environment uses `pull_policy: if-not-present` (i.e. a local Docker
> environment test) then resolves **only** when a local image at that exact tag
> exists; otherwise the
> launcher tries to pull the (often placeholder) registry and fails. Before running
> such a committed Mode-2 test, rebuild **and** re-publish at the current commit
> (`make -C steps/<step>/<env> image publish-step`) so both sides agree — or pin a stable
> `IMAGE_TAG` (e.g. `IMAGE_TAG=local`) on both the build and the publish-step so the
> committed assets don't drift. Mode 1 (`make test`) is immune: it builds and renders
> at HEAD together. Public-image steps (e.g. byoc) have no such coupling.

## Generated artifacts and drift

`make publish-step` writes **three committed trees derived from the step's source** under
`steps/<step>/<env>/` (its `step-template.yaml`, `src/`, and per-cluster
`test/`+`test-data/`):

1. the assets step — `configurations/assets/environments/<env>/steps/<name>/`
   (`step.yaml` + bundled `src/`);
2. the Mode-2 tests — `test/steps/<name>/<env>/<cluster>/`;
3. their fixtures — `test-data/steps/<name>/<env>/<cluster>/`.

These are **generated, not hand-edited** — edit the source under `steps/<step>/<env>/`
and re-run `make publish-step`, then commit the regenerated trees. Nothing in the build
re-publishes automatically, so a source edit that isn't followed by `make publish-step`
leaves the committed copies stale.

Run **`make check-published`** to catch that drift: it re-renders `publish-step` into a
temp dir and diffs it against the three committed trees, exiting non-zero if they
differ. It re-renders image steps with the image reference **already committed** in
`step.yaml`, so a routine `IMAGE_TAG` (git-SHA) bump alone is *not* flagged — only a
genuine template/`src/`/test change that was never re-published (contrast the
image-tag coupling note above). It is not wired into CI; run it locally after editing a
step's source, or add it to a pre-commit / CI check if you want the guarantee enforced.
