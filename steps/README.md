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
* **`Makefile`** — a thin file that sets a couple of variables and includes the
  shared [`common.mk`](common.mk). It exposes the conventional targets below.
* **`README.md`** — documents that step's function, its `config` contract, and its
  inputs/outputs (see [`byoc/skypilot`](byoc/skypilot/README.md) and
  [`eval/skypilot`](eval/skypilot/README.md) for the two step-type examples).
* **`step/`** — *generated* by `make step`; the deployable bundle (a `step.yaml`
  and any bundled `src/`). This directory is git-ignored.

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
* **`step`** — render `step-template.yaml` → `step/step.yaml` and bundle `src/`.
  Cheap and offline; it does *not* rebuild/push.
* **`all`** — the full pipeline: `image` + `publish-image` + `step` for image steps,
  or just `step` for public-image steps.
* **`clean`** — remove the generated `step/`.
* **`help`** — display this README (rendered with `glow`/`mdcat`/`bat` if one is
  installed, otherwise printed plain).

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

`make step` renders the template with `envsubst` restricted to a single variable,
`$IMAGE_REF`:

```sh
IMAGE_REF='<full image ref>' envsubst '$IMAGE_REF' < step-template.yaml > step/step.yaml
```

Because the allowlist names only `IMAGE_REF`, everything else passes through
untouched — importantly, the **runtime Jinja** `{{ ... }}` (resolved later by the
build, e.g. `{{ config.eval_config.model_path }}`) and shell expansions like
`${GB_BUILD_WORKDIR}` / `$(hostname)` inside `run:`/`setup:` blocks.

* Image steps put `image_id: "docker:${IMAGE_REF}"` in the template; `${IMAGE_REF}`
  becomes the published image reference (`$(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)`)
  at render time.
* Public-image steps carry no `${IMAGE_REF}` token — they select their image via
  runtime Jinja from `config.*`, so rendering is effectively a copy plus bundling.

> Requires `envsubst` (from gettext). macOS: `brew install gettext`; Debian/Ubuntu:
> `apt-get install gettext-base`. `make check-tools` verifies it.

## Referencing a generated step from a build.yaml

After `make step`, reference the bundle by an **absolute** `file://` URI:

```yaml
steps:
  - step_uri: file:///abs/path/to/steps/byoc/skypilot/step
    config:
      byoc_config:
        image: "python:3.12-slim"
        repo: "https://github.com/org/repo"
        ref: "main"
        command: "python main.py"
```

End-to-end execution requires a real SkyPilot cluster; submit the build.yaml via
the `gbserver` MCP tools (see the `run-gbserver` and `create-step` skills).
