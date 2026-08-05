# Step Implementation Framework — SkyPilot exemplars (`byoc` + `eval`)

## Context

We're building a convention-driven framework under `steps/` for authoring Granite.build
custom steps per compute environment. Layout is `steps/<step>/<env>/` (e.g.
`steps/byoc/skypilot`). Each env dir holds the machinery to **generate** a deployable
`step/` directory (a `step.yaml` plus any `src/` assets) that a `build.yaml` references by an
absolute `file://` URI.

Two step styles must be supported by one set of Makefile conventions:
1. **Custom-image steps** — build a Dockerfile, publish the image, insert its ref into
   `step.yaml`. Exemplar: **`eval`** (single fixed output; the `run:` block emits the
   artifact line, keeping the workload script free of the Granite.build convention).
2. **Public-container + code steps** — run in a public image, bring code at runtime.
   Exemplar: **`byoc`** (clones a public git repo in `setup:`, runs a user-defined command).

The `steps/byoc/skypilot` scaffold and `steps/README.md` already exist (README fixes the target
names: `image`, `publish-image`, `step`). SkyPilot launch consumes four inputs we target:
`run`, `setup`, `image_id`, `file_mounts` (see `StepSkypilotConfig` in
`src/gbserver/types/environment/skypilot.py` and the launcher in
`src/gbserver/environment/skypilot.py`).

Decisions locked with the user:
- **byoc code delivery**: `git clone` in `setup:` (repo URL + ref as config params); `run:`
  executes a user command. `src/` holds at most a small optional helper (file-mounted to
  demonstrate the `file_mounts` input).
- **Registry**: configurable `REGISTRY ?= quay.io/<your-org>` placeholder, overridable.
- **Makefiles**: a shared `steps/common.mk` include; each step Makefile is thin.
- **Rendering**: `envsubst` with an explicit allowlist (`$IMAGE_REF`) so runtime Jinja `{{ }}`
  and shell `${VARS}` in `run:`/`setup:` are left untouched.

## Design

### Generation pipeline (per env dir)
`step-template.yaml` → (`envsubst '$IMAGE_REF'`) → rendered **directly into** `step/step.yaml`,
alongside a copy of `src/`. There is no top-level intermediate `step.yaml` (the user is
deleting the redundant placeholder). The `step/` dir is the deployable bundle referenced as
`file:///abs/.../steps/<step>/skypilot/step`, and is a generated artifact (add a `.gitignore`).

### Reference model
- Image-conditional `image_id` pattern comes from the builtin command step
  (`src/gbserver/builtins/steps/skypilot/command/step.yaml`).
- Image + workload + artifact-emit + `outputs` pattern from
  `configurations/assets/environments/skypilot/lsf/ibm-bluevela/steps/openinstruct-sft/step.yaml`.
- Minimal `resources`/`run`/monitor-ref from
  `configurations/assets/environments/skypilot/aws/steps/hello/step.yaml`.
- Dockerfile precedent: `configurations/assets/environments/skypilot/steps/sage/Dockerfile`.
- Container tool default `DOCKER ?= podman` and tag/push idioms from the root `Makefile`.
- Monitor is referenced, not redefined: `ref: space://monitors/skypilot`.

## Files

### 0. `step-framework-plan.md` (new — save this plan)
Write this plan document to `step-framework-plan.md` at the repo root as the first action, so
it persists in the working tree for review. *(User handles any git add/commit.)*

### 1. `steps/common.mk` (new — shared conventions)
Thin, included by each step Makefile which first sets a few vars. Defines:

**Variables (overridable):**
- `STEP_NAME` (required, set by includer), `ENV ?= skypilot`
- `STEP_USES_IMAGE := $(if $(wildcard Dockerfile),true,false)` — auto-detected from
  the presence of a `Dockerfile`; `true` enables real `image`/`publish-image` (not set by includer)
- `DOCKER ?= podman`
- `REGISTRY ?= quay.io/your-org`
- `IMAGE_NAME ?= gb-step-$(STEP_NAME)`
- `IMAGE_TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo latest)`
- `IMAGE_REF = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)`
- `STEP_DIR = step`, `SRC_DIR = src`, `TEMPLATE = step-template.yaml`

**Targets:**
- `image` — if `STEP_USES_IMAGE=true`: `$(DOCKER) build . -t $(IMAGE_REF)`; else echo "no custom image; skipping".
  (`image` builds for a configurable `PLATFORM`, default `linux/amd64`, so it cross-builds on an Apple Silicon host for the x86 GPU clusters SkyPilot provisions).
- `publish-image` — if uses image: `$(DOCKER) push $(IMAGE_REF)`; else no-op. (Assumes prior `docker/podman login`.)
- `step` — depends on `image publish-image` for image steps (byoc: no image prereqs). `mkdir -p $(STEP_DIR)`; render `$(TEMPLATE)` → `$(STEP_DIR)/step.yaml` via `IMAGE_REF='$(IMAGE_REF)' envsubst '$$IMAGE_REF'` (IMAGE_REF empty when non-image); then `cp -r $(SRC_DIR) $(STEP_DIR)/` when `$(SRC_DIR)` exists & non-empty. Renders straight into `step/` — no top-level intermediate.
- `clean` — `rm -rf $(STEP_DIR)`.
- `all` — `step` (image steps get `image publish-image step`; byoc gets `step`). Order enforced via prereqs, not `.PHONY` chaining that races.
- `help` — display the shared `steps/README.md` (located via `MAKEFILE_LIST`), using `glow`/`mdcat`/`bat` if available else `cat`.
- `check-tools` — verify `envsubst` present (fail with a `brew install gettext` hint).

### 2. `steps/byoc/skypilot/` (fill in placeholders)
- **`Makefile`** — thin: `STEP_NAME := byoc`, `include ../../common.mk` (no Dockerfile → non-image step).
- **`step-template.yaml`** — `type: exec`; `config.byoc_config` with `image` (public container, e.g. `python:3.12-slim`), `repo`, `ref`, `command`. Launcher `type: skypilot`:
  - `image_id: '{{ ("docker:" ~ config.byoc_config.image) if config.byoc_config.image else "" }}'` (runtime Jinja — a public image, **not** a Makefile-time token)
  - `resources: {}` (cloud-agnostic; build supplies via `config.launcher_config.resources`)
  - `file_mounts: { src: src }` (mounts the optional helper dir)
  - `setup:` — `git clone` the repo, `git checkout` the ref if set
  - `run:` — run `{{ config.byoc_config.command }}` in the cloned dir; emit the `LLMB_ARTIFACT_ID:/LLMB_ARTIFACT_PATH:` line for any output
  - `monitors: { skypilot_monitor: { ref: space://monitors/skypilot } }`
- **`src/`** — add a minimal optional helper (e.g. `README` note or a tiny `helpers.sh`) so the file_mount is demonstrated; keep it small per the "at most a small wrapper" decision.
- **`Dockerfile`** — byoc needs no custom image; replace the empty placeholder with a short comment explaining byoc runs in a public image (so the file documents intent rather than dangling empty). `image`/`publish-image` are no-ops here.
- The redundant empty top-level `step.yaml` is being deleted by the user; the framework never writes there.

### 3. `steps/eval/skypilot/` (new — image exemplar)
- **`Makefile`** — `STEP_NAME := eval`, `REGISTRY := ...`, `include ../../common.mk` (Dockerfile present → image step).
- **`Dockerfile`** — modeled on the sage Dockerfile: python base, `COPY src/ /opt/eval`, install deps, entrypoint into the eval script.
- **`src/`** — the eval code baked into the image (e.g. `eval.py`), reads params from CLI args, writes a single `results.json`. It does **not** print the artifact line — the fixed output path is known to the step. `requirements.txt` lives beside the Dockerfile (build-time only; not bundled into `step/`).
- **`step-template.yaml`** — `type: EVAL`; `config.eval_config` (model_path, tasks, output_dir, batch_size); launcher `type: skypilot` with:
  - `image_id: docker:${IMAGE_REF}` (**Makefile-time** substituted to the published ref)
  - `run:` invokes the baked entrypoint with `{{ config.eval_config.* }}`, then **emits the `LLMB_ARTIFACT_ID:results` line itself** for `$OUTPUT_DIR/results.json` (single fixed output ⇒ step registers it, not the workload)
  - `outputs.optional.results: { type: dataset }`
  - `monitors: { skypilot_monitor: { ref: space://monitors/skypilot } }`

### 4. `steps/.gitignore` (new)
Ignore the generated deployable bundles across all steps: `*/*/step/`.

### 5. `steps/README.md` (expand)
Add: the generation pipeline diagram, the canonical variable + target contract (from
`common.mk`), the two step-type recipes (byoc vs eval), how `envsubst`'s allowlist protects
runtime Jinja, and the `file:///abs/.../step` reference usage in a `build.yaml`.

## Verification
1. **byoc**: `cd steps/byoc/skypilot && make step` → `step/step.yaml` + `step/src/` produced.
   Confirm the rendered YAML parses:
   `python -c "from gbserver.types.stepconfig import StepConfig; StepConfig.from_yaml('steps/byoc/skypilot/step/step.yaml')"`.
2. **eval render**: `cd steps/eval/skypilot && make step IMAGE_TAG=test` (skip push by
   overriding `publish-image`, or run with a local-only tag) → confirm `step/step.yaml` has
   `image_id: docker:quay.io/your-org/gb-step-eval:test` (Makefile-time substitution worked,
   runtime `{{ }}` elsewhere untouched); parse with `StepConfig.from_yaml`. Confirm the `run:`
   block still contains the `LLMB_ARTIFACT_ID:results` echo (emitted by the step, not the code).
3. **eval image**: `make image` builds the Dockerfile locally with podman (no push) to prove
   the Dockerfile is valid; `make publish-image` is manual (needs `podman login quay.io`).
   `python src/eval.py --model-path x --output-dir /tmp/ev` writes `results.json` and prints
   **no** artifact line.
4. **negative/allowlist check**: grep the rendered `step.yaml` files to confirm `{{ ... }}` and
   `${SKYPILOT_*}`/`$(...)` in `run:`/`setup:` survived rendering unmodified.
5. End-to-end SkyPilot execution needs a real cluster and is out of local scope; note in README
   how to reference `file:///abs/.../steps/byoc/skypilot/step` from a `build.yaml` and run via the
   `gbserver` MCP tools when a cluster is available.
