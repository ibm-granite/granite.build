# eval (SkyPilot)

Evaluation step for SkyPilot clusters. Its evaluation code is baked into a
**custom image** built from [`Dockerfile`](Dockerfile), published to a registry,
and referenced from the generated `step.yaml` via `image_id`. The `run` block
invokes the baked entrypoint ([`src/eval.sh`](src/eval.sh)) with parameters from
`config.eval_config`, then registers the single results file as the step's output.

This is a custom-image counterpart to the public-image
[byoc](../../byoc/skypilot/README.md) step. It is *generated* from the sources in
this directory by the shared Makefile conventions — see the framework overview:
[steps/README.md](../../README.md).

> **This is an exemplar, not a working evaluator.** The shipped
> [`src/eval.sh`](src/eval.sh) is a **placeholder shell script** — it writes a
> `results.json` recording its parameters but performs no real evaluation, so the
> image needs no Python or dependencies (just a minimal Fedora base). When you
> implement eval for real, replace the script body with a real harness and give
> the image a suitable runtime + dependencies (see
> [the base image](#building-publishing-and-deploying-the-step) below); the flag
> contract and the fixed `results.json` output path are what the step depends on.

## Who emits the artifact line?

This step demonstrates the **preferred** pattern for a workload with a single,
fixed output: **the step registers the output, not the workload.**

- `eval.sh` always writes its results to `<output-dir>/results.json` — a path the
  step already knows. It prints no Granite.build marker and has no dependency on
  the artifact convention.
- The `run:` block in `step.yaml`, after the eval command, emits the registration
  line for that known path:

  ```sh
  RESULT_FILE="$(cd "$OUTPUT_DIR" && pwd)/results.json"
  echo "LLMB_ARTIFACT_ID:results LLMB_ARTIFACT_PATH:${RESULT_FILE}"
  ```

Contrast this with a **training** workload, whose output path (a checkpoint dir)
is decided by the code at run time — there the *script* must print the line,
because only it knows the path. Prefer registering from `step.yaml` whenever the
output location is fixed and known ahead of time.

## Referencing the step

`make space` renders a self-contained Space into `space/` (see `SPACE_DIR` in the
[framework overview](../../README.md)). Point the build's Space at that directory
and reference the step by the stable `space://steps/eval` URI:

```yaml
steps:
  - step_uri: space://steps/eval
```

## Config contract (`eval_config`)

All fields live under the step's `config.eval_config` and are templated into the
`run` block as CLI arguments to `eval.sh`.

| Field | Type | Required | Purpose |
|---|---|---|---|
| `model_path` | string | **required** | Model to evaluate (path or hub id). Passed as `--model-path`. |
| `tasks` | string | optional (default empty ⇒ a `placeholder` task) | Comma-separated benchmark/task names. Passed as `--tasks`. |
| `output_dir` | string | optional (default `output`) | Directory the results file is written into; `results.json` is created here. Passed as `--output-dir`. |
| `per_device_eval_batch_size` | int | optional (default `8`) | Per-device eval batch size. Passed as `--batch-size`. |

## Inputs and outputs

- **Inputs** — the exemplar takes the model to evaluate as the `model_path`
  *config* string rather than a bound artifact input, so it declares no target
  `inputs:`. (To consume a resolved artifact instead, add a target input and
  reference its path in the `run` block.)
- **Outputs** — declared on the step as `outputs.optional.results` (`type:
  dataset`), a single file at `<output_dir>/results.json`. It is registered by the
  `run:` block (see [Who emits the artifact line?](#who-emits-the-artifact-line)),
  not by `eval.sh`. Bind a matching `outputs.results` on the target to persist it.

## Env vars the step provides to your commands

The SkyPilot launcher exports `$GB_BUILD_WORKDIR` into `run`: the per-run workdir
and the run script's initial CWD. Relative `output_dir` values resolve here, giving
per-run isolation.

The eval script runs inside the **image** (built from `Dockerfile`); the exemplar
is a shell script, so the image needs no Python interpreter or runtime venv.

## Building, publishing, and deploying the step

Because a `Dockerfile` is present, this is an image step: `make all` runs
`image` → `publish-image` → `space`. For the full target list, variables, and
[registry credentials](../../README.md#registry-credentials), see the shared
[Makefile target conventions](../../README.md#makefile-target-conventions).

To promote the step into the repo's committed assets tree
(`configurations/assets/environments/skypilot/steps/eval/`) and copy its Docker
build test into `test/steps/eval/skypilot/` so it is runnable from VSCode against
the published step, run `make publish-step`. See
[Two test modes](../../README.md#two-test-modes) for how the same test runs both
against the locally rendered `space/` (Mode 1, `make test`) and against the
published step (Mode 2, under `test/steps/`).

Eval-specific notes:

- `REGISTRY` ships as a **placeholder** (`quay.io/your-org`) so the offline
  targets work out of the box; replace it in the `Makefile`, or override per
  release, e.g. `make all REGISTRY=quay.io/myorg IMAGE_TAG=0.1.0`.
  `make publish-image` against the placeholder will fail auth — set a real
  registry first. `IMAGE_TAG` defaults to the git short SHA.
- At `make space` time the image reference
  `$(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)` is substituted into **both** launcher
  blocks — the Skypilot `image_id: "docker:${IMAGE_REF}"` and the Docker
  launcher's `image` (see below).

### Running locally with no publish (the `Docker` launcher)

The step's `step-template.yaml` also carries a **`Docker`** environment config
(a `docker` launcher running the same image and `eval.sh`). This lets the image
be **built and exercised locally with no registry publish**: `make test` renders
the Space and builds the image locally (`make image`), then the docker build test
in [`test/docker/`](test/docker/) runs it against the local Docker daemon. The
`docker` environment's `pull_policy` is `if-not-present`, so the just-built local
image is used as-is — no push, no pull, no container-capable cluster needed. Run
it (with the repo-root `.venv` active) via:

```sh
make -C steps/eval/skypilot test
```

> **Running the *committed* Mode-2 docker test — tag coupling.** The Mode-1 flow
> above always works because `make test` builds the image and renders the Space at
> the same commit, so their tags agree. The **committed** Mode-2 test
> ([`test/steps/eval/skypilot/docker/`](../../../test/steps/eval/skypilot/docker/))
> instead runs against the **published** `step.yaml` under
> `configurations/assets/…/steps/eval/`, whose `image`/`image_id` bake in
> `IMAGE_TAG` (the git short SHA by default) **as of the commit `make publish-step` was
> last run at**. Because `pull_policy` is `if-not-present`, that test only resolves
> if a local image at that exact tag exists — otherwise it tries to pull the
> placeholder `quay.io/your-org` registry and fails. So before running it, rebuild
> **and** re-publish at your current commit so both sides agree:
>
> ```sh
> make -C steps/eval/skypilot image publish-step   # builds gb-step-eval:<HEAD-sha> and re-renders the assets to match
> ```
>
> (Then commit the regenerated `step.yaml`.) Pin a stable `IMAGE_TAG` — e.g. `make
> … image publish-step IMAGE_TAG=local` on both sides — if you would rather the
> committed assets not drift per commit.

## Example build.yaml

```yaml
granite.build:
  name: eval-example
  version: 0.0.1
  targets:
    evaluate:
      environment_uri: space://environments/skypilot/aws
      outputs:
        results:
          uri: lh://prod/myspace/datasets/shared/eval-{{ run_metadata.targetsteprun_id | short_hash }}/1
      steps:
        - step_uri: space://steps/eval
          config:
            compute_config: { num_nodes: 1, num_gpus_per_node: 1 }
            eval_config:
              model_path: "ibm-granite/granite-4.0-h-350m"
              tasks: "hellaswag,arc_easy"
              output_dir: "output"
              per_device_eval_batch_size: 8
```

## Notes and limitations

- **Placeholder evaluation.** The shipped `eval.sh` records its parameters into
  `results.json` but runs no harness. Implementing eval for real means a real
  evaluation loop plus a base image that carries its runtime and dependencies
  (the current placeholder needs neither), then choosing the proper image.
- **Single, fixed output.** `results.json` is the one artifact, registered by the
  step. A workload whose output path varies at run time should instead print the
  `LLMB_ARTIFACT_ID` line itself.
- **Image is required at run time.** On a real remote cluster (the Skypilot
  launcher) the image must be **published and reachable** — run `make
  publish-image` (after `podman login`) before submitting such a build. The
  `Docker` launcher is the exception: it uses the **local** image, so `make
  image` (done for you by `make test`) is enough — no publish.
