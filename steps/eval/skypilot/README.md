# eval (SkyPilot)

Evaluation step for SkyPilot clusters. Its evaluation code and dependencies are
baked into a **custom image** built from [`Dockerfile`](Dockerfile), published to
a registry, and referenced from the generated `step.yaml` via `image_id`. The
`run` block invokes the baked entrypoint ([`src/eval.py`](src/eval.py)) with
parameters from `config.eval_config`, then registers the single results file as
the step's output.

This is a custom-image counterpart to the public-image
[byoc](../../byoc/skypilot/README.md) step. It is *generated* from the sources in
this directory by the shared Makefile conventions — see the framework overview:
[steps/README.md](../../README.md).

> The shipped `eval.py` is a **stdlib-only placeholder** — it writes a
> `results.json` with placeholder metrics but performs no real evaluation.
> Replace the body of `evaluate()` with a real harness; the CLI argument contract
> and the fixed `results.json` output path are what the step depends on.

## Who emits the artifact line?

This step demonstrates the **preferred** pattern for a workload with a single,
fixed output: **the step registers the output, not the workload.**

- `eval.py` always writes its results to `<output-dir>/results.json` — a path the
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

`eval` is a generated bundle referenced by an **absolute `file://` URI** to the
`step/` directory produced by `make step`:

```yaml
steps:
  - step_uri: file:///abs/path/to/steps/eval/skypilot/step
```

## Config contract (`eval_config`)

All fields live under the step's `config.eval_config` and are templated into the
`run` block as CLI arguments to `eval.py`.

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
  not by `eval.py`. Bind a matching `outputs.results` on the target to persist it.

## Env vars the step provides to your commands

Exported by the SkyPilot launcher into `run`:

| Variable | Set when | Meaning |
|---|---|---|
| `$GB_BUILD_WORKDIR` | always | Per-run workdir; the run script's initial CWD. Relative `output_dir` values resolve here, giving per-run isolation. |
| `$GB_SHARED_WORKDIR` | env sets `shared_workdir` | Env-level shared dir mounted on every worker, for cross-step state. |
| `$GB_BUILD_ID`, `$GB_TARGETRUN_ID` | run metadata present | Build / target-run identifiers. |
| `$GB_SKYPILOT_LAUNCH_ID`, `$GB_SKYPILOT_CLUSTER_NAME` | always | This launch's id and the SkyPilot cluster name. |

The eval code and its Python interpreter come from the **image** (built from
`Dockerfile`), not from a runtime venv.

## Building, publishing, and deploying the step

Because a `Dockerfile` is present, this is an image step: `make all` runs
`image` → `publish-image` → `step`. For the full target list, variables, and
[registry credentials](../../README.md#registry-credentials), see the shared
[Makefile target conventions](../../README.md#makefile-target-conventions).

Eval-specific notes:

- `REGISTRY` ships as a **placeholder** (`quay.io/your-org`) so the offline
  targets work out of the box; replace it in the `Makefile`, or override per
  release, e.g. `make all REGISTRY=quay.io/myorg IMAGE_TAG=0.1.0`.
  `make publish-image` against the placeholder will fail auth — set a real
  registry first. `IMAGE_TAG` defaults to the git short SHA.
- At `make step` time the published reference
  `$(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)` is substituted into the template's
  `image_id: "docker:${IMAGE_REF}"`.

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
        - step_uri: file:///abs/path/to/steps/eval/skypilot/step
          config:
            compute_config: { num_nodes: 1, num_gpus_per_node: 1 }
            eval_config:
              model_path: "ibm-granite/granite-4.0-h-350m"
              tasks: "hellaswag,arc_easy"
              output_dir: "output"
              per_device_eval_batch_size: 8
```

## Notes and limitations

- **Placeholder evaluation.** The shipped `eval.py` records config and writes
  placeholder metrics but runs no harness; add real dependencies to
  [`requirements.txt`](requirements.txt) (baked into the image) and a real loop.
- **Single, fixed output.** `results.json` is the one artifact, registered by the
  step. A workload whose output path varies at run time should instead print the
  `LLMB_ARTIFACT_ID` line itself.
- **Image is required at run time.** The step's `image_id` must point at a
  published, reachable image; run `make publish-image` (after `podman login`)
  before submitting a build that references it on a real cluster.
