# eval (SkyPilot)

Evaluation step for SkyPilot clusters. Its evaluation code is baked into a container
image; the `run` block invokes that entrypoint with parameters from `config.eval_config`
and registers the single results file as the step's output.

> **This is an exemplar, not a working evaluator.** The shipped evaluation script
> ([`src/eval.sh`](src/eval.sh)) is a **placeholder** — it writes a `results.json`
> recording its parameters but performs no real evaluation. When you implement eval for
> real, replace the script body with a real harness (and give the image a suitable
> runtime + dependencies); the flag contract and the fixed `results.json` output path
> are what the step depends on.

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

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/eval` URI:

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

## Working directory and paths

`run` starts in the step's per-run **working directory**, so the step never needs to
know its absolute location. Relative `output_dir` values resolve there, giving per-run
isolation; derive an absolute path with `$(pwd)` when you need one.

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
  evaluation loop plus a base image that carries its runtime and dependencies.
- **Single, fixed output.** `results.json` is the one artifact, registered by the
  step. A workload whose output path varies at run time should instead print the
  `LLMB_ARTIFACT_ID` line itself.
- **Image is required at run time.** On a real remote cluster the evaluation image must
  be published and reachable before submitting a build.
