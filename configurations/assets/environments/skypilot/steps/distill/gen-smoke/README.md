# gen-smoke (SkyPilot / LSF)

Asks whether a trained model still writes usable text. It greedy-decodes eight raw
prompts (code in five languages and three short prose prompts) from each checkpoint in a
ladder. For each checkpoint it reports the fraction of generated lines that sit inside a
run of three or more identical lines, and by default fails the target when the last
checkpoint is degenerate.

> **Developing or testing this step?** See `steps/distill/gen-smoke/skypilot/README.md` in
> the granite.build repository for how the step is generated, tested, and published.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/gen-smoke` URI:

```yaml
steps:
  - step_uri: space://steps/distill/gen-smoke
```

## Config contract (`gen_smoke_config`)

All fields in the first two tables live under the step's `config.gen_smoke_config`.

### Required

| Field | Type | Purpose |
|---|---|---|
| `rungs` | list | One `"<step>:<hf_model_dir>"` string per checkpoint, in ladder order; `<step>` is an integer. The **last** entry is the one `gate_final_rung` judges. With the empty default `src/gen_smoke.py` fails with `FATAL: no rungs given`. |

### Optional

| Field | Type | Purpose |
|---|---|---|
| `max_repetition` | number | A checkpoint is degenerate when its mean looped fraction is above this. Default: `0.15`. |
| `new_tokens` | integer | Greedy tokens generated per prompt. Default: `256`. |
| `gate_final_rung` | boolean | `true` fails the target when the **last** rung is degenerate; `false` only reports. Default: `true`. |
| `output_dir` | string | Where `repetition.json` is written. A relative path lands under `GB_BUILD_WORKDIR`. Default: `gen-smoke`. |
| `python` | string | Interpreter in the image. Default: `/stage/.venv/bin/python`. |

### Top-level step config

| Field | Type | Purpose |
|---|---|---|
| `compute_config` | mapping | Node and GPU counts; the step needs one GPU, e.g. `{num_nodes: 1, num_gpus_per_node: 1}`. |
| `launcher_config` | mapping | SkyPilot launcher overrides, notably `resources` (`accelerators`, `cluster`, `zone`, `memory`) and `image_id`. |
| `poll_interval_seconds` | integer | Read by the shared `space://monitors/skypilot` monitor. Default: `300`. |
| `log_retrieval_mode` | string | Read by the shared monitor. Default: `on_completion`. |

## Inputs and outputs

### Inputs

The step declares no named inputs (`inputs: {allow_unknown: true}`): its rungs are a
variable-length list, so there is no fixed set of names to declare. It reads the model
directories named in `rungs`, each of which must be a complete local HF export — the
`hf_model` output of the `space://steps/distill/hf-export` step. A recipe declares one input
per export on the target (`binding: export-<N>.hf_model`) and lists the same directories in
`rungs`. Binding the export targets as inputs makes gen-smoke wait for every checkpoint it
reads.

### Outputs

| Output | Type | What it is |
|---|---|---|
| `repetition_report` | `fileset` | `<output_dir>/repetition.json`: the threshold and, per rung, the step, path, mean and worst looped fraction, mean adjacent-repeat rate, the `degenerate` verdict, and each prompt's completion (first 400 characters) with its looped fraction. |

`src/gen_smoke.py` prints the marker once `repetition.json` is written, before the gate is
applied, so the report is registered even when the final rung fails the target:

```
GB_ARTIFACT_ID:repetition_report GB_ARTIFACT_PATH:<absolute output_dir>/repetition.json
```

Declare `repetition_report` on the target (typically `uri: "env://{{ binding.path }}"`,
`type: fileset`).

## Working directory and paths

- The run block sets `WORK="${GB_BUILD_WORKDIR:-$PWD}"`. A relative `output_dir` is
  absolutised to `$WORK/<output_dir>` and created before the script runs.
- Rung paths are passed to the script unchanged. Use absolute paths.
- The step's `src/` directory is mounted at `./src`; the workload is
  `<python> ./src/gen_smoke.py <output_dir>/repetition.json <max_repetition> <new_tokens> <gate_final_rung> <rung>...`.
- `HF_HUB_OFFLINE=1` and `HF_HOME=/opt/hf-cache`. Nothing is fetched from the Hub.
- No trainer source is cloned: the step carries no `code_config`.

## Example build.yaml

One `gen-smoke` target for a two-rung ladder. `export-500` and `export-1000` are targets
running the `space://steps/distill/hf-export` step; they are elided here.

```yaml
granite.build:
  name: gen-smoke-example
  version: 0.0.1
  targets:
    # train, export-500, export-1000: elided.

    gen-smoke:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        export_500:  {binding: export-500.hf_model}
        export_1000: {binding: export-1000.hf_model}
      outputs:
        repetition_report:
          uri: "env://{{ binding.path }}"
          type: fileset
      steps:
        - step_uri: space://steps/distill/gen-smoke
          config:
            compute_config: {num_nodes: 1, num_gpus_per_node: 1}
            launcher_config:
              resources: {accelerators: "H100:1", cluster: "bluevela", zone: "normal", memory: 256}
            gen_smoke_config:
              rungs:
                - "500:/proj/run/export-500"
                - "1000:/proj/run/export-1000"
              output_dir: /proj/run/gen-smoke
```

## Notes and limitations

### Why this step exists

Runs of identical lines are the failure a distilled model showed in build df8512e0: one
line repeated about fifty times running. No training-time metric caught it. A few minutes on one GPU here is
much cheaper than finding it in a full eval.

### Only the last rung gates

Only the last rung can fail the target, because it is the checkpoint a later target would
pick up. A degenerate earlier rung is a finding to read in the report, not a reason to
fail a build whose later checkpoints may be fine. Set `gate_final_rung: false` for a
recipe whose job is to measure a whole ladder.

### What the verdict measures

- The verdict is on runs of three or more identical lines, not on adjacent repeats. The
  adjacent-repeat rate is reported but never gates: a correct short function can score
  high on it. Why, and the measurements behind the threshold, are in `src/gen_smoke.py`.
- The prompts are raw completions with no chat template, matching how the benchmark that
  caught the collapse (MultiPL-E) prompts the model.

### Log output

The log has one `GEN-SMOKE step=… looped=… DEGENERATE|ok` line per rung, then a table of
all of them.

### Runtime

- The step runs with `HF_HUB_OFFLINE=1`. Each `<hf_model_dir>` must be a complete local
  export.
- Needs one GPU, which the build supplies, and no trainer source. It runs in
  `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` (whose venv carries
  torch and transformers), on SkyPilot's LSF backend only (`subtypes: [lsf]`).
- One run covers the whole ladder rather than one per rung: for a 350m model, loading
  several checkpoints serially on one GPU is still minutes.
