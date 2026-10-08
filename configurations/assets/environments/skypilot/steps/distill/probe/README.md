# probe (SkyPilot / LSF)

Runs one of three cheap checks on a checkpoint, used to settle questions before a
distillation run is spent on them. It trains nothing and produces no artifact: the
answer is a `PROBE VERDICT` line in the target's log.

| probe | question | needs |
|---|---|---|
| `load-student` | does the model load in this image, with FlashAttention-2 and with eager attention? | 1 GPU |
| `tokenizer` | does the checkpoint's declared `tokenizer_class` change how text is split into tokens? | CPU |
| `tokenizer-fit` | which way of splitting do the checkpoint's weights prefer, measured by NLL per byte? | 1 GPU |

> **Developing or testing this step?** See `steps/distill/probe/skypilot/README.md` in
> the granite.build repository for how the step is generated, tested, and published.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/probe` URI:

```yaml
steps:
  - step_uri: space://steps/distill/probe
```

## Config contract (`probe_config`)

All fields live under the step's `config.probe_config`.

### Required

| Field | Type | Purpose |
|---|---|---|
| `probe` | string | Which probe to run: `load-student`, `tokenizer` or `tokenizer-fit`. Any other value, including the empty default, fails the step. |
| `checkpoint` | string | An HF checkpoint directory. It is only read: the tokenizer probes edit a scratch copy. An empty value fails the step. |

### Optional

| Field | Type | Purpose |
|---|---|---|
| `corpus` | string | A JSONL of `{"messages": [...]}` rows. **Required for `tokenizer-fit`**, which fails without it; ignored by the other two probes. Default: `""`. |
| `rows` | integer | `tokenizer-fit` only: how many rows of `corpus` to score; at least 1. Default: `200`. |
| `python` | string | Interpreter in the image that runs the probe. Default: `/stage/.venv/bin/python`. |

## Inputs and outputs

### Inputs

The step declares no `inputs:`. It reads the `checkpoint` directory and, for
`tokenizer-fit`, the `corpus` file, both given as plain paths in `probe_config` that
must be visible on the compute node (for example under `/proj/...` on BlueVela). A
target that wants to probe an upstream target's output can set either field to
`{{ bindings.<name>.binding.path }}` after declaring the input on the target.

### Outputs

None. The step registers no artifact. The result is the `PROBE VERDICT` line in the
workload log (along with the other `PROBE ...` lines each probe prints):

| probe | verdict line | meaning |
|---|---|---|
| `load-student` | `PROBE VERDICT loaded=flash_attention_2,eager` | the attention implementations that loaded; `loaded=NONE` means neither did |
| `tokenizer` | `PROBE VERDICT differing=N/M` | `N > 0`: the declared class splits text differently, so any eval recorded with it needs a tokenizer-control comparison |
| `tokenizer-fit` | `PROBE VERDICT nll_per_byte asis=… pinned=… prefers=… ratio=…` | `prefers=pinned`: the declared class is the mismatch and pinning fixes it; `prefers=asis`: pinning would be a regression |

The target succeeds whatever the verdict is. A wrong answer is a finding to read, not
a failure. It fails only on bad arguments, for example `tokenizer-fit` without a
`corpus`, or when the probe itself crashes.

## Working directory and paths

`run` starts in the step's per-run working directory. The step's `src/` directory is
mounted at `./src` there, and `run` invokes `<python> ./src/distill_probe.py <probe>
--checkpoint ... --corpus ... --rows ...`. `checkpoint` and `corpus` are passed through
unchanged, so give absolute paths. The step writes nothing to the working directory;
the tokenizer probes make their scratch copy of the checkpoint in a temporary
directory.

## Example build.yaml

One target per probe. All three can run against the same checkpoint in one build:

```yaml
granite.build:
  name: probe-example
  targets:
    load-student:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      steps:
        - step_uri: space://steps/distill/probe
          config:
            compute_config: {num_nodes: 1, num_gpus_per_node: 1}
            launcher_config:
              resources: {accelerators: "H100:1", cluster: "bluevela", zone: "normal", memory: 256}
            probe_config:
              probe: load-student
              checkpoint: /proj/run/checkpoints/epoch_hf_2
    tokenizer:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      steps:
        - step_uri: space://steps/distill/probe
          config:
            compute_config: {num_nodes: 1, num_cpus_per_node: 2}
            launcher_config:
              resources: {cluster: "bluevela", zone: "normal", memory: 32}
            probe_config:
              probe: tokenizer
              checkpoint: /proj/run/checkpoints/epoch_hf_2
    tokenizer-fit:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      steps:
        - step_uri: space://steps/distill/probe
          config:
            compute_config: {num_nodes: 1, num_gpus_per_node: 1}
            launcher_config:
              resources: {accelerators: "H100:1", cluster: "bluevela", zone: "normal", memory: 256}
            probe_config:
              probe: tokenizer-fit
              checkpoint: /proj/run/checkpoints/epoch_hf_2
              corpus: /proj/run/corpus/eval.jsonl
              rows: 200
```

## Notes and limitations

- **Needs** torch and transformers from the image, one GPU for `load-student` and
  `tokenizer-fit`, and no trainer source. Because it uses no trainer source, the step
  does not carry the `code_config` / source-delivery contract of the other distill
  steps.
- **No `HF_HUB_OFFLINE`.** The step deliberately does not set it. `load-student` asks
  whether transformers can fetch its kernels lazily in this image, and an offline hub
  would answer a different question.
- **What each probe does, and why,** is in [`src/distill_probe.py`](src/distill_probe.py).
