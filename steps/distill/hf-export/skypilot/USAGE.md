# hf-export (SkyPilot / LSF)

Packages one `checkpoint-N` from a distillation run into a self-contained, downloadable
HuggingFace model directory. It selects the checkpoint, prunes resume state, normalises
the tokenizer and model config for publishing, and asserts the result loads. It is not a
weight converter, needs no GPU, and runs in a prebuilt image on SkyPilot's LSF backend
only.

> **Developing or testing this step?** See `steps/distill/hf-export/skypilot/README.md` in
> the granite.build repository for how the step is generated, tested, and published —
> including the end-to-end test that needs a BlueVela allocation.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/hf-export` URI:

```yaml
steps:
  - step_uri: space://steps/distill/hf-export
```

## Config contract (`export_config`)

All fields in the first two tables live under the step's `config.export_config`.

### Required

| Field | Type | Purpose |
|---|---|---|
| `train_output_dir` | string | The **trainer's** `output_dir` — the parent containing `checkpoint-25/`, `checkpoint-50/`, … — not a single checkpoint. With the empty default the step looks for `checkpoint-*` in the current directory and fails with "no checkpoint-* directory". |

### Optional

| Field | Type | Purpose |
|---|---|---|
| `dest` | string | Where the packaged model is written; this is what gets published as `hf_model`. A relative value resolves against `$GB_BUILD_WORKDIR`. Default: `hf_model`. |
| `checkpoint` | string | Which checkpoint to publish: a bare `checkpoint-N` (resolved under `train_output_dir`) or a path. Empty selects the **highest step number**, not the newest mtime. Default: `""`. |
| `padding_side` | string | `right` (publish-correct, the reason this step exists) \| `left` (knowingly publishing a generation-only artifact) \| `keep` (copy the trainer's value verbatim). Default: `right`. |
| `allow_unknown` | boolean | `false` refuses checkpoint files the keep list does not recognise; `true` drops them and records them in `export_manifest.json`. Default: `false` — but see [allow_unknown and distributed checkpoints](#allow_unknown-and-distributed-checkpoints). |
| `chat_template_thinking` | string | `keep` \| `default-off`: which generation prompt `apply_chat_template(..., add_generation_prompt=True)` hands out. Default: `keep`. |
| `verify` | boolean | Load the result with `AutoConfig`/`AutoTokenizer` before declaring success. Costs seconds; catches an export missing a file. Default: `true`. |
| `expect_tokenizer_from` | string | A tokenizer directory the export must agree with, token id for token id — normally the `retagged_student` output of the `space://steps/distill/tokenizer-align` step. Empty skips the check. Default: `""`. |

> **Wire `expect_tokenizer_from` rather than leaving it empty.** It is the only check in the
> pipeline that catches a checkpoint trained against a different tokenizer than the recipe
> believes, which no amount of "does it load" can detect.

### Source delivery (`code_config`)

The step imports the shared distillation package from a source checkout resolved at run
time. `code_config` is identical in every ported distillation step and is documented once —
see "Source delivery" in the `space://steps/distill/tokenizer-align` step's README. The
defaults clone a public repository at a pinned commit, so a recipe normally sets nothing
here.

| Field | Type | Purpose |
|---|---|---|
| `code_dir` | string | Existing checkout to use instead of cloning. Default: `""` (clone). |
| `expect_ref` | string | Commit `code_dir` must be at; a mismatch fails the step. Default: `a5d59bc45524a8d75706e20d44ae1a254f273f23`. |
| `repo` | string | Repository cloned when `code_dir` is empty. Default: `https://github.com/laminair/gb-steps-distillation.git`. |
| `ref` | string | Branch, tag or commit checked out after the clone. Default: the same commit as `expect_ref`. |
| `workdir` | string | Clone destination, relative to the working directory. Default: `distill-code`. |
| `token_secret` | string | Name of a space secret holding a read credential for a private `repo`. Default: `""` (public clone). |
| `python` | string | Interpreter that runs the workload. Default: `/stage/.venv/bin/python`. |
| `setup_command` | string | Bash run after the source resolves and before the workload. Default: `""` (skipped). |

### Top-level step config

| Field | Type | Purpose |
|---|---|---|
| `compute_config` | mapping | Node and CPU counts, e.g. `{num_nodes: 1, num_cpus_per_node: 4}`. No GPU is needed. |
| `launcher_config` | mapping | SkyPilot launcher overrides, notably `resources` (`cluster`, `zone`, `memory`) and `image_id`. |
| `poll_interval_seconds` | integer | Read by the shared `space://monitors/skypilot` monitor. Default: `300`. |
| `log_retrieval_mode` | string | Read by the shared monitor. Default: `on_completion`. |

## Inputs and outputs

### Inputs

The step declares no `inputs:` of its own. It reads `train_output_dir` and, optionally,
`expect_tokenizer_from` from `export_config`; a recipe supplies both by declaring inputs on
the target and passing their paths in with `{{ bindings.<name>.binding.path }}`:

- a `binding:` to the trainer target's `checkpoint` output (e.g. `train.checkpoint`, from
  the `space://steps/distill/gold` or `space://steps/distill/sft` step), passed as
  `train_output_dir`;
- a `binding:` to `align.retagged_student`, passed as `expect_tokenizer_from`. Binding it
  means the cross-check cannot name a tokenizer this build did not produce.

Nothing is deleted, moved or written under `train_output_dir`.

### Outputs

| Output | Type | What it is |
|---|---|---|
| `hf_model` | `model` | The `dest` directory: the kept model and tokenizer files plus `export_manifest.json`, which records the source checkpoint, what was kept, pruned and dropped, every normalisation, and the `padding_side` and `chat_template_thinking` policies. |

The step template prints the marker after the export script returns successfully:

```
GB_ARTIFACT_ID:hf_model GB_ARTIFACT_PATH:<absolute dest>
```

Declare `hf_model` on the target (typically `uri: "env://{{ binding.path }}"`,
`type: model`). Consumers are the `space://steps/distill/eval` step (`student_model`), the
`space://steps/distill/gen-smoke` step (one rung per export) and `space://steps/bfcl-eval`
(`model_path`).

## Working directory and paths

- The run block sets `WORK="${GB_BUILD_WORKDIR:-$PWD}"`. A relative `dest` is absolutised to
  `$WORK/<dest>` before the marker, because the monitor hands the path to the `env://` store,
  possibly from another host, and a relative `env:` URI is rejected at config load.
- `train_output_dir`, `checkpoint` and `expect_tokenizer_from` are passed to the script
  unchanged. Use absolute paths; bindings already resolve to absolute paths.
- The source is cloned into `$WORK/<code_config.workdir>` (default `distill-code/`), or taken
  from `code_config.code_dir`, and `$CODE_DIR/src` is prepended to `PYTHONPATH`.
- The step's `src/` directory is mounted at `./src`; the workload is
  `<code_config.python> ./src/export_hf_model.py`.
- `HF_HUB_OFFLINE=1` and `HF_HOME=/opt/hf-cache`: a missing local file fails here rather than
  silently fetching.

## Example build.yaml

The `export` target after the trainer. `align` and `train` are targets running the
`space://steps/distill/tokenizer-align` and `space://steps/distill/gold` steps; they are
elided here.

```yaml
granite.build:
  name: distill-export-example
  version: 0.0.1
  targets:
    # align, corpus, train: elided.

    export:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        checkpoint:
          binding: train.checkpoint
        # Bound so the cross-check below cannot name a tokenizer this build did not produce.
        tokenizer:
          binding: align.retagged_student
      outputs:
        hf_model:
          uri: "env://{{ binding.path }}"
          type: model
      steps:
        - step_uri: space://steps/distill/hf-export
          config:
            compute_config:
              num_nodes: 1
              num_cpus_per_node: 4
            launcher_config:
              resources:
                cluster: "bluevela"
                zone: "normal"
                memory: 64
            export_config:
              train_output_dir: "{{ bindings.checkpoint.binding.path }}"
              dest: "/proj/run/hf_model"
              # Needed for a multi-GPU training run; see Notes and limitations.
              allow_unknown: true
              expect_tokenizer_from: "{{ bindings.tokenizer.binding.path }}"
```

## Notes and limitations

### What it is, and what it deliberately is not

**It is not a weight converter.** The trainer's accelerate config sets
`zero3_save_16bit_model: true`, so every `checkpoint-N/` already contains HF-native
`model.safetensors` **at its root**, beside the DeepSpeed `global_step*/` shard tree. Anyone
expecting `zero_to_fp32.py` to be invoked here will not find it: that tree is **pruned, not
converted**.

It has five jobs, each of which exists because the trainer cannot do it:

| | Why the trainer can't |
|---|---|
| **SELECT** which `checkpoint-N` is the release | The trainer saves every `save_steps` and has no notion of "the one we ship". Eval may well prefer an earlier checkpoint over the last. |
| **PRUNE** resume state | `global_step*/`, optimizer, scheduler and RNG state are needed *during* a run and are noise in a release — and publishing them invites someone to resume from a release. ~4.7 G → ~679 M measured. |
| **NORMALISE** `padding_side`, strip `local_files_only`/`is_local` | The trainer sets `left` because it **generates** during on-policy rollout. Correct for a trainer, wrong for a published model, where left padding silently corrupts batched non-generative use. |
| **REWRITE** a `tokenizer_class` of `TokenizersBackend` to `PreTrainedTokenizerFast` | transformers 5 records its fast-tokenizer backend under a name transformers 4 cannot resolve, so a v4 consumer raises `ValueError: Tokenizer class TokenizersBackend does not exist` before reading a token. These steps run transformers 5.8.0, so every published model carried the pin — `bfcl-eval`, a v4 image, died on it (build 30a99c4b). `PreTrainedTokenizerFast` exists in both generations and loads `tokenizer.json` directly, so the spelling changes and the tokenizer does not. Only that one name is rewritten; any other class is left alone. |
| **ASSERT** the result loads | A directory missing one file it needs looks fine until a user downloads it. |

Grafting a tokenizer from somewhere else is **not** one of its jobs. If the trainer's tokenizer
disagrees with the corpus the run trained on, that is a defect to fail on, not to paper over at
packaging time — which is what `expect_tokenizer_from` is for.

**The prune is by omission.** Files reach `dest` through `shutil.copy2` against an explicit keep
list; nothing is deleted, moved or written under `train_output_dir`. So pointing this step at a
checkpoint you did not produce is safe.

The NORMALISE job also covers the model config and `tokenizer.json`; `src/export_hf_model.py`
documents each rule and the measurement behind it. In short: numeric values that transformers 5
nests under a `*_parameters` dict (notably `rope_parameters.rope_theta`) are also written at top
level, because transformers-4-schema consumers such as vLLM and `bfcl-eval` read only the
top-level key; the trainer's runtime `truncation`/`padding` state is stripped from
`tokenizer.json`; a `use_cache: false` left by the trainer (which disables the KV cache for
gradient checkpointing) is restored to `true`; and an infinite `time_step_limit` sentinel that
transformers 4 cannot read is dropped.

### `checkpoint` selection

Empty selects the **highest step number**, which is *not* newest mtime: a resumed run rewrites
older checkpoints' mtimes, so mtime ordering can pick one that is behind. Set it explicitly
(`checkpoint-500`) to publish a specific one — for instance when eval preferred an earlier
checkpoint over the last.

### `padding_side` and `chat_template_thinking`

- `padding_side: keep` is for reproducing an existing release byte-for-byte, or for diagnosing
  whether `padding_side` is the culprit. Whichever value is chosen, the change (or its absence)
  is recorded in `export_manifest.json`, so a published model can be traced back to what was
  done.
- `chat_template_thinking` is **not** the `think_policy` of the
  `space://steps/distill/corpus-prep` step, which decides whether reasoning traces stay in the
  training data. This key touches no data, only which generation prompt a consumer gets. `keep`
  copies `chat_template.jinja` verbatim and is the default, because which prompt a published
  model hands out is a decision about the artifact, and this step's job is to make it explicit
  and recorded, not to make it. `default-off` flips the default prompt to
  `<|im_start|>assistant\n<think></think>` instead of `<|im_start|>assistant\n<think>\n`; it is
  right for a student distilled on a corpus with no think traces, whose weights were never
  conditioned on the newline the reasoning prompt ends with.

### `allow_unknown` and distributed checkpoints

`false` is the default on purpose: the keep list is explicit, so a future transformers or TRL
release that starts writing a new file would otherwise have it silently published in a
directory users download. A failure here means "someone must classify this file" — a one-line
change, not an incident.

In practice, a checkpoint from a **multi-GPU** training run fails that check: HF Trainer writes
one `rng_state_<rank>.pth` per rank, while the classifier's known-prune list carries only the
single-process `rng_state.pth`. The distillation recipes therefore set `allow_unknown: true`;
`export_manifest.json` still records exactly what was dropped.

### Resume

A restarted recipe re-runs every step. If `dest` already holds an export made under the same
expectation (source checkpoint, kept files and policy flags), the step prints
`nothing to do` and still emits the `hf_model` marker. If `dest` holds an export made under a
different expectation, the step refuses and fails rather than overwrite it. An existing export
whose `config.json` no longer meets the current normalisation rules is re-exported rather than
skipped. The completion record is written only after `verify` passes.

### Why the export is required before eval

`bfcl-eval` needs an **HF-native model directory** loadable by vLLM, not a raw `checkpoint-N`
carrying DeepSpeed state. So `export` is not optional decoration between training and
capability evaluation — it is what makes the checkpoint loadable by anything but the trainer.

### Runtime

- **Type** `data_processing`; **environment** SkyPilot, LSF only (`subtypes: [lsf]`).
- **Image** `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv`, prebuilt; this
  step builds none.
- **No GPU**, by design — if this step appears to need one, something has been misdiagnosed.
  CPUs and memory come from the build.yaml.
