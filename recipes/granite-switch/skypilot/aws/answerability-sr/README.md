# Shadow Residual answerability adapter (AWS EC2)

A recipe **template**: it trains a
[Shadow Residual](https://github.com/generative-computing/shadow-residual)
(SR) answerability adapter on `ibm-granite/granite-4.2-8b` and scores it, on
EC2 GPUs through SkyPilot, with the build's data moving between instances
through a shared EFS filesystem. It ships **without data**: you bring
training and eval sets in the formats below (see [Your data](#your-data)).

```bash
gb build start -f build.yaml --param RUN_ID=<unique> --param DATA_URI=<your data root>
```

`parameters.yaml` beside `build.yaml` is picked up automatically. Without
`DATA_URI` the build fails before it is submitted, with
`'DATA_URI' is undefined`.

SR runs two hidden-state streams per decoder layer: a frozen base stream that
stays bit-exact with the unadapted model, and a trainable stream carrying the
LoRA deltas plus a per-layer rank-R `cross_stream` injection from the base.
Only the adapter stream reaches the LM head.

## How it runs

```
train ──(adapter)──> eval
```

Every step runs on its own EC2 instance in the region of the EFS
(`us-east-1` for the default environment template; the environment pins the
`byoc` and `hfpull` steps there):

| Step | Instance | What it does |
|---|---|---|
| hfpull × 2 (per target) | small CPU instance | Downloads your data root and the base model into the EFS hf cache; later pulls find them there. |
| `train` | `ACCELERATORS` (1×H100 by default) | Clones shadow-residual, writes the SR config, runs `sr-train`, writes the adapter to `$ADAPTER_ROOT/$RUN_ID`. |
| `eval` | `EVAL_ACCELERATORS` (1×H100) | Scores that adapter with `answerability_eval --mode peft`, writes the report to `$REPORT_ROOT/$RUN_ID`. |

`train` generates its SR training config from `parameters.yaml` and runs
`sr-train` (or, with `SR_NUM_PROCESSES` > 1, the same trainer under
`accelerate launch`). `eval` binds the adapter that produced and runs
`shadow_residual.eval.answerability_eval --mode peft`, which rebuilds the SR
base and attaches the adapter through stock PEFT. Both targets bind the same
base-model artifact: they have to, or the adapter is served on a different
graph than it was trained on, and nothing in the output would show it.

The adapter and the report are on the EFS, outside the per-run workdirs that
gbserver removes when a target ends. That is what lets `eval`, on a different
instance, read the adapter `train` wrote. A re-run with the same `RUN_ID`
clears and replaces that run's adapter and report.

## Your data

`DATA_URI` names one **root** that holds both datasets; two parameters name
the subdirectories inside it:

| Parameter | Default | Holds |
|---|---|---|
| `DATA_URI` | none (required) | the root: `hf:///datasets/<org>/<repo>` for a Hugging Face dataset repo, or `env://<path>` already on the EFS |
| `TRAIN_DATA_SUBDIR` | `train` | `train.jsonl` and `test.jsonl` (the validation set) |
| `EVAL_DATA_SUBDIR` | `eval` | `gold.jsonl`, `rag.jsonl`, `unanswerable_v2.{a,b,c}.jsonl`, `collection.txt`, `setting.txt` |

**Training rows** (`train.jsonl`, `test.jsonl`) are shadow-residual's chat
format, one JSON object per line, with the label as the assistant turn:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<label>"}],
 "documents": [...]}
```

`documents` (and `tools`) are optional and passed as they are to the chat
template, which renders them.

**Eval rows** are the format `answerability_eval.py` reads (see its module
docstring): a `messages` list with the user turn, optional prior turns, a
`documents` turn whose content is a list of strings, and an `answerability`
turn whose content is `answerable` or `unanswerable`. `collection.txt` and
`setting.txt` are Python lists, one entry per sample: the collection name and
the setting (`gold`/`rag`/`a`/`b`/`c`).

The train command checks that `train.jsonl` and `test.jsonl` exist, so a wrong
subdirectory fails in seconds. For an `hf://` repo the pull downloads the
**whole repo** to one cache directory and hands the step its root, which is
why the subdirectories are parameters and not part of the URI. Both targets
name the same repo, so the eval target's pull finds the cache warm. Three
slashes in `hf:///datasets/...` matter: with two, `datasets` parses as the
host. A private repo needs an `HF_TOKEN` secret (setup guide, step 3); the
token never appears in a recipe file or on a command line.

### The base model is an input too

`BASE_MODEL_URI` (`hf:///ibm-granite/granite-4.2-8b`, public) is an input of
both targets, not something the trainer downloads: the weights land in the
EFS cache once, and train and eval bind the same artifact. PEFT writes the
path it loaded the base from into `adapter_config.json`, which here is a cache
path on the EFS, so the train command rewrites `base_model_name_or_path` to
`BASE_MODEL_ID` after training. Keep the two naming the same weights.

## The parameters that are load-bearing in a non-obvious way

Every config section `sr-train` validates rejects unknown keys, so a
misspelled key is a startup error. These are the ones whose *values* decide
whether the run means anything:

- **`CROSS_STREAM_RANK` must stay set.** Listing `cross_stream` in
  `target_modules` is what engages SR's dual-stream forward. Drop it and the
  run degrades to plain LoRA without an error.
- **`k_proj` and `v_proj` are deliberately absent** from `target_modules`: SR
  computes K/V once from the frozen base stream and shares it, so a K/V LoRA
  delta has nowhere to land and the model rejects it.
- **`ASSISTANT_MARKER: <|im_start|>assistant\n<think></think>`.**
  granite-4.2's chat template has no `{% generation %}` tags, so the loss is
  masked by SR's `ResponseOnlyCollator`, whose default marker is Granite 4.1's
  `<|start_of_role|>assistant<|end_of_role|>`. That never occurs in a 4.2 row,
  so left at the default the loss mask is *wrong* rather than missing. 4.2's
  assistant opener carries an empty think block, so the marker includes it and
  supervision starts after `</think>`.
- **`LAST_CONTEXT_TOKEN: </think>`**, the last token of `ASSISTANT_MARKER`. It
  must be a single token (it is, on the 4.2 tokenizer); the trainer checks
  every row's supervised-region boundary against it, so the marker and this
  token change together.
- **`LAST_TOKEN: <|im_end|>`**, 4.2's end-of-sequence token.

`GRADIENT_CHECKPOINTING` is on: two hidden-state streams roughly double
activation memory for an 8B model at 8,192 tokens, and SR's forward is pure,
so recompute is exact.

## What to expect

- **Transfers to other data:** the parts tied to the model and to SR: the
  marker and token settings, rank 16 / alpha 32 / dropout 0.05 on q, o and the
  MLP projections plus a rank-16 `cross_stream`, learning rate 5e-5 with a
  cosine schedule, and an effective batch of 8.
- **Tied to the answerability task:** 3 epochs, `MAX_LENGTH` 8,192 for long
  multi-document prompts, and the eval itself: `answerability_eval` scores
  only the answerable / unanswerable label. For another task, replace the
  `eval` target.
- **Scores.** On an answerability eval set of this layout, an earlier SR build
  on the same base model reached about 90% accuracy and weighted F1, at rough
  parity with a plain LoRA. Treat a number near chance as the loss mask or
  the marker being wrong. These scores are a sanity range for answerability
  data, not a target, and say nothing about another task.

### Checking the run did real work

A green build is not evidence on its own. Both targets already check the
artifact they produce (`adapter_config.json` after training, `eval_summary.md`
after eval), and the eval target prints the summary into the build log.
Beyond that:

- **train:** a training loss that moves across steps (`LOGGING_STEPS: 1`),
  `checkpoint-*` dirs at each epoch, and an `adapter_config.json` whose
  `target_modules` includes `cross_stream` and excludes `k_proj`/`v_proj`
  (the difference between having trained SR and plain LoRA), and whose
  `base_model_name_or_path` is `BASE_MODEL_ID` (the command prints it).
- **eval:** `eval_summary.{md,csv,jsonl}` and `result_df.csv` with the
  per-row predictions.

## Prerequisites

The one-time AWS setup is shared by the recipes in this folder and is in
[Running the granite-switch recipes on AWS](../README.md): your AWS key and
GPU quota, an EFS in your region, where to store the EFS id and the
credentials under `~/.granite.build/`, and the private copy of the
environment you point `ENVIRONMENT_URI` at
(`--param ENVIRONMENT_URI=space://environments/skypilot-local/aws-gb-shared`).
For this recipe specifically, your data in the [formats above](#your-data),
and an `HF_TOKEN` secret if it is in a private Hugging Face repo.

**GPU memory:** SR holds two hidden-state streams, so an 8B base does not fit
on a 24 GB L4 even at `MAX_LENGTH=2048`. A run on g6.4xlarge (1×L4) cloned,
installed and built the SR base, then ran out of memory on the first backward
pass. Use a GPU with at least 48 GB (L40S, A100-80GB, H100, H200).

## GPUs

`ACCELERATORS` and `EVAL_ACCELERATORS` pick the GPUs; SkyPilot picks the
instance unless `INSTANCE_TYPE` / `EVAL_INSTANCE_TYPE` names one (another size
with the same GPUs, when the cheapest one has no capacity).

| Setup | `ACCELERATORS` | `SR_NUM_PROCESSES` | `GRADIENT_ACCUMULATION` | Notes |
|---|---|---|---|---|
| One GPU (default) | `H100:1` (p5.4xlarge) | `1` | `8` | `sr-train`. Effective batch 1 × 1 × 8 = 8. |
| 8 GPUs, DDP | `H200:8` or `H100:8` | `8` | `1` | `accelerate launch --multi_gpu`, one process per GPU. Effective batch 8 × 1 × 1 = 8. |

`SR_NUM_PROCESSES` must equal the GPU count in `ACCELERATORS`. Above 1, the
train runs the same trainer, `shadow_residual.training.train`, under
`accelerate launch --num_processes N --multi_gpu --mixed_precision bf16`, the
DDP launch shadow-residual's README documents. Each GPU holds a full copy of
the 8B base, which fits in 80 GB with `GRADIENT_CHECKPOINTING: true`.
**Untested:** no SR run has used DDP yet. Smoke-test it (below) before a full
run. FSDP (shadow-residual's `fsdp_4gpu.yaml`) is not wired in.

A smoke run, to try the plumbing before a full run:

```bash
gb build start -f build.yaml --param RUN_ID=sr-smoke1 --param DATA_URI=<your data root> \
  --param MAX_STEPS=4 --param EVAL_DATASET_SIZE=20
```

## Parameters

The comments in [`parameters.yaml`](parameters.yaml) explain each value. The
ones you are most likely to set:

| Parameter | Default | Meaning |
|---|---|---|
| `RUN_ID` | `sr-r1` | Names the build, the adapter dir and the report dir. |
| `DATA_URI` / `TRAIN_DATA_SUBDIR` / `EVAL_DATA_SUBDIR` | none / `train` / `eval` | Your data; see [Your data](#your-data). |
| `ENVIRONMENT_URI` | `space://environments/skypilot/aws/gb-shared` | The us-east-1 EFS template, with a placeholder `file_system_id`. Point it at your own copy; see [The EFS filesystem](../answerability-lora/README.md#the-efs-filesystem). |
| `SR_REPO` / `SR_REF` | the public shadow-residual repo / `main` | The `byoc` step clones it without authentication, so it must be public. Pin `SR_REF` to a tag or SHA for runs you will compare. |
| `MAX_STEPS` / `NUM_EPOCHS` | `0` / `3` | A positive `MAX_STEPS` caps training. |
| `ADAPTER_ROOT` / `REPORT_ROOT` | `/mnt/gb-shared/granite-switch/answerability-sr/{adapters,reports}` | Where the adapter and the report are written, on the EFS, under `$RUN_ID`. |
| `ACCELERATORS` / `INSTANCE_TYPE` | `H100:1` / empty | The train's GPUs, and optionally its instance type. |
| `SR_NUM_PROCESSES` | `1` | Training processes; see [GPUs](#gpus). |
| `EVAL_ACCELERATORS` / `EVAL_INSTANCE_TYPE` | `H100:1` / empty | The eval's GPUs, and optionally its instance type. |
| `EVAL_DATASET_SIZE` | `100000` | Rows loaded per eval file; lower it for a quick pass. |
| `POLL_INTERVAL_SECONDS` | `120` | How often gbserver polls each step. |

## Limits

- **No container image for now (`IMAGE: ""`).** The `byoc` step `git clone`s
  the repo before `setup_command` runs, and the `pytorch/pytorch` images have
  no `git`, so with such an image the step fails at setup with
  `bash: git: command not found`
  ([#529](https://github.com/ibm-granite/granite.build/issues/529)). With an
  empty `IMAGE` the step runs on the bare EC2 instance, which has `git`, and
  `setup_command`'s `pip install '.[train]'` brings torch from PyPI (a few
  minutes per target). Once `byoc` installs `git` itself, set
  `IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`. shadow-residual
  pins `transformers<5.10` and `peft==0.19.1`, so it cannot share the
  hf-peft-sft image either way.
- **No baseline arm.** There is no `--mode base` target to compare the adapter
  against in the same build; add a third target running
  `answerability_eval --mode base` on the same `DATA_URI` / `EVAL_DATA_SUBDIR`.
- **No dependency caching.** `byoc` installs from PyPI on every run.
- **EFS traffic is not encrypted in transit**, and the EFS root is
  world-writable; see [the LoRA recipe on AWS](../answerability-lora/README.md#limits).
