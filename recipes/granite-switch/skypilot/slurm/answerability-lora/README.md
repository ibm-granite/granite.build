# Answerability LoRA / aLoRA adapter (hf-peft-sft, SkyPilot/SLURM)

Trains an answerability adapter on `ibm-granite/granite-4.2-8b`, either a
regular LoRA (`METHOD=lora`, the default) or an activated LoRA
(`METHOD=alora`, see [aLoRA](#alora)), with the [`hf-peft-sft`](../../../../../steps/train/hf-peft-sft/skypilot/USAGE.md)
step, on SkyPilot/SLURM. It registers the **adapter**, not a merged
model, as the `adapter` artifact. It runs on any SLURM cluster SkyPilot can
reach: everything site-specific comes from a private site file (see
[Setup](#setup)).

```bash
gb build start -f build.yaml --param-file "$SITE" --param RUN_ID=<unique>
```

The build has two targets:

```
prepare ──(dataset)──> train
```

- **`prepare`** (CPU, on `CPU_PARTITION`) pulls the answerability parquet from the
  private HF dataset repo named by `SOURCE_DATA_URI` and writes a prompt/completion
  copy to the cluster's shared filesystem. The `hf://` URI ends in a path inside the repo, so
  only that directory is downloaded, not the whole repo. It also binds the
  base model, for its tokenizer.
- **`train`** (`NUM_GPUS`, 4 by default, of the site's `GPU_ACCELERATOR`, with DDP) stages the base model, binds `prepare.dataset`,
  and trains the adapter.

## Setup

**Requires `gb` v0.3.10 or later**, for layered `--param-file`
([#663](https://github.com/ibm-granite/granite.build/issues/663), PR
[#665](https://github.com/ibm-granite/granite.build/pull/665)): the site file
is layered over the recipe's `parameters.yaml`, not in place of it. The setup
commands run from the repo root; the `gb build start` commands, from this
directory.

**1. The site file.** Copy the example to a private location outside this repo
and fill it in. The real one names your hosts, users, accounts, paths and data,
so never commit it.

```bash
mkdir -p ~/.granite.build/local-env
cp recipes/granite-switch/skypilot/slurm/answerability-lora/site.example.yaml \
  ~/.granite.build/local-env/granite-switch-slurm-site.yaml
SITE=~/.granite.build/local-env/granite-switch-slurm-site.yaml
```

Its comments describe each key. The `Environment` keys describe the cluster:
partitions, GPU constraint, account and QOS, shared paths and container mounts,
login nodes, and the **name** of the space secret holding the SSH key. The
`Recipe` keys are what `parameters.yaml` used to default to one cluster's
values: `ENVIRONMENT_URI`, `SOURCE_DATA_URI`, `DATA_ROOT`, `ADAPTER_ROOT`, and
`GPU_ACCELERATOR`, the GPU model (the recipe keeps the count, `NUM_GPUS`).
`prepare` also reads `CPU_PARTITION`. `parameters.yaml` has none
of these, so a run without a site file fails to render instead of running
somewhere else.

**2. The environment.** Render the shared SLURM template with the site file:

```bash
scripts/setup-local-space.sh \
  --template recipes/environments/skypilot-slurm.template.yaml \
  --site "$SITE" --name <env-name>
```

It writes the environment into the standalone space
`~/.granite.build/local-space` (or, with `--out DIR`, into a private repo
backing a shared gbserver) and sets `ENVIRONMENT_URI` in the site file to it.
[The distill recipes' Setup](../../../../distill/README.md#setup) describes
what it writes and its options; the same template serves both.

**3. gbserver**, for the standalone case:

```bash
gbserver standalone --space-dir ~/.granite.build/local-space
```

`gbserver standalone` sets `GB_ENVIRONMENT=STANDALONE` for itself only. Export
it in the shell you run `gb` from too (as in
[Getting started](../../../../../docs/getting-started.md#submit-a-build)):

```bash
export GB_ENVIRONMENT=STANDALONE
```

Without it, `gb` does not talk to the server as a standalone client, and the
server can reject the build with a 401 that suggests `gb auth login`.

**4. Secrets**, with gbserver running. The space needs the secret
`SSH_KEY_SECRET` names, holding the SSH private key (see
[Secrets in `cluster_ssh_configs`](../../../../../docs/environments/skypilot-slurm.md#secrets-in-cluster_ssh_configs)).
If the dataset repo is private (the one this recipe was built for is), it
also needs an `HF_TOKEN`: a read-only token from an account with read access
to the repo. The download runs on a cluster node, which can't see
your shell, so register the token once by name:

```bash
gb secret create HF_TOKEN --personal --value "$HF_TOKEN" --skip-version-check
gb secret list --personal --skip-version-check    # -> HF_TOKEN
```

`HF_TOKEN` is the credential name the Hugging Face assetstore looks for
(`token_secretname` in its `store.yaml`), so a secret under another name is not
found. The token never appears in `build.yaml`, `parameters.yaml`, or a command
line.

The source is a **dataset** repo, so the URI needs `datasets/`:
`hf:///datasets/<org>/<dataset>/...`. Without it, the name resolves to a
model repo and the Hub answers 404 "Repository not found".

## The data

The recipe was built for the answerability parquet at
`training_data/answerability/parquet_granite4.2_8b` in its source dataset repo
(the path `site.example.yaml` keeps):

| file | sha256 |
|---|---|
| `train.parquet` (~114 MB, 12,849 rows) | `b81f26558bd09933bd1125ee81fcf3116d2f1b9ae973efdbb30c94e9865f91a1` |
| `test.parquet` (~12 MB, 1,428 rows) | `7c9d38dc402f936f8b610a7c886be74c1cc45f5f4767df63c1eda396c8a05bca` |

`prepare` records these hashes in the dataset's own `README.md`. To pin the
data, replace `main` in the URI with a commit hash.

### What `prepare` does

The parquet holds pre-tokenized `input_ids` / `labels` / `attention_mask` next
to the conversational `prompt` and `completion` message lists they were
tokenized from. hf-peft-sft accepts TRL's `prompt` + `completion` format, so
`prepare` keeps exactly those two columns.

It must drop `input_ids`. TRL treats a dataset that already has `input_ids` as
tokenized and skips its own chat-template tokenization and completion masking.
The step's collator then has no `completion_mask`, and the run trains on the
whole sequence.

It also drops any row whose prompt doesn't end in a user turn, because the
completion must answer one. Six train rows (507, 695, 997, 5041, 7418, 10176)
end with an `agent` turn. If more than 1% of a split would be dropped this
way, `prepare` fails instead, since that points to a data problem rather than
stray rows.

Last, it drops rows longer than `MAX_LENGTH` (8,192) tokens, counted as TRL
counts them: prompt + completion through the base model's chat template. TRL
keeps the start of a row, so in a longer row the label at the end is cut off
and TRL drops the row anyway, silently. 853 train rows (up to 14,472 tokens)
and 76 test rows go, which leaves **11,990 train and 1,352 test rows**. The
source parquet had cut these rows at 8,192 tokens with no trained label, so
the original pipeline didn't train on them either. If more than 10% of a split
is too long, `prepare` fails: `MAX_LENGTH` is then too short for the data.

Every dropped row is logged and listed, by its row in the source parquet, in
the dataset's `README.md` (`dropped_rows`, `too_long_rows`).

The output is `$DATA_ROOT/$RUN_ID/data/{train,test}.parquet` on the shared
filesystem. `DATA_ROOT` must be under one of the site's `CONTAINER_MOUNTS` (or
`CLUSTER_WORKDIR`): SkyPilot's SLURM backend bind-mounts only those and home
into a container, not the whole shared filesystem. The same holds for
`ADAPTER_ROOT`.

### Checking the tokenization (optional)

`check_tokenization.py` replays the step's tokenization on the source rows
and compares the result with the stored `input_ids` / `labels`. It needs
`transformers` and `jinja2`, no GPU and no weights:

```bash
hf download <org>/<dataset> --repo-type dataset \
  --include 'training_data/answerability/parquet_granite4.2_8b/*' --local-dir .
python -I check_tokenization.py ibm-granite/granite-4.2-8b \
  training_data/answerability/parquet_granite4.2_8b/train.parquet 500
```

All rows should match. See [What is trained](#what-is-trained).

## What is trained

The loss is on the completion only. TRL's `completion_only_loss` defaults on
for `prompt` / `completion` data.

Granite 4.2's template ends a generation prompt with
`<|im_start|>assistant\n<think>\n`, but renders a finished assistant turn as
`<|im_start|>assistant\n<think></think>"answerable"<|im_end|>`. TRL starts the
completion where the two diverge, so the trained span is
`</think>"answerable"<|im_end|>\n` (or `"unanswerable"`). This should equal the
stored `labels`; `check_tokenization.py` verifies that.

## aLoRA

`METHOD=alora` trains an activated LoRA: the adapter acts only from the
invocation on, so a server can reuse the base model's cache for the prompt
and switch the adapter on just for the answer. It can't be merged into the
base model; load it as an adapter (vLLM needs a release with activated-LoRA
support).

```bash
gb build start -f build.yaml --param-file "$SITE" \
  --param METHOD=alora --param RUN_ID=<unique>
```

The adapter goes to `$ADAPTER_ROOT/answerability-alora/<RUN_ID>`, apart from
the LoRA adapters in `answerability-lora/`.

- **Invocation: `<|im_start|>assistant`**, Granite 4.2's ChatML assistant
  marker (tokens `[100256, 78191]`). It is the 4.2 form of the
  `<|start_of_role|>assistant<|end_of_role|>` marker used for the 4.0 / 4.1
  answerability aLoRA adapters. The 4.1 string does not work on 4.2: 4.2 has
  no `<|start_of_role|>` token, so it becomes 19 plain text tokens that occur
  in no row. On all 11,990 prepared train rows, the last occurrence of the
  4.2 invocation opens the answer turn
  (`…<|im_start|>assistant\n<think></think>"answerable"<|im_end|>`), which is
  where peft activates the adapter. The step re-checks the first 32 tokenized
  rows and fails before training if the invocation is missing from all of
  them.
- **Why not `<|im_start|>assistant\n`.** The Granite 4.2 query rewrite and
  query clarification aLoRA adapters use the marker with its newline (tokens
  `[100256, 78191, 198]`), and it fits this data equally well. But gbserver
  renders every step config string as a Jinja template, and Jinja drops one
  trailing newline, so a value ending in `\n` reaches the trainer without it.
  The aLoRA smoke run (build `911ee612`) asked for the newline and trained
  with `[100256, 78191]`. Until gbserver keeps trailing newlines
  (`keep_trailing_newline` in `src/gbserver/utils/template.py`, a change to
  every build, so out of scope here), the default is the form that actually
  arrives. The adapter then activates one token earlier than the QR/QC
  adapters, on the newline before `<think></think>`.
- **`gradient_checkpointing` is off for aLoRA.** Recomputing activations loses
  peft's aLoRA offsets and zeroes the gradients (peft issue #2826), so the
  build turns it off whenever `METHOD=alora`. That uses more GPU memory per
  row than the LoRA run.
- **Hyperparameters are the LoRA ones by default** (`LORA_R` 16, `LORA_ALPHA`
  32, `LEARNING_RATE` 5e-5, 3 epochs), so the first aLoRA run is directly
  comparable with the LoRA run. None of these is tuned for aLoRA on 4.2.

## Parameters

`parameters.yaml` holds the defaults; the rows marked *site file* have none
there.

| Parameter | Default | Meaning |
|---|---|---|
| `RUN_ID` | `it1-r1` | Names the build, the dataset dir (`$DATA_ROOT/$RUN_ID`), the adapter dir (`$ADAPTER_ROOT/answerability-$METHOD/$RUN_ID`) and the ClearML task. Use a new one per run: re-using one replaces that run's outputs. The dataset dir does not include the method, so builds running at the same time (a LoRA and an aLoRA run, say) need different `RUN_ID`s, or their `prepare` targets write the same files. |
| `ENVIRONMENT_URI` | *site file* | The environment both targets run in; `setup-local-space.sh` sets it. |
| `SOURCE_DATA_URI` | *site file* | The source parquet dir, `hf:///datasets/<org>/<dataset>/<revision>/<path>`. The path after the revision limits the pull to it. |
| `DATA_ROOT` | *site file* | Where `prepare` writes, on the shared filesystem: a plain path, not a URI. |
| `CPU_PARTITION` | *site file* | The partition `prepare` runs on; the environment sends its HF I/O jobs there too. |
| `PREP_IMAGE` / `PREP_CPUS` / `PREP_MEMORY` | the hf-peft-sft image / `2` / `16` | `prepare` runs in the step image (its venv has pyarrow), so the node's image cache serves both targets. If the CPU partition caps memory per CPU (`MaxMemPerCPU`), keep `PREP_MEMORY` ≤ cap × `PREP_CPUS`, or SLURM raises the CPU count and SkyPilot's `srun` aborts. |
| `BASE_MODEL_URI` | `hf:///ibm-granite/granite-4.2-8b` | Staged by an hfpull job on the CPU partition. If that job fails (it runs in a small cgroup), point this at an existing copy: `env://<shared-fs>/...`. |
| `DATASET_SPLIT` | `train` | `load_dataset` split; `train[:256]` for a smoke run. |
| `MAX_LENGTH` | `8192` | Tokens per row: `prepare` drops longer rows, and `train` uses it as `max_length`. |
| `METHOD` | `lora` | `lora` or `alora`. Also names the build, the ClearML task and the adapter subdirectory. |
| `ALORA_INVOCATION_STRING` | `<\|im_start\|>assistant` | Only for `METHOD=alora`. A trailing newline would be dropped (see [aLoRA](#alora)). |
| `LORA_R` / `LORA_ALPHA` / `LEARNING_RATE` | `16` / `32` / `5e-5` | Adapter rank, scale and learning rate. |
| `ADAPTER_ROOT` | *site file* | A plain path on the shared filesystem, so the adapter survives teardown. Each method gets its own subdirectory, `answerability-lora/` or `answerability-alora/`, so a LoRA and an aLoRA run with the same `RUN_ID` don't collide. |
| `GPU_ACCELERATOR` | *site file* | `train`'s GPU model, e.g. `A100`; the recipe appends `:<NUM_GPUS>`. |
| `NUM_GPUS` | `4` | DDP over that many GPUs. Effective batch `NUM_GPUS` × 1 × 8 = 32; another count trains at another effective batch. |
| `MEMORY` | `512` | Host GiB for `train`, set as `resources.memory` so it wins on every cluster. SLURM rejects a request above a node's `RealMemory`, so lower it on smaller nodes. |
| `NUM_TRAIN_EPOCHS` / `MAX_STEPS` | `3` / `-1` | A positive `MAX_STEPS` caps the run. |
| `CLEARML` | `false` | `true` turns tracking on and declares space secrets `CLEARML_API_HOST`, `CLEARML_API_ACCESS_KEY` and `CLEARML_API_SECRET_KEY`, which must exist. |

A smoke run, to try the plumbing on one GPU before the full run:

```bash
gb build start -f build.yaml --param-file "$SITE" \
  --param RUN_ID=smoke1 --param NUM_GPUS=1 \
  --param DATASET_SPLIT='train[:64]' --param MAX_STEPS=4
```

## What this recipe does not do

- **No evaluation, and no checkpoint per epoch.** hf-peft-sft does not
  evaluate and owns `save_strategy` (it passes `--save_strategy no`), so one
  adapter is registered at the end. `prepare` converts `test.parquet`, but
  `train` doesn't use it.
- **TRL re-tokenizes `prompt` / `completion`.** The step accepts only text
  formats, so the stored `input_ids` / `labels` are not used. Pre-tokenized
  input is planned for iteration 2.
- **Nothing is truncated.** `prepare` already dropped the rows longer than
  `MAX_LENGTH`, so `max_length` never cuts a row.
- **`use_reentrant` is set by the step** (`false` for LoRA + DDP), and
  `mixed_precision: bf16` is how this step asks for bf16.

## Findings from iteration 1

Runs on SkyPilot/SLURM, gbserver standalone, before hf-peft-sft was
merged (#596) and with the `peft-v0.21.2-4` image; the environment has since
been renamed (#590). Training loss here is only a plumbing check: the
completion is one label plus `</think>` and `<|im_end|>`, so loss falls to
near zero quickly and says nothing about held-out accuracy. No evaluation has
been run yet.

| Build | What | Result |
|---|---|---|
| `d6c10b3f` (smoke1) | LoRA, 1×H200, `train[:64]`, 4 steps | Path-limited HF pull OK (`hf download … --include <dir> --include <dir>/*`, 2 files). `prepare` never started: cpu partition memory cap (below). Cancelled. |
| `0f88a0f5` (smoke2) | LoRA, same | SUCCESS. 4 steps in 19.8 s, adapter registered. |
| `a255acdc` (it1-r1) | LoRA, 4×H200 DDP, full data, 3 epochs | Training completed (1,125 steps, 5,935 s, final loss ≈0.0015, mean 0.0866), adapter at `.../adapters/answerability-lora/it1-r1`. Build marked FAILED by gbserver (below); adapter not registered. |
| `911ee612` (asmoke1) | aLoRA, 1×H200, longest-row slice `train[11069:11133]`, 4 steps, no gradient checkpointing | SUCCESS. Rows up to 8,136 tokens fit without checkpointing; all 280 `lora_B` matrices nonzero (no #2826 zero gradients). Invocation stored as `[100256, 78191]`, not the requested `\n` form (below). |
| `2aaa4f17` (it1-a1) | aLoRA, 4×H200 DDP, 11,990 rows (over-length rows dropped by `prepare`), 3 epochs, invocation `<\|im_start\|>assistant` | SUCCESS. 1,125 steps in 5,462 s (≈8% faster than LoRA: no recomputation), final loss ≈0.0076, mean 0.0947; all `lora_B` nonzero; adapter registered at `.../adapters/answerability-alora/it1-a1`. |

What these runs found:

- **That cluster's CPU partition caps memory at 8 GiB per CPU.** Asking for more memory
  than 8 × CPUs makes SLURM raise cpus-per-task past SkyPilot's `srun --cpus`,
  and every job aborts at container start (`srun: fatal: cpus-per-task set by
  two different environment variables`). SkyPilot reports only "Failed to
  acquire resources" and retries. Fixed: `prepare` asks for 2 CPUs and 16 GiB.
- **853 train / 76 test rows exceed 8,192 tokens.** TRL keeps a row's start,
  so their label was cut and TRL dropped them silently: it1-r1 trained on
  11,990 rows, not 12,843. Fixed: `prepare` drops and records them, so the
  data a run trains on is stated, and LoRA and aLoRA runs see the same rows.
- **gbserver failed a running job when DNS dropped.** The laptop running
  gbserver lost DNS for the login nodes for ~24 min; the status checks failed,
  the retry handler treated that as a job failure, and gbserver marked the
  build FAILED and dropped the cluster while training kept running. The SLURM
  job then held its 4 H200s, idle, until cancelled by hand, and the two
  post-build cleanup VMs failed, orphaning
  `${shared_workdir}/builds/a255acdc-.../runs/*`. Not fixed here: run gbserver
  from a host with stable access to the login nodes, and check `squeue` after
  a FAILED build.
- **gbserver drops a trailing newline from config strings** (see
  [aLoRA](#alora)). Not fixed here; the aLoRA default avoids it.
