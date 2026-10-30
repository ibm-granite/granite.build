# Shadow Residual answerability adapter

Trains a [Shadow Residual](https://github.com/generative-computing/shadow-residual)
(SR) answerability adapter on `ibm-granite/granite-4.2-8b` and scores it, as two
targets joined by one artifact:

    train ──(adapter)──> eval

`train` runs `sr-train` against a config this recipe generates from
[`parameters.yaml`](parameters.yaml). `eval` binds the adapter it produced and
runs `shadow_residual.eval.answerability_eval --mode peft`, which rebuilds the
SR base and attaches the adapter through stock PEFT. Both targets bind the same
base-model artifact — they have to, or the adapter is served on a different
graph than it was trained on, and that is the one error nothing in the output
would reveal.

SR runs two hidden-state streams per decoder layer: a frozen base stream that
stays bit-exact with the unadapted model, and a trainable stream carrying the
LoRA deltas plus a per-layer rank-R `cross_stream` injection from the base. Only
the adapter stream reaches the LM head.

## Data

Both datasets come from one Hugging Face dataset repo set up for testing the
recipe, `ibm-research/touma`. Data, base model and code are all fetched by URI.
`DATA_URI` names the **repo**; the two subdirectory parameters name what to use
inside it:

| Parameter | Value | Holds |
|---|---|---|
| `DATA_URI` | `hf:///datasets/ibm-research/touma` | the repo |
| `TRAIN_DATA_SUBDIR` | `training_data/answerability/jsonl/granite-4.2-8b` | `train.jsonl`, `test.jsonl` |
| `EVAL_DATA_SUBDIR` | `eval_data/answerability` | `gold.jsonl`, `rag.jsonl`, `unanswerable_v2.{a,b,c}.jsonl`, `collection.txt`, `setting.txt` |

Everything both targets read — training, validation and eval — is already
staged in that repo, in the form the code expects. There is nothing to export,
convert or copy before a run. The repo holds other representations of the same
task beside the two directories above (`raw/`, `parquet_granite4.2_8b/`);
`sr-train` reads neither, which is why the subdirectory parameters point where
they do.

The train command asserts `train.jsonl` and `test.jsonl` exist under the
resolved directory, so a wrong subdirectory fails in seconds and costs queue
time and nothing else.

The split between repo and subdirectory is not cosmetic: the pull downloads the
**whole repo** to one cache directory keyed by owner/repo/revision and hands the
step that directory's root. A sub-path written into the URI would be recorded in
the lineage and then ignored by the download, so the subdirectory has to be
appended by the command — which is what `build.yaml` does. Because both targets
name the same repo, the eval target's pull finds the cache warm and transfers
nothing.

Nothing in the recipe, the steps, or the shadow-residual code knows these files
came from the Hub. granite.build resolves the URI to a local directory before a
step starts; `sr-train` and `answerability_eval.py` only ever open local paths,
which is all they can do.

Three slashes in `hf:///datasets/...` are load-bearing. With two
(`hf://datasets/...`) the word `datasets` parses as the *host*, and the pull
targets a hostname that does not exist.

### The base model is an input too

`BASE_MODEL_URI: hf:///ibm-granite/granite-4.2-8b` is an input of **both**
targets, not something the trainer downloads on its own. The weights land in
the shared asset cache once and every later run reuses them, and train and eval
bind the *same artifact*, so the fact that the adapter was scored on the base it
was trained against is recorded rather than left to two parameters happening to
agree.

PEFT writes whatever path it loaded the base from into the adapter's
`adapter_config.json` as `base_model_name_or_path`, which here would be a cache
directory on this cluster — a dead reference to anyone else. So the train
command rewrites that one field to `BASE_MODEL_ID` after training and echoes the
result. Keep `BASE_MODEL_ID` and `BASE_MODEL_URI` naming the same weights;
nothing checks that they do.

### The one prerequisite: an `HF_TOKEN` secret

For this first round the data sits in a **private** repo, so the pull needs a
**read-only** token — read-only being enough because nothing here writes to the
Hub. An anonymous download of a private repo gets a 401, and the download runs
on a cluster node with no access to whatever is in your shell, so the token has
to be registered once, by name, where granite.build can find it:

```bash
gb secret create HF_TOKEN --personal --value "$HF_TOKEN" --skip-version-check
gb secret list --personal --skip-version-check    # -> HF_TOKEN
```

`HF_TOKEN` is not an arbitrary label: it is the credential name the Hugging Face
assetstore looks for (`token_secretname` in its `store.yaml`), so a secret under
any other name is simply not found. `--personal` registers it against your user
rather than the space, and personal secrets merge over space secrets user-wins,
so the pull runs under your Hub identity.

`--value "$HF_TOKEN"` reads the token out of the environment rather than typing
it, which keeps it out of your shell history and out of this file. The token has
to belong to an account with access to the `ibm-research` org.

From there its whole path is server-side: granite.build resolves the secret,
hands it to the download step as an environment variable, and `hf download`
picks it up. It never appears in `build.yaml`, in `parameters.yaml`, or on a
command line. The same mechanism covers a gated model in `BASE_MODEL_URI`, which
for `ibm-granite/granite-4.2-8b` is public and needs no credential.

## Run

For testing the recipe, we are using an IBM BlueVela environment; to run
elsewhere, see the comment on `ENVIRONMENT_URI` in
[`parameters.yaml`](parameters.yaml).

The commands below use the `public` space, which a standalone gbserver
registers when started as in the top-level
[Quick start (standalone)](../../../../README.md#quick-start-standalone):

    gbserver standalone --space-dir configurations/spaces/local

    gb build start -f recipes/granite-switch/lsf/answerability-sr/build.yaml \
      --space public

Explicitly, or with a different parameter set:

    gb build start -f recipes/granite-switch/lsf/answerability-sr/build.yaml \
      --parameters-path recipes/granite-switch/lsf/answerability-sr/parameters.yaml \
      --space public

The configuration is rank 16, alpha 32, dropout 0.05, lr 5e-5 cosine with 0.05
warmup, 3 epochs, `max_length` 8192, batch 1 × grad accum 8, seed 0, one H100.

## The parameters that are load-bearing in a non-obvious way

Every config sub-model `sr-train` validates is Pydantic `extra="forbid"`, so a
stale or misspelled key is a hard startup error rather than a silently ignored
field. These are the ones whose *values* decide whether the run means anything:

- **`CROSS_STREAM_RANK` must stay set.** Listing `cross_stream` in
  `target_modules` is what engages SR's dual-stream forward. Drop it and the run
  degrades to plain LoRA without erroring.
- **`k_proj` and `v_proj` are deliberately absent** from `target_modules` and
  must stay absent: SR computes K/V once from the frozen base stream and shares
  it, so a K/V LoRA delta has nowhere to land and the model rejects it at build
  time.
- **`ASSISTANT_MARKER`.** Loss is masked to assistant turns by one of two paths,
  picked automatically: TRL's `assistant_only_loss` when the chat template
  carries `{% generation %}` tags, otherwise SR's own `ResponseOnlyCollator`.
  granite-4.2's template has no such tags, so the collator path is taken — and
  its default marker is Granite 4.1's
  `<|start_of_role|>assistant<|end_of_role|>`, which never occurs in a 4.2 row.
  Left at the default the loss mask is *wrong* rather than missing.
  `<|im_start|>assistant\n<think></think>` is the 4.2 equivalent: 4.2's
  assistant opener carries an empty think block, so the marker includes it and
  supervision starts after `</think>`. This matches shadow-residual's own 4.2
  config (`sr-qo-mlp-r32-c32-sharedkv-g42.yaml`).
- **`LAST_CONTEXT_TOKEN: </think>`**, the last token of `ASSISTANT_MARKER` and
  so the final context token before generation. It must encode to exactly one
  token, and `</think>` is a single added token on the 4.2 tokenizer.
  `train.py` validates every row's supervised-region boundary against it and
  fails loudly when it disagrees, so the marker and this token have to change
  together.
- **`LAST_TOKEN: <|im_end|>`**, 4.2's eos and a single added token, which is what
  the trainer requires of this field.

`GRADIENT_CHECKPOINTING` is on by default because two hidden-state streams
roughly double activation memory and this is an 8B model at 8192 context. SR's
dual-stream decoder forward is pure, so recompute is exact — checkpointing costs
wall clock, not correctness.

## Why the code is cloned from github.com

The `byoc` step does an **unauthenticated** `git clone`, so the repo it clones
must be publicly readable. `SR_REPO` therefore points at the public github.com
repo and should stay there.

Pin `SR_REF` to a tag or SHA for anything you intend to compare against later.
`main` is the default because that is the branch whose API the generated config
is written against, not because tracking a moving branch is desirable.

Everything else the run needs is public too — the base model from the Hub, the
dependencies from PyPI. The data is the sole exception, and only because this
first round reads it from a private repo.

## Checking the run did real work

A green build is not evidence on its own. Both targets already assert the
artifact they claim to produce exists — `adapter_config.json` after training,
`eval_summary.md` after eval — and the eval target prints the summary table into
the build log. Beyond that:

- **train**: a training loss that *moved* across steps (`LOGGING_STEPS: 1`, so
  every step is in the log), `checkpoint-*` dirs at each epoch, and an
  `adapter_config.json` whose `target_modules` includes `cross_stream` and
  excludes `k_proj`/`v_proj` — that check is the difference between having
  trained SR and having trained plain LoRA — and whose `base_model_name_or_path`
  reads `ibm-granite/granite-4.2-8b` rather than a cache path (the command
  prints this line).
- **eval**: `eval_summary.md` / `.csv` / `.jsonl` plus `result_df.csv` with the
  per-row predictions.

Expected scores on this eval set, measured earlier on the same base model with
an older SR build:

| Adapter | Unanswerable F1 | Answerable F1 | Accuracy | Weighted F1 |
|---|---|---|---|---|
| LoRA (baseline) | 90.8% | 89.5% | 90.2% | 90.2% |
| SR (shared-KV + cross_stream) | 90.4% | 88.7% | 89.6% | 89.6% |

SR is at rough parity with LoRA here, not ahead. Treat a number in that
neighbourhood as the run having worked, and anything near chance as the loss
mask or the marker being wrong rather than the science being interesting. These
are the right order of magnitude to expect, not a target to reproduce exactly.

## What this recipe does not do

- **No baseline arm.** There is no `--mode base` target, so the adapter's
  accuracy has nothing in-build to be compared against; use the table above, or
  add a third target that runs `answerability_eval --mode base` against the same
  `DATA_URI` / `EVAL_DATA_SUBDIR`.
- **No multi-GPU training.** One GPU. shadow-residual ships exactly one
  accelerate FSDP config and it is a 4-process variant, so the only other
  supported width is 4 — reaching it means launching `accelerate launch
  --config_file <pkg>/config/accelerate/fsdp_4gpu.yaml -m
  shadow_residual.training.train` instead of `sr-train`, which cannot fan out by
  itself. Under FSDP you would also want to keep `--skip-post-train-generate`
  (already passed): in-job generation gathers the model on rank 0 while the
  other ranks sit at a collective barrier, and a full eval set outlives the
  default 30-minute timeout.
- **No dependency caching.** `byoc` installs from PyPI on every run; the image
  supplies torch and CUDA, so this is a couple of minutes, not a rebuild.
