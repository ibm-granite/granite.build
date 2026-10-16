# Shadow Residual answerability adapter

Trains a [Shadow Residual](https://github.com/generative-computing/shadow-residual)
(SR) answerability adapter on `ibm-granite/granite-4.2-8b` and scores it, as two
targets joined by one artifact:

    train ──(adapter)──> eval

`train` runs `sr-train` against a config this recipe generates from
[`parameters.yaml`](parameters.yaml); `eval` binds the adapter it produced and
runs the same repo's `shadow_residual.eval.answerability_eval --mode peft`,
which rebuilds the SR base and attaches the adapter through stock PEFT. Both
targets bind the same base-model artifact — they have to, or the adapter is
served on a different graph than it was trained on, and that is the one error
nothing in the output would reveal.

SR runs two hidden-state streams per decoder layer: a frozen base stream that
stays bit-exact with the unadapted model, and a trainable stream carrying the
LoRA deltas plus a per-layer rank-R `cross_stream` injection from the base. Only
the adapter stream reaches the LM head. The practical consequence for this
recipe is that `cross_stream` must appear in `target_modules` — it is what
engages the dual-stream forward, and dropping it degrades the run to plain LoRA
without erroring.

## Data

Both datasets come from one private Hugging Face dataset repo,
`ibm-research/touma`. Nothing in this recipe names a `/proj` path any more —
data, base model and code are all fetched by URI. `DATA_URI` names the
**repo**; the two subdirectory parameters name what to use inside it:

| Parameter | Value | Holds |
|---|---|---|
| `DATA_URI` | `hf:///datasets/ibm-research/touma` | the repo |
| `TRAIN_DATA_SUBDIR` | `training_data/answerability/jsonl/granite-4.2-8b` | `train.jsonl`, `test.jsonl` |
| `EVAL_DATA_SUBDIR` | `eval_data/answerability` | `gold.jsonl`, `rag.jsonl`, `unanswerable_v2.{a,b,c}.jsonl`, `collection.txt`, `setting.txt` |

Everything both targets read — `train.jsonl`, `test.jsonl` and the eval set —
is already staged in that repo, in the form the code expects. There is nothing
to export, convert or copy before a run. The repo holds other representations
of the same task beside the ones named above (`raw/`,
`parquet_granite4.2_8b/`); `sr-train` reads neither, which is why the
subdirectory parameters point where they do.

The train command asserts both files exist under the resolved directory and
fails in seconds if they do not, so a wrong subdirectory costs queue time and
nothing else.

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

### The base model travels the same road

`BASE_MODEL_URI: hf:///ibm-granite/granite-4.2-8b` is an input of **both**
targets, not something the trainer downloads on its own. The weights land in
the shared asset cache once and every later run reuses them, and — the reason
that matters more — train and eval bind the *same artifact*, so the fact that
the adapter was scored on the base it was trained against is recorded rather
than left to two parameters happening to agree.

One wrinkle comes with that. PEFT writes whatever path it loaded the base from
into the adapter's `adapter_config.json` as `base_model_name_or_path`, which
here would be a cache directory on this cluster — a dead reference to anyone
else. So the train command rewrites that one field to `BASE_MODEL_ID` after
training and echoes the result into the log. Keep `BASE_MODEL_ID` and
`BASE_MODEL_URI` naming the same weights; nothing checks that they do.

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
so the pull runs under your Hub identity without touching anyone else's.

`--value "$HF_TOKEN"` reads the token out of the environment rather than typing
it, which keeps it out of your shell history and out of this file. A **read**
token is sufficient — nothing here pushes to the Hub — and it has to belong to
an account with access to the `ibm-research` org.

From there the token's whole path is server-side: granite.build resolves the
secret, hands it to the download step as an environment variable, and `hf
download` picks it up for the private repo. It never appears in `build.yaml`, in
`parameters.yaml`, or on a command line. The same mechanism covers any other
private or gated Hub repo a recipe names — including a gated model in
`BASE_MODEL_URI`, which for `ibm-granite/granite-4.2-8b` happens to be public
and so needs no credential at all.

## Run

    gb build start -f recipes/granite-switch/lsf/answerability-sr/build.yaml \
      --space <your-space>

Explicitly, or with a different parameter set:

    gb build start -f recipes/granite-switch/lsf/answerability-sr/build.yaml \
      --parameters-path recipes/granite-switch/lsf/answerability-sr/parameters.yaml \
      --space <your-space>

## Why the code is cloned from github.com

The `byoc` step does an **unauthenticated** `git clone`, so the repo it clones
must be publicly readable. That rules out both IBM-internal candidates: the
internal GitHub Enterprise refuses anonymous clones outright, and its
`shadow-residual` mirror is in any case about six weeks behind the github.com
repo. `SR_REPO` therefore points at github.com and should stay there.
Everything else the run needs is public too — the base model comes from the Hub
and the dependencies from PyPI. The data is the sole exception, and only
because this first round reads it from a private repo.

Pin `SR_REF` to a tag or SHA for anything you intend to compare against later.
`main` is the default because that is the branch whose API the generated config
is written against, not because tracking a moving branch is desirable.

The data is the one exception to "nothing internal": `ibm-research/touma` is a
private Hub repo, which is why the `HF_TOKEN` secret above is required. The
clone, the base model and the dependencies need no credential at all.

## Where this deviates from `intrinsics_training`, and why

`intrinsics_training` has its own SR answerability run
(`configs/answerability/shadow_residual/sr_sharedkv_4.2_8b.yaml` plus
`scripts/shell/sr/ad/`), and the hyperparameters here are taken from it: rank 16,
alpha 32, dropout 0.05, lr 5e-5, 3 epochs, `max_length` 8192, batch 1 × grad
accum 8, seed 0, one GPU. What is **not** carried over is the parts of its config
that no longer exist.

That repo pins shadow-residual to an internal-mirror commit from the aLoRA era.
Public `main` has since moved to 100% stock PEFT, and three keys went with it:

| In `intrinsics_training`'s config | Here |
|---|---|
| `adapter.invocation_tokens` | gone — the adapter is always active, so there is nothing to gate |
| `model.shared_base_kv: true` | gone — shared base-only K/V is now the only topology |
| loaded via `peft_shadow_residual.load_shadow_residual_peft_model` | `build_sr_base` + `PeftModel.from_pretrained`, which `answerability_eval.py` already does |

These are not cosmetic omissions. Every config sub-model is Pydantic
`extra="forbid"`, so carrying either key across would be a hard validation error
at startup, not a silently ignored field.

Two further deliberate differences:

- **`GRADIENT_CHECKPOINTING: true`, where that repo ran `false`.** Its value was
  forced off by the aLoRA gating it used. Current SR has no gating and its
  dual-stream decoder forward is pure, so recompute is exact — checkpointing
  costs wall clock, not correctness. It is on by default because two
  hidden-state streams roughly double activation memory, and this is an 8B model
  at 8192 context.
- **`last_context_token` is not set.** It is optional, it does not gate anything,
  and it must encode to exactly one token. On ChatML the prompt ends
  `<|im_start|>assistant\n`, whose final token is not a safe single-token
  constant, and `train.py` validates the supervised-region boundary against it
  and fails loudly when it disagrees. `last_token` *is* set, to `<|im_end|>` —
  4.2's eos, and a single added token.

## The one parameter that is load-bearing in a non-obvious way

`ASSISTANT_MARKER`. Loss is masked to assistant turns by one of two paths, picked
automatically: TRL's `assistant_only_loss` when the chat template carries
`{% generation %}` tags, otherwise SR's own `ResponseOnlyCollator`.
granite-4.2's template has no such tags, so the collator path is taken — and its
default marker is Granite 4.1's `<|start_of_role|>assistant<|end_of_role|>`,
which never occurs in a 4.2 row. Left at the default the loss mask is *wrong*
rather than missing. `<|im_start|>assistant\n` is the 4.2 equivalent, matching
what `intrinsics_training` sets.

## Checking the run did real work

A green build is not evidence on its own. Both targets already assert the
artifact they claim to produce actually exists — `adapter_config.json` after
training, `eval_summary.md` after eval — and the eval target prints the summary
table into the build log. Beyond that:

- **train**: a training loss that *moved* across steps (`LOGGING_STEPS: 1`, so
  every step is in the log), `checkpoint-*` dirs at each epoch, and an
  `adapter_config.json` whose `target_modules` includes `cross_stream` and
  excludes `k_proj`/`v_proj`, and whose `base_model_name_or_path` reads
  `ibm-granite/granite-4.2-8b` rather than a cache path (the command prints
  this line). That last check is worth making: it is the
  difference between having trained SR and having trained plain LoRA.
- **eval**: `eval_summary.md` / `.csv` / `.jsonl` plus `result_df.csv` with the
  per-row predictions.

For reference, `intrinsics_training` measured this on granite-4.2-8b against the
same eval set, same base model for both rows:

| Adapter | Unanswerable F1 | Answerable F1 | Accuracy | Weighted F1 |
|---|---|---|---|---|
| LoRA (baseline) | 90.8% | 89.5% | 90.2% | 90.2% |
| SR (shared-KV + cross_stream) | 90.4% | 88.7% | 89.6% | 89.6% |

So SR is at rough parity with LoRA here, not ahead. Treat a number in that
neighbourhood as the run having worked, and anything near chance as the loss
mask or the marker being wrong rather than the science being interesting. Those
figures came from the older aLoRA-gated SR, so they are the right order of
magnitude to expect and not a target to reproduce exactly.

## Smoke-testing

`MAX_STEPS` caps training at N optimizer steps, which turns this recipe into its
own gate without a second copy to keep in sync:

    gb build start -f recipes/granite-switch/lsf/answerability-sr/build.yaml \
      --space <your-space> --param MAX_STEPS=2 --param NUM_EPOCHS=1

Two steps rather than one on purpose: one proves the trainer starts, two prove it
can take a second step carrying the first one's state. That clears the HF pull,
the data paths, the generated config, the loss mask, the artifact wiring and the
whole eval leg. What it cannot clear is memory at the full schedule — the geometry is
unchanged, so an OOM would still show up here, but a slow leak across 3 epochs
would not.

## What this recipe does not do

- **No baseline arm.** There is no `--mode base` target, so the adapter's
  accuracy has nothing in-build to be compared against; use the table above, or
  add a third target that runs `answerability_eval --mode base` against the same
  `DATA_URI` / `EVAL_DATA_SUBDIR`.
- **No multi-GPU training.** One GPU, which is what `intrinsics_training` used
  for this config and model size. shadow-residual ships exactly one accelerate
  FSDP config and it is a 4-process variant, so the only other supported width
  is 4 — reaching it means launching
  `accelerate launch --config_file <pkg>/config/accelerate/fsdp_4gpu.yaml -m
  shadow_residual.training.train` instead of `sr-train`, and `sr-train` cannot
  fan out by itself. Under FSDP you would also want to keep
  `--skip-post-train-generate` (already passed): in-job generation gathers the
  model on rank 0 while the other ranks sit at a collective barrier, and a full
  eval set outlives the default 30-minute timeout.
- **No dependency caching.** `byoc` installs from PyPI on every run; the image
  supplies torch and CUDA, so this is a couple of minutes, not a rebuild.
