# Shadow Residual answerability adapter

Trains a [Shadow Residual](https://github.com/generative-computing/shadow-residual)
(SR) answerability adapter on `ibm-granite/granite-4.2-8b` and scores it, as two
targets joined by one artifact:

    train ──(adapter)──> eval

`train` runs `sr-train` against a config this recipe generates from
[`parameters.yaml`](parameters.yaml); `eval` binds the adapter it produced and
runs the same repo's `shadow_residual.eval.answerability_eval --mode peft`,
which rebuilds the SR base and attaches the adapter through stock PEFT. Both
targets pin the same `BASE_MODEL` — they have to, or the adapter is served on a
different graph than it was trained on.

SR runs two hidden-state streams per decoder layer: a frozen base stream that
stays bit-exact with the unadapted model, and a trainable stream carrying the
LoRA deltas plus a per-layer rank-R `cross_stream` injection from the base. Only
the adapter stream reaches the LM head. The practical consequence for this
recipe is that `cross_stream` must appear in `target_modules` — it is what
engages the dual-stream forward, and dropping it degrades the run to plain LoRA
without erroring.

## Before the first run

**One prerequisite, and it is the only thing here that is not already in
place.** `sr-train` reads `{"messages": [...]}` JSONL. The answerability data on
`/proj` is not in that form: `/proj/dmfexp/8cc/data/answerability/train_data/`
holds raw classification JSONL and `train_data_parquet/` holds tokenized
parquet, and the trainer reads neither. The messages-JSONL has to be exported
once, from [`intrinsics_training`](https://github.ibm.com/touma/intrinsics_training):

```bash
# in an intrinsics_training checkout, with git-lfs data fetched
conda activate py311
bash scripts/shell/sr/ad/export_answerability_jsonl.sh
# -> data/train/answerability/jsonl/granite-4.2-8b/{train,test}.jsonl
```

Copy that pair to a `/proj` directory the build can read and set
`TRAIN_DATA_DIR` to it. `parameters.yaml` defaults to
`/proj/dmfexp/8cc/data/answerability/jsonl/granite-4.2-8b`, which is where it
would naturally live beside the other answerability assets — **confirm the path
before the first run rather than assuming it.** The train command asserts both
files exist and fails in seconds if they do not, so a wrong path costs queue
time and nothing else.

The eval side needs no preparation.
`/proj/dmfexp/8cc/data/answerability/test_data/` already has exactly the layout
`answerability_eval.py` expects — `gold.jsonl`, `rag.jsonl`,
`unanswerable_v2.{a,b,c}.jsonl` plus `collection.txt` and `setting.txt` — and is
the provenance `intrinsics_training` records for its own bundled copy.

## Run

    gb build start -f recipes/granite-switch/lsf/answerability-sr/build.yaml \
      --space <your-space>

Explicitly, or with a different parameter set:

    gb build start -f recipes/granite-switch/lsf/answerability-sr/build.yaml \
      --parameters-path recipes/granite-switch/lsf/answerability-sr/parameters.yaml \
      --space <your-space>

## Why the code is cloned from github.com

The `byoc` step does an **unauthenticated** `git clone`, so the repo it clones
must be publicly readable. That rules out both IBM-internal candidates:
`github.ibm.com` refuses anonymous clones outright, and its
`generative-computing/shadow-residual` mirror is in any case about six weeks
behind the github.com repo. `SR_REPO` therefore points at github.com and should
stay there. Everything else the run needs is public too — the base model comes
from the Hub and the dependencies from PyPI — so the only IBM-internal inputs
are the two data paths.

Pin `SR_REF` to a tag or SHA for anything you intend to compare against later.
`main` is the default because that is the branch whose API the generated config
is written against, not because tracking a moving branch is desirable.

## Where this deviates from `intrinsics_training`, and why

`intrinsics_training` has its own SR answerability run
(`configs/answerability/shadow_residual/sr_sharedkv_4.2_8b.yaml` plus
`scripts/shell/sr/ad/`), and the hyperparameters here are taken from it: rank 16,
alpha 32, dropout 0.05, lr 5e-5, 3 epochs, `max_length` 8192, batch 1 × grad
accum 8, seed 0, one GPU. What is **not** carried over is the parts of its config
that no longer exist.

That repo pins shadow-residual to a `github.ibm.com` commit from the aLoRA era.
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
  excludes `k_proj`/`v_proj`. That last check is worth making: it is the
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
can take a second step carrying the first one's state. That clears the data
paths, the generated config, the loss mask, the artifact wiring and the whole
eval leg. What it cannot clear is memory at the full schedule — the geometry is
unchanged, so an OOM would still show up here, but a slow leak across 3 epochs
would not.

## What this recipe does not do

- **No baseline arm.** There is no `--mode base` target, so the adapter's
  accuracy has nothing in-build to be compared against; use the table above, or
  add a third target that runs `answerability_eval --mode base` against the same
  `EVAL_DATASET_DIR`.
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
