# `distill-gold` — GOLD knowledge distillation (SkyPilot / LSF)

Distils a small **student** from a larger **teacher** with the GOLD trainer
(generalized JSD, `gb_steps_post_training.distillation.gold`), multi-node via the
SkyPilot LSF cloud.

## What supplies what

| Piece | Source |
|---|---|
| Pinned deps (torch, transformers, trl, vllm, deepspeed) | the container image, venv at `/stage/.venv` |
| Trainer (`gold.py`, `custom_gold_trainer.py`, DeepSpeed config) | `code_config`'s clone of the public source repo |
| `bsub`, `blaunch`, enroot, per-node rank/master | SkyPilot's LSF provisioner |
| Config rendering, node roles, `accelerate` launch | this step |

The trainer's commit is pinned by `code_config.expect_ref`, the same pin every
other ported distillation step uses — recorded as `distill_code_commit` step
metadata on every run.

## Minimal build

```yaml
steps:
  - step_uri: space://steps/distill/distill-gold
    config:
      compute_config:
        num_nodes: 2            # accelerators below are PER NODE
        num_gpus_per_node: 8
      launcher_config:
        resources:
          accelerators: "H100:8"
      gold_config:
        model_name_or_path: /proj/.../student_overlays/granite-4.1-3b-base-hub
        teacher_model_name_or_path: /proj/.../teacher_overlays/granite-4.2-30b
        dataset_name: /proj/.../subsampled_0.4_shuffled_nothink.jsonl
        gradient_accumulation_steps: 12
```

## Choosing a student/teacher pair

**In this step's configuration the two must share a tokenizer** — a same-sized
vocabulary is not sufficient, the token IDs must agree.

That is a limitation of *this step*, not of GOLD. Upstream GOLD
(`trl.experimental.gold`, which `CustomGOLDTrainer` subclasses) exists precisely to
distil across *differing* tokenizers, aligning them by byte offsets via its ULD
loss; the trainer carries that path in full (`use_uld_loss`,
`teacher_tokenizer_name_or_path`, and ~15 `uld_*` options in
`custom_gold_config.py`). This step exposes **none** of those keys, so no build can
reach the cross-tokenizer path today, and the shared-vocabulary JSD path is the
only one it can run. Adding them is a config-surface change, not a redesign — see
[Adding a hyperparameter](README.md#adding-a-hyperparameter).

A mismatched pair on the path this step *does* run is a **loud** failure, not a silent
one: `gold.py` calls `verify_tokenizer_consistency()`, which raises `RuntimeError`
naming both sources when the two tokenizers disagree on class, pre-tokenizer,
post-processor, normalizer, decoder, added-token table or a battery of encoding probes.
(The trainer skips that check only when `use_uld_loss` is set, which this step cannot
set.) Confirmed by direct measurement.

Model directories fall into several tokenizer families, and the family does **not**
track the version number — two models with different names can ship byte-identical
tokenizers, and two models sharing a name prefix may not. Group by the hash of
`tokenizer.json`, not by name:

```shell
for d in path/to/{student,teacher}_overlays/*; do
  echo "$(md5sum "$d/tokenizer.json" | cut -c1-8)  $(basename "$d")"
done | sort
```

A second, **independent** constraint is the one that really is silent, and it is about
chat markup rather than tokenizers. The default `response_template`
(`<|im_start|>assistant`) only works for families whose vocabulary contains
`<|im_start|>`. Others (the granite-4 markup families, using `<think_off>`,
`<documents>`, …) need a `response_template` matching their own chat format —
otherwise the template matches nothing, the completion span is never located, and the
loss is computed over the wrong tokens **without any error**. Nothing checks this: a
pair can pass `verify_tokenizer_consistency` and still train against the wrong span.

As of this writing the reference pair (`granite-4.1-3b-base-hub` +
`granite-4.2-30b`) is the only pair in its family, so there is no smaller
drop-in substitute. Scale down context and data instead.

## Things that will bite

- **Effective batch is `per_device x grad_accum x nodes x gpus_per_node`.** The
  reference run is 192 (1 x 6 x 4 x 8) and the LR schedule was tuned against it,
  so halving the nodes wants `gradient_accumulation_steps: 12` to hold it. Change
  the node count without this and you have changed the optimization, not just the
  throughput.
- **The dataset must be think-filtered** (`*_nothink.jsonl`). An inline
  `<think>...</think>` in an assistant turn breaks completion extraction.
- **Leave `use_liger_fused_jsd: false`.** granite's `logits_scaling=10` overflows
  the fused bf16 JSD kernel and gives NaN loss.
- **Keep `save_total_limit` high.** An early, less-forgotten checkpoint is often
  the best one to evaluate; a small limit deletes it irrecoverably.
- **A refusal that says `[residency] REFUSING to start`** is not a config error:
  GPFS has migrated those weight files to tape, and the first read would block
  on a recall of tens of GB with the GPUs already held. Stage them in (`dd` to
  `/dev/null` on a login node, or `mmrestripefile`) and resubmit, or pass
  `allow_offline_weights: true` to hold the allocation through the recall
  deliberately. `check_weight_residency: false` turns the check off entirely —
  appropriate on a cluster where `mmlsattr` is absent, though there it already
  warns and proceeds. Note it is **not** a completeness check: a resident shard
  can still be truncated.
- **The image is an SM90 (H100) build** and will not run on A100.
- **Multi-node needs a SkyPilot with LSF multi-node support.** gbserver refuses
  the launch otherwise rather than silently running on one node.

## Outputs

`checkpoint` (type `model`) — `$GB_BUILD_WORKDIR/checkpoints/<run_name>_node<N>`,
registered by rank 0. The node count in that path comes from the allocation, not
from a parameter, so it always describes the run that produced it.

## On-policy

`vllm_num_servers > 0` dedicates the **last** N nodes to serving the student under
vLLM and trains on the rest; the renderer rejects a count that leaves no trainers,
or on-policy on a single node. On-policy continues from a good off-policy
checkpoint — point `model_name_or_path` at that checkpoint, not the base overlay.
