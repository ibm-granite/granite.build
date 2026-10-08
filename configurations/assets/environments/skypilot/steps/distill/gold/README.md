# gold (SkyPilot / LSF)

Distils a small **student** from a larger **teacher** with the GOLD trainer (generalized
JSD, `gb_steps_post_training.distillation.gold`), multi-node via the SkyPilot LSF cloud.
The step renders the trainer's config, assigns each node its role (trainer, or vLLM server
on the on-policy path) and starts `accelerate` with that node's rank. It registers the
checkpoint directory as a `model` artifact, and optionally each intermediate checkpoint as
soon as it is written.

> **Developing or testing this step?** See `steps/distill/gold/skypilot/README.md` in the
> granite.build repository for how the step is generated, tested, and published — including
> the opt-in that runs the real two-node BlueVela build test.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/gold` URI:

```yaml
steps:
  - step_uri: space://steps/distill/gold
```

## Config contract (`gold_config`)

The trainer's settings live under `config.gold_config`. Checkpoint emission, resume and
the monitor are top-level `config` keys; the trainer source is `config.code_config`.

### Required

| Field | Type | Purpose |
|---|---|---|
| `model_name_or_path` | string | Student init. On the on-policy path this is also the model the vLLM server nodes serve. Must share a tokenizer with the teacher (see [Choosing a student/teacher pair](#choosing-a-studentteacher-pair)). |
| `teacher_model_name_or_path` | string | The teacher: frozen, forward-only in the training loop. |
| `dataset_name` | string | The training corpus. **Must be think-filtered** (`*_nothink.jsonl`): an assistant completion containing an inline `<think>...</think>` breaks GOLD's completion extraction. |

All three default to `""`. The renderer passes them through without checking for an empty
value, so an empty one is not caught before the trainer starts on a held allocation.

### Optional

#### Code and run naming

| Field | Type | Purpose |
|---|---|---|
| `ds_config` | string | DeepSpeed ZeRO-3 `accelerate` config, relative to `code_config`'s clone. Default: `steps/distill-gold-train/configs/deepspeed/accelerate_deepspeed_zero3.yaml`. |
| `run_name` | string | Names the checkpoint directory and the rendered config. The node count is appended at run time, from the allocation (`<run_name>_node<N>`), so the two cannot disagree. Default: `gold`. |

#### Weight residency

| Field | Type | Purpose |
|---|---|---|
| `check_weight_residency` | boolean | Refuse to start when the student or teacher path names weights GPFS has migrated to tape. See [Weight residency refusal](#weight-residency-refusal). Default: `true`. |
| `allow_offline_weights` | boolean | Proceed through the OFFLINE refusal anyway; announces itself in the log. For when you know the recall is already running. Default: `false`. |

#### Training hyperparameters

| Field | Type | Purpose |
|---|---|---|
| `num_train_epochs` | number | Epochs over `dataset_name`. Default: `1.0`. |
| `max_steps` | integer | Cap the run by optimizer steps rather than epochs. `0` leaves the key out of the rendered config, so an epoch-bounded run is unaffected; any positive value overrides `num_train_epochs` in the trainer. One epoch of the 802,027-row reference corpus is 4,177 steps at effective batch 192. Default: `0`. |
| `learning_rate` | number | Peak learning rate. Rendered as a YAML float. Default: `1.0e-05`. |
| `min_lr` | number | Floor of the `cosine_with_min_lr` schedule; rendered under `lr_scheduler_kwargs`, not at the top level. Default: `1.0e-06`. |
| `warmup_ratio` | number | Warmup fraction of the schedule. Default: `0.05`. |
| `lr_scheduler_type` | string | Scheduler. Default: `cosine_with_min_lr`. |
| `per_device_train_batch_size` | integer | Per-GPU batch. Default: `1`. |
| `gradient_accumulation_steps` | integer | Effective batch is `per_device x grad_accum x nodes x gpus_per_node`; see [Things that will bite](#things-that-will-bite). Default: `6`. |
| `max_completion_length` | integer | Maximum completion tokens. Default: `4096`. |
| `max_length` | integer | Maximum sequence length; also the vLLM server's `--max-model-len` on the on-policy path. Default: `16384`. |
| `gradient_checkpointing` | boolean | Activation checkpointing. Default: `true`. |
| `save_strategy` | string | `steps` pairs with `save_steps` and is what every validated reference config sets; `no` turns checkpointing off without editing the step. Default: `steps`. |
| `save_steps` | integer | Save interval in optimizer steps. Default: `500`. |
| `save_total_limit` | integer | Checkpoints kept. Keep it high; see [Things that will bite](#things-that-will-bite). Default: `20`. |
| `logging_steps` | integer | Logging interval in optimizer steps. Default: `5`. |
| `dataset_num_proc` | integer | Processes for dataset preprocessing. Default: `64`. |

#### GOLD loss

| Field | Type | Purpose |
|---|---|---|
| `temperature` | number | Distillation temperature. Default: `1.0`. |
| `lmbda` | number | Fraction of on-policy (student-generated) training. `0.0` => off-policy: every node trains on the fixed dataset. Default: `0.0`. |
| `beta` | number | Generalized JSD interpolation: `0.0` = forward KL, `1.0` = reverse KL. Default: `0.0`. |
| `use_liger_fused_jsd` | boolean | Fused Liger JSD kernel. **Leave false:** granite's `logits_scaling=10` overflows the fused bf16 kernel and gives NaN loss. Default: `false`. |
| `loss_arm` | string | The objective as a validated preset instead of free boolean combinations; see [Loss arms](#loss-arms). Empty means unchanged — the flags above stay authoritative. Default: `""`. |
| `response_template` | string | Locates the completion span for loss masking (the nothink chat template has no `{% generation %}` tag). Must match the pair's chat markup; see [Choosing a student/teacher pair](#choosing-a-studentteacher-pair). A trailing line boundary must be written as the two-character escape `\n`; see [Response template escape](#response-template-escape). Default: `<|im_start|>assistant`. |

#### CE anchor and entropy guard

These keys are rendered into the trainer's config **only when non-default**, and they need
a trainer that carries the CE-anchor patch, which the default `code_config` pin does not;
see [CE anchor and entropy guard](#ce-anchor-and-entropy-guard-1).

| Field | Type | Purpose |
|---|---|---|
| `ce_coef` | number | Adds cross entropy on the corpus tokens to the divergence: `loss = divergence + ce_coef * CE(student, labels)`. Must be `>= 0`. Default: `0.0`. |
| `log_student_entropy` | boolean | Log mean student entropy and reverse KL (nats, supervised positions) every `logging_steps`. Default: `false`. |
| `entropy_guard_drop_frac` | number | Trip when entropy falls this fraction below a baseline; `0` disables. Must be in `[0.0, 1.0)`; `> 0` requires `log_student_entropy: true`. Default: `0.0`. |
| `entropy_guard_baseline_steps` | integer | Steps over which the baseline is measured; `>= 1`. Default: `20`. |
| `entropy_guard_patience` | integer | Consecutive low readings before a trip; `>= 1`. Default: `3`. |
| `entropy_guard_action` | string | `stop` ends the run at the trip; `warn` reports it, checkpoints it and continues to `max_steps`. Default: `stop`. |

#### Pre-launch checks

| Field | Type | Purpose |
|---|---|---|
| `check_corpus_tokenizer` | boolean | Verify the student's tokenizer is the one the corpus was built with, before the config is rendered. Worth turning on for any build that chains the `space://steps/distill/corpus-prep` step into this one, because a mismatch does not error — it trains on mis-segmented text at full speed with a falling loss. Default: `false`. |
| `verify_trainer_accepts_keys` | boolean | Check every emitted key that upstream trl's `GOLDConfig` does not define (the CE anchor, the collapse guard, the on-policy shaping pair) against the delivered trainer's dataclass, and refuse before `accelerate` launches. Only emitted keys are checked, so a build leaving them at their defaults never triggers it. Set false only to render deliberately against a trainer you know differs. Default: `true`. |

#### NCCL and distributed diagnostics

| Field | Type | Purpose |
|---|---|---|
| `nccl_debug` | string | `NCCL_DEBUG`. Empty disables NCCL's own logging. `INFO` (with `nccl_debug_subsys: INIT,NET`) shows which transport and HCA each rank selected — the only way to tell an IB path from a silent TCP fallback, or to see where a collective stalls. Default: `""`. |
| `nccl_debug_subsys` | string | `NCCL_DEBUG_SUBSYS`. Default: `INIT,NET`. |
| `nccl_timeout_ms` | integer | Collective timeout (`TORCH_NCCL_TIMEOUT_MS` and `NCCL_TIMEOUT`). 1 h survives a slow 30B teacher forward without spurious aborts; lower it for debugging so a hang reports in minutes. Default: `3600000`. |
| `nccl_enable_monitoring` | boolean | `TORCH_NCCL_ENABLE_MONITORING`: whether torch's monitor thread may abort a stalled collective. The reference launcher disables this; it defaults on here because with it off a deadlock hangs forever holding the allocation, with no diagnostic. Default: `true`. |
| `nccl_ib_hca` | string | `NCCL_IB_HCA` for this run, overriding the environment's `nccl_tuning_file`. Empty => not touched. Keep the tuning file's exclusions when setting it, e.g. on BlueVela `^=mlx5_1,mlx5_6,mlx5_8` drops the mlx5_8 rail on top of the storage NICs. Default: `""`. |

#### On-policy

All default to off-policy. See [On-policy](#on-policy-1).

| Field | Type | Purpose |
|---|---|---|
| `vllm_num_servers` | integer | `> 0` selects on-policy. Without `vllm_server_url`, dedicates the **last** N nodes of the allocation to serving the student under vLLM and trains on the rest. Default: `0`. |
| `vllm_server_url` | string | An **external** vLLM server, reached by URL, instead of carving nodes out of this allocation; typically another target's `mem://` binding, `{{ bindings.<name>.binding.state }}`. When set, every node trains. Requires `vllm_num_servers > 0` and `lmbda > 0`. Not yet run on a cluster. Default: `""`. |
| `vllm_mode` | string | `server` = talk to a vLLM HTTP server (the reference launcher's value). Passed to the trainer only on the external-server path. Default: `server`. |
| `vllm_sync_frequency` | integer | How often, in optimizer steps, the trainer pushes updated weights to the server; a larger value samples from a staler policy. Passed to the trainer only on the external-server path. Default: `1`. |
| `top_p` | number | Rollout sampling `top_p`; emitted only on-policy. Default: `0.95`. |
| `use_sampled_opd_loss` | boolean | Sampled on-policy-distillation loss; emitted only on-policy. Prefer `loss_arm: sampled_opd`, which enforces its requirements. Default: `false`. |
| `last_message_only` | boolean | Trainer `last_message_only`; emitted only on-policy. Required `true` by the `sampled_opd` arm. Default: `false`. |
| `clip_alpha` | number | Trainer `clip_alpha`; emitted only on-policy. Default: `0.1`. |
| `opd_importance_sampling` | boolean | Truncated importance sampling for the sampled OPD loss; emitted only on-policy. Default: `false`. |
| `lmbda_schedule` | string | `constant`, or `linear` to ramp `lmbda` from `lmbda_init` up to `lmbda` across training — off-policy early, on-policy later. `linear` needs `vllm_num_servers > 0`; the renderer refuses it otherwise. Default: `constant`. |
| `lmbda_init` | number | Starting `lmbda` for the linear schedule, in `[0.0, 1.0]`. Default: `0.0`. |
| `min_completion_length` | integer | vLLM `min_tokens` for the student's rollout; `> 0` blocks the immediate-EOS mode collapse. Needs `vllm_num_servers > 0`. With `vllm_mode: server` the server must agree, via `GOLD_MIN_TOKENS`. Default: `0`. |

### Checkpoint emission and resume (top-level `config`)

| Field | Type | Purpose |
|---|---|---|
| `emit_checkpoint_artifacts` | boolean | Off: exactly one artifact, `checkpoint`, at the end. On: a watcher on rank 0 emits `checkpoint_<N>` for each complete `checkpoint-<N>` directory as it is written. See [Per-checkpoint artifacts](#per-checkpoint-artifacts). Default: `false`. |
| `checkpoint_watch_interval_seconds` | integer | How often that watcher polls. The monitor's log-retrieval interval is the other half of the latency. Default: `60`. |
| `resume_from_checkpoint_dir` | string | The `<run_name>_node<N>` checkpoint directory of an earlier run of this step to resume from. See [Resume](#resume). Default: `""` (no resume). |
| `resume_emit_seeded` | boolean | Whether seeded checkpoints are emitted as `checkpoint_<N>` again. Only has an effect with both `resume_from_checkpoint_dir` and `emit_checkpoint_artifacts` set. Default: `true`. |

### Monitor (top-level `config`)

| Field | Type | Purpose |
|---|---|---|
| `poll_interval_seconds` | integer | Monitor poll interval. Default: `900`. |
| `log_retrieval_mode` | string | Default: `periodic`, because a distillation run lasts hours and `on_completion` surfaces nothing until the end. |
| `log_retrieval_interval_seconds` | integer | Periodic log retrieval interval. Default: `900`. |

### Source delivery (`code_config`)

`code_config` (`code_dir`, `expect_ref`, `repo`, `ref`, `workdir`, `token_secret`, `python`,
`setup_command`) is identical in every ported distillation step and documented once — see
"Source delivery" in the `space://steps/distill/tokenizer-align` step's README. The trainer
(`gold.py`, `custom_gold_trainer.py`, `run_vllm_serve.py` and their shared helpers) comes
from that clone. Its commit is pinned by `code_config.expect_ref` and recorded as
`distill_code_commit` step metadata on every run. In this step the renderer, the residency
check and the vLLM server run with `/stage/.venv/bin/python` directly, not with
`code_config.python`.

## Inputs and outputs

### Inputs

The step declares no `inputs:`; it reads the paths in `gold_config`. A recipe declares the
inputs on the target and passes their paths in:

```yaml
inputs:
  student:
    binding: align.retagged_student   # or an SFT target's checkpoint
  corpus:
    binding: corpus.corpus
  teacher_model:
    uri: "env:///proj/.../teacher"
    type: model
# ...and in the step's config:
gold_config:
  model_name_or_path: "{{ bindings.student.binding.path }}"
  teacher_model_name_or_path: "{{ bindings.teacher_model.binding.path }}"
  dataset_name: "{{ bindings.corpus.binding.path }}"
```

Typical sources are the `retagged_student` output of `space://steps/distill/tokenizer-align`
(or the `checkpoint` of `space://steps/distill/sft`) for the student, and the `corpus` output
of `space://steps/distill/corpus-prep`. On the external-server on-policy path,
`vllm_server_url` comes from another target's `mem://` binding as
`{{ bindings.<name>.binding.state }}`.

### Outputs

| Output | Type | What it is |
|---|---|---|
| `checkpoint` | `model` | `$GB_BUILD_WORKDIR/checkpoints/<run_name>_node<N>`, the trainer's output directory holding its `checkpoint-<N>` subdirectories. Declared optional in the step. |
| `checkpoint_<N>` | — | Only with `emit_checkpoint_artifacts: true`: one per complete `checkpoint-<N>` directory. Not declared by the step; the target must declare a matching `checkpoint_<N>` output for every rung it binds. |

Markers, printed on rank 0 only (the executor streams every node into one driver log, so an
unguarded marker would register N artifacts):

```
GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:<CKPT_DIR>
GB_ARTIFACT_ID:checkpoint_<N> GB_ARTIFACT_PATH:<CKPT_DIR>/checkpoint-<N>
```

The `<N>` in `<run_name>_node<N>` comes from the allocation, not from a parameter, so it
always describes the run that produced it. The trainer's ZeRO-3 config saves 16-bit weights,
so a checkpoint is HF-native; downstream, the `space://steps/distill/hf-export` step selects
and checks one.

## Working directory and paths

`run` executes on **every** node with that node's own rank (`RANK`, `TOTAL_NODES`,
`MASTER_ADDR`, `MASTER_PORT`, `NUM_GPUS_PER_NODE` from SkyPilot's LSF provisioner). All
paths are under `$GB_BUILD_WORKDIR` (the per-run workdir, on `/proj` on BlueVela and
identity-mounted into the container), falling back to the current directory when unset:

- `./src` — the step's `src/` (config renderer and residency check), file-mounted.
- `distill-code/` (`code_config.workdir`) — the trainer source clone. Only rank 0 clones;
  the other ranks wait (up to 600 s) for a completion marker naming this attempt.
  `ds_config` resolves relative to it.
- `gold_config_<run_name>_node<N>.yaml` — the rendered trainer config, printed to the log by
  rank 0.
- `checkpoints/<run_name>_node<N>/` — the trainer's `--output_dir`, and the `checkpoint`
  artifact.

`model_name_or_path`, `teacher_model_name_or_path`, `dataset_name` and
`resume_from_checkpoint_dir` are used as given; pass absolute paths.

## Example build.yaml

Two-node off-policy run; `accelerators` is **per node**:

```yaml
granite.build:
  name: gold-example
  targets:
    train-gold:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      outputs:
        checkpoint:
          uri: "env://{{ binding.path }}"
          type: model
      steps:
        - step_uri: space://steps/distill/gold
          config:
            compute_config:
              num_nodes: 2            # accelerators below are PER NODE
              num_gpus_per_node: 8
            launcher_config:
              resources:
                accelerators: "H100:8"
                memory: 256           # host memory per node; the LSF default is 16G
                cluster: "bluevela"
                zone: "normal"
            gold_config:
              model_name_or_path: /proj/.../student_overlays/granite-4.1-3b-base-hub
              teacher_model_name_or_path: /proj/.../teacher_overlays/granite-4.2-30b
              dataset_name: /proj/.../subsampled_0.4_shuffled_nothink.jsonl
              gradient_accumulation_steps: 12
```

`gradient_accumulation_steps: 12` holds the reference effective batch of 192 on two nodes
instead of four. For a full chain (align, corpus-prep, optional SFT, gold, export), see
`recipes/granite4-gold-distillation/lsf/distill-pipeline-smoke` in the granite.build
repository.

## Notes and limitations

### What supplies what

| Piece | Source |
|---|---|
| Pinned deps (torch, transformers, trl, vllm, deepspeed) | the container image, venv at `/stage/.venv` |
| Trainer (`gold.py`, `custom_gold_trainer.py`, DeepSpeed config) | `code_config`'s clone of the public source repo |
| `bsub`, `blaunch`, enroot, per-node rank/master | SkyPilot's LSF provisioner |
| Config rendering, node roles, `accelerate` launch | this step |

### Choosing a student/teacher pair

**In this step's configuration the two must share a tokenizer** — a same-sized vocabulary is
not sufficient, the token IDs must agree.

That is a limitation of *this step*, not of GOLD. Upstream GOLD (`trl.experimental.gold`,
which `CustomGOLDTrainer` subclasses) exists precisely to distil across *differing*
tokenizers, aligning them by byte offsets via its ULD loss; the trainer carries that path in
full (`use_uld_loss`, `teacher_tokenizer_name_or_path`, and ~15 `uld_*` options in
`custom_gold_config.py`). This step exposes `use_uld_loss` only through `loss_arm: uld`,
and exposes neither `teacher_tokenizer_name_or_path` nor any `uld_*` option, so the
cross-tokenizer path cannot be configured end to end; the shared-vocabulary path is the one
this step is built for. Adding the missing keys is a config-surface change, not a redesign —
see "Adding a hyperparameter" in the step's source README.

A mismatched pair on the shared-vocabulary path is a **loud** failure, not a silent one:
`gold.py` calls `verify_tokenizer_consistency()`, which raises `RuntimeError` naming both
sources when the two tokenizers disagree on class, pre-tokenizer, post-processor,
normalizer, decoder, added-token table or a battery of encoding probes. (The trainer skips
that check only when `use_uld_loss` is set, i.e. under `loss_arm: uld`.) Confirmed by direct
measurement.

Model directories fall into several tokenizer families, and the family does **not** track the
version number — two models with different names can ship byte-identical tokenizers, and two
models sharing a name prefix may not. Group by the hash of `tokenizer.json`, not by name:

```shell
for d in path/to/{student,teacher}_overlays/*; do
  echo "$(md5sum "$d/tokenizer.json" | cut -c1-8)  $(basename "$d")"
done | sort
```

A second, **independent** constraint is the one that really is silent, and it is about chat
markup rather than tokenizers. The default `response_template` (`<|im_start|>assistant`) only
works for families whose vocabulary contains `<|im_start|>`. Others (the granite-4 markup
families, using `<think_off>`, `<documents>`, …) need a `response_template` matching their
own chat format — otherwise the template matches nothing, the completion span is never
located, and the loss is computed over the wrong tokens **without any error**. Nothing checks
this: a pair can pass `verify_tokenizer_consistency` and still train against the wrong span.

As of this writing the reference pair (`granite-4.1-3b-base-hub` + `granite-4.2-30b`) is the
only pair in its family, so there is no smaller drop-in substitute. Scale down context and
data instead.

### Things that will bite

- **Effective batch is `per_device x grad_accum x nodes x gpus_per_node`.**
  The reference run is 192 (1 x 6 x 4 x 8) and the LR schedule was tuned against it, so
  halving the nodes wants `gradient_accumulation_steps: 12` to hold it. Change the node count
  without this and you have changed the optimization, not just the throughput.
- **The dataset must be think-filtered** (`*_nothink.jsonl`). An inline `<think>...</think>`
  in an assistant turn breaks completion extraction.
- **Leave `use_liger_fused_jsd: false`.** granite's `logits_scaling=10` overflows the fused
  bf16 JSD kernel and gives NaN loss.
- **Keep `save_total_limit` high.** An early, less-forgotten checkpoint is often the best one
  to evaluate; a small limit deletes it irrecoverably.
- **The image is an SM90 (H100) build** and will not run on A100.
- **LSF only.** enroot and the LSF topology contract this step reads are LSF-specific
  (`subtypes: [lsf]`).
- **Multi-node needs a SkyPilot with LSF multi-node support.** gbserver refuses the launch
  otherwise rather than silently running on one node.
- **State host memory.** The LSF provisioner defaults to 16G per node, and a host-memory kill
  presents as a cross-node NCCL hang.

### Weight residency refusal

A refusal that says `[residency] REFUSING to start` is not a config error: GPFS has migrated
those weight files to tape, and the first read would block on a recall of tens of GB with
the GPUs already held. Stage them in (`dd` to `/dev/null` on a login node, or
`mmrestripefile`) and resubmit, or pass `allow_offline_weights: true` to hold the allocation
through the recall deliberately. `check_weight_residency: false` turns the check off
entirely — appropriate on a cluster where `mmlsattr` is absent, though there it already warns
and proceeds.

The check is metadata-only (`mmlsattr`, and `st_blocks` arithmetic where `mmlsattr` cannot
answer) and never touches file contents. It refuses on OFFLINE — GPFS's own verdict, or a
large file with almost nothing allocated — or on a path it cannot stat at all; the
cannot-stat refusal is not overridden by `allow_offline_weights`. If it cannot measure, it
warns and proceeds, and a hub id rather than a path is not a finding. On BlueVela the
environment bind-mounts `/usr/lpp/mmfs` so the container gets `mmlsattr`'s answer rather than
only the arithmetic's. It is **not** a completeness check: a resident shard can still be
truncated.

### Loss arms

`loss_arm` names the objective as a preset instead of nine booleans a build may combine
freely. Empty means unchanged: the flags stay authoritative and every existing recipe renders
exactly the config it rendered before (so `use_liger_fused_jsd: true` remains a direct escape
hatch). Naming an arm buys two things a boolean cannot: the arm sets switches the renderer
has no flag for (`use_kl_interpolation`, `use_adaptive_kld`, `use_distillm2`,
`use_distillm2_like`, `use_reversed_distillm2_like`, `use_uld_loss`, `use_ce_loss`), and it
enforces that arm's requirements — several exist because outside them the arm does not fail,
it silently becomes a different arm.

| Arm | What it is | Requirement enforced |
|---|---|---|
| `jsd` | mixture generalized JSD; `beta` 0.0 = forward KL, 1.0 = reverse KL | — |
| `kl_interpolation` | convex `(1-beta)*FKL + beta*RKL` | `0 < beta < 1` (else identical to `jsd`) |
| `adaptive_kld` | adaptive KL divergence weighting | — |
| `liger_fused_jsd` | same JSD math via Liger's fused kernel; lower peak memory | — |
| `liger_fused_kl_interpolation` | fused kernel computing the convex FKL/RKL interpolation | `0 < beta < 1` |
| `distillm2` | DistiLLM-2, comparative: reverse KL on on-policy tokens, forward KL off | `0 < lmbda < 1` |
| `distillm2_like` | DistiLLM-2 shape, non-comparative; policy class per microbatch | `0 < lmbda < 1` |
| `reversed_distillm2_like` | `distillm2_like` with the divergence roles exchanged | `0 < lmbda < 1` |
| `sampled_opd` | REINFORCE-style policy gradient on sampled tokens; optional truncated IS | `lmbda == 1.0`, `last_message_only: true` |
| `uld` | Universal Logit Distillation over sorted logits; the cross-tokenizer arm | — |
| `ce` | cross entropy only — in-trainer SFT control; the teacher is still loaded | `lmbda == 0.0` |

The renderer also refuses an arm that contradicts a standalone flag (`use_liger_fused_jsd`,
`use_sampled_opd_loss`), and an arm whose flag is on-policy-only on an off-policy render.
`LOSS_ARMS` in [`src/render_gold_config.py`](src/render_gold_config.py) lists each arm and
why.

### Response template escape

A trailing line boundary in `response_template` must be written as the two-character escape
`\n`, not as a real newline: gbserver fills every config string through Jinja, which strips
exactly one trailing newline from each value, so a real one never reaches this step.
`render_gold_config.py` decodes the escape once, in the container. The default wants no line
boundary and so carries none.

### CE anchor and entropy guard

`ce_coef`, `log_student_entropy` and the four `entropy_guard_*` keys need the **delivered**
trainer to carry the `ce_anchor_and_entropy_guard` patch
(`steps/distill/gold/skypilot/patches/ce_anchor_and_entropy_guard.diff` in the granite.build
repository; its header has the invocation and the directory remap). The default
`code_config` pin does **not** carry it — at `a5d59bc4` the trainer has `lmbda_schedule` but
not `ce_coef` — so a recipe using these keys points `code_config.code_dir` at a checkout that
does, as `recipes/granite4-350m/lsf/distill-stage1-v2` does. Because the keys are rendered
only when non-default, a recipe that leaves them alone runs against any checkout; when one
does not, `verify_trainer_accepts_keys` turns "TrlParser rejected your config after the
allocation was held" into a refusal that names the patch. Without that check the two
failures are both expensive: TrlParser rejects an unknown top-level key only after the
teacher has loaded on every node, and `min_completion_length` fails the other way, silently,
by never reaching the sampler.

- **`ce_coef` is not the trainer's `use_ce_loss`**, which *replaces* the divergence. Without an
  anchor the objective has no opinion about real text, so an already-SFT'd student — one that
  already matches the teacher closely — can only improve by sharpening, and sharpening ends in
  repetition loops. Additive rather than a convex mix because JSD runs ~0.07 here against
  CE ~1, so a convex 50/50 is ~20x CE-dominated. Both terms are logged, so the ratio is a
  reading. Build `df8512e0` ran the unanchored objective for 8,150 steps.
- **The train loss is blind to collapse** — it moved 2.7% across the 7,640 steps in which
  `df8512e0`'s student lost 42% of its entropy. `log_student_entropy` is the instrument.
- **A tripped guard is a finding, not a failed build.** With `entropy_guard_action: stop` the
  run stops gracefully: the checkpoint is saved, the process exits 0, and export/eval still
  run. `warn` exists because of build `d1acf1c0`: the guard stopped it at step 77 of 2000,
  correctly by its own rule, and in doing so destroyed the checkpoint curve the run existed
  to measure and left its four fixed export rungs naming steps that never happened. Where
  `max_steps` already bounds the waste, the trip is worth a reading rather than a stop.

### Per-checkpoint artifacts

With `emit_checkpoint_artifacts: true`, a background watcher on rank 0 polls the checkpoint
directory during training and emits `checkpoint_<N>` per HF `checkpoint-<N>` directory as
soon as it is completely written (`model.safetensors`, `tokenizer.json` and
`trainer_state.json` all present). A recipe that declares matching `checkpoint_<N>` outputs
can then export and evaluate rung N while the run is still training — see
`recipes/granite4-350m/lsf/distill-checkpoint-eval`, where that turns ~36 GPU-h of eval from
a serial tail into work overlapped with a 13-hour epoch. After the trainer exits, the step
sweeps once more, which catches the final checkpoint of an epoch-bounded run.

The emitted ids are a **contract** with the recipe: a rung whose checkpoint directory the
trainer never writes is an artifact never emitted, and a target bound to it waits forever
with no error. Hence every rung should be a multiple of `save_steps`, and hence the step
fails loudly if it emitted **zero** checkpoints.

### Resume

The checkpoint directory is under `GB_BUILD_WORKDIR`, which is new for every build **and**
every retry, so the trainer's own auto-resume (`gold.py` resumes when its `output_dir` already
holds a `checkpoint-*`) never finds anything. `resume_from_checkpoint_dir` names the
`<run_name>_node<N>` directory of an earlier run, e.g.
`/proj/.../builds/<build>/runs/<run>/checkpoints/<run_name>_node2`. Before the trainer
starts, rank 0 hardlinks every **complete** `checkpoint-*` from there into this run's
checkpoint directory (falling back to a copy across filesystems; incomplete ones are skipped)
and the trainer resumes from the highest — weights, optimizer, scheduler, RNG and data
position. The other ranks wait up to 900 s for rank 0 to finish. The step records
`resumed_from` step metadata. Hardlinks, not copies: no data is duplicated and the source is
never modified.

- **The world size must match the source's.** A ZeRO-3 checkpoint holds one optimizer shard
  per rank, so the step refuses a source saved by a different number of training ranks rather
  than letting DeepSpeed fail after allocation.
- **The schedule must match too** (`max_steps`, batch sizes, learning rate). The trainer
  restores the scheduler's state, not its shape; nothing checks this.
- **No complete checkpoint is an error,** not a fresh start: starting from step 0 would
  silently spend the whole schedule again.
- **With `emit_checkpoint_artifacts` on,** every seeded checkpoint is emitted as
  `checkpoint_<N>` again (`resume_emit_seeded: true`), so its downstream export and eval
  targets re-run — which is also what keeps them from waiting forever on a rung this run never
  writes. Set `resume_emit_seeded: false` when the stopped run already exported and evaluated
  the seeded rungs and the relaunch's ladder names only the rungs still to come.

### On-policy

`vllm_num_servers > 0` dedicates the **last** N nodes to serving the student under vLLM and
trains on the rest; the renderer rejects a count that leaves no trainers, or on-policy on a
single node. On-policy continues from a good off-policy checkpoint — point
`model_name_or_path` at that checkpoint, not the base overlay. `--use_vllm` is passed to the
trainer explicitly in both directions from `vllm_num_servers`: emitted only negatively, an
on-policy run would allocate a vLLM node and then generate locally, leaving the server idle
(build `d77546a9` did exactly that and died in the local path).

With `vllm_server_url` set, the server lives in another target's allocation and every node
here trains. This path has **not yet run on a cluster**: the trainer pushes updated student
weights to the server over NCCL, so an external server means a NCCL process group spanning two
LSF allocations, and whether that works is the open question this path exists to answer. The
renderer refuses `vllm_server_url` with `vllm_num_servers: 0` (the run would train off-policy
while a server sat idle) or with `lmbda: 0` (the student would never generate).
