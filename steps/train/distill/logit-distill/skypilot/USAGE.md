# gold (SkyPilot / LSF or SLURM)

Distils a small **student** from a larger **teacher** with the GOLD trainer (generalized
JSD, `gb_steps_post_training.distillation.gold`), multi-node via the SkyPilot LSF or SLURM
cloud (one step, both subtypes).
The step renders the trainer's config, assigns each node its role (trainer, or vLLM server
on the on-policy path) and starts `accelerate` with that node's rank. It registers the
checkpoint directory as a `model` artifact, and optionally each intermediate checkpoint as
soon as it is written.

> **Developing or testing this step?** See `steps/train/distill/logit-distill/skypilot/README.md` in the
> granite.build repository for how the step is generated, tested, and published — including
> the opt-in that runs the real two-node BlueVela build test.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/train/distill/logit-distill` URI:

```yaml
steps:
  - step_uri: space://steps/train/distill/logit-distill
```

## Config contract (`gold_config`)

The trainer's settings live under `config.gold_config`. Checkpoint emission, resume and
the monitor are top-level `config` keys. The trainer source has no config field at all —
it's baked into the shared GPU image (see [Source delivery](#source-delivery)).

The student, teacher and corpus are no longer config fields; they are the required
`student`, `teacher` and `corpus` target inputs — see [Inputs](#inputs) below. The
student is also, on the on-policy path, the model the vLLM server nodes serve, and must
share a tokenizer with the teacher (see
[Choosing a student/teacher pair](#choosing-a-studentteacher-pair)). The corpus **must be
think-filtered** (`*_nothink.jsonl`): an assistant completion containing an inline
`<think>...</think>` breaks GOLD's completion extraction.

### Optional

#### Code and run naming

| Field | Type | Purpose |
|---|---|---|
| `ds_config` | string | DeepSpeed ZeRO-3 `accelerate` config, relative to `$DISTILL_HOME` (`/opt/distill`, where the image bakes the trainer and the shared configs). Default: `configs/deepspeed/accelerate_deepspeed_zero3.yaml`. |
| `run_name` | string | Names the checkpoint directory and the rendered config. The node count is appended at run time, from the allocation (`<run_name>_node<N>`), so the two cannot disagree. Default: `gold`. |

#### Weight residency

| Field | Type | Purpose |
|---|---|---|
| `check_weight_residency` | boolean | Refuse to start when the student or teacher path names weights GPFS has migrated to tape. See [Weight residency refusal](#weight-residency-refusal). Default: `true`. |
| `allow_offline_weights` | boolean | Proceed through the OFFLINE refusal anyway; announces itself in the log. For when you know the recall is already running. Default: `false`. |

#### Training hyperparameters

| Field | Type | Purpose |
|---|---|---|
| `num_train_epochs` | number | Epochs over the `corpus` input. Default: `1.0`. |
| `max_steps` | integer | Cap the run by optimizer steps rather than epochs. `0` leaves the key out of the rendered config, so an epoch-bounded run is unaffected; any positive value overrides `num_train_epochs` in the trainer. One epoch of the 802,027-row reference corpus is 4,177 steps at effective batch 192. Default: `0`. |
| `data_seed` | integer | Seed for the training data sampler only. `-1` leaves the key out, so the sampler follows the trainer's `seed` (42). Set it when initialising from another run's checkpoint over the same corpus, or the new run draws the very batches the old one trained on. `0` is a valid seed. Default: `-1`. |
| `learning_rate` | number | Peak learning rate. Rendered as a YAML float. Default: `1.0e-05`. |
| `min_lr` | number | Floor of the `cosine_with_min_lr` schedule; rendered under `lr_scheduler_kwargs`, not at the top level. Default: `1.0e-06`. |
| `warmup_ratio` | number | Warmup fraction of the schedule. Default: `0.05`. |
| `adam_beta2` | number | Adam's `beta2`, in `(0, 1)`. `-1` leaves the key out and the trainer keeps `0.999` (a second-moment horizon of ~1,000 steps; `0.95` is ~20). Default: `-1.0`. |
| `train_sampling_strategy` | string | The training sampler: `random`, `sequential` or `group_by_length`. Empty leaves the key out (the trainer's `random`). `sequential` feeds the corpus in file order, the only way a corpus whose row order is part of its design reaches the trainer in that order. Default: `""`. |
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

#### Pre-launch checks

| Field | Type | Purpose |
|---|---|---|
| `check_corpus_tokenizer` | boolean | Verify the student's tokenizer is the one the corpus was built with, before the config is rendered. Worth turning on for any build that chains the `space://steps/train/distill/data-prep` step into this one, because a mismatch does not error — it trains on mis-segmented text at full speed with a falling loss. Default: `false`. |
| `verify_trainer_accepts_keys` | boolean | Check every emitted key that upstream trl's `GOLDConfig` does not define (the on-policy shaping pair, `min_completion_length`, the lmbda ramp window, `stop_at_step`) against the delivered trainer's dataclass, and refuse before `accelerate` launches. Only emitted keys are checked, so a build leaving them at their defaults never triggers it. Set false only to render deliberately against a trainer you know differs. Default: `true`. |

#### NCCL and distributed diagnostics

| Field | Type | Purpose |
|---|---|---|
| `nccl_debug` | string | `NCCL_DEBUG`. Empty disables NCCL's own logging. `INFO` (with `nccl_debug_subsys: INIT,NET`) shows which transport and HCA each rank selected — the only way to tell an IB path from a silent TCP fallback, or to see where a collective stalls. Default: `""`. |
| `nccl_debug_subsys` | string | `NCCL_DEBUG_SUBSYS`. Default: `INIT,NET`. |
| `nccl_timeout_ms` | integer | Collective timeout (`TORCH_NCCL_TIMEOUT_MS` and `NCCL_TIMEOUT`). 1 h survives a slow 30B teacher forward without spurious aborts; lower it for debugging so a hang reports in minutes. Default: `3600000`. |
| `nccl_enable_monitoring` | boolean | `TORCH_NCCL_ENABLE_MONITORING`: whether torch's monitor thread may abort a stalled collective. The reference launcher disables this; it defaults on here because with it off a deadlock hangs forever holding the allocation, with no diagnostic. Default: `true`. |
| `nccl_ib_hca` | string | `NCCL_IB_HCA` for this run, overriding the environment's `nccl_tuning_file`. Empty => not touched. Keep the tuning file's exclusions when setting it, e.g. on BlueVela `^=mlx5_1,mlx5_6,mlx5_8` drops the mlx5_8 rail on top of the storage NICs. Default: `""`. |

#### On-policy

All default to off-policy. See [On-policy](#on-policy-1). The external vLLM server is no
longer a config field; it is the optional `vllm` mem:// input (see [Inputs](#inputs)) —
binding it requires `vllm_num_servers > 0` and `lmbda > 0`, and makes every node train.
Not yet run on a cluster.

On the SLURM subtype, the vLLM-serving branch runs `ldconfig` before starting the server:
SLURM's pyxis/enroot builds the container in a separate `srun` from the one that injects
the NVIDIA driver, so the image's own `/etc/ld.so.cache` misses `libcuda.so` and Triton's
driver discovery fails with "libcuda.so cannot found!" otherwise. Unconditional and
harmless where the gap doesn't exist — see `steps/train/distill/images/gpu/README.md`.

| Field | Type | Purpose |
|---|---|---|
| `vllm_num_servers` | integer | `> 0` selects on-policy. Without the optional `vllm` input bound, dedicates the **last** N nodes of the allocation to serving the student under vLLM and trains on the rest. Default: `0`. |
| `vllm_mode` | string | `server` = talk to a vLLM HTTP server (the reference launcher's value). Passed to the trainer only on the external-server path. Default: `server`. |
| `vllm_sync_frequency` | integer | How often, in optimizer steps, the trainer pushes updated weights to the server; a larger value samples from a staler policy. Passed to the trainer only on the external-server path. Default: `1`. |
| `top_p` | number | Rollout sampling `top_p`; emitted only on-policy. Default: `0.95`. |
| `use_sampled_opd_loss` | boolean | Sampled on-policy-distillation loss; emitted only on-policy. Prefer `loss_arm: sampled_opd`, which enforces its requirements. Default: `false`. |
| `last_message_only` | boolean | Trainer `last_message_only`; emitted only on-policy. Required `true` by the `sampled_opd` arm. Default: `false`. |
| `clip_alpha` | number | Trainer `clip_alpha`; emitted only on-policy. Default: `0.1`. |
| `opd_importance_sampling` | boolean | Truncated importance sampling for the sampled OPD loss; emitted only on-policy. Default: `false`. |
| `lmbda_schedule` | string | `constant`, or `linear` to ramp `lmbda` from `lmbda_init` up to `lmbda` across training — off-policy early, on-policy later. `linear` needs `vllm_num_servers > 0`; the renderer refuses it otherwise. Default: `constant`. |
| `lmbda_init` | number | Starting `lmbda` for the linear schedule, in `[0.0, 1.0]`. Default: `0.0`. |
| `lmbda_ramp_start_step` | integer | Linear schedule only: the optimizer step the ramp starts at; `lmbda` is `lmbda_init` before it. Lets a resumed run ramp over its own steps. Refused with a constant schedule. Default: `0`. |
| `lmbda_ramp_steps` | integer | Linear schedule only: the ramp's length in steps; `lmbda` holds after it. `0` ramps until `max_steps`. Refused with a constant schedule. Default: `0`. |
| `stop_at_step` | integer | Save and end training at this step, leaving `max_steps` (and the LR schedule built from it) unchanged: what a resume of a longer run needs to train a fixed number of extra steps on its original schedule. Refused past a positive `max_steps`. `0` disables. A retry that resumes from a checkpoint at or past it does not launch the trainer; see [Already at the limit](#already-at-the-limit). Default: `0`. |
| `min_completion_length` | integer | vLLM `min_tokens` for the student's rollout; `> 0` blocks the immediate-EOS mode collapse. Needs `vllm_num_servers > 0`. The trainer sends it as `min_tokens` in every vLLM request (and as `SamplingParams.min_tokens` in colocate mode). Default: `0`. |

### Checkpoint emission and resume (top-level `config`)

| Field | Type | Purpose |
|---|---|---|
| `emit_checkpoint_artifacts` | boolean | Off: exactly one artifact, `checkpoint`, at the end. On: a watcher on rank 0 emits `checkpoint_<N>` for each complete `checkpoint-<N>` directory as it is written. See [Per-checkpoint artifacts](#per-checkpoint-artifacts). Default: `false`. |
| `checkpoint_watch_interval_seconds` | integer | How often that watcher polls. The monitor's log-retrieval interval is the other half of the latency. Default: `60`. |
| `resume_from_checkpoint_dir` | string | The `<run_name>_node<N>` checkpoint directory of an earlier run of this step to resume from. See [Resume](#resume). Default: `""` (no resume). |
| `resume_emit_seeded` | boolean | Whether seeded checkpoints are emitted as `checkpoint_<N>` again. Only has an effect with both `resume_from_checkpoint_dir` and `emit_checkpoint_artifacts` set. Default: `true`. |
| `train_dir` | string | A training directory that is stable for the whole build: the checkpoint directory becomes `<train_dir>/checkpoints/<run_name>_node<N>` and the rendered config sits under `<train_dir>` too. Empty keeps both under `GB_BUILD_WORKDIR`. See [Stable train_dir](#stable-train_dir). Default: `""`. |
| `rollout_stop_file` | string | A path on a filesystem the rollout server shares. When this step exits non-zero, rank 0 writes it, so a separate rollout server (`student-rollout-generator`'s `vllm_config.stop_file`, set to the same path) stops at once. See [Stopping the rollout server](#stopping-the-rollout-server). Empty turns it off and renders nothing. Default: `""`. |
| `require_progress` | boolean | With `train_dir` set: refuse to train when the two previous attempts in this `train_dir` both failed to raise the highest complete checkpoint, and refuse every attempt after a refusal. Build-level retries only: `retry_workload` relaunches are not checked. See [No-progress refusal](#no-progress-refusal). Default: `true`. |

### Monitor (top-level `config`)

| Field | Type | Purpose |
|---|---|---|
| `poll_interval_seconds` | integer | Monitor poll interval. Default: `900`. |
| `log_retrieval_mode` | string | Default: `periodic`, because a distillation run lasts hours and `on_completion` surfaces nothing until the end. |
| `log_retrieval_interval_seconds` | integer | Periodic log retrieval interval. Default: `900`. |

### Source delivery

Nothing is cloned or mounted at job-launch time. The trainer (`gold.py`,
`custom_gold_trainer.py`, `run_vllm_serve.py` and their shared helpers) is baked into the
shared GPU image (`steps/train/distill/images/gpu/Dockerfile` COPYs `steps/train/distill/src/`,
copied from `gb-steps-distillation` commit `a5d59bc45524a8d75706e20d44ae1a254f273f23`), with
the local patches listed in `steps/train/distill/src/VENDORED.md` folded into that copy. `$PYBIN` and `$CODE_DIR` come from that image
(the venv `python` and `$DISTILL_HOME`, i.e. `/opt/distill`), not from a config field.

### Image

This step runs in the shared GPU image used by every GPU distillation step (`gold`, `sft`,
`logit-precompute`, `eval`, `vllm-server`, `gen-smoke`, `probe`) — see
`steps/train/distill/images/gpu/README.md`. The image bakes in CUDA, a uv-managed Python 3.12, the
trainer's own pinned deps (torch, transformers, trl, vllm, deepspeed, etc.) via
`uv sync --locked` at build time, and the vendored trainer source itself, so nothing
installs or clones at job-launch time. A dependency-pin bump means rebuilding and
republishing this image (and bumping every GPU step's `image_id`), not a per-run resync.

## Inputs and outputs

### Inputs

The step declares its inputs; the run script reads path inputs as
`{{ bindings.<name>.binding.path }}` and the `mem://` `vllm` input as
`{{ bindings.vllm.binding.state }}`:

| Input | Required | Type | Typical source |
|---|---|---|---|
| `student` | yes | model | `align.retagged_student`, or the `checkpoint` of `space://steps/train/distill/student-sft` |
| `teacher` | yes | model | a direct `uri:` to the teacher directory |
| `corpus` | yes | dataset | `corpus.corpus` (the `space://steps/train/distill/data-prep` step) |
| `vllm` | no | `mem://` | `<server>.vllm_url` (the `space://steps/train/distill/student-rollout-generator` step) |

```yaml
inputs:
  student:
    binding: align.retagged_student   # or an SFT target's checkpoint
  corpus:
    binding: corpus.corpus
  teacher:
    uri: "env:///proj/.../teacher"
    type: model
  vllm:                               # external-server on-policy path only
    binding: vllm-server.vllm_url
```

`gold_config.model_name_or_path`, `teacher_model_name_or_path`, `dataset_name` and
`vllm_server_url` are no longer read; a target that still sets one fails at run time with a
message naming the input to bind instead.

### Outputs

| Output | Type | What it is |
|---|---|---|
| `checkpoint` | `model` | `$GB_BUILD_WORKDIR/checkpoints/<run_name>_node<N>` (`<train_dir>/checkpoints/<run_name>_node<N>` with `train_dir` set), the trainer's output directory holding its `checkpoint-<N>` subdirectories. Declared optional in the step. |
| `checkpoint_<N>` | — | Only with `emit_checkpoint_artifacts: true`: one per complete `checkpoint-<N>` directory. Not declared by the step; the target must declare a matching `checkpoint_<N>` output for every rung it binds. |

Markers, printed on rank 0 only (the executor streams every node into one driver log, so an
unguarded marker would register N artifacts):

```
GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:<CKPT_DIR>
GB_ARTIFACT_ID:checkpoint_<N> GB_ARTIFACT_PATH:<CKPT_DIR>/checkpoint-<N>
```

The `<N>` in `<run_name>_node<N>` comes from the allocation, not from a parameter, so it
always describes the run that produced it. The trainer's ZeRO-3 config saves 16-bit weights,
so a checkpoint is HF-native; downstream, the `space://steps/misc/hf-export` step selects
and checks one.

## Working directory and paths

`run` executes on **every** node with that node's own rank, read from SkyPilot's own
generic topology vars (`SKYPILOT_NODE_RANK`, `SKYPILOT_NUM_NODES`, `SKYPILOT_NODE_IPS`,
`SKYPILOT_NUM_GPUS_PER_NODE` — present on both the lsf and slurm subtypes, unlike LSF's
provisioner-specific `RANK`/`TOTAL_NODES`/`MASTER_ADDR`/`MASTER_PORT`). All paths are under
`$GB_BUILD_WORKDIR` (the per-run workdir, on `/proj` on BlueVela and identity-mounted into
the container), falling back to the current directory when unset:

- `$DISTILL_HOME/steps/logit-distill` — the step's `src/` (config renderer and residency
  check), baked into the image.
- `/opt/distill` (`$DISTILL_HOME`, also `$CODE_DIR`) — the trainer package (`src/`) and the
  shared `configs/`, baked into the image. No per-run clone, so no rank-0-clones/others-wait
  delay. `ds_config` resolves relative to it.
- `gold_config_<run_name>_node<N>.yaml` — the rendered trainer config, printed to the log by
  rank 0.
- `checkpoints/<run_name>_node<N>/` — the trainer's `--output_dir`, and the `checkpoint`
  artifact.

The resolved `student`, `teacher` and `corpus` input paths, and
`resume_from_checkpoint_dir`, are used as given; pass/bind absolute paths.

## Example build.yaml

Two-node off-policy run; `accelerators` is **per node**:

```yaml
granite.build:
  name: gold-example
  targets:
    train-gold:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        student:
          uri: "env:///proj/.../student_overlays/granite-4.1-3b-base-hub"
          type: model
        teacher:
          uri: "env:///proj/.../teacher_overlays/granite-4.2-30b"
          type: model
        corpus:
          uri: "env:///proj/.../subsampled_0.4_shuffled_nothink.jsonl"
          type: dataset
      outputs:
        checkpoint:
          uri: "env://{{ binding.path }}"
          type: model
      steps:
        - step_uri: space://steps/train/distill/logit-distill
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
              gradient_accumulation_steps: 12
```

`gradient_accumulation_steps: 12` holds the reference effective batch of 192 on two nodes
instead of four. For a full chain (align, corpus-prep, optional SFT, gold, export), see
`recipes/distill/distill` in the granite.build repository, and its `README.md`.

## Notes and limitations

### What supplies what

| Piece | Source |
|---|---|
| Pinned deps (torch, transformers, trl, vllm, deepspeed) | baked into the shared GPU image via `uv sync --locked` at build time (see "Image" above) |
| Trainer (`gold.py`, `custom_gold_trainer.py`, DeepSpeed config) | baked into the shared GPU image, from `steps/train/distill/src/` (see [Source delivery](#source-delivery)) |
| Job submission, container import, per-node rank/master | the provisioner (LSF's `bsub`/`blaunch`/enroot, or SLURM's `sbatch`/`srun`/pyxis) |
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
- **LSF and SLURM only** (`subtypes: [lsf, slurm]`) — not yet validated on aws/kubernetes.
- **Multi-node needs a SkyPilot build with multi-node support for the subtype in use.**
  gbserver refuses the launch otherwise rather than silently running on one node.
- **State host memory on LSF.** The LSF provisioner defaults to 16G per node, and a
  host-memory kill presents as a cross-node NCCL hang.

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

Under `lmbda_schedule: linear`, `0 < lmbda < 1` is judged over the whole ramp: a ramp from
`lmbda_init` 0 to `lmbda` 1 has both policy classes at every step but its two ends, so it
satisfies the requirement, while a ramp that never moves is judged by its one value.

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

### Per-checkpoint artifacts

With `emit_checkpoint_artifacts: true`, a background watcher on rank 0 polls the checkpoint
directory during training and emits `checkpoint_<N>` per HF `checkpoint-<N>` directory as
soon as it is completely written (`model.safetensors` or, for sharded weights,
`model.safetensors.index.json`, plus `tokenizer.json` and `trainer_state.json`: the same
rule the move-aside, the seed, the re-emit and the progress check use). A recipe that declares matching `checkpoint_<N>` outputs
can then export and evaluate rung N while the run is still training — see
`recipes/distill/distill`, whose `export-N` targets do exactly that (on an earlier 350m run
it turned ~36 GPU-h of eval from a serial tail into work overlapped with a 13-hour epoch). After the trainer exits, the step
sweeps once more, which catches the final checkpoint of an epoch-bounded run.

The emitted ids are a **contract** with the recipe: a rung whose checkpoint directory the
trainer never writes is an artifact never emitted, and a target bound to it waits forever
with no error. Hence every rung should be a multiple of `save_steps`, and hence the step
fails loudly if it emitted **zero** checkpoints.

### Resume

The checkpoint directory is under `GB_BUILD_WORKDIR`, which is new for every build **and**
target run, so the trainer's own auto-resume (`gold.py` resumes when its `output_dir` already
holds a `checkpoint-*`) never finds an earlier run's checkpoints. `resume_from_checkpoint_dir`
names the `<run_name>_node<N>` directory of an earlier run, e.g.
`/proj/.../builds/<build>/runs/<run>/checkpoints/<run_name>_node2`. Before the trainer
starts, rank 0 hardlinks every **complete** `checkpoint-<N>` from there into this run's
checkpoint directory (falling back to a copy across filesystems; incomplete ones, and names
HF never loads such as `checkpoint-final`, are skipped) and the trainer resumes from the
highest `checkpoint-<N>` there — weights, optimizer, scheduler, RNG and data position. The
other ranks wait for rank 0 to finish, and stop at once if it fails (at most 900 s if it
dies outright). The step records `resumed_from` step metadata: the source's highest complete
checkpoint. Hardlinks, not copies: no data is duplicated and the source is never modified.

- **The world size must match the source's.** A ZeRO-3 checkpoint holds one optimizer shard
  per rank, so the step refuses a source saved by a different number of training ranks rather
  than letting DeepSpeed fail after allocation.
- **The schedule must match too** (`max_steps`, batch sizes, learning rate). The trainer
  restores the scheduler's state, not its shape; nothing checks this.
- **The save and logging grid are this run's.** HF Trainer restores `trainer_state.json`
  whole, `save_steps` and `logging_steps` included, so a run seeded from a source saved every
  1000 steps would save every 1000 whatever its own `save_steps`. Rank 0 rewrites both in the
  resumed checkpoint's `trainer_state.json` and logs each change. That one file is replaced
  rather than edited, so it is no longer a hardlink and the source keeps its own.
- **An automatic retry of a resumed run resumes from its own progress.** A `retry_workload`
  relaunch reuses `GB_BUILD_WORKDIR`, so the checkpoint directory still holds what the
  failed attempt wrote, and HF resumes from its highest `checkpoint-<N>`, not necessarily
  the source's. The world-size check and the rewrite apply to that one. A checkpoint the
  failed attempt was still writing (rank 0's files missing, or fewer ZeRO shards than
  ranks) is moved aside to `incomplete-checkpoint-<N>.<timestamp>`, so the retry resumes
  from the highest complete one rather than failing on it; the per-run workdir, and these
  with it, is reaped when the target run tears down. This all happens only when
  `resume_from_checkpoint_dir` or `train_dir` is set: a retry of a run with neither goes
  straight to `gold.py`'s own resume.
- **A seeded checkpoint is never moved aside for its shard count.** On a later attempt the
  move-aside meets the seeded checkpoints again; one whose name is complete in the source
  carries the source's ZeRO shards, so the shard rule is skipped for it.
- **No complete checkpoint is an error,** not a fresh start: starting from step 0 would
  silently spend the whole schedule again.
- **With `emit_checkpoint_artifacts` on,** every seeded checkpoint is emitted as
  `checkpoint_<N>` again (`resume_emit_seeded: true`), so its downstream export and eval
  targets re-run — which is also what keeps them from waiting forever on a rung this run never
  writes. Set `resume_emit_seeded: false` when the stopped run already exported and evaluated
  the seeded rungs and the relaunch's ladder names only the rungs still to come.

### Already at the limit

With `resume_from_checkpoint_dir` or `train_dir` set and a positive `stop_at_step` (or,
when that is `0`, a positive `max_steps`), rank 0 compares the checkpoint the trainer
would resume from against that limit. At or past it there is nothing left to train:
the trainer's `stop_at_step` check runs at the end of a step and HF trains a step before
looking (and stops at `max_steps` on resume only on an epoch boundary), so launching it
would train one step past the limit and save an off-ladder checkpoint. Instead rank 0
logs `logit-distill: checkpoint-<N> is at stop_at_step <S> or past it: training is
already done; not launching the trainer`, still re-emits and emits as usual, and hands
the other ranks `done:<token>`; every rank, the in-allocation vLLM nodes included, skips
`accelerate launch` and exits 0. A retry after a run that saved its last checkpoint and
then failed (in export, say, or in teardown) therefore finishes at once.

### Stable train_dir

`GB_BUILD_WORKDIR` is per target run: a build-level retry (same build id) gets a new one,
and the target run's teardown reaps it. With `train_dir` empty, a retried build therefore
restarts training at step 0. A recipe sets `train_dir` to a directory that is stable for the
build and outside the per-run tree, e.g. `<WORKDIR_ROOT>/<RUN_NAME>/<build_id>/train`: a
build-level retry finds what the failed attempt wrote and `gold.py` resumes from its highest
`checkpoint-<N>`, and post-build cleanup leaves it in place. A new build gets a new
directory. With `train_dir` set, every attempt, before anything else:

1. **moves a half-written checkpoint aside** (rank 0's files missing, or fewer ZeRO
   shards than ranks) to `incomplete-checkpoint-<N>.<timestamp>`, as a resumed run always
   has. Under `train_dir` these are kept: nothing reaps them.
2. **seeds from `resume_from_checkpoint_dir` only when the directory holds no
   `checkpoint-*` yet.** Otherwise the step logs
   `logit-distill: resume: train_dir already holds checkpoints; ignoring resume_from_checkpoint_dir`
   and skips seeding: what an earlier attempt of this build wrote is further along the
   same lineage. The world-size check and the `trainer_state.json` rewrite described under
   [Resume](#resume) apply to the checkpoint the trainer resumes from, either way.
3. **checks progress** (`require_progress`; see below).
4. **re-emits on resume.** With `emit_checkpoint_artifacts` on, rank 0 emits
   `GB_ARTIFACT_ID:checkpoint_<N> GB_ARTIFACT_PATH:<path>` for every complete
   checkpoint already in the directory, at the same paths, in ascending step order,
   before the trainer starts; the watcher then announces only new ones. This is required,
   not cosmetic: gbserver decides whether to reuse a downstream export only once its input
   binding is delivered again on the retry, and a consumer whose binding is not re-delivered
   is never dispatched, while the build still reports SUCCESS. A checkpoint moved aside in
   step 1 is never emitted. Seeded rungs withheld by `resume_emit_seeded: false` stay
   withheld.

The other ranks wait for rank 0 on the same handshake as a resume and stop at once if it
fails. The final `checkpoint` artifact is unchanged.

### No-progress refusal

With `train_dir` set and `require_progress: true`, rank 0 appends one record per attempt to
`<train_dir>/attempts.jsonl` (its id, the highest complete checkpoint it starts from, and a
`status` of `started` or `refused`), after the move-aside and the seed and before anything is
emitted or `accelerate` launches. An attempt is a **target run** (`GB_TARGETRUN_ID`): a
build-level retry is a new one. An attempt is refused with exit 3 when the **two** most
recent `started` attempts both began at a step at or above the one it would start from,
that is, when two consecutive attempts made no progress:

```
logit-distill: FATAL: no progress over the last 2 attempts (<a>, <b>; still at checkpoint-<N>); not retrying
```

It means both attempts failed before their next save, and this one would fail at the same
point: a deterministic failure (a bad batch, an OOM at a fixed length, a key the trainer
rejects after loading) would otherwise burn every retry the build has, each holding the
allocation until it got there. A **single** attempt that dies before its next save is let
through: on our clusters an NCCL/IB hang or a dead node does that often, and the next
attempt usually gets past it. Progress by any attempt in between resets the count.

The refusal is itself recorded (`status: refused`), and once one attempt has been refused
every later attempt in that `train_dir` is refused at once, naming the refusal:

```
logit-distill: FATAL: attempt <id> was refused for no progress (at checkpoint-<N>); not retrying
```

Those later retries fail fast: they still pay the queue wait to reach rank 0, but train
nothing. Read the failed attempts' logs, fix the cause, and start a new build: a new build
has a new `train_dir`. `require_progress: false` turns the check off.

The cost: after its first failure, a deterministic failure now costs **two** extra
attempts, where it used to cost one: a retry that trains to the same point and fails again
(let through as possibly transient), then one refused at rank 0 after its queue wait.

**What it does not check.** Only build-level retries are judged. A `retry_workload` relaunch
(the environment's `retry` block, which relaunches the workload inside one target run) keeps
its `GB_TARGETRUN_ID`, is let through unrecorded, and is not counted, so it is bounded only
by that block's own retry count. That step-level retry is off unless the environment enables
it.

### Stopping the rollout server

With an external rollout server (the `vllm` input bound to `student-rollout-generator`),
the teardown target that ends the server waits on this target's checkpoint, which a
failed run never emits; gbserver waits for every running target, so the build would sit
RUNNING until the server's `max_lifetime_seconds`. Set `rollout_stop_file` here and the
server's `vllm_config.stop_file` to the same path on a shared filesystem, e.g. under the
build's directory. The step then arms an EXIT handler before anything else that, on
**rank 0** and only on a **non-zero** exit, writes the file atomically (to a temporary
name, then `mv`):

```
<GB_TARGETRUN_ID>
logit-distill rank 0 exited with status <N>
```

The server sees it within its `stop_poll_seconds` and fails its own target. The handler
survives the resume handshake's and the checkpoint watcher's own EXIT traps, and leaves
the exit status unchanged. A successful run never writes it. A rank 0 killed outright
(SIGKILL, a dead node) runs no handler, so the server's lifetime cap stays the backstop.
Empty (the default) renders nothing.

### On-policy

`vllm_num_servers > 0` dedicates the **last** N nodes to serving the student under vLLM and
trains on the rest; the renderer rejects a count that leaves no trainers, or on-policy on a
single node. On-policy continues from a good off-policy checkpoint — bind the `student`
input to that checkpoint, not the base overlay. `--use_vllm` is passed to the trainer
explicitly in both directions from `vllm_num_servers`: emitted only negatively, an
on-policy run would allocate a vLLM node and then generate locally, leaving the server idle
(build `d77546a9` did exactly that and died in the local path).

With the optional `vllm` input bound, the server lives in another target's allocation and
every node here trains. The trainer pushes updated student weights to the server over NCCL,
so an external server means a NCCL process group spanning two allocations. This path has
**run on LSF** (build `9973e766`, against `student-rollout-generator`: all 362 parameter
tensors synced across the two allocations) and has **not yet run on SLURM**. The renderer refuses the `vllm` input with `vllm_num_servers: 0` (the run
would train off-policy while a server sat idle) or with `lmbda: 0` (the student would never
generate).
