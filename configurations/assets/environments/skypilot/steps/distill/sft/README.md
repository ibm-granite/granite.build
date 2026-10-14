# sft (SkyPilot / LSF)

Plain SFT on the distillation corpus, with **no teacher**. It is a `training` step that runs on
the LSF SkyPilot environment only (`subtypes: [lsf]`), on one GPU node, in the prebuilt
`docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` image; it builds no image.
With `precomputed_logits_dir` set, the same step becomes forward-KL distillation against a
frozen teacher — see [Two uses](#two-uses-and-the-second-one-is-not-a-control).

> **Developing or testing this step?** See `steps/distill/sft/skypilot/README.md` in the
> granite.build repository for how the step is generated, tested, and published — including the
> ported upstream tests that need a distillation checkout (`GB_DISTILL_CODE_DIR`).

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/sft` URI:

```yaml
steps:
  - step_uri: space://steps/distill/sft
```

## Config contract (`sft_config`)

All fields in these tables live under the step's `config.sft_config`.

The student and the corpus are no longer config fields; they are the required `student`
and `corpus` target inputs — see [Inputs](#inputs) below. The student must be the
**retagged** student (empty or a non-directory binding is refused): a control trained on
the base student would also differ in its embedding rows for the control tokens, so the
comparison would measure two things at once. The corpus must be the tokenizer the student
was built for.

### Optional

| Field | Type | Purpose |
|---|---|---|
| `output_dir` | string | Trainer output directory and the `checkpoint` artifact; relative values resolve against `$GB_BUILD_WORKDIR`. Default: `sft`. |
| `check_weight_residency` | boolean | Runs `src/check_weight_residency.py` on the student before launch, to refuse when GPFS has migrated its weights to tape. See [Weight residency](#weight-residency). Default: `true`. |
| `allow_offline_weights` | boolean | Proceed through a migrated-weights refusal, loudly. For when the recall is already under way. Does not override a weight path that cannot be stat'd. Default: `false`. |
| `response_template` | string | Fallback masking string, used only when the student's chat template has no `{% generation %}` markers. Carried as a literal backslash-n; see [The response template's trailing newline](#the-response-templates-trailing-newline). Default: `'<\|im_start\|>assistant\n'` (single-quoted). |
| `max_length` | integer | Must be the same parameter the corpus was filtered at and the eval steps measure at. Default: `4096`. |
| `per_device_train_batch_size` | integer | Default: `1`. |
| `gradient_accumulation_steps` | integer | Default: `8`. |
| `learning_rate` | string | A number, quoted so that `1e-6` survives YAML 1.1 (which parses bare `1e-6` as a string); converted to a float by the renderer, which refuses a non-number. Default: `"1e-6"`. |
| `num_train_epochs` | number | Must be > 0. Default: `1`. |
| `save_steps` | integer | Must be >= 1: this step runs on preemptable capacity, so a run that never checkpoints loses everything on the first preemption. Default: `100`. |
| `seed` | integer | Default: `42`. |
| `resume` | string | `auto` \| `never` \| `require`. The trainer resumes on the mere **presence** of a checkpoint directory, so this is enforced as preflight validation rather than trusted. Default: `auto`. |
| `deepspeed_config` | string | Accelerate DeepSpeed config. **Relative resolves against the delivered checkout**; a missing file fails the step before launch. Default: `steps/distill-gold-train/configs/deepspeed/accelerate_deepspeed_zero3.yaml`. |
| `use_liger_memory_opt` | boolean | Fused linear+CE loss, no model-internal patching. Default: `false`. |
| `use_liger_swiglu_mlp` | boolean | Liger tiled-MLP patch to `GraniteSWAMLP`. **`true` is refused**: the patch is not part of this collection. Default: `false`. |
| `precomputed_logits_dir` | string | **Setting this stops the step being a control**: it becomes forward-KL distillation against the teacher logits in this directory (an output of the `space://steps/distill/logit-precompute` step). Must be a directory when set. Default: `""`. |
| `kd_top_k` | integer | KD only. Must be >= 1. Default: `256`. |
| `kd_weight` | number | KD only. Must be >= 0, and > 0 when `precomputed_logits_dir` is set. Default: `1.0`. |
| `ce_weight` | number | KD only. Must be >= 0; `kd_weight` and `ce_weight` may not both be 0. Default: `0.0`. |
| `kd_temperature` | number | KD only. Must be > 0. Default: `1.0`. |
| `max_dataset_size` | integer | Debug knob: `-1` = the whole corpus, or a positive row count. `0` and other negatives are refused. Default: `-1`. |
| `extra_config_yaml` | string | YAML mapping for the long tail of `CustomSFTConfig`. Merged **under** the explicit keys, so a collision is an error rather than a silent winner. Default: `""`. |

> **The `kd_*` keys are refused when changed without `precomputed_logits_dir`.** They are read
> only on the precomputed-logits path, so a changed value with no logits directory expresses an
> intent nothing would honour; the renderer stops instead of training plain CE under a config
> that reads as KD.

### Workload (`workload`)

| Field | Type | Purpose |
|---|---|---|
| `gpus_per_node` | integer | **Asserted** against the GPUs visible in the container (`CUDA_VISIBLE_DEVICES`, else `nvidia-smi -L`); a mismatch fails the step. Must match the allocation the target requests. Default: `8`. |
| `nodes` | integer | Only `1` is accepted; more is **refused**. Default: `1`. See [Single node only](#single-node-only). |
| `hf_home` | string | Overrides `HF_HOME` for this step only; empty leaves the launcher's value. Default: `""`. |

### Experiment tracking (`tracking`)

| Field | Type | Purpose |
|---|---|---|
| `clearml_project` | string | Default: `""`. |
| `clearml_run_name` | string | Default: `""`. |
| `wandb_entity` | string | Default: `""`. |
| `wandb_project` | string | Default: `""`. |
| `wandb_run_name` | string | Default: `""`. |

All five empty means tracking off; see [Tracking](#tracking) before setting any of them.

### Monitor (top-level `config`)

| Field | Type | Purpose |
|---|---|---|
| `poll_interval_seconds` | integer | Monitor poll interval. Default: `900`. |
| `log_retrieval_mode` | string | Log retrieval mode. Default: `periodic` — a training run lasts far longer than the library's default `on_completion` mode surfaces anything for, so a stalled run would look identical to a slow one. |
| `log_retrieval_interval_seconds` | integer | Log retrieval interval. Default: `900`. |

### Source delivery (`code_config`)

`code_config` is identical in every ported distillation step and is documented once — see
"Source delivery" in the `space://steps/distill/tokenizer-align` step's README. No field is
required; the defaults clone a public repository unauthenticated.

| Field | Type | Purpose |
|---|---|---|
| `code_dir` | string | Filesystem checkout to use instead of cloning. Default: `""` (clone). |
| `expect_ref` | string | Commit a `code_dir` checkout must be at. Default: `a5d59bc45524a8d75706e20d44ae1a254f273f23`. |
| `repo` | string | Repository cloned when `code_dir` is empty. Default: `https://github.com/laminair/gb-steps-distillation.git`. |
| `ref` | string | Ref checked out after the clone. Default: same commit as `expect_ref`. |
| `workdir` | string | Clone destination, relative to the working directory. Default: `distill-code`. |
| `token_secret` | string | Name of a space secret for a private clone. Default: `""`. |
| `python` | string | Interpreter. Default: `/stage/.venv/bin/python`. |
| `setup_command` | string | Bash run after the source resolves, before the workload. Default: `""` (skipped). |

## Inputs and outputs

### Inputs

The step declares two required inputs, read as `{{ bindings.<name>.binding.path }}`:

| Input | Required | Type | Typical source |
|---|---|---|---|
| `student` | yes | model | `align.retagged_student` (the `space://steps/distill/tokenizer-align` step) |
| `corpus` | yes | dataset | `corpus.corpus` (the `space://steps/distill/corpus-prep` step) |

For the KD arm, `precomputed_logits_dir` stays a config key, wired from the
`space://steps/distill/logit-precompute` step.

`sft_config.student_model_path` and `sft_config.corpus_path` are no longer read; a target
that still sets either fails at run time with a message naming the input to bind instead.

### Outputs

| Output | Type | What it is |
|---|---|---|
| `checkpoint` | `model` | The trainer's `output_dir`. Emitted as `GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:<abs output_dir>`. |

Declare it on the target as `uri: "env://{{ binding.path }}"`, `type: model`. In epic 61's
pipeline it is bound (`train-sft.checkpoint`) as the student of the `space://steps/distill/gold`
step.

## Working directory and paths

The run starts in the step's working directory, `$GB_BUILD_WORKDIR` (falling back to `$PWD`).
`src/` is mounted at `./src`; `./src/run-sft.sh` renders the trainer config with
`./src/render_sft_config.py` and launches upstream's `sft.py` through `accelerate`. The shared
package and `sft.py` come from the delivered checkout, cloned into
`$GB_BUILD_WORKDIR/<code_config.workdir>` (default `distill-code`) or taken from
`code_config.code_dir`; the step points `run-sft.sh` at it with
`SFT_SRC=$CODE_DIR/src/gb_steps_post_training/distillation` and
`LIB_DIR=$CODE_DIR/steps/distill-sft/src/lib`.

- A relative `output_dir` is absolutised against `$GB_BUILD_WORKDIR` before the artifact marker
  is printed, because the monitor hands the path to the `env://` store, possibly from another
  host, and a relative `env:` URI is rejected at config load.
- A relative `deepspeed_config` resolves against the delivered checkout (`$CODE_DIR`).
- The resolved `student` and `corpus` input paths, and `precomputed_logits_dir`, are passed
  through unchanged, so bind/give them as absolute paths.
- `run-sft.sh` puts its preprocessing cache in `sft-preprocess-cache/` beside `output_dir` (its
  parent directory), so runs writing different output dirs under one parent share it; set
  `SFT_PREPROCESS_CACHE_ROOT` to override.

## Example build.yaml

The warm-up stage, after align and corpus-prep (their targets as in the
`space://steps/distill/corpus-prep` step's example):

```yaml
    train-sft:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        student:
          binding: align.retagged_student
        corpus:
          binding: corpus.corpus
      outputs:
        checkpoint:
          uri: "env://{{ binding.path }}"
          type: model
      steps:
        - step_uri: space://steps/distill/sft
          config:
            compute_config:
              num_nodes: 1
              num_gpus_per_node: $${NUM_GPUS_PER_NODE}
            workload:
              gpus_per_node: $${NUM_GPUS_PER_NODE}   # must match the allocation
              nodes: 1
            sft_config:
              output_dir: "$${RUN_NAME}/train-sft"
              max_length: $${MAX_LENGTH}             # the corpus's own budget
```

`max_length` must be the same parameter the corpus was filtered at and the eval steps measure at.

## Notes and limitations

### Two uses, and the second one is not a control

With `precomputed_logits_dir` **empty** this is the SFT baseline the distillation arms are
measured against. Every number an arm reports is a comparison; without this step the honest
statement about a distillation run is *"it got better at something, possibly just from more
training on this data."* The `arm` line near the top of the log says which of the two ran.

**Set `precomputed_logits_dir` and the same step becomes forward-KL distillation** against a
frozen teacher — the `space://steps/distill/logit-precompute` arm, wearing this step's name. A
recipe that sets it should say so in its own README.

In epic 61's pipeline this step's role is the **SFT warm-up stage** rather than a parallel
control: its checkpoint becomes the GOLD student. That is a recipe-level choice; the step is the
same either way.

The optimization keys (`per_device_train_batch_size`, `gradient_accumulation_steps`,
`learning_rate`, `seed`) have the same names and defaults as the `space://steps/distill/gold`
step, deliberately: a control that trains at a different batch size or learning rate than the
treatment is not a control.

### The response template's trailing newline

The single easiest thing to get silently wrong here, and it is a **transport** problem rather than
a config one.

`sft.py` builds assistant masks from `apply_chat_template`'s `{% generation %}` markers and raises
if the mask is all zeros. `response_template` is its *fallback*: a literal string searched for in
the rendered token ids. Its trailing newline **is the data** — that is where loss masking begins.

But gbserver fills every config string through Jinja, and its `SandboxedEnvironment`s are built
without `keep_trailing_newline`, so **Jinja strips exactly one trailing newline from each value**:

```
'<|im_start|>assistant\n'   ->  '<|im_start|>assistant'
```

A run then masks a span one token off, trains, and reports success — nothing errors, no NaN,
checkpoints written, loss plausible. That was measured on the `space://steps/distill/gold` step
(build `d8470f14`) and is fixed the same way here: the default is **single-quoted YAML holding a
literal backslash-n**, which has no trailing whitespace for anything to strip, and the launcher
decodes it once inside the container.

The decode uses a sentinel — `"$(printf '%b.' "$V")"` then strip the `.` — because `$(...)`
*also* strips trailing newlines, so the obvious form loses exactly the character the mechanism
exists to carry. A test executes both forms and asserts the sentinel version keeps one more byte.

With the `space://steps/distill/tokenizer-align` step in the pipeline the `{% generation %}`
markers are present, so the fallback should never be reached. It matters anyway: it is what runs
if a student arrives without them. A student with no markers **and** an empty
`response_template` is refused before launch.

### Weight residency

`check_weight_residency` refuses to launch when GPFS has migrated the student's weights to tape,
where the first read blocks on a recall with the GPUs already held. It reads metadata only
(`mmlsattr -L`, else `st_blocks` arithmetic) and never reads a shard:

- `mmlsattr` reporting `OFFLINE` is authoritative and refuses; reporting resident clears the file.
- Without `mmlsattr`, the `st_blocks` heuristic can only condemn: a weight file of 16 MiB or more
  with under 50% of its size allocated refuses; otherwise it warns and starts normally.
- A weight path that does not exist or cannot be stat'd also refuses, and `allow_offline_weights`
  does not override that — it is not a tape recall.
- A hub id rather than a path is skipped, not condemned.

`precomputed_logits_dir` is deliberately not checked: it holds no weight shards and is streamed.

### `deepspeed_config`

gb-steps-distillation ships exactly one deepspeed config, under `distill-gold-train`'s own
directory (`steps/distill-gold-train/configs/deepspeed/`) — `distill-sft` there has none of its
own and reuses that one. `configs/distillation/deepspeed/` was the pre-port path (an in-image
`/opt` copy that does not exist in this image) and stayed as a stale default through the port,
until a real run confirmed it missing (build `30e46247`: "deepspeed config not found"). It is an
**accelerate** config, not a bare DeepSpeed one, and it is load-bearing — its own header records a
predecessor whose `offload_param  offload_param_device: none` parsed as *one* key, so the setting
was silently dead. A typo in this file does not fail loudly.

### Boolean flags are rendered as pairs

`use_liger_memory_opt` / `use_liger_swiglu_mlp` are rendered as `--flag`/`--no-flag` pairs.
**This step is where that house rule was learned**: Jinja renders a YAML boolean with Python
casing, so `--use-liger-memory-opt {{ … }}` reached the script as the literal string `False`;
nothing errors, and a shell test against `"true"` is then false forever, silently, in both
directions.

`use_liger_swiglu_mlp` monkey-patches `GraniteSWAMLP.forward`, is needed only for long-context
(>=64K) GraniteSWA training, costs ~2-3x MLP forward, and applies to no model this collection
trains. Its module is not in the collection, so `true` is refused at render time rather than
failing with an ImportError minutes into an allocated run.

### Single node only

`nodes > 1` is **refused**. The script gets no host list, so a multi-node `accelerate launch`
would have no `main_process_ip`, port or `machine_rank` per host. Refusing beats launching one
node while reporting N — which would train a control at 1/N of the declared global batch and
report the declared one. Lifting it needs the provisioner's `RANK`/`TOTAL_NODES`/`MASTER_ADDR`
fed in as the `space://steps/distill/gold` step does.

`gpus_per_node` is asserted for a related reason: a mismatch between what the recipe asked for and
what LSF granted is a run that trains at a different effective batch size than the config says.

### Tracking

All five `tracking` keys empty means off. A step that invented a project name would log every
recipe's runs into one bucket. A *partial* config aborts rather than being completed with a
guess, because neither backend fails on an incomplete config — they log to their own defaults,
which is how a multi-day run ends up somewhere nobody looks.

**Measured for this image: `clearml` and `wandb` are both absent.** `tracking.py` imports them
lazily and only when a project is configured, so empty is safe — but setting any of these five
would fail at import.

### Hub kernels

The launcher does not set `HF_HUB_OFFLINE`, and sets `GOLD_PREFETCH_KERNELS=1` so `run-sft.sh`'s
kernel preflight fetches any required hub kernel missing from the image's cache inside the job.
The image caches `kernels-community/{mamba-ssm,causal-conv1d}` but not
`kernels-community/flash-attn2` (build `0e60920a`), which transformers needs because the renderer
selects `flash_attention_2` for a granite student. This costs a hub fetch per run, since
`/opt/hf-cache` does not persist; a warm shared cache (upstream's
`scripts/bluevela/prefetch-kernels.sh` against a persistent `workload.hf_home`) avoids it.
