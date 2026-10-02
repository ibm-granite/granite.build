# distill-sft

Plain SFT on the distillation corpus, with **no teacher**.

| | |
|---|---|
| **Type** | `training` |
| **Environment** | SkyPilot, LSF only (`subtypes: [lsf]`) |
| **Image** | `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` (prebuilt; this step builds none) |
| **GPU** | yes. `workload.gpus_per_node` is **asserted** against the GPUs visible in the container |
| **Nodes** | 1 only — the script refuses more, see below |
| **Emits** | `checkpoint` (`type: model`) → `<output_dir>` |

## Two uses, and the second one is not a control

With `precomputed_logits_dir` **empty** this is the SFT baseline the distillation arms are
measured against. Every number an arm reports is a comparison; without this step the honest
statement about a distillation run is *"it got better at something, possibly just from more
training on this data."*

**Set `precomputed_logits_dir` and the same step becomes forward-KL distillation** against a
frozen teacher — the `distill-logit-precompute` arm, wearing this step's name. A recipe that sets
it should say so in its own README.

In epic 61's pipeline this step's role is the **SFT warm-up stage** rather than a parallel
control: its checkpoint becomes the GOLD student. That is a recipe-level choice; the step is the
same either way.

## The response template's trailing newline

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
checkpoints written, loss plausible. That was measured on `distill-gold` (build `d8470f14`) and is
fixed the same way here: the default is **single-quoted YAML holding a literal backslash-n**,
which has no trailing whitespace for anything to strip, and the launcher decodes it once inside
the container.

The decode uses a sentinel — `"$(printf '%b.' "$V")"` then strip the `.` — because `$(...)`
*also* strips trailing newlines, so the obvious form loses exactly the character the mechanism
exists to carry. A test executes both forms and asserts the sentinel version keeps one more byte.

With `distill-tokenizer-align` in the pipeline the `{% generation %}` markers are present, so the
fallback should never be reached. It matters anyway: it is what runs if a student arrives without
them.

## Config

Source delivery (`code_config`) is identical in every ported distillation step — see
[distill-tokenizer-align's USAGE.md](../../distill-tokenizer-align/skypilot/USAGE.md#source-delivery).

**`workload`**

| Key | Default | Notes |
|---|---|---|
| `hf_home` | `""` | Overrides `HF_HOME` for this step only. |
| `gpus_per_node` | `8` | **Asserted** against the visible GPUs, not trusted: a mismatch between what the recipe asked for and what LSF granted is a run that trains at a different effective batch size than the config says. |
| `nodes` | `1` | More is **refused**. The script gets no host list, so a multi-node `accelerate launch` would have no `main_process_ip`, port or `machine_rank` per host. Refusing beats launching one node while reporting N — which would train a control at 1/N of the declared global batch and report the declared one. Lifting it needs the provisioner's `RANK`/`TOTAL_NODES`/`MASTER_ADDR` fed in as `distill-gold` does. |

**`sft_config`** — the keys worth commentary:

| Key | Default | Notes |
|---|---|---|
| `student_model_path` | `""` | The **retagged** student. A control trained on the base student would also differ in its embedding rows for the control tokens, so the comparison would measure two things at once. |
| `check_weight_residency` | `true` | Refuses to launch when GPFS has migrated the student's weights to tape, where the first read blocks on a recall with the GPUs already held. Metadata only (`mmlsattr`), never reads a shard, and refuses **only** on an authoritative `OFFLINE` — no `mmlsattr` means a warning and a normal start, and a hub id rather than a path is skipped, not condemned. `precomputed_logits_dir` is deliberately not checked: it holds no weight shards and is streamed. |
| `allow_offline_weights` | `false` | Proceed through the refusal, loudly. For when the recall is already under way. |
| `corpus_path` | `""` | `distill-corpus-prep`'s `corpus`. |
| `deepspeed_config` | `steps/distill-gold-train/configs/deepspeed/accelerate_deepspeed_zero3.yaml` | **Relative resolves against the delivered checkout.** gb-steps-distillation ships exactly one deepspeed config, under `distill-gold-train`'s own directory — `distill-sft` has none of its own and reuses that one. It is an **accelerate** config, not a bare DeepSpeed one, and it is load-bearing — its own header records a predecessor whose `offload_param  offload_param_device: none` parsed as *one* key, so the setting was silently dead. |
| `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, `seed` | `1`, `8`, `1e-6`, `42` | Same keys and defaults as `distill-gold`, deliberately: a control that trains at a different batch size or learning rate than the treatment is not a control. |
| `resume` | `auto` | `auto` \| `never` \| `require`. The trainer resumes on the mere **presence** of a checkpoint directory, so this is enforced as preflight validation rather than trusted. |
| `use_liger_memory_opt` / `use_liger_swiglu_mlp` | `false` / `false` | Rendered as `--flag`/`--no-flag` pairs. **This step is where that house rule was learned**: Jinja renders a YAML boolean with Python casing, so `--use-liger-memory-opt {{ … }}` reached the script as the literal string `False`; nothing errors, and a shell test against `"true"` is then false forever, silently, in both directions. |
| `extra_config_yaml` | `""` | Escape hatch for the long tail of `CustomSFTConfig`. Merged **under** the explicit keys, so a collision is an error rather than a silent winner. |

**`tracking`** — all five empty, meaning off. A step that invented a project name would log every
recipe's runs into one bucket. A *partial* config aborts rather than being completed with a guess,
because neither backend fails on an incomplete config — they log to their own defaults, which is
how a multi-day run ends up somewhere nobody looks.

⚠️ **Measured for this image: `clearml` and `wandb` are both absent.** `tracking.py` imports them
lazily and only when a project is configured, so empty is safe — but setting any of these five
would fail at import.

## Wiring it as the warm-up

```yaml
  train-sft:
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
      - step_uri: space://steps/distill/distill-sft
        config:
          workload:
            gpus_per_node: $${NUM_GPUS_PER_NODE}   # must match the allocation
          sft_config:
            student_model_path: "{{ bindings.student.binding.path }}"
            corpus_path: "{{ bindings.corpus.binding.path }}"
            max_length: $${MAX_LENGTH}             # the corpus's own budget
```

`max_length` must be the same parameter the corpus was filtered at and the eval steps measure at.

## Tests

```bash
make test                                                   # 37 contract tests, no checkout needed
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + 94 ported upstream tests
```

The ported suite includes upstream's parity tests, which compare this renderer against
`distill-gold-train`'s. That renderer is not in granite.build — `steps/distill/distill-gold` ships a
different one, because it drives kd-sandbox's trainer — so they are pointed at the **upstream**
renderer in the delivered checkout. What they then assert is still worth knowing: upstream's
control and its treatment share their guards, their optimization defaults and their tracking
surface.
