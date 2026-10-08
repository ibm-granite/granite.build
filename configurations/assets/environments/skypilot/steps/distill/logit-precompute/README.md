# logit-precompute (SkyPilot / LSF)

Runs the teacher **once** over the corpus and keeps the top-K logits per assistant token, so a
later training run can do forward KL against a file instead of holding the teacher in memory.
It is an optimization of the **off-policy** path only, and it is wired into **no recipe** on
purpose — see [Notes and limitations](#notes-and-limitations) before using it.

> **Developing or testing this step?** See `steps/distill/logit-precompute/skypilot/README.md`
> in the granite.build repository for how the step is generated, tested, and published, and
> which parts are ported from upstream.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/logit-precompute` URI:

```yaml
steps:
  - step_uri: space://steps/distill/logit-precompute
```

## Config contract (`precompute_config`)

The workload fields live under `config.precompute_config`; the allocation under
`config.workload`; the trainer source under `config.code_config`.

### Required

| Field | Type | Purpose |
|---|---|---|
| `corpus_path` | string | The `corpus` output of the `space://steps/distill/corpus-prep` step. **The same corpus the arm will train on** — the index is keyed to it, so a precompute against a different corpus is a silent mismatch. Empty, or not an existing file, fails the step. |
| `teacher_model_path` | string | The teacher model directory, loaded once. This is the only step that holds the teacher without a student. Empty, or not an existing directory, fails the step. |

### Optional

| Field | Type | Purpose |
|---|---|---|
| `teacher_tokenizer_path` | string | The tokenizer that defines the token ids in the index. Kept **separate** from the model path on purpose, as in the trainer: it need not be the model directory's own (for example the `teacher_overlay` output of the `space://steps/distill/tokenizer-align` step). Default: `""`, which uses `teacher_model_path`. |
| `check_weight_residency` | boolean | Refuse to launch when GPFS has migrated the teacher's weights to tape — this step reads the teacher and nothing else, so that is the whole step waiting on a recall. See [Weight residency check](#weight-residency-check). Default: `true`. |
| `allow_offline_weights` | boolean | Proceed through the OFFLINE refusal, loudly. For when the recall is already under way. Default: `false`. |
| `output_dir` | string | Where the artifact is written. Relative values resolve against `$GB_BUILD_WORKDIR`. Default: `teacher-logits`. |
| `top_k` | integer | Top-K logits kept per assistant token. The whole size/fidelity trade: 256 is upstream's measured default, and raising it multiplies the artifact. Default: `256`. |
| `dtype` | string | Dtype the teacher is loaded in. Stored logits are float16 regardless. Default: `bfloat16`. |
| `max_length` | integer | Maximum sequence length; longer records are skipped. Deliberately above the trainer's 4096: a logit file can serve a **longer** training budget than the one it was made for, never a shorter one. Default: `8192`. |
| `shard_target_tokens` | integer | Target tokens per shard. An implementation detail of reading the artifact back, not a knob with a scientific meaning. Default: `4000000`. |
| `batch_size` | integer | Teacher forward-pass batch size. Default: `4`. |
| `seed` | integer | Random seed. Default: `0`. |
| `ignore_documents` | boolean | Drop (`true`) or keep (`false`) records whose `documents` list is non-empty; such a record depends on grounding text granite's chat template does not render. Default: `false`. |
| `response_template` | string | The completion boundary. The trailing newline is **data**, carried as a two-character escape `\n` in single-quoted YAML. See [Response template escape](#response-template-escape). Default: `'<|im_start|>assistant\n'`. |
| `max_skip_fraction` | number | A **refusal threshold**, not a report: the step fails when more than this fraction of records is skipped. Raise it only to measure a short slice on purpose. Default: `0.05`. |
| `allow_tokenizer_mismatch` | boolean | Compute even when the teacher's model and tokenizer disagree on token ids. Keep it false: the index would be keyed to ids that mean something else to the trainer, and the arithmetic would succeed. Default: `false`. |

### Allocation (`workload`)

| Field | Type | Purpose |
|---|---|---|
| `gpus_per_node` | integer | Tensor-parallel width. **Asserted** against the GPUs actually visible in the container, not trusted; a mismatch fails the step. Default: `8`. |
| `nodes` | integer | Must be `1`; `nodes > 1` is **refused** (see [Single node only](#single-node-only)). Default: `1`. |
| `hf_home` | string | If set, exported as `HF_HOME` for the workload, overriding the launcher's `/opt/hf-cache`. Default: `""`. |

### Source delivery (`code_config`)

`code_config` (`code_dir`, `expect_ref`, `repo`, `ref`, `workdir`, `token_secret`, `python`,
`setup_command`) is identical in every ported distillation step and documented once — see
"Source delivery" in the `space://steps/distill/tokenizer-align` step's README. The defaults
clone the public trainer source at a pinned commit, so nothing needs setting.

### Monitor (top-level `config`)

| Field | Type | Purpose |
|---|---|---|
| `poll_interval_seconds` | integer | Monitor poll interval. Default: `900`. |
| `log_retrieval_mode` | string | Log retrieval mode. Default: `periodic` — one forward pass of a 30B teacher over a real corpus is an hours-long job, and `on_completion` would surface nothing until the end, so a stalled run would look identical to a slow one. |
| `log_retrieval_interval_seconds` | integer | Periodic log retrieval interval. Default: `900`. |

## Inputs and outputs

### Inputs

The step declares no `inputs:`; it reads the paths given in `precompute_config`. To take
them from upstream targets, declare the inputs on the target and pass
`{{ bindings.<name>.binding.path }}` into the config fields:

- `corpus_path` ← the `corpus` output of `space://steps/distill/corpus-prep`.
- `teacher_tokenizer_path` ← typically the `teacher_overlay` output of
  `space://steps/distill/tokenizer-align`.
- `teacher_model_path` ← a teacher checkpoint directory, usually a plain path.

### Outputs

| Output | Type | What it is |
|---|---|---|
| `teacher_logits` | `fileset` | `<output_dir>/` holding `shards/` (`logits_*.bin`, `indices_*.bin`), `index.jsonl` and `meta.json`. |

`src/run-precompute.sh` prints the marker once, after the pass and its completeness check
succeed, and also on the already-done skip path:

```
GB_ARTIFACT_ID:teacher_logits GB_ARTIFACT_PATH:<absolute output_dir>
```

The target must declare `teacher_logits` in its `outputs:`; an undeclared output makes the
resolver drop the artifact event and the target completes with no output and no error. The
only consumer is the `precomputed_logits_dir` key of the `space://steps/distill/sft` step —
and setting that key changes what that step is (see [Why no recipe wires it](#why-no-recipe-wires-it)).

## Working directory and paths

`run` starts in the step's per-run working directory; `$GB_BUILD_WORKDIR` (or that directory
when unset) is the base for relative paths. The step's `src/` is mounted at `./src`. The
trainer source is cloned into `<workdir>/distill-code` (`code_config.workdir`) unless
`code_config.code_dir` points at an existing checkout. A relative `output_dir` is
absolutised against `$GB_BUILD_WORKDIR` before the script runs, so the marker always carries
an absolute path. `corpus_path`, `teacher_model_path` and `teacher_tokenizer_path` are passed
through unchanged; give absolute paths.

## Example build.yaml

```yaml
granite.build:
  name: logit-precompute-example
  targets:
    precompute:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        corpus:
          binding: corpus.corpus            # a space://steps/distill/corpus-prep target
        teacher_tok:
          binding: align.teacher_overlay    # a space://steps/distill/tokenizer-align target
      outputs:
        teacher_logits:
          uri: "env://{{ binding.path }}"
          type: fileset
      steps:
        - step_uri: space://steps/distill/logit-precompute
          config:
            compute_config: {num_nodes: 1, num_gpus_per_node: 8}
            launcher_config:
              resources: {accelerators: "H100:8", cluster: "bluevela", zone: "normal"}
            workload:
              gpus_per_node: 8
              nodes: 1
            precompute_config:
              corpus_path: "{{ bindings.corpus.binding.path }}"
              teacher_model_path: /proj/run/teacher
              teacher_tokenizer_path: "{{ bindings.teacher_tok.binding.path }}"
```

…and then, in the arm that consumes it, say plainly in that recipe's README that
`precomputed_logits_dir` makes it a distillation arm rather than a control.

## Notes and limitations

### At a glance

| | |
|---|---|
| **Type** | `data_generation` — it samples from a model (the distinction the `space://steps/distill/corpus-prep` step draws from the other side) |
| **Environment** | SkyPilot, LSF only (`subtypes: [lsf]`) |
| **Image** | `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` (prebuilt; builds none) |
| **GPU** | yes — this is the only step that holds the teacher without a student |
| **Nodes** | 1 only; more is refused, and the reason is unusual (below) |
| **Emits** | `teacher_logits` (`type: fileset`) → `<output_dir>/{shards/,index.jsonl,meta.json}` |
| **Wired into** | **no recipe** — deliberately |

### What this is, and what it can never be

It is an optimization of the **off-policy** path, nothing more. The teacher's top-K distribution
over the corpus is fixed once the corpus and the teacher are fixed, so computing it once and
reading it back is strictly cheaper than loading a 30B teacher alongside the student every epoch.

It is **not** a route to on-policy GOLD, and cannot be turned into one. On-policy means the
*student* generates and the teacher scores text that does not exist until training time — there is
nothing to precompute. The step's name invites exactly that misreading.

### Why no recipe wires it

Its only consumer is the `space://steps/distill/sft` step's `precomputed_logits_dir`, and
**setting that key turns that step from the SFT control into a forward-KL distillation arm.** A
recipe wiring both is no longer running a control, so it has to say so in its own README.
Upstream ships it unwired for the same reason, and this port keeps that. It is a deliberate
shape, not a gap in verification.

### The failure mode it is built around

**A partial precompute is structurally valid.** The output is `shards/` + `index.jsonl` +
`meta.json`, and the index describes whatever was actually written. A run that covered a tenth of
the corpus, or skipped every example over `max_length`, or ran against the wrong tokenizer,
produces a directory that loads, memmaps and trains. **Nothing downstream fails.** The symptom is
a slightly worse student, discovered weeks later, with no error anywhere in the pipeline.

Everything unusual about this step follows from that sentence:

- `max_skip_fraction` (default `0.05`) is a **refusal threshold**, not a report.
- `allow_tokenizer_mismatch` defaults false: an index keyed to ids that mean something else to the
  trainer would compute happily.
- `nodes > 1` is **refused** (next section).

### Single node only

`nodes > 1` is refused, and here that protects something subtler than a rendezvous: sharding is
`i % num_nodes == node_id`, so a single node started under a `num_nodes` it does not have
silently precomputes **one residue class** of the corpus — a structurally valid directory
covering a fraction of the data. `gpus_per_node` is the tensor-parallel width (the teacher is
sharded across exactly that many ranks), which is why it is asserted against the visible GPUs.

### Resume

A failed pass writes no marker and leaves `output_dir` in place on purpose: a re-run picks up
every row already indexed and recomputes only what is missing. Deleting the directory discards
finished work. A re-run against an already-complete directory re-verifies it, skips the pass and
still emits the `teacher_logits` marker.

### Weight residency check

With `check_weight_residency: true`, `src/check_weight_residency.py` checks the teacher's weight
files before the allocation is spent. It reads metadata only (`mmlsattr`, never a shard). It
refuses on an authoritative `OFFLINE`, and also on a weight path it cannot stat at all (that
refusal is not overridable by `allow_offline_weights`). With no `mmlsattr` it warns and proceeds,
and a hub id rather than a path is skipped. `teacher_tokenizer_path` is not checked — a
tokenizer overlay has no shards.

### Response template escape

The trailing newline of `response_template` is where the labelled span begins, and it cannot
cross gbserver's config fill as a real character: Jinja strips exactly one trailing newline from
each value, so `'<|im_start|>assistant\n'` with a real newline would arrive without it and key
the index to a span one token off. It is therefore carried as single-quoted YAML holding a
literal backslash-n, decoded once by the launcher inside the container. The same transport is
used by the `space://steps/distill/sft` step, whose README has the full account.

### Verified on BlueVela

Build `ed6894ee` — one H100, the 64-row smoke corpus, a dense 3B teacher, the
`teacher_overlay` output of `space://steps/distill/tokenizer-align` as the separate tokenizer:

```
[rank 0] done: processed=64 skip_noassist=0 skip_toolong=0
[rank 0] peak alloc = 8.35 GB
[verify] 64 index rows over 64 corpus rows, 64 with logits, 0 skipped (0.00%),
         1 shard(s), 25964 teacher rows x top_k 256
```

`shards/{logits_000000.bin,indices_000000.bin}` + `index.jsonl` + `meta.json`, 39 MB, registered
as a `fileset`. `meta.json` records `response_template` as `'<|im_start|>assistant\n'` — 22 bytes
with a real trailing newline, so the escape transport survives into the artifact's own metadata.
The tokenizer logged `sha256:883975314d587437`, the same hash the `space://steps/distill/eval`
step reports for `retagged_student`, which is independent confirmation that the overlay and the
retagged student are the same tokenizer.

**One path that run did NOT exercise:** it produced a single shard. `shard_target_tokens` was set
to 40000 specifically to force several, and 25,964 teacher rows did not reach it — so the
multi-shard write, and the `index_part_*` merge across more than one part, remain untested here.
Forcing them needs a larger corpus or a far smaller `shard_target_tokens`.
