# distill-logit-precompute

Runs the teacher **once** over the corpus and keeps the top-K logits per assistant token, so a
later training run can do forward KL against a file instead of holding the teacher in memory.

| | |
|---|---|
| **Type** | `data_generation` |
| **Environment** | SkyPilot, LSF only (`subtypes: [lsf]`) |
| **Image** | `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` (prebuilt; builds none) |
| **GPU** | yes — this is the only step that holds the teacher without a student |
| **Nodes** | 1 only; more is refused, and the reason is unusual (below) |
| **Emits** | `teacher_logits` (`type: fileset`) → `<output_dir>/{shards/,index.jsonl,meta.json}` |
| **Wired into** | **no recipe** — deliberately |

## What this is, and what it can never be

It is an optimization of the **off-policy** path, nothing more. The teacher's top-K distribution
over the corpus is fixed once the corpus and the teacher are fixed, so computing it once and
reading it back is strictly cheaper than loading a 30B teacher alongside the student every epoch.

It is **not** a route to on-policy GOLD, and cannot be turned into one. On-policy means the
*student* generates and the teacher scores text that does not exist until training time — there is
nothing to precompute. The step's name invites exactly that misreading.

## Why no recipe wires it

Its only consumer is `distill-sft`'s `precomputed_logits_dir`, and **setting that key
turns that step from the SFT control into a forward-KL distillation arm.** A recipe wiring both is
no longer running a control, so it has to say so in its own README. Upstream ships it unwired for
the same reason, and this port keeps that.

## The failure mode it is built around

**A partial precompute is structurally valid.** The output is `shards/` + `index.jsonl` +
`meta.json`, and the index describes whatever was actually written. A run that covered a tenth of
the corpus, or skipped every example over `max_length`, or ran against the wrong tokenizer,
produces a directory that loads, memmaps and trains. **Nothing downstream fails.** The symptom is
a slightly worse student, discovered weeks later, with no error anywhere in the pipeline.

Everything unusual about this step follows from that sentence:

- `max_skip_fraction` (default `0.05`) is a **refusal threshold**, not a report.
- `allow_tokenizer_mismatch` defaults false: an index keyed to ids that mean something else to the
  trainer would compute happily.
- `nodes > 1` is **refused**, and here that protects something subtler than a rendezvous:
  sharding is `i % num_nodes == node_id`, so a single node started under a `num_nodes` it does not
  have silently precomputes **one residue class** of the corpus.

## Config

Source delivery (`code_config`) is identical in every ported distillation step — see
[distill-tokenizer-align's USAGE.md](../../distill-tokenizer-align/skypilot/USAGE.md#source-delivery).

| Key | Default | Notes |
|---|---|---|
| `precompute_config.corpus_path` | `""` | `distill-corpus-prep`'s `corpus`. **The same corpus the arm will train on** — the index is keyed to it. |
| `precompute_config.teacher_model_path` | `""` | The teacher, loaded once. |
| `precompute_config.teacher_tokenizer_path` | `""` | Kept **separate** from the model path on purpose, as in the trainer: the tokenizer defining the index's token ids need not be the model directory's own. |
| `precompute_config.check_weight_residency` | `true` | Refuses to launch when GPFS has migrated the teacher's weights to tape — this step reads the teacher and nothing else, so that is the whole step waiting on a recall. Metadata only (`mmlsattr`), never reads a shard, and refuses **only** on an authoritative `OFFLINE`: no `mmlsattr` warns and proceeds, and a hub id rather than a path is skipped. `teacher_tokenizer_path` is not checked — a tokenizer overlay has no shards. |
| `precompute_config.allow_offline_weights` | `false` | Proceed through the refusal, loudly. For when the recall is already under way. |
| `precompute_config.output_dir` | `teacher-logits` | Relative resolves against `$GB_BUILD_WORKDIR`. |
| `precompute_config.top_k` | `256` | The whole size/fidelity trade. Raising it multiplies the artifact. |
| `precompute_config.max_length` | `8192` | Deliberately above the trainer's 4096: a logit file can serve a **longer** training budget than the one it was made for, never a shorter one. |
| `precompute_config.response_template` | `'<|im_start|>assistant\n'` | The trailing newline is **data**, carried as a two-character escape because gbserver's Jinja fill strips a real one. See distill-sft's USAGE for the full account. |
| `precompute_config.max_skip_fraction` | `0.05` | See above. |
| `precompute_config.allow_tokenizer_mismatch` | `false` | Keep it false. |
| `workload.gpus_per_node` / `nodes` | `8` / `1` | `gpus_per_node` is asserted against the visible GPUs; `nodes > 1` is refused. |

## If you do wire it

```yaml
  precompute:
    inputs:
      corpus:
        binding: corpus.corpus
    outputs:
      teacher_logits:
        uri: "env://{{ binding.path }}"
        type: fileset
    steps:
      - step_uri: space://steps/distill/distill-logit-precompute
        config:
          precompute_config:
            corpus_path: "{{ bindings.corpus.binding.path }}"
            teacher_model_path: "$${TEACHER_MODEL}"
            teacher_tokenizer_path: "{{ bindings.teacher_tok.binding.path }}"
```

…and then, in the arm that consumes it, say plainly in that recipe's README that
`precomputed_logits_dir` makes it a distillation arm rather than a control.

## Status in this repo

**Verified on BlueVela**, build `ed6894ee` — one H100, the 64-row smoke corpus, a dense 3B
teacher, align's `teacher_overlay` as the separate tokenizer:

```
[rank 0] done: processed=64 skip_noassist=0 skip_toolong=0
[rank 0] peak alloc = 8.35 GB
[verify] 64 index rows over 64 corpus rows, 64 with logits, 0 skipped (0.00%),
         1 shard(s), 25964 teacher rows x top_k 256
```

`shards/{logits_000000.bin,indices_000000.bin}` + `index.jsonl` + `meta.json`, 39 MB, registered
as a `fileset`. `meta.json` records `response_template` as `'<|im_start|>assistant\n'` — 22 bytes
with a real trailing newline, so the escape transport survives into the artifact's own metadata.
The tokenizer logged `sha256:883975314d587437`, the same hash `distill-eval` reports for
`retagged_student`, which is independent confirmation that the overlay and the retagged student
are the same tokenizer.

**One path that run did NOT exercise:** it produced a single shard. `shard_target_tokens` was set
to 40000 specifically to force several, and 25,964 teacher rows did not reach it — so the
multi-shard write, and the `index_part_*` merge across more than one part, remain untested here.
Forcing them needs a larger corpus or a far smaller `shard_target_tokens`.

It is still wired into **no recipe**, for the reason above — that is a deliberate shape, not a gap
in verification.

## Tests

```bash
make test                                                   # 32 contract tests, no checkout needed
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + 51 ported upstream tests
```
