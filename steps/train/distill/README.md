# Logit Distillation Steps

Knowledge distillation: a teacher model's next-token distribution is
transferred to a smaller student. The steps run on SkyPilot (SLURM; with legacy support of LSF that will no longer be maintained) from two shared
images; nothing is cloned or mounted at job launch.

## Pipeline order

```
  stage        steps (left to right; a | b = either one)
  ─────        ────────────────────────────────────────────────────────────────
  0 check      preflight-check                                      (optional)
  1 prepare    tokenizer-align  +  data-mixture                  (independent)
  2 data       data-prep  |  data-validate        (validate = reuse a corpus)
  3 train      student-sft (optional)  ->  logit-distill
               └ with student-rollout-generator (on-policy) or teacher-logits (offline)
  4 export     misc/hf-export
  5 measure    teacher-divergence  +  repetition-check
```

1. **Check** (optional) — `preflight-check` answers cheap questions before a long run.
2. **Prepare the student and the data** — `tokenizer-align` and `data-mixture` have no
   dependency on each other; `data-prep` needs both (the aligned student's tokenizer defines
   the corpus). A recipe that reuses an existing corpus runs `data-validate` instead of
   `data-prep`.
3. **Train** — optionally warm-start with `student-sft`, then distill with `logit-distill`.
   On-policy runs add `student-rollout-generator` alongside the trainer; offline runs can use
   `teacher-logits` instead of holding the teacher in memory.
4. **Export** — [`steps/misc/hf-export`](../../misc/hf-export) turns a checkpoint into a
   publishable HF model directory. It is generic, so it lives outside this collection.
5. **Measure** — `teacher-divergence` on the exported model (and once on the untrained
   student, as the baseline); `repetition-check` on each export.

## The steps

| Step | Image | What it does | Inputs | Outputs |
|---|---|---|---|---|
| `preflight-check` | GPU | Answers one cheap question before a long run: does the student load in this image (`load-student`), does its declared tokenizer class change how text segments (`tokenizer`), and which segmentation its weights prefer (`tokenizer-fit`). Produces no artifact; the answer is the `PROBE VERDICT` log line. | — (paths are config) | — |
| `data-mixture` | CPU | Samples several raw SFT splits proportionally into one shuffled corpus file, renaming `conversations` to `messages`. | — | `corpus_source` |
| `tokenizer-align` | CPU | Re-maps the student's token ids onto the teacher's tokenizer and installs a chat template with generation markers, so the two models score the same tokens. | `teacher`, `student`, `chat_template` (opt.) | `retagged_student`, `teacher_overlay`, `student_overlay` |
| `data-prep` | CPU | Turns the source data into deterministic training and eval splits: applies the think-tag policy, checks the completion boundary, and drops rows whose rendered length exceeds `max_length` under the student's tokenizer. | `source_dataset`, `tokenizer` | `corpus` |
| `data-validate` | CPU | When a recipe reuses an existing corpus (its directory is config): refuses it unless its manifest matches this build's tokenizer and policies. | `tokenizer` | `pin_check` |
| `student-sft` | GPU | Supervised fine-tuning of the student on the corpus, so it has seen the chat format before distillation. | `student`, `corpus` | `checkpoint` |
| `teacher-logits` | GPU | Precomputes the teacher's top-K logits over the corpus, for offline forward-KL training without the teacher in memory. No shipped recipe wires it. | `corpus`, `teacher`, `teacher_tokenizer` | `teacher_logits` |
| `student-rollout-generator` | GPU | Serves the student with vLLM so the trainer can sample on-policy completions; publishes its URL. Pair it with a `skypilot-teardown` target. | `model` | `vllm_url`, `cluster_name` |
| `logit-distill` | GPU | The distillation trainer (TRL's GOLD trainer, vendored): generalized JSD, forward/reverse KL, AKL, ULD (cross-tokenizer) and DistiLLM-2-style objectives, off- or on-policy, multi-GPU and multi-node with DeepSpeed ZeRO-3. | `student`, `teacher`, `corpus`, `vllm` (opt.) | `checkpoint` |
| `teacher-divergence` | GPU | Measures how far the student's next-token distribution is from the teacher's on held-out data (JSD, KL, reverse KL, entropy). Not one of the `steps/eval/` benchmarks. | `student`, `teacher` (opt.) | `eval_metrics` |
| `repetition-check` | GPU | Generates from each exported checkpoint and flags degenerate, looping text. | — (rungs are config) | `repetition_report` |

Every step's own `USAGE.md` documents its full config. The GPU steps refuse to start when
the container sees fewer GPUs than were allocated, and log the GPU view either way
(ibm-granite/granite.build#604).

## What else lives here

| Path | What it is |
|---|---|
| `images/gpu/`, `images/cpu/` | The two shared images every step runs on: Dockerfiles, locked dependencies, and the Accelerate/DeepSpeed ZeRO-3 config the trainers launch with. See [`images/gpu/README.md`](images/gpu/README.md) to build and push. |
| `src/` | The trainer package (baked into both docker images). Provenance and local patches: [`src/VENDORED.md`](src/VENDORED.md). |
| `<step>/skypilot/` | One step: `step-template.yaml` (the source of truth), `USAGE.md`, its launcher scripts under `src/`, and its tests. |

## Restarts

Each step writes a completion record next to its output. A restarted recipe re-runs every
target, and a step whose output already matches its inputs and settings skips instead of
redoing the work; one whose output came from different settings refuses rather than
overwrite it.
