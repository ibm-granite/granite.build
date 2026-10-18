# Upstream artifacts: `retagged_student` and `corpus/train.jsonl`

The `distill-gold-colocated` and `distill-gold-trainonly` recipes skip the `align` and
`corpus` targets by binding directly to a **prior build's** outputs via `env://` URIs
(e.g. `RETAGGED_STUDENT_URI` / `CORPUS_URI` default to build `cc33e2e2`). This file
documents what those two artifacts are and how the full-pipeline recipe
(`distill-gold`) produces them, so a reuse binding is an informed choice, not a
black box.

Both are produced by single-node, CPU-only steps run inside the SM90 image. Both are
**tokenizer-specific**, and `RETAGGED_STUDENT_URI` + `CORPUS_URI` **must come from the
same build** — the corpus is filtered against the retagged student's tokenizer, and
nothing downstream re-checks it.

Example paths (from build `cc33e2e2`):
```
.../distill/distill-gold-slurm/cc33e2e2-.../align/retagged_student
.../distill/distill-gold-slurm/cc33e2e2-.../corpus/train.jsonl
```

---

## `align/retagged_student` — step `tokenizer-align-slurm`

**Goal:** make a raw Granite base student able to speak the teacher's ChatML, and emit
the tokenizer artifacts every later step reads.

**Inputs:** the raw **4.1 3B base student** directory (pre-retag) + the **4.2 30B
teacher** directory (its tokenizer is what the student is retagged *onto*) + an optional
chat-template file.

**Entrypoint:** `./src/run-align.sh --teacher-model … --student-model … --out-dir … --chat-template … --copy-mode … [--require-chatml] [--verify] [--no-dry-run]`
(from `github.com/laminair/gb-steps-distillation` @ the pinned SHA). The heavy lifting
is in the shared package: `build_overlay`, `retag_student`, `align_state`, masking.

**What it actually does to the student:**
1. **Pins the tokenizer class.** A Granite dir's `tokenizer_config.json` declares
   `tokenizer_class: "GPT2Tokenizer"`. `AutoTokenizer` honours that and rebuilds a plain
   `ByteLevel(use_regex=True)` backend, **discarding** the `pre_tokenizer` stored in
   `tokenizer.json`. For the 4.1 base student that pre_tokenizer is
   `Sequence[Split(regex), ByteLevel]`, and the override drops the `Split` — silently
   mis-segmenting. Deleting the key isn't enough either (with `config.json` present,
   `model_type: granite` still resolves to `GPT2Tokenizer`). So `build_overlay` **pins
   the tokenizer to `PreTrainedTokenizerFast`**, which keeps the rule stored in
   `tokenizer.json`. (For the 4.1/4.2 family that stored rule is the one the model was
   trained with, so the pin is the whole fix. A teacher like `granite-5.0-20b-sft` would
   additionally need a pre_tokenizer transplant — decided by measuring TOTAL NLL, not by
   reading files.)
2. **Retags the student onto the teacher's ChatML** (`retag_student`) + applies the chat
   template, so the student emits/consumes the teacher's special tokens and role markup.
3. **Aligns model state** (`align_state`) for the retag (e.g. embedding/vocab handling),
   and sets up assistant-token masking.
4. **Verifies** with four cheap encoding probes (`--verify`) that segmentation is correct
   before writing a teacher-sized copy.

**Outputs (all three declared; an undeclared output silently drops the artifact):**
- `retagged_student` → the student directory the trainer loads as `model_name_or_path`,
  **and** the tokenizer `corpus-prep` tokenizes with. This is the `RETAGGED_STUDENT_URI`
  the colocated/trainonly recipes bind.
- `teacher_overlay` → the teacher tokenizer for eval steps (kept separate from the
  teacher *model* path on purpose).
- `student_overlay` → `corpus-prep`'s trustworthy comparison point for its
  one-tokenizer-per-run assertion.

---

## `corpus/train.jsonl` — step `corpus-prep-slurm`

**Goal:** turn a raw conversation dataset into the GOLD training corpus.

**THE CORPUS IS TEXT, NOT TOKENS.** `train.jsonl` is a JSONL of `messages`
conversations and contains **no token ids** — the trainer re-tokenizes at train time
(this is why "reuse" is text-level; see the re-tokenization note in
[README.md](README.md)). It is nevertheless **tokenizer-specific**: every *keep/drop*
decision is measured in tokens, so a corpus built with the wrong tokenizer is not a
crash — it quietly contains examples the trainer will truncate.

**Inputs:** a `source_dataset` (raw conversations) + the `tokenizer` (bound to the
`retagged_student` from the same build — this is what makes the two artifacts a matched
pair).

**Entrypoint:** `./src/prep_corpus.py --dataset … --dataset-split train --tokenizer … --out-dir corpus --max-length 4096 --length-policy drop --think-policy keep --completion-boundary last_message --min-messages 2 --documents-policy drop --eval-fraction 0.0 --seed 42 --max-examples 0 [--no-emit-row-id]`

**What it does to each conversation (defaults shown):**
1. **Length filter** (`--max-length 4096`, `--length-policy drop`): render the
   conversation with the tokenizer and drop it if it exceeds `max_length`. `max_length`
   MUST equal the trainer's and the eval steps' — they are filtered/trained/evaluated as
   a set and nothing re-checks it.
2. **Think-tag policy** (`--think-policy keep`): keep/strip `<think>` reasoning segments.
3. **Completion-boundary enforcement** (`--completion-boundary last_message`): render
   with `return_assistant_tokens_mask=True` and require a **non-empty** assistant mask,
   so there is something to train on.
4. **Minimum messages** (`--min-messages 2`) and **documents policy**
   (`--documents-policy drop`): drop too-short / document-style rows.
5. **Deterministic split** (`--seed 42`, `--eval-fraction 0.0`): carve an eval split if
   asked (here none), reproducibly.
6. **Optional `row_id`** (`--no-emit-row-id` by default): stamp a recomputable id so any
   holder of `train.jsonl` can verify which rows survived filtering.
7. Writes a **manifest** recording what was done (the trainer checks it) plus the
   `train.jsonl` (and an eval split if `eval_fraction > 0`).

**Output:** `corpus` (type `dataset`) → the directory holding `train.jsonl`. This is the
`CORPUS_URI` the colocated/trainonly recipes bind.

It is `DATA_PROCESSING`, not `DATA_GENERATION`: nothing is sampled from a model here
(that is `distill-logit-precompute`'s job).

---

## Why reuse requires the SAME build

`corpus-prep` tokenizes/filters with the `retagged_student`'s tokenizer. Binding
`RETAGGED_STUDENT_URI` and `CORPUS_URI` from **different** builds pairs a corpus with a
tokenizer it wasn't filtered against — no error, just a training set whose length filter
and completion boundaries no longer match what the trainer sees. Keep them from one build
(the defaults both point at `cc33e2e2`).
