# distill-tokenizer-align

Makes a raw Granite base student able to speak the teacher's ChatML, and emits the three
tokenizer artifacts every later distillation step reads.

| | |
|---|---|
| **Type** | `data_processing` |
| **Environment** | SkyPilot, LSF only (`subtypes: [lsf]`) |
| **Image** | `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` (prebuilt; this step builds none) |
| **GPU** | none needed — two tokenizers and one embedding matrix, on CPU |
| **Emits** | `retagged_student`, `teacher_overlay`, `student_overlay` (all `type: model`) |

## Why this step exists

A Granite directory's `tokenizer_config.json` declares `tokenizer_class: "GPT2Tokenizer"`.
`AutoTokenizer` honours that and builds **that class**, which rebuilds its backend from
`vocab`+`merges` and installs a plain `ByteLevel(use_regex=True)` — discarding whatever
`pre_tokenizer` `tokenizer.json` stored. Nothing errors, and the result still reports
`is_fast=True`: **class identity is the mechanism here, not fast-versus-slow.**

The 4.1 base **student** is where this bites: its `pre_tokenizer` is
`Sequence[Split(regex), ByteLevel]`, and the override discards that `Split`. Two fixes that
look obvious and are both wrong:

- **It is not the legacy `vocab.json` / `merges.txt` sidecars.** Under transformers 5.8 they are
  inert. The overlay excludes them for hygiene, not protection.
- **Deleting `tokenizer_class` is not sufficient.** With a `config.json` present,
  `model_type: granite` resolves through `TOKENIZER_MAPPING_NAMES` to `GPT2Tokenizer` anyway.
  So the overlay builder **pins** the key to `PreTrainedTokenizerFast`.

### What the pin does not do

It stops the class lookup from replacing the rule stored in `tokenizer.json`. It does **not**
decide whether that stored rule is the one the model was **trained** with. For the 4.1/4.2
family it is, which is why the pin is the whole fix for this pairing. For
`granite-5.0-20b-sft` it is not: that directory's stored
`Sequence[Split(regex), ByteLevel(use_regex=False)]` is vestigial, and the model's own
likelihood prefers the imposed plain `ByteLevel` by **17.0–19.8% of total NLL** over 512
documents — so pinning alone would hand a trainer a segmentation the model never saw. A
teacher like that needs the pin **plus** a `pre_tokenizer` transplant, and which rule it was
trained with is a **measurement**, not a reading of its files. Rank candidates on **total
NLL**, never PPL/token: two pre-split rules emit different token counts, so a per-token mean
is not comparable across them.

## The three outputs, and who consumes them

| Artifact | What it is | Consumer |
|---|---|---|
| `retagged_student` | the base student re-embedded onto the teacher's tokenizer, with a chat template installed | `distill-gold`'s `model_name_or_path`; the tokenizer `distill-corpus-prep` tokenizes with |
| `teacher_overlay` | the teacher's tokenizer files only, `tokenizer_class` pinned | the teacher tokenizer for the eval steps — kept separate from the teacher MODEL path on purpose |
| `student_overlay` | the *pre-retag* student's tokenizer files, pinned the same way | `distill-corpus-prep`, as the trustworthy comparison point when it asserts one-tokenizer-per-run |

**All three must be declared in the consuming target's `outputs:`.** An undeclared output makes
the buildrun resolver drop the `NEWARTIFACT` event, and the target then completes with no
output *and* no error — a silent failure that costs a full training run.

## Source delivery

This step ships **no image and no Python of its own** beyond `src/run-align.sh`. The work is done
by `gb_steps_post_training.distillation`, which is delivered at run time from the public repo
[`github.com/laminair/gb-steps-distillation`](https://github.com/laminair/gb-steps-distillation),
pinned to one commit — the same arrangement every other ported distillation step uses.

```yaml
code_config:
  code_dir: ""
  expect_ref: "a5d59bc45524a8d75706e20d44ae1a254f273f23"
  repo: "https://github.com/laminair/gb-steps-distillation.git"
  ref: "a5d59bc45524a8d75706e20d44ae1a254f273f23"
```

**The default is an unauthenticated HTTPS clone** of `repo` at `ref`, into `workdir` under the
step's working directory. No `/proj` checkout and no BlueVela-specific path, so it resolves the same
on any environment that can reach github.com. **No credential reaches the container:**
`token_secret` is empty on purpose, because the repo is public.

On this path the clone is checked out at `ref`, and that is what pins the code: `ref` is a full
commit rather than a branch, so two runs a week apart run the same code. `expect_ref` carries the
same commit so the two cannot be read differently, but it is only *checked* on the pre-staged path
below.

### Bumping the pin

Set `ref` and `expect_ref` to the same new commit of `gb-steps-distillation` — in the recipe, or in
the step default. The default is asserted byte-identical across the ported steps by
`test/test_source_contract.py`, so a default bump has to land in all of them together.

### Using a pre-staged checkout instead

Set `code_dir` to an existing checkout and the clone is skipped. There, `expect_ref` is checked
against the checkout's actual `HEAD` and the step **fails loudly** on a mismatch, because a
silently-moved shared checkout is how two runs that report the same pin end up on different code.
Uncommitted changes are not fatal but are warned about and recorded as `distill_code_dirty` step
metadata.

## Config

| Key | Default | Notes |
|---|---|---|
| `align_config.teacher_model` | `""` | The 4.2 30B teacher **directory**, not an HF repo id. Its tokenizer is what the student is retagged onto. |
| `align_config.student_model` | `""` | The **raw, pre-retag** 4.1 3B base student — the one directory that genuinely mis-segments. |
| `align_config.out_dir` | `align` | Relative values resolve against `$GB_BUILD_WORKDIR`. The three outputs are derived from it and are not separately configurable. |
| `align_config.chat_template` | `templates/chatml_granite_42_generation.jinja` | **Relative resolves against the checkout.** Empty ⇒ `--no-chat-template` plus a loud warning: the student then has no template and GOLD cannot build assistant masks. A template *without* `{% generation %}` markers is worse than none — it makes the masks all-zero, which raises inside the trainer instead of here. |
| `align_config.require_chatml` | `true` | Whether the **teacher's** tokenizer and the retagged student's must speak ChatML. Set it `false` only together with a same-family `chat_template` — see [Markup families](#markup-families-and-require_chatml). |
| `align_config.copy_mode` | `copy` | `copy` \| `symlink` \| `hardlink`. Only `copy` is safe across filesystems. Never symlink an overlay a later step may write to. |
| `align_config.verify` | `true` | Keep it true: `verify()` is what catches a resolved backend disagreeing with `tokenizer.json`, and it is four cheap encoding probes. Gates the retagged student's post-condition, which asserts the ChatML markers only when `require_chatml` is true but checks the pre_tokenizer either way. |
| `align_config.dry_run` | `false` | Resolve and report what would be written, without writing it. Skips the resume marker entirely, in both directions. |

## Markup families and `require_chatml`

Upstream hard-codes `--require-chatml` on the teacher overlay and `require_chatml=True`
on the retagged student's post-condition. That is correct for the pair this step was
written against — granite-4.2 **is** a ChatML family — and wrong as a universal.

granite 4.0 and 4.1 carry `<|start_of_role|>` and `<|end_of_role|>` in
`added_tokens_decoder` and no `<|im_start|>` anywhere; the raw backend returns
`<|im_start|>` as the six bytes `[27, 91, 318, 5011, 91, 29]`. So a granite-4.1 teacher
aborts in stage `[1/4]` under the upstream default, and the message is about ChatML
rather than about the pair, which is the wrong place to start debugging.

`require_chatml: false` is the knob for a **same-family pair** — a granite-4.0 student
with a granite-4.1 teacher, say, where the two `tokenizer.json` files are byte-identical
and there are no new turn tokens for the retag to introduce.

Three things to know before setting it:

* **Set it false only together with a chat template of the teacher's family.** False with
  the default `templates/chatml_granite_42_generation.jinja` is the worst of both:
  alignment succeeds, and the teacher then scores a prompt format it has never seen. The
  step cannot catch that for you — the template is a path, and its contents are not
  compared against the vocabulary.
* **The turn boundary is still checked**, just elsewhere. Stage `[4/4]` derives the
  masking contract from the installed template and fails the step if the marker cannot be
  recovered, so a template whose boundary is undiscoverable never records a completed
  align. Do not read `false` as "unchecked".
* **The pre_tokenizer half of the post-condition is unaffected.** That is the half that
  catches the mis-segmentation this step exists to prevent (26.1 vs 3.29 PPL/token), and
  it is family-independent. It runs whenever `verify` is true.

Note that the student overlay in stage `[2/4]` is built with `--no-require-chatml`
regardless, and always has been: it comes from the **pre-retag** base student, whose
vocabulary contains no turn tokens of any family. Demanding them there would fail a
correct artifact.

## Wiring it

```yaml
targets:
  align:
    environment_uri: space://environments/skypilot/lsf/ibm-bluevela
    outputs:
      retagged_student:
        uri: "env://{{ binding.path }}"
        type: model
      teacher_overlay:
        uri: "env://{{ binding.path }}"
        type: model
      student_overlay:
        uri: "env://{{ binding.path }}"
        type: model
    steps:
      - step_uri: space://steps/distill/distill-tokenizer-align
        config:
          align_config:
            teacher_model: "$${TEACHER_MODEL}"
            student_model: "$${STUDENT_MODEL}"
            out_dir: "$${RUN_NAME}/align"
          launcher_config:
            resources:
              cluster: "bluevela"
              zone: "normal"
```

A consumer then binds one of them:

```yaml
  corpus:
    inputs:
      tokenizer:
        binding: align.retagged_student
    steps:
      - step_uri: space://steps/distill/distill-corpus-prep
        config:
          corpus_config:
            tokenizer: "{{ bindings.tokenizer.binding.path }}"
```

## Resume

A recipe on a preemptable queue is **restarted, not resumed**, so every step must answer "has
this already been done, and with what?" for itself. `align_state` is this step's answer:

| Exit | Meaning |
|---|---|
| `0` | nothing recorded; do the work |
| `64` | recorded under this **exact** expectation; publish the paths and exit |
| `65` | recorded under a **different** expectation, or a declared output is gone — refuse |

`65` refuses rather than rebuilding because this step's output is what four later steps compare
their tokenizers against, so quietly replacing it with something built from different inputs is
the expensive mistake. The completion marker is written **last**, after the ChatML post-condition,
so a marker can never describe an unverified student.

## Tests

```bash
make test                                                   # 33 contract tests, no checkout needed
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + 39 ported upstream tests
```

The contract tests read the template and `src/run-align.sh` directly and assert, among other
things, that every declared artifact id is actually printed (and vice versa), that booleans are
rendered as `--flag`/`--no-flag` pairs rather than `--flag {{ value }}`, and that every flag the
template passes is one the script parses. A green run with no checkout means the **step contract**
holds — not that the upstream code does.
