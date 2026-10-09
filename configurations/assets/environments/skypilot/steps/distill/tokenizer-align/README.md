# tokenizer-align (SkyPilot / LSF)

Makes a raw Granite base student able to speak the teacher's chat format, and emits the
three tokenizer artifacts every later distillation step reads: the student re-embedded
onto the teacher's tokenizer with a chat template installed, and pinned tokenizer
overlays of the teacher and of the pre-retag student. It is a `data_processing` step,
gated to SkyPilot on LSF (`subtypes: [lsf]`), and needs no GPU: it loads two tokenizers
and one embedding matrix on CPU. It runs in the prebuilt image
`docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv` and builds none.

> **Developing or testing this step?** See `steps/distill/tokenizer-align/skypilot/README.md`
> in the granite.build repository for how the step is generated, tested, and published —
> including which tests need a trainer checkout (`GB_DISTILL_CODE_DIR`) and which need
> BlueVela.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/tokenizer-align` URI:

```yaml
steps:
  - step_uri: space://steps/distill/tokenizer-align
```

## Config contract (`align_config`)

All fields in this section live under the step's `config.align_config`. The step also
carries a `config.code_config` block, described in
[`code_config`](#code_config) below.

The teacher and student models are no longer config fields; they are target inputs — see
[Inputs](#inputs) below. `teacher` must be a **directory**, not an HF repo id (the
teacher for the reference pairing is not publicly fetchable), and `student` must be the
**raw, pre-retag** base student directory — the one directory in the pipeline that
genuinely mis-segments.

### Required

| Field | Type | Purpose |
|---|---|---|
| `chat_template` | string | The chat template installed on the retagged student, used when the target does not bind the optional `chat_template` input (a bound input wins). A relative path resolves against the delivered checkout; an absolute path is used as-is. **Set it explicitly:** the default, `templates/chatml_granite_42_generation.jinja`, does not exist in `gb-steps-distillation` at the pinned commit, so the default resolves to a missing file. See [Chat template](#chat-template). |

### Optional

| Field | Type | Purpose |
|---|---|---|
| `out_dir` | string | Where the three outputs are written, as `<out_dir>/{teacher_overlay,student_overlay,retagged_student}`. A relative value resolves against `$GB_BUILD_WORKDIR`. The three paths are derived from it and are not separately configurable. Default: `align`. |
| `require_chatml` | boolean | Whether the **teacher's** tokenizer and the retagged student's must carry ChatML turn tokens. Set it `false` only together with a same-family `chat_template` — see [Markup families and `require_chatml`](#markup-families-and-require_chatml). Default: `true`. |
| `copy_mode` | string | `copy`, `symlink` or `hardlink`. Only `copy` is safe across filesystems. Never symlink an overlay a later step may write to. Default: `copy`. |
| `verify` | boolean | Runs `verify()` on every artifact, plus the post-condition on the retagged student. Keep it `true`: `verify()` is what catches a resolved backend disagreeing with `tokenizer.json`, and it is four cheap encoding probes. The post-condition asserts the ChatML markers only when `require_chatml` is true, but checks the pre_tokenizer either way. Default: `true`. |
| `dry_run` | boolean | Resolve and report what would be written, without writing it; checks a recipe's paths before spending a teacher-sized copy. Skips the resume marker entirely, in both directions, and publishes no artifacts. Default: `false`. |

The booleans are rendered as `--flag` / `--no-flag` pairs, so `false` reaches the script
as an explicit `--no-<flag>`.

### `code_config`

Where the shared distillation package comes from. The block is identical in every ported
distillation step; see [Source delivery](#source-delivery) for how the fields interact.

| Field | Type | Purpose |
|---|---|---|
| `code_dir` | string | An existing checkout to use instead of cloning. Used only when it contains `src/gb_steps_post_training`. Default: `""` (clone). |
| `expect_ref` | string | The commit `code_dir` is expected to be at; checked only on the `code_dir` path, where a mismatch fails the step. Empty disables the check. Default: `a5d59bc45524a8d75706e20d44ae1a254f273f23`. |
| `repo` | string | Git repository cloned when `code_dir` is not used. Default: `https://github.com/laminair/gb-steps-distillation.git`. |
| `ref` | string | Branch, tag or commit checked out after the clone. Default: `a5d59bc45524a8d75706e20d44ae1a254f273f23`. |
| `workdir` | string | Clone destination, relative to the step's working directory. Default: `distill-code`. |
| `token_secret` | string | The **name** of a space secret holding a read-scoped credential, never a value. Consulted only on the clone path. Default: `""` (the repo is public). |
| `python` | string | The interpreter. Not a bare `python`: the image's venv is not on `PATH`, and the system interpreter has no transformers. Default: `/stage/.venv/bin/python`. |
| `setup_command` | string | Bash run after the source resolves and before the workload, from the step's working directory. Empty => skipped. Not needed on this image. Default: `""`. |

## Inputs and outputs

### Inputs

The step declares its inputs, read as `{{ bindings.<name>.binding.path }}`:

| Input | Required | Type | Typical source |
|---|---|---|---|
| `teacher` | yes | model | a direct `uri:` to the teacher directory |
| `student` | yes | model | a direct `uri:` to the RAW base student directory |
| `chat_template` | no | fileset | a template file; unbound uses `align_config.chat_template` |

```yaml
inputs:
  teacher: { uri: "env:///path/to/teacher", type: model }
  student: { uri: "env:///path/to/raw-base-student", type: model }
```

Any scheme the environment can resolve to a directory works (`env://` for a path already
on a reachable filesystem, `hf:///org/repo`, object storage); the step reads the resolved
binding path, never the URI. Both must be **directories**: `retag_student` needs the
teacher's `config.json` (vocab size and the bos/eos/pad ids live there and nowhere else),
so a tokenizer-only directory is not enough for the teacher.

`align_config.teacher_model` and `align_config.student_model` are no longer read; a target
that still sets either fails at run time with a message naming the input to bind instead.

### Outputs

All three are declared under `outputs.required` with `type: model`, and `src/run-align.sh`
registers each with one marker line:

```
GB_ARTIFACT_ID:retagged_student GB_ARTIFACT_PATH:<out_dir>/retagged_student
GB_ARTIFACT_ID:teacher_overlay GB_ARTIFACT_PATH:<out_dir>/teacher_overlay
GB_ARTIFACT_ID:student_overlay GB_ARTIFACT_PATH:<out_dir>/student_overlay
```

| Output | What it is | Consumer |
|---|---|---|
| `retagged_student` | The base student re-embedded onto the teacher's tokenizer, with the chat template installed. Also holds `retag_manifest.json`, `tokenizer_identity.json` and, when a template was installed, `masking.json`. | The `student` input of the `space://steps/distill/gold` step; the tokenizer the `space://steps/distill/corpus-prep` step tokenizes with. |
| `teacher_overlay` | The teacher's tokenizer files only, `tokenizer_class` pinned. | The teacher tokenizer for the eval steps — kept separate from the teacher MODEL path on purpose. |
| `student_overlay` | The **pre-retag** student's tokenizer files, pinned the same way. Not consumed by the retag. | The `space://steps/distill/corpus-prep` step, as the trustworthy comparison point when it asserts one-tokenizer-per-run. |

**All three must be declared in the target's `outputs:`.** An undeclared output makes the
buildrun resolver drop the `NEWARTIFACT` event, and the target then completes with no
output *and* no error — a silent failure that costs a full training run.

The markers are printed from one place, on the normal path, on the already-retagged-input
path, and on a resume skip (exit 64, see [Resume](#resume)), so a restarted step still
hands its consumers the paths. Under `dry_run: true` none are printed.

## Working directory and paths

- `$WORK` is `$GB_BUILD_WORKDIR` (falling back to the current directory). On BlueVela it is
  on `/proj`, identity-mounted into the container at the same path.
- `src/` is file-mounted beside the step and the workload runs `./src/run-align.sh`.
- On the clone path the source lands at `$WORK/<code_config.workdir>` (default
  `distill-code/`), and is removed and re-cloned on every run. `$CODE_DIR/src` is prepended
  to `PYTHONPATH` (the package root, because every module is imported by its full dotted
  path).
- A relative `out_dir` becomes `$WORK/<out_dir>`, and is absolutised before any marker is
  printed: the monitor hands the paths to the `env://` store, possibly from another host.
- A relative `chat_template` becomes `$CODE_DIR/<chat_template>`; an absolute one is used
  as-is.
- The image runs with `HF_HUB_OFFLINE=1` and `HF_HOME=/opt/hf-cache`, so a missing local
  file fails here rather than silently fetching.
- The resolved commit is recorded as step metadata `distill_code_commit`, the source kind
  as `distill_code_source` (`git` or `filesystem`), and on the `code_dir` path the count of
  uncommitted changes as `distill_code_dirty`.

## Example build.yaml

An `align` target on BlueVela, and a consumer that binds its `retagged_student`:

```yaml
granite.build:
  name: tokenizer-align-example
  version: 0.0.1
  targets:
    align:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        teacher:
          uri: "env:///proj/models/granite-4.2-30b"
          type: model
        student:
          uri: "env:///proj/models/granite-4.1-3b-base"
          type: model
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
        - step_uri: space://steps/distill/tokenizer-align
          config:
            compute_config:
              num_nodes: 1
              num_cpus_per_node: 4
            launcher_config:
              resources:
                cluster: "bluevela"
                zone: "normal"
                memory: 64
            align_config:
              out_dir: "my-run/align"
              chat_template: "/proj/granite-build/g4os/chat_templates/granite_4_role_generation.jinja"
              require_chatml: false
              verify: true
              dry_run: false
    corpus:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        tokenizer:
          binding: align.retagged_student
      steps:
        - step_uri: space://steps/distill/corpus-prep
          config:
            corpus_config:
              # ...
```

`require_chatml: false` here is paired with a granite-native (`<|start_of_role|>`)
template, which is how the shipped recipes run it. For a ChatML pair, keep
`require_chatml: true` and point `chat_template` at a ChatML template with
`{% generation %}` markers.

## Notes and limitations

### Why this step exists

A Granite directory's `tokenizer_config.json` declares `tokenizer_class: "GPT2Tokenizer"`.
`AutoTokenizer` honours that and builds **that class**, which rebuilds its backend from
`vocab`+`merges` and installs a plain `ByteLevel(use_regex=True)` — discarding whatever
`pre_tokenizer` `tokenizer.json` stored. Nothing errors, and the result still reports
`is_fast=True`: **class identity is the mechanism here, not fast-versus-slow.**

The 4.1 base **student** is where this bites: its `pre_tokenizer` is
`Sequence[Split(regex), ByteLevel]`, and the override discards that `Split`. Two fixes that
look obvious and are both wrong:

- **It is not the legacy `vocab.json` / `merges.txt` sidecars.** Under transformers 5.8 they
  are inert. The overlay excludes them for hygiene, not protection.
- **Deleting `tokenizer_class` is not sufficient.** With a `config.json` present,
  `model_type: granite` resolves through `TOKENIZER_MAPPING_NAMES` to `GPT2Tokenizer` anyway.
  So the overlay builder **pins** the key to `PreTrainedTokenizerFast`.

The retag itself (`retag_student`) never goes through `AutoTokenizer` — it reads
`tokenizer.json` as raw JSON and pins `tokenizer_class` on its own output — so it is
immune to this by construction, and takes the teacher MODEL directory rather than the
teacher overlay. The overlays exist for the downstream consumers that do load a tokenizer
through transformers.

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

### Stages

`src/run-align.sh` runs, in order: `[0/3]` resume check (skipped under `dry_run`), `[1/4]`
teacher overlay, `[2/4]` student overlay, `[3/4]` retag, the post-condition on the
retagged student (when `verify` is true and not a dry run), `[4/4]` masking contract
(when a chat template is set), and finally the completion marker.

### Chat template

- **Relative resolves against the checkout.** The default,
  `templates/chatml_granite_42_generation.jinja`, was repointed from an upstream in-image
  `/opt` path that does not exist in this image — but `gb-steps-distillation` does not ship
  a `templates/` directory at the pinned commit either, so the default currently resolves to
  a missing file. The shipped recipes override it with an absolute path, e.g.
  `/proj/granite-build/g4os/chat_templates/granite_4_role_generation.jinja` (granite-4.0's
  own template with `{% generation %}` markers added around the assistant span and nothing
  else changed).
- **Empty** => `--no-chat-template` plus a loud warning: the student then has no template,
  the `[4/4]` masking contract is skipped, and GOLD training cannot build assistant masks
  from it (`sft.py:909`). This is only correct if a later step installs one.
- **A template without `{% generation %}` markers is worse than none** — it makes the
  assistant masks all-zero, which raises inside the trainer instead of here.

### Markup families and `require_chatml`

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

- **Set it false only together with a chat template of the teacher's family.** False with
  a ChatML template such as the default is the worst of both: alignment succeeds, and the
  teacher then scores a prompt format it has never seen. The step cannot catch that for
  you — the template is a path, and its contents are not compared against the vocabulary.
- **The turn boundary is still checked**, just elsewhere. Stage `[4/4]` derives the
  masking contract from the installed template and fails the step if the marker cannot be
  recovered, so a template whose boundary is undiscoverable never records a completed
  align. Do not read `false` as "unchecked".
- **The pre_tokenizer half of the post-condition is unaffected.** That is the half that
  catches the mis-segmentation this step exists to prevent (26.1 vs 3.29 PPL/token), and
  it is family-independent. It runs whenever `verify` is true.

The student overlay in stage `[2/4]` is built with `--no-require-chatml` regardless, and
always has been: it comes from the **pre-retag** base student, whose vocabulary contains no
turn tokens of any family. Demanding them there would fail a correct artifact.

### Resume

A recipe on a preemptable queue is **restarted, not resumed**, so every step must answer
"has this already been done, and with what?" for itself. `align_state` is this step's
answer:

| Exit | Meaning |
|---|---|
| `0` | nothing recorded; do the work |
| `64` | recorded under this **exact** expectation; publish the paths and exit |
| `65` | recorded under a **different** expectation, or a declared output is gone — refuse |

`65` refuses rather than rebuilding because this step's output is what four later steps
compare their tokenizers against, so quietly replacing it with something built from
different inputs is the expensive mistake. Any other exit from the check fails the step:
a step that cannot tell whether it has already run must not guess. The completion marker
is written **last**, after the post-condition and the masking contract, so a marker can
never describe an unverified student.

### Already-retagged student input

If the `student` input already contains `retag_manifest.json` — `retag_student`'s own
signature, so it is another run's `retagged_student`, not a raw base — the step skips
`[3/4]` and reuses that directory as this run's `retagged_student` unchanged. Retagging
again is not idempotent: the second template install can land on a tokenizer that already
carries a different template, and the masking contract can then disagree with itself
(measured: a two-assistant-turn probe disagreed at `[96]` vs `[96, 158]`). On this path:

- Both overlays are still built, and `[4/4]` still runs (it only reads the tokenizer).
- `tokenizer_identity.json` is backfilled if absent, for retags that predate it.
- `copy_mode` must be `copy` or `hardlink`; `symlink` fails with `unknown --copy-mode`.
- `dry_run` is not honoured: the copy, overlays and marker are written, and the
  artifacts are published.

The shipped recipes that bind an already-retagged student pay this path's runtime for what
is otherwise a no-op.

### Source delivery

This step ships **no image and no Python of its own** beyond `src/run-align.sh`. The work is
done by `gb_steps_post_training.distillation`, which is delivered at run time from the public
repo `https://github.com/laminair/gb-steps-distillation`, pinned to one commit — the same
arrangement every other ported distillation step uses.

```yaml
code_config:
  code_dir: ""
  expect_ref: "a5d59bc45524a8d75706e20d44ae1a254f273f23"
  repo: "https://github.com/laminair/gb-steps-distillation.git"
  ref: "a5d59bc45524a8d75706e20d44ae1a254f273f23"
```

**The default is an unauthenticated HTTPS clone** of `repo` at `ref`, into `workdir` under
the step's working directory. No `/proj` checkout and no BlueVela-specific path, so it
resolves the same on any environment that can reach github.com. **No credential reaches the
container:** `token_secret` is empty on purpose, because the repo is public. If
`token_secret` is set, the clone authenticates through a short-lived `GIT_ASKPASS` helper
(never a token in the URL), and a `token_secret` naming a secret the space does not have
fails the step.

On this path the clone is checked out at `ref`, and that is what pins the code: `ref` is a
full commit rather than a branch, so two runs a week apart run the same code. `expect_ref`
carries the same commit so the two cannot be read differently, but it is only *checked* on
the pre-staged path below. A clone without `src/gb_steps_post_training` fails the step.

#### Bumping the pin

Set `ref` and `expect_ref` to the same new commit of `gb-steps-distillation` — in the recipe,
or in the step default. The default is asserted byte-identical across the ported steps by
the step's source-contract test, so a default bump has to land in all of them together.

#### Using a pre-staged checkout instead

Set `code_dir` to an existing checkout and the clone is skipped. There, `expect_ref` is
checked against the checkout's actual `HEAD` and the step **fails loudly** on a mismatch,
because a silently-moved shared checkout is how two runs that report the same pin end up on
different code. Uncommitted changes are not fatal but are warned about and recorded as
`distill_code_dirty` step metadata. A `code_dir` that does not contain
`src/gb_steps_post_training` is ignored and the step falls back to cloning `repo`; if `repo`
is empty too, the step fails.

### Image

The image ships no `clearml` and no `wandb`. They are imported lazily, only when a tracking
project is configured, so tracking must stay off.
