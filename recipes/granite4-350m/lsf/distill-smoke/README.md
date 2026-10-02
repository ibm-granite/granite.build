# distill-smoke — granite4-350m distillation, end to end, at smoke scale

[`distill-pipeline-smoke`](../../../granite4-gold-distillation/lsf/distill-pipeline-smoke/README.md)'s
seven-target graph retargeted from the 4.1-3b/4.2-30b reference pair to ours:

| | |
|---|---|
| **student** | the granite-4.0-350m **SFT checkpoint** at `epoch_hf_2` |
| **teacher** | `granite-4.1-3b`, via a pinned-tokenizer copy |
| **objective** | off-policy GOLD, `lmbda 0.0`, `beta 0.5` |
| **scale** | 64 rows, 48 train / 16 eval, 4096 context, 2 optimizer steps, 1 node × 2 H100 |

It is stage 1's graph at smoke scale. Nothing it reports is a measurement.

**Run [`distill-probe`](../distill-probe/README.md) first.** It answers, for about four
GPU-minutes, whether the student loads in this image at all.

## Why the student is a checkpoint and not a base model

Two epochs of SFT on the `all_combined` mixture already happened, and that model's
27-benchmark eval row already exists. Initialising distillation from *those weights*
makes distillation the only difference between the recorded row and the new result.

That is the whole experimental design, and it has three consequences worth stating
because each removes something you might otherwise expect to see here:

- **There is no SFT arm and no control arm.** The control was already run and already
  measured. `INCLUDE_SFT` is kept rather than deleted so a same-corpus control stays
  one `--param` away, but turning it on replaces a measured control with an unmeasured
  one.
- **There is no "SFT warmup" phase.** Your SFT run is the warmup. Within distillation,
  stage 1 (off-policy) is in turn the warmup for stage 2 (on-policy) — which is the
  ordering the published sweep never tested, because its λ=1.0 arm started from a base
  student.
- **`epoch_hf_2` specifically**, because that is the checkpoint `full-eval` and
  `ifrl-full` already point at. If your spreadsheet's after-SFT row is a different
  epoch, change `STUDENT_MODEL` to match it — the recipe test asserts `epoch_hf_2`, so
  it will tell you rather than let the two drift.

## Three things the reference pair could not have found

### 1. align's teacher gate refuses granite 4.x

`run-align.sh` stage `[1/4]` demanded ChatML of the teacher unconditionally. No granite
4.x model has `<|im_start|>` in `added_tokens_decoder` — the raw backend returns it as
the six bytes `[27, 91, 318, 5011, 91, 29]`. The reference pair only worked because
granite-4.2 **is** a ChatML family.

Fixed by adding `align_config.require_chatml` (default `true`, preserved), which this
recipe sets `false`. It required no upstream Python change: `build_overlay` already
accepted both `--require-chatml` and `--no-require-chatml`, and `verify()` already took
it as a keyword.

**`REQUIRE_CHATML: false` is only safe because `CHAT_TEMPLATE` is granite-native.**
False with the default ChatML template is the one combination that fails silently:
alignment succeeds and the teacher then scores a prompt format it has never seen. The
step cannot catch that — the template is a path, and its contents are never compared
against the vocabulary. `test_not_requiring_chatml_forces_a_granite_native_template`
is what catches it instead.

It does not leave the turn boundary unchecked. Stage `[4/4]` derives the masking
contract from the installed template and fails the step if it cannot, and `verify()`'s
`pre_tokenizer` half — the half that catches mis-segmentation — is family-independent
and runs either way.

### 2. The teacher's tokenizer is the broken one this time

`tokenizer.json` is **byte-identical** between `granite-4.0-350m-base` and
`granite-4.1-3b` (sha256 match, 7,153,421 bytes). That reads like a simplification and
is the opposite: both declare `tokenizer_class: GPT2Tokenizer` over the trained
`Sequence[Split(regex), ByteLevel]` pre_tokenizer, so both are identically exposed.

In the reference pair the 30B teacher's pre_tokenizer was *already* plain ByteLevel, so
the override was a no-op on the teacher side and align only had to fix the student.
Here it bites both, and align fixes only the student — so `TEACHER_MODEL` must be a
pinned copy. See [The pinned teacher](#the-pinned-teacher).

Once both sides are pinned there is no cross-tokenizer problem left, which matters:
`distill-gold` exposes no `teacher_tokenizer_path` and none of the trainer's `uld_*`
keys, so the shared-vocabulary JSD path is the only one reachable from any build.

### 3. The student is granitemoehybrid

The first hybrid student in this family. See [`distill-probe`](../distill-probe/README.md).

## The response template

    RESPONSE_TEMPLATE: '<|start_of_role|>assistant<|end_of_role|>'

Note what is absent: a trailing newline, and therefore the entire escaping apparatus
the reference recipes carry. ChatML puts a line boundary after the role header, that
boundary is where loss masking begins, and gbserver fills every config string through
a Jinja environment built without `keep_trailing_newline` — which strips exactly one
real trailing newline per value. Build `d8470f14` lost one that way and trained two
steps against a span one token off *while reporting success*. Hence
`'<|im_start|>assistant\\n'`, single-quoted with a doubled backslash.

granite 4.x closes its role header with `<|end_of_role|>` and puts nothing after it. So
there is no boundary to smuggle: no backslash for `printf '%b'` to reinterpret, nothing
for the fill to strip. The test asserts the *absence* of the escape, so nobody
reintroduces it as cargo cult.

It must still be stated in every target. `render_gold_config.py` defaults it to
`<|im_start|>assistant`, and a wrong marker matches nothing, leaves every label at
`ignore_index`, and trains on **nothing** while producing a loss curve and checkpoints
— `utils.py` has no post-condition on this, unlike `sft.py`.

## Two values that differ from the reference on purpose

**`BETA: 0.5`** rather than the smoke recipe's `0.0`. The trainer short-circuits `0.0`
to forward KL and `1.0` to reverse KL, so `0.5` is the only one of the three that
actually mixes. It is the right default for *this* pair rather than a copied one: the
student has an eighth of the teacher's hidden size and a different architecture class,
so it cannot represent the teacher's distribution and has to choose what to cover.
That is what reverse KL's mode-seeking half buys.

**`GOLD_LEARNING_RATE: 2.0e-06`**, a fifth of the reference recipes' `1e-5`. The
student is a converged SFT checkpoint, not a base model — this is fine-tuning from an
optimum, and that optimum is the baseline row. Irrelevant at two steps; set here so
stage 1 inherits a schedule that was chosen rather than copied.

`USE_LIGER_FUSED_JSD` stays `false`: the teacher's `logits_scaling` is 10.0, which
overflows the fused bf16 JSD kernel and yields NaN loss.

## Before the first run

### The pinned teacher

Built once, out of band. Weights hardlinked, tokenizer files copied, `tokenizer_class`
rewritten. On a login node:

```bash
SRC=<the granite-4.1-3b directory>          # see "finding the teacher" below
DST=/proj/granite-build/g4os/models/granite-4.1-3b-pinned
mkdir -p "$DST"
# Hardlink the weights (same filesystem), copy everything small.
for f in "$SRC"/*.safetensors "$SRC"/*.safetensors.index.json; do
  [ -e "$f" ] && ln -f "$f" "$DST/$(basename "$f")"
done
for f in config.json generation_config.json tokenizer.json vocab.json merges.txt \
         special_tokens_map.json chat_template.jinja; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/"
done
python3 - "$SRC" "$DST" <<'PY'
import json, sys, pathlib
src, dst = (pathlib.Path(p) for p in sys.argv[1:3])
d = json.loads((src / "tokenizer_config.json").read_text())
print("was:", d.get("tokenizer_class"))
d["tokenizer_class"] = "PreTrainedTokenizerFast"
(dst / "tokenizer_config.json").write_text(json.dumps(d, indent=2))
print("now: PreTrainedTokenizerFast ->", dst / "tokenizer_config.json")
PY
```

Finding the teacher — it may already be on the shared filesystem:

```bash
ls -d /proj/granite-build/g4os/models/granite-4.1-3b* 2>/dev/null
ls -d /proj/data-eng/hew/.cache/hf/hub/models--ibm-granite--granite-4.1-3b/snapshots/* 2>/dev/null
```

Take the **instruct** model, `ibm-granite/granite-4.1-3b`, not
`granite-4.1-3b-base`. The base model has no chat template and has not been
instruction-tuned, which makes it a poor teacher for an SFT corpus.

### The chat template

`CHAT_TEMPLATE` points at a granite-native template carrying `{% generation %}`
markers. There is no such template in this repo's upstream checkout or on the cluster
— the only marked one is ChatML — so it is authored here:

    steps/distill/distill-tokenizer-align/skypilot/test-data/granite_4_role_generation.jinja

It is **granite-4.0-350m's own** chat template with generation markers added around the
assistant span and nothing else changed. Derived from the *student's* template rather
than the teacher's on purpose: the two differ — the 350m template injects a default
system message when none is given and the 3b one does not — and `full-eval` renders
prompts with the model's own template, so deriving from the teacher's would change
every prompt relative to the SFT run whose eval row is the baseline.

The markers emit nothing, so rendered output is byte-identical to the unmarked
template. That is asserted rather than assumed:
`steps/distill/distill-tokenizer-align/skypilot/test/test_granite_role_template.py` renders both
over nine conversation shapes — including the default-system-message branch, tool calls
and documents — and diffs them.

Copy it into place:

```bash
mkdir -p /proj/granite-build/g4os/chat_templates
cp steps/distill/distill-tokenizer-align/skypilot/test-data/granite_4_role_generation.jinja \
   /proj/granite-build/g4os/chat_templates/
```

**Then check the assumption underneath it.** The vendored base is the Hub's copy of
granite-4.0-350m's template. The template your SFT mixture was actually tokenized with
is `/proj/granite-build/g4os/chat_template.jinja` (`sft-eval-full-dataset`'s
`CHAT_TEMPLATE_PATH`). Those *should* be the same file — but a difference moves every
prompt:

```bash
diff /proj/granite-build/g4os/chat_template.jinja \
     steps/distill/distill-tokenizer-align/skypilot/test-data/granite_4_role_base.jinja \
  && echo "SAME — the vendored base is the SFT template"
grep -c 'generation' /proj/granite-build/g4os/chat_template.jinja
```

If they differ, re-derive: add the two marker lines to `/proj`'s template instead, and
update the vendored base so the render-identity test keeps guarding the right file.

### Corpus discovery, for stage 1 rather than for this recipe

This recipe's `DATASET` is the gold smoke slice, reused as a plumbing fixture — its
content is irrelevant to every question asked here. **Stage 1 needs the raw ancestor of
the SFT mixture**, which nothing in this repo records, so run this while the smoke
build is queued:

```bash
ls -la /proj/granite-build/g4os/datasets/ \
       /proj/granite-build/g4os/datasets/tokenized/all_combined/
find /proj/granite-build/g4os/datasets -maxdepth 3 -name '*.jsonl' | head -50

python3 -c "
import json
for p in ('/proj/granite-build/g4os/RLHF/data/long_context/rlnothink_300m/mix16_if/train.jsonl',
          '/proj/granite-build/g4os/RLHF/data/long_context/general_data/general_identity/train.jsonl'):
    r = json.loads(open(p).readline()); print(p); print('  keys:', sorted(r))
    m = r.get('messages'); print('  roles:', [x.get('role') for x in m] if m else None)"
```

`distill-corpus-prep` needs raw text with a `messages` list — it renders and tokenizes
itself, because every keep/drop decision is measured on the rendered conversation. A
pre-tokenized directory of 8192-token blocks cannot be un-rendered.

The RL schema dump answers a different question: those two are open-instruct GRPO sets,
so they almost certainly carry prompts and verifiers with no gold assistant turn. That
is not a problem — prompt-only is exactly what stage 2's on-policy arm needs, and it is
why the split between the stages falls where it does. But under
`completion_boundary: last_message` a prompt-only row is dropped and counted, so a
prompt-only file would drop 100% and raise `PrepError: 0 of N records survived`.

Fallback if no raw ancestor exists: re-prep
`/proj/data-eng/hew/gb-steps-collection-post-training/data/distillation/prepped/en-sft-4.1-0.2-16K-v2/merged/train.jsonl`
(802k rows, `{messages, row_id}`, readable from the granitebuild account) from its
`messages` against our tokenizer and template. It is **not** the corpus your SFT stage
used, so that becomes a named divergence rather than a silent one.

## Run

```bash
gb build start -f recipes/granite4-350m/lsf/distill-smoke/build.yaml --space <your-space>
```

Override anything with `--param KEY=VALUE`; `parameters.yaml` sits next to `build.yaml`
and is picked up automatically.

## What to read in the log

In order, per target:

| Target | Look for | A failure here means |
|---|---|---|
| `align` | `--- [1/4] teacher overlay` **completes** | the `require_chatml` fix works — this is the gate |
| | `verified:` lines under the post-condition | the pinned tokenizer round-trips |
| | `--- [4/4] masking contract -> .../masking.json` | the boundary is derivable from the template |
| | 3 × `GB_ARTIFACT_ID:` | outputs published; an undeclared one is dropped silently |
| `corpus` | `kept` > 0, no `empty_assistant_mask` drops | the `{% generation %}` block landed on the assistant span |
| | manifest `target_tokens` non-zero | all-zero means the markers missed |
| `train-gold` | a **finite, non-zero** loss at step 1 | NaN ⇒ liger or `logits_scaling` |
| | no `verify_tokenizer_consistency` refusal | the pinned teacher matches the retagged student |
| | no `ManifestDrift` | prep and the trainer agree on the row count |
| `export` | `normalised: tokenizer_config.json: tokenizer_class ...` | the transformers-4 BFCL image can load it |
| `eval-bfcl` | the handler loads the tokenizer | the `adfdc64` regression has not returned |

`gb build log <id>` gives gbserver's event view — statuses, artifact markers,
exceptions. It does **not** carry the workload's stdout. For steps that use
`$GB_BUILD_WORKDIR` that lands in
`$GB_HOME_DIR/workdir/llm-build-<id>/**/outputs/job.log`; for a generic `command` step
it lands only in SkyPilot's own log on the gbserver host:

```bash
for c in $(gb build log <build-id> | grep -oE 'gb-[0-9a-f]{8}-[0-9a-z]{3}' | sort -u); do
  sed 's/\x1b\[[0-9;]*m//g' ~/sky_logs/$c/*/run.log
done
```

## Known cluster fault

`p2-r03-n1` cannot initialise CUDA — three runs, three `CUDA unknown error` at
`cuda_init` (commit `7bf2051`). Exclude it in `~/.sky/config.yaml`; no surrounding
quotes, no spaces:

    bsub_options:
      R: "select[hname!='p2-r03-n1']"

`MAX_RETRIES: 1` with `target_reuse_enabled: true` covers a single draw of a bad host
without re-running the targets that already succeeded.
