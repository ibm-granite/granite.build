# distill-probe

Two cheap questions that gate the whole granite4-350m distillation family. One
H100, about two minutes each, nothing trained, nothing produced — the answers are
in the log.

**Run this before [`distill-smoke`](../distill-smoke/README.md), and that before
stage 1.**

```bash
gb build start -f recipes/granite4-350m/lsf/distill-probe/build.yaml --space <your-space>
```

## Reading the output

**`gb build log` will NOT show the `PROBE` lines.** That command returns gbserver's
event log — statuses, artifact markers, exceptions. A `command` step's stdout is not in
it, and `build_job_log` does not find it either: that tool looks for
`$GB_HOME_DIR/workdir/llm-build-<id>/**/outputs/job.log`, which is written by steps that
use `$GB_BUILD_WORKDIR`, and the generic `command` step does not.

The workload's real stdout is in SkyPilot's own log on the gbserver host, under the
cluster name that appears in the event log:

```bash
BUILD=<build-id>
for c in $(gb build log "$BUILD" | grep -oE 'gb-[0-9a-f]{8}-[0-9a-z]{3}' | sort -u); do
  echo "=== $c ==="
  sed 's/\x1b\[[0-9;]*m//g' ~/sky_logs/$c/*/run.log | grep PROBE
done
```

The `sed` strips SkyPilot's ANSI colour prefixes, which otherwise sit in front of every
line and defeat a plain `grep -o`.

## 1. `load-student` — does the student load in this image at all?

`granite-4.0-350m` is `granitemoehybrid`. A hybrid model makes transformers lazily
fetch `kernels-community/mamba-ssm`, and that kernel's **cached** build imports
`GreedySearchDecoderOnlyOutput`, which does not exist in this image's transformers
5.8.0. It was measured: build `96868683`.

Every model the reference distillation recipes use — `granite-4.1-3b-base`,
`granite-4.2-30b` — is dense, which is why none of them hit this. Our student is the
first hybrid one, so this is the single largest threat to the family and the cheapest
to retire.

There is counter-evidence worth knowing: `distill-sft`'s `run-sft.sh` states
that upstream's student *is* granitemoehybrid and that job `1136209` failed only on a
cold cache — which suggests the cached build is stale and a run-time Hub fetch of a
newer revision works.

**Measured 2026-09-22, build `b70cd903`: clear.**

    PROBE transformers 5.8.0 torch 2.8.0+cu128
    PROBE model_type granitemoehybrid logits_scaling 4 tie_word_embeddings True
    PROBE LOAD_OK attn=flash_attention_2 params=352379904
    PROBE LOAD_OK attn=eager params=352379904
    PROBE VERDICT loaded=flash_attention_2,eager

The mamba-ssm failure did **not** reproduce, on either attention implementation, so the
cached kernel build is no longer the stale one build `96868683` hit. Both arms are
viable and no `HF_HOME` prefetch is needed. Note `logits_scaling 4` on the student
against the teacher's `10.0` — they differ by 2.5x, so the effective KD temperature is
not the `1.0` the config states.

| Log line | Means |
|---|---|
| `PROBE LOAD_OK attn=flash_attention_2` | Clear. Both arms can run. |
| `PROBE LOAD_OK attn=eager` only | The model loads, the FA2 kernel does not. GOLD is fine (it never sets `attn_implementation`); the SFT control arm is **not** — `sft.py`'s `verify_optimization_stack` aborts every rank on a mismatch. |
| `PROBE LOAD_FAIL` both, with `GreedySearchDecoderOnlyOutput` | The measured failure. Fix by prefetching a newer kernel revision into a persistent `HF_HOME` (`workload.hf_home` exists on every step for this); upstream has `scripts/bluevela/prefetch-kernels.sh`. **Do not** switch to a dense student — that changes the experiment. |
| `PROBE LOAD_FAIL` with `OfflineModeIsEnabled` | The image is offline and the kernel is not cached. Same fix. |

It also prints `logits_scaling` and `tie_word_embeddings`, both of which matter later:
the teacher's `logits_scaling` is 10.0 and overflows the fused bf16 JSD kernel, and
tied embeddings interact with ZeRO-3's 16-bit gather.

## 2. `tokenizer` — is the tokenizer confound real?

This one decides whether the comparison against your recorded after-SFT eval row
needs an extra column.

Both granite 4.x models declare `tokenizer_class: GPT2Tokenizer` in
`tokenizer_config.json`, over a `tokenizer.json` whose `pre_tokenizer` is the trained
`Sequence[Split(regex), ByteLevel]`. `distill-tokenizer-align`'s premise is that
`AutoTokenizer` honours the declared class, imposes a plain ByteLevel, and discards
the `Split` — silently mis-segmenting, at 26.1 vs 3.29 PPL/token. That is why the step
pins `PreTrainedTokenizerFast`.

**Why it is worth checking rather than assuming:** `GPT2TokenizerFast.from_pretrained`
normally loads the backend verbatim from `tokenizer.json`, including its
`pre_tokenizer`. If the override always fired, every public granite 4.x benchmark
number would be depressed, and they look sane. So the effect may be confined to a
narrower path than the step's header implies.

**Why it matters here:** our exported model pins `PreTrainedTokenizerFast`
(`distill-hf-export`, commit `adfdc64`). Your recorded row was produced from a
directory declaring `GPT2Tokenizer`. If those tokenize differently, then part of any
delta you measure is *a tokenizer fix*, not distillation.

The probe copies only the small files to scratch, rewrites `tokenizer_class` on the
copy, and tokenizes five strings chosen to hit the `Split` regex — contractions, a
leading space, digits, a newline, and the granite role marker.

**Measured 2026-09-22, build `b70cd903`: real, and narrower than feared.**

    PROBE declared_class GPT2Tokenizer
    PROBE as-is  class=GPT2Tokenizer     pre_tokenizer=ByteLevel(use_regex=True)
    PROBE pinned class=TokenizersBackend pre_tokenizer=Sequence[Split(regex), ByteLevel(use_regex=False)]
    PROBE VERDICT differing=1/5

The override fires exactly as the step's header says: the declared `GPT2Tokenizer`
discards the trained `Split`. But `GPT2TokenizerFast`'s replacement is
`ByteLevel(use_regex=True)`, which applies GPT-2's own regex — close enough to the
trained one that 4 of 5 probes agree, including the role marker.

The one that disagrees is **digits**:

    "don't 1234 tokenize"
      as-is  [15357, 956, 220,  717, 1958, 78751]
      pinned [15357, 956, 220, 4513,   19, 78751]

The trained pre_tokenizer splits digit runs at `\p{N}{1,3}` — `1234` becomes `123|4`.
GPT-2's regex uses `\p{N}+` and groups the whole run, giving `12|34`. So the confound is
confined to numeric text, which puts the **five math targets** (GSM8K, GSM-Symbolic,
Minerva Math, DeepMind Math, MGSM) and plausibly code's numeric literals at risk, while
leaving most of the other 22 comparable.

**But five probes under-sampled it.** Over 200 real conversations the two tokenizers
produce 272,276 vs 264,684 tokens for the same 1,326,879 bytes — a 2.8% difference. If
only digit runs differed, conversational text would show far less than that, so the
segmentations diverge more broadly than the probe set suggested. Treat "confined to
digits" as the clearest case rather than the whole extent.

That makes the *direction* the open question, which `tokenizer-fit` below answers.

| `PROBE VERDICT` | Means |
|---|---|
| `differing=0/N` | The override is inert on this path; the recorded row is comparable as-is. |
| `differing>0/N` | Real. Run `tokenizer-fit`, then add the tokenizer-control eval column: `full-eval` on the **same SFT weights** with only `tokenizer_class` rewritten. 27 preemptable 1×H100 jobs, no training. |

## 3. `tokenizer-fit` — which segmentation do the weights prefer?

`tokenizer` shows the two paths disagree. It cannot say which is *right for this
checkpoint*, and that decides whether pinning is a fix or a regression:

* granite 4.x **pretraining** used `tokenizer.json`, so the base model's embeddings
  correspond to the **pinned** segmentation.
* the **SFT** stage trained on a pre-tokenized dataset, and whatever produced it most
  likely went through `AutoTokenizer` — the **as-is** segmentation.

If both hold, the SFT stage introduced a mismatch and pinning restores consistency with
pretraining — in which case the math numbers should *improve*, and your recorded row is
understated. If the SFT baked the as-is segmentation in deeply enough, pinning is a
regression and the export must not pin.

The probe scores 200 real conversations under both tokenizers and reports **NLL per
byte**, not per token: two tokenizations give different token counts, so
perplexity-per-token is not comparable between them — the coarser segmentation wins by
construction. Bytes are the invariant. (Worth noting the step header's "26.1 vs 3.29
PPL/token" has that normalisation problem, though a gap that size likely survives it.)

**Measured 2026-09-22, build `8803bcf9`: the weights prefer the pinned tokenizer.**

    PROBE fit rows=200 from .../smoke_2000_nothink.jsonl
    PROBE fit asis:   class=GPT2Tokenizer     tokens=272276 bytes=1326879 nll_per_byte=0.58593
    PROBE fit pinned: class=TokenizersBackend tokens=264684 bytes=1326879 nll_per_byte=0.52224
    PROBE VERDICT nll_per_byte asis=0.58593 pinned=0.52224 prefers=pinned ratio=1.122

**10.9% lower NLL per byte under the pinned tokenizer, and 2.8% more token-efficient.**
Both directions agree, and per-byte is the fair comparison. In perplexity-per-byte terms
that is 1.797 vs 1.686 — a real gap, not a rounding difference.

So the reading is the first one: **pretraining used `tokenizer.json`, the SFT stage
introduced a mismatch, and pinning restores consistency with pretraining.** Pinning is a
fix. Three consequences:

1. **`distill-hf-export`'s pin is correct** and needs no override. Keep it.
2. **The recorded after-SFT eval row is understated**, so a naive comparison against it
   would credit distillation with a tokenizer fix it did not perform.
3. **The tokenizer-control eval column is required**, and is a standalone finding rather
   than bookkeeping: it measures what pinning alone recovers, at zero training cost.

It very likely generalises beyond the SFT row. Every recorded granite4-350m eval —
after-SFT, after-IFRL, after-IDRL — went through the sage harness, which loads with
`AutoTokenizer`, so all of them are measured under the segmentation the weights like
less.

| `PROBE VERDICT` | Means |
|---|---|
| `prefers=pinned` | The SFT stage introduced the mismatch. Pin, and read the control column as a finding rather than a correction. **This is what was measured.** |
| `prefers=asis`, ratio near 1.0 | Barely distinguishable; treat the affected targets as not comparable and say so. |
| `prefers=asis`, ratio well above 1.0 | Pinning is a regression. The export must not pin, and `distill-hf-export`'s normalisation needs an override. |

## The tokenizer-control column

Needs no new recipe — `full-eval` with two overrides against a pinned copy of the SFT
checkpoint. Build the copy once, on a login node; the block is the same shape as the
pinned teacher in [`distill-smoke`](../distill-smoke/README.md#the-pinned-teacher), with
`SRC` and `DST` changed to:

```
SRC=/proj/granite-build/g4os/skypilot-test/sft/checkpoints/v0-20260529-20260529_212150-hf/epoch_hf_2
DST=/proj/granite-build/g4os/skypilot-test/sft/checkpoints/epoch_hf_2-tokpinned
```

Nothing but `tokenizer_config.json` differs — the weights are the same inodes, so it is
the same model by construction.

```bash
gb build start -f recipes/granite4-350m/lsf/full-eval/build.yaml --space <your-space> \
  --param MODEL_PATH=/proj/granite-build/g4os/skypilot-test/sft/checkpoints/epoch_hf_2-tokpinned \
  --param EXPERIMENT=sft-epoch2-tokpinned-r1 \
  --param OE_EVAL_BCB_API_URL=http://<fresh-bcb-host>:7860/evaluate/
```

`EXPERIMENT` must be unique per submission: it is the sole output namespace for all 27
targets, and a colliding artifact URI makes the target report **SUCCESS with an empty
output list** (build `b5f030cd`). Start `bcb-server` first and pass its host, or accept
that 1 of the 27 (`olmes-bigcodebench`) ran against a stale server.

Note the probe deliberately does **not** write to the checkpoint. If you need the
control column, build the rewritten copy as a separate directory with the weights
hardlinked, the way the pinned teacher is built in
[`distill-smoke`](../distill-smoke/README.md#the-pinned-teacher).

## Why no outputs

Neither target declares one. There is nothing to hand downstream: the answers are
decisions for a human, not artifacts for a step. `max_retries: 0` for the same
reason — a probe that needed a retry has already told you something, and a silently
retried transient would hide exactly the flakiness worth seeing.
