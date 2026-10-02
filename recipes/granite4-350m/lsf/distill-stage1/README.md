# distill-stage1 — off-policy GOLD, the real run

Off-policy GOLD distillation of the granite-4.0-350m SFT checkpoint towards a
granite-4.1-3b teacher. `lmbda 0.0`, `beta 0.5`, 2 nodes x 8 H100, effective batch 96,
8192 context.

**Run [`distill-probe`](../distill-probe/README.md) and
[`distill-smoke`](../distill-smoke/README.md) first.** Both are minutes; this is hours.

## The comparison this exists to make

The baseline is the **existing after-SFT eval row**, not a new control run. The student
is the checkpoint that produced that row (`epoch_hf_2`), so the only difference between
the recorded numbers and this run's is the distillation. That is why there is no SFT arm
and no control arm — the control was already run and already measured.

One correction is required before the comparison is honest. Probe build `8803bcf9`
measured the checkpoint preferring the pinned tokenizer by **10.9% NLL/byte**, and our
export pins `tokenizer_class` while the recorded row does not. So run the
tokenizer-control column too — `full-eval` on the *same* SFT weights with only
`tokenizer_class` rewritten. The procedure is in
[`distill-probe`](../distill-probe/README.md#the-tokenizer-control-column). Without it,
a chunk of any measured gain is a tokenizer fix rather than distillation.

## The corpus, and why there is a `sources` target

Stage 1 trains on the raw ancestor of the SFT mixture:

| split | rows | share |
|---|---|---|
| `general/general.jsonl` | 4,311,969 | 84.86% |
| `tools/tools.jsonl` | 710,290 | 13.98% |
| `rag/rag.jsonl` | 59,245 | 1.17% |
| **total** | **5,081,504** | |

Those row counts match `datasets/tokenized/{general,tools,rag}/data.jsonl` exactly
(verified 2026-09-22), and `tokenized/all_combined` is their merge — so this is the same
text the baseline was trained on, one transformation earlier.

Three facts about these files drive the `sources` target, and **each one fails silently**
if left unhandled:

1. **The rows are spelled `conversations`, not `messages`.** `prep_corpus.py:367` reads
   `record.get("messages")` and returns `no_messages` otherwise. Handed these files
   unchanged, prep drops all 5,081,504 rows and raises `PrepError`. The rename is the
   single most load-bearing line in the recipe.
2. **`prep` takes one path, not a directory.** `prep_corpus.py:294` refuses one outright:
   *"Pass the .jsonl itself, or an HF dataset id."* There is no glob and no list form.
3. **The splits are wildly unequal, and prep's `max_examples` stops after N kept rows
   *in input order*.** So concatenating and capping would hand you a subset of pure
   `general` with no tools and no rag in it, and nothing would say so. `sources` samples
   each split with its own seeded reservoir, which holds the shares *exactly* rather
   than in expectation, then interleaves.

`SHUFFLE_SEED` alone reproduces the exact corpus. `MAX_EXAMPLES` in prep is therefore 0:
the cap belongs where it can be applied proportionally.

## `TARGET_ROWS` is the cost knob

| rows | steps at batch 96 | est. GPU-h | wall on 2 nodes |
|---|---|---|---|
| **800,000** (default) | 8,333 | 57–115 | 3.6–7.2 h |
| 5,081,504 (one epoch) | 52,932 | 365–729 | 23–46 h |

Derived from the reference arm's measured 554 GPU-h per epoch over 802,027 rows at
3B←30B, scaled by the per-token cost ratio `(6·3+2·30)/(6·0.35+2·3)` = 9.6× cheaper, then
doubled for the non-fused JSD path's `[batch, seq, 100352]` reduction, which does **not**
shrink with the student and becomes the dominant term at 350M.

**The default is a subset on merit, not only on budget.** The student has already seen
all of this text twice under hard labels. What is new in stage 1 is the teacher's
distribution over it, and that is learnable from a representative sample — the published
sweep reported its headline result after 100 steps. 800,000 also puts this run at roughly
the reference corpus's scale (802,027), which keeps the two comparable. Raise it with one
`--param` once a first run says the objective is working.

## Geometry

    effective batch = per_device x grad_accum x nodes x gpus = 2 x 3 x 2 x 8 = 96

96 rather than the reference's 192: that was tuned for a 3B student at 16384 context, and
96 sits nearer the 350m's own SFT geometry (4 x 2 x 8 = 64), which is where this
student's optimum came from.

**Hold the product when changing the node count.** Two nodes at `grad_accum 3` and one
node at `grad_accum 6` are the same optimization at different wall-clock; two nodes at
`grad_accum 6` is a different run that merely finishes in the same time. The test asserts
the product, so a compensating edit passes and an uncompensated one fails.

Two nodes because off-policy GOLD is pure data parallelism — no vLLM server, no weight
sync, no generation — so it scales well; `gold-smoke` has proven a 2-node allocation and
the upstream reference ran 4. It halves wall clock at the same GPU-h. It does **not**
relax memory: the binding constraint is the per-device `[batch, seq, 100352]` logits
tensor, which is local. If step 1 OOMs, go `per_device 1` / `grad_accum 6`.

`GOLD_MAX_STEPS: 0` means the run is bounded by **epochs**, not by a step count
(`render_gold_config.py:103` emits `max_steps` only when positive, and a positive value
overrides `num_train_epochs`). So the step count follows `TARGET_ROWS` automatically — a
hand-maintained step count is how a run silently covers a quarter of its corpus.

## Making it faster

Three levers, in increasing order of cost and decreasing order of confidence. All of them
must hold effective batch at 96, and all of them should be tried on a short run
(`--param TARGET_ROWS=50000`) rather than on the real one.

**1. Fill the GPUs you already hold — free.** At `per_device 2` / 8192 context the
footprint is roughly 25 GB of each 80 GB H100: ~11.6 GB of teacher weights, student
weights, grads and Adam state, plus ~13 GB of `[batch, seq, 100352]` logit tensors and JSD
intermediates. There is a lot of headroom. On the same two nodes, `(per_device 6,
grad_accum 1)` is also 96 and estimates at ~51 GB — bigger matmuls and no accumulation
loop at all, on the identical allocation and identical GPU-h.

    --param GOLD_PER_DEVICE_TRAIN_BATCH_SIZE=6 --param GOLD_GRADIENT_ACCUMULATION_STEPS=1

Those numbers exclude activations and assume 4x for the JSD intermediates, so treat them
as a starting point and expect to back off to 4 or 3 if step 1 OOMs.

**2. Stop sharding parameters — probably the biggest win, at no extra GPU-h.** `DS_CONFIG`
is ZeRO-3, inherited from the reference pair where a 3B student and a 30B teacher genuinely
needed it. This pair does not: the entire unsharded weight-plus-optimizer footprint is
~11.6 GB. ZeRO-3 all-gathers parameters on every forward and backward, and with a 350M
model split across 16 or 32 ranks each shard is ~11-22M parameters — small enough that the
collectives are latency-bound rather than bandwidth-bound. That overhead is also exactly
what limits multi-node scaling, so removing it helps more than adding nodes.

    ls /proj/granite-build/g4os/kd-sandbox/configs/deepspeed/
    --param DS_CONFIG=configs/deepspeed/<a zero2 or zero1 config>

Unverified, and it needs the short run rather than a leap: the checkpoint layout changes
(`zero3_save_16bit_model` is a ZeRO-3 setting) and `distill-hf-export` asserts what it
loads, so a wrong guess surfaces at export rather than at step 1. Off-policy has no
weight-sync path, so the on-policy NCCL machinery is not a constraint here.

**3. More nodes — real, but sub-linear and it costs more.** At effective batch 96 the
geometry gets tight fast, because 96 has to divide by the GPU count:

| nodes | GPUs | rows per GPU per step | `(per_device, grad_accum)` |
|---|---|---|---|
| 1 | 8 | 12 | (1,12) (2,6) (3,4) (4,3) (6,2) |
| **2** | 16 | 6 | (1,6) **(2,3)** (3,2) (6,1) |
| 4 | 32 | 3 | (1,3) **(3,1)** |
| 6 | 48 | 2 | (1,2) (2,1) |

At four nodes prefer `(3,1)` over `(1,3)`: same effective batch, but a per-device batch of
3 keeps the matmuls worth doing and drops the accumulation loop, where `per_device 1` wastes
an H100 on a single sequence.

    --param GOLD_NUM_NODES=4 \
      --param GOLD_PER_DEVICE_TRAIN_BATCH_SIZE=3 --param GOLD_GRADIENT_ACCUMULATION_STEPS=1

Expect well under the ideal 2x from doubling: `per_device` is already small, the ZeRO-3
collectives grow with rank count, queue wait for 4x8 H100 is longer than for 2x8, and more
nodes means more exposure to the `p2-r03-n1` CUDA-init fault. **GPU-h goes up, not down** —
you are buying wall clock, not capacity. Do levers 1 and 2 first; they are cheaper and
likely larger.

## Run

```bash
gb build start -f recipes/granite4-350m/lsf/distill-stage1/build.yaml --space <your-space>
```

A shorter first pass, to see the loss move before committing hours:

```bash
gb build start -f recipes/granite4-350m/lsf/distill-stage1/build.yaml --space <your-space> \
  --param TARGET_ROWS=50000 --param GOLD_SAVE_STEPS=100
```

## What to read

| Target | Look for |
|---|---|
| `sources` | `SOURCES count TOTAL 5081504`, then `SOURCES quota` showing all three splits non-zero, then `SOURCES renamed=<N> already_messages=0 bad=0`. **`renamed` must be non-zero** — a zero there with a non-zero `already_messages` means the schema changed under us. |
| `align` | `[1/4] teacher overlay` completes; `[4/4] masking contract -> .../masking.json` |
| `corpus` | `kept` > 0 and no `empty_assistant_mask` drops; manifest `target_tokens` non-zero |
| `train-gold` | a finite, non-zero loss at step 1; no `verify_tokenizer_consistency` refusal; no `ManifestDrift` |
| `export` | `normalised: tokenizer_config.json: tokenizer_class ...` |

Workload stdout for `command` steps (that is `sources`) is only in SkyPilot's log:

```bash
for c in $(gb build log <build-id> | grep -oE 'gb-[0-9a-f]{8}-[0-9a-z]{3}' | sort -u); do
  sed 's/\x1b\[[0-9;]*m//g' ~/sky_logs/$c/*/run.log | grep SOURCES
done
```

## Then

Evaluate the export through `full-eval` against the recorded after-SFT row *and* the
tokenizer-control row, with a fresh `EXPERIMENT` per submission — a colliding artifact
URI makes a target report SUCCESS with an empty output list (build `b5f030cd`).

Stage 2 (on-policy, on the IFRL and IdentityRL prompt sets) continues from this run's
`export`.

## The chat template is verified

`CHAT_TEMPLATE` is granite-4.0-350m's own template with `{% generation %}` markers added
and nothing else changed, and both halves of that claim are now checked:

* **It is the right base.** `sha256sum /proj/granite-build/g4os/chat_template.jinja` —
  the template the SFT mixture was actually tokenized with — returned
  `9524df67b77a7b25...` on 2026-09-22, byte-identical to the vendored
  `granite_4_role_base.jinja`. The hash is pinned in
  `steps/distill/distill-tokenizer-align/skypilot/test/test_granite_role_template.py`, so
  re-vendoring from a different granite release fails a test instead of silently
  changing what the marked template is derived from.
* **The markers change no rendered byte.** Asserted over nine conversation shapes,
  including the default-system-message branch, tool calls, consecutive tool turns and
  documents — plus a structural check that the only edit is the two marker lines and the
  one split emission they required.

So the distilled student renders prompts exactly the way the model behind the recorded
after-SFT eval row did, which is the assumption the whole comparison rests on.
