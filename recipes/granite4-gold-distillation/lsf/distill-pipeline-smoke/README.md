# Distillation pipeline smoke recipe

The whole distillation chain in one build, at a scale where a wrong path costs
minutes:

    align → corpus → [train-sft] → train-gold → export → eval-transfer ×2 → eval-bfcl

Seven targets, eight with the SFT arm switched on. It is the first recipe here
that wires the ported distillation steps together rather than exercising one of
them, so its subject is the **plumbing**: the bindings, the shared corpus root,
the artifact declarations, and the four separate places that have to agree about
the length budget and the teacher. Nothing it produces is a measurement — the
student underneath the two transfer numbers has taken two optimizer steps over 48
rows.

## Run

`parameters.yaml` sits next to `build.yaml`, so it is picked up automatically:

    gb build start -f recipes/granite4-gold-distillation/lsf/distill-pipeline-smoke/build.yaml \
      --space <your-space>

With the SFT arm:

    gb build start -f recipes/granite4-gold-distillation/lsf/distill-pipeline-smoke/build.yaml \
      --space <your-space> --param INCLUDE_SFT=true

Override any value with `--param KEY=VALUE`. The pair and the data:

    --param TEACHER_MODEL=/proj/granite-build/g4os/kd-sandbox/teacher_overlays/<teacher> \
    --param STUDENT_MODEL=/proj/granite-build/g4os/kd-sandbox/student_overlays/<student> \
    --param DATASET=/proj/granite-build/g4os/gbtest/gold-distill-smoke/<data>_nothink.jsonl

## Why the targets are in this order

Every edge is a binding, and every binding exists because the downstream step
cannot be *correct* without the upstream artifact — not merely because it needs a
file.

| Edge | Why |
|---|---|
| `align → corpus` | Lengths and the assistant-mask check are counted in tokens, so a corpus measured with a different tokenizer is filtered against the wrong distribution. Prep also needs align's chat template to locate assistant spans at all: measured, none of the 13 hand-built `/proj` overlays carries a `{% generation %}` tag. |
| `align → train-*` | The student is the **retagged** one. Its embedding rows for the control tokens differ from the base model's. |
| `corpus → train-*` | Plus the manifest the trainers' row-count guard reads. |
| `train-gold → export` | Selects a checkpoint, prunes it, and verifies it loads. |
| `align → export` | `expect_tokenizer_from`, so the export asserts the tokenizer it ships is the one that trained rather than whatever landed in the checkpoint directory. |
| `align → eval-transfer-baseline` | The t=0 read, taken **before** training. A divergence with no baseline is a number without a direction. |
| `export → eval-transfer`, `export → eval-bfcl` | Both post-training reads measure the published model, not a raw checkpoint. |

The two evaluations answer different questions and neither substitutes for the
other: a student can move toward the teacher's distribution while losing
tool-calling accuracy, and the transfer metric cannot see that.

## One bad host: p2-r03-n1

Three GPU targets across two builds died with `RuntimeError: CUDA unknown error`
raised inside torch at `cuda_init`, after their own "1 GPU visible via
CUDA_VISIBLE_DEVICES" gate had passed — so the allocation arrived and the
initialisation is what failed. All three were on **p2-r03-n1**:

| host | runs in `~/sky_logs` | cuda_init failures |
|---|---|---|
| `p2-r03-n1` | 3 | **3** |
| every other host | several hundred | 0 |

In build b5f030cd the same `distill-eval` step succeeded on `p3-r27-n2` and then
failed on `p2-r03-n1` two minutes later, same config, same image. The node takes
the allocation and cannot initialise CUDA. **Report it**; nothing in this
repository fixes it.

Two mitigations, in order of usefulness:

1. **Exclude the host.** `-R` accumulates in the LSF provisioner
   (`sky/provision/lsf/instance.py:150`), so a `bsub_options` entry is added to the
   derived resource requirements rather than replacing them. In `~/.sky/config.yaml`:

   ```yaml
   lsf:
     cluster_configs:
       bluevela:
         bsub_options:
           R: "select[hname!='p2-r03-n1'&&hname!='p2-r03-n1.bluevela.rmf.ibm.com']"
   ```

   **No surrounding double quotes and no spaces**, both load-bearing and both
   learned the hard way (builds 8d5000fd and 1884d108 died before submitting a
   job). `sky/templates/lsf-ray.yml.j2:75` emits each entry as `{{key}}: "{{val}}"`,
   so a value carrying its own quotes yields `R: ""select[...]""` and the generated
   cluster YAML fails to parse. And `provision/lsf/instance.py:758` emits the
   directive unquoted as `#BSUB -R <val>`, so a space would end the argument early —
   hence `&&` with no spaces, which LSF's select grammar accepts. The derived
   directives (`-R "span[ptile=N]"`) carry their own quotes because they never pass
   through the cluster YAML; `bsub_options` values do.

   This is global to every build from this machine, which is right for a node that
   is broken for everybody. Remove it once the node is fixed.

2. **`retries: max_retries: 1`**, which this recipe sets. With
   `target_reuse_enabled` a retry re-runs only the failed target and reuses what
   already succeeded, so a fault at `eval-transfer` does not re-train. Drop
   `MAX_RETRIES` to 0 while debugging a failure of your own, so it fails once and
   stays failed.

### A wrong turn worth recording

The first build lost `train-sft` and `eval-transfer-baseline`, which by data
dependency were the pipeline's only two concurrent GPU targets. That, plus the
provisioner's `mode=exclusive_process` request, made "two jobs collided on one
host" a plausible reading, and this recipe briefly chained its GPU targets with
ordering-only bindings to prevent the overlap.

It was wrong. Both failures were simply on the bad node, and the next build —
fully serialised — failed on that node anyway. The chain is gone rather than kept
as harmless belt-and-braces: a workaround whose stated reason is false invites the
same wrong diagnosis next time, and it cost the baseline evaluation its
independence for nothing.

Concurrency is therefore still **untested** on a healthy pair of hosts. If it ever
does prove to be a real constraint, the fix belongs in the environment or in
retries, not in a hand-built chain.

## The SFT arm is a switch, not a sibling recipe

`INCLUDE_SFT` (default `false`) inserts a plain-SFT pass between the aligned
student and GOLD, and GOLD then continues from the SFT checkpoint instead of from
the aligned student. Off, both the target and every reference to it disappear from
the rendered build — they have to, because a binding naming a target that is not
in the build is a validation failure.

No export is needed between the two trainers: the SFT step's ZeRO-3 config sets
`zero3_save_16bit_model` and `sft.py` calls `save_model` on the output directory
with the tokenizer attached, so the registered checkpoint is already HF-native and
loadable as a student init.

The switch reads `true/false`, `yes/no`, `1/0` and `on/off`, in any case, because
`--param INCLUDE_SFT=false` delivers the **string** `"false"` — the CLI does no
YAML parsing of parameter values, and a bare Jinja truth test on a non-empty
string is true. A recipe that tested truthiness would turn the arm on while the
command line said to turn it off. `test_distill_pipeline_recipe.py` pins every
spelling.

## Two things here are deliberate and look wrong

**`allow_unknown: true` on the export.** HF Trainer writes one
`rng_state_<rank>.pth` per rank for a distributed run, while the ported
classifier's `PRUNE_KNOWN` carries only the single-process `rng_state.pth` — it
was validated against a 1-GPU checkpoint. So *every* checkpoint this pipeline
produces contains files it cannot classify, and with `false` the export refuses
after the training run has been paid for. `export_manifest.json` still records
exactly what was dropped. The fix is one line upstream; until it lands, this is
what lets the pipeline complete. Do not copy the setting into a release recipe
once it is fixed.

**A 3B teacher, not the 30B reference one.** An overlay is tokenizer files only
and this recipe's subject is the wiring, so the teacher's size buys nothing here,
while a 30B teacher forward would dominate the run and force a much larger
allocation. `granite-4.2-3b-nothink` carries the same ChatML surface.

## What this recipe found in the steps it chains

**The published tokenizer was unreadable by transformers 4.** Build 30a99c4b got
seven of eight targets green and then `eval-bfcl` died at
`base_oss_handler.py:109` with `ValueError: Tokenizer class TokenizersBackend does
not exist or is not currently imported`. The distillation steps run transformers
5.8.0, which records its fast-tokenizer backend under that v5-only name; the BFCL
harness ships a transformers 4 image and could not resolve it, so the evaluation
failed before generating anything.

The bug was not BFCL's: *any* transformers 4 consumer of the published model would
have hit it. `distill-hf-export` now rewrites that one class name to
`PreTrainedTokenizerFast` — which exists in both generations and loads
`tokenizer.json` directly — and records the rewrite in `export_manifest.json`
beside its other normalisations. Chaining the capability evaluation onto the export
is what surfaced it; the two transfer evaluations run in the same image as the
export and never would have.

## Known divergence from upstream

Upstream's `distill-gold-train` step takes a `teacher_tokenizer_path` and is given
`align.teacher_overlay`, because a trainer that loads the teacher tokenizer from
the teacher *directory* silently re-acquires the `tokenizer_class` trap — it does
not error, it mis-segments. The `distill-gold` step used here exposes no such key:
the kd-sandbox trainer reads the teacher tokenizer from `TEACHER_MODEL`. On this
recipe's pairing that is benign, because the teacher path is itself a prepared
overlay or a snapshot whose `tokenizer_class` is correct — but it is a real gap,
and closing it means either a `teacher_tokenizer_path` key on `distill-gold` or
porting upstream's step.

## Where the results land

Everything composed hangs off `WORKDIR_ROOT/RUN_NAME/BUILD_SUBDIR`, one directory
per build:

    /proj/granite-build/g4os/gbtest/distill/distill-pipeline-smoke/<build-id>/
      align/                     retagged student + the two tokenizer overlays
      corpus/train.jsonl         the declared corpus artifact (+ manifest sibling)
      corpus/eval.jsonl          the held-out split, composed rather than bound
      train-sft/                 only when the SFT arm is on
      export/                    the published HF model
      eval-transfer-baseline/    t=0 divergence
      eval-transfer/             post-training divergence
      eval-bfcl/<run>/bfclv3/    BFCL scores

The absolute root is forced rather than chosen: `eval.jsonl` is deliberately not a
declared artifact of corpus-prep — it exists only when `eval_fraction > 0`, and a
declared-but-absent output is a resolver failure — so the eval targets must
compose its path, and every target run gets its own `GB_BUILD_WORKDIR`.

### Why the per-build segment is correctness, not tidiness

`BUILD_SUBDIR` defaults to `{{ run_metadata.build_id }}`, which gbserver fills per
step (`targetsteprun.py:399`) with the same value for every target in a build.
Sharing one directory across builds instead looks harmless and is not:

1. gbserver refuses to register an artifact whose URI **another build in the space**
   already registered: `ValueError: Same artifact URI found in space ... for
   another build in this space`.
2. The target that emitted it still reports **SUCCESS** — with an empty output
   list. Nothing fails, and the current run works, because downstream bindings
   resolve against the artifact the *earlier* build registered.
3. The damage lands on the next retry or restart. Reuse replays each skipped
   target's **stored** outputs as bindings, so a target with no stored outputs
   propagates nothing: its consumers never become ready, are never dispatched, and
   the build finishes with no pending work — reporting **SUCCESS**.

Build b5f030cd did exactly that. Its `align` and `corpus` targets wrote paths
already registered by build 7853d33b, so both ended SUCCESS with no artifacts. On
restart the six finished targets were reused, `corpus` propagated no `corpus`
binding, `eval-transfer` — the one target that still needed running — was never
dispatched, `eval-bfcl` never existed, and the build reported SUCCESS with
`eval-transfer` still FAILED.

**Two things follow.** Do not pin `BUILD_SUBDIR` to a fixed string unless you want
that (the only reason to is deliberately resuming a previous build's directory).
And a gbserver build reporting SUCCESS while a declared target has no successful
run is a bug worth filing on its own: a false green is worse than a failure.

## Tests

    GB_ENVIRONMENT=STANDALONE .venv/bin/python -m pytest test/unit/recipes/test_distill_pipeline_recipe.py

Those checks are all about the class of error that produces a run which completes
and reports something other than what it says: one teacher for the whole
pipeline, one length budget, the two transfer evals differing only in the student,
every emitted artifact declared as an output, the response template's trailing
line boundary surviving as a two-character escape, and the per-build run
directory above.
