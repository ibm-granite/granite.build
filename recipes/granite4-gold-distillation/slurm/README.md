# GOLD distillation on BlueVela SLURM (SkyPilot native SLURM backend)

GOLD / JSD knowledge distillation of a Granite student from a Granite teacher,
driven by granite.build on the **BlueVela SLURM** cluster via SkyPilot's native
SLURM backend. This is the SLURM counterpart of the `lsf/` recipes, ported from the
hand-run `gbansible/skypilot/kd_distill_bluevela_slurm*.yaml` SkyPilot YAMLs.

Teacher `granite-4.1-30b` → student `granite-4.0-350m`, off-policy by default,
2 nodes × 8 H200 (16 GPUs). The smoke-scale defaults (20k dataset, `MAX_STEPS=30`,
4K context) prove the plumbing end to end; raise them for a real run.

---

## What lives where

### Recipes — `recipes/granite4-gold-distillation/slurm/`
| Recipe | Targets | Use |
|---|---|---|
| [`distill-gold/`](distill-gold/) | align → corpus → train-gold (+ vllm-server, teardown on-policy) | Full pipeline from the raw student/dataset. |
| [`distill-gold-trainonly/`](distill-gold-trainonly/) | train-gold (+ vllm-server, teardown on-policy) | Skips align+corpus by binding the trainer to a **prior build's** align/corpus `env://` outputs. For fast train-gold iteration. |
| [`distill-gold-colocated/`](distill-gold-colocated/) | train-gold **only** | On-policy with an **in-allocation** vLLM serve node (no separate server/teardown). The one recommended on-policy path on BlueVela — keeps all NCCL inside one allocation, sidestepping the cross-allocation deadlock. Binds to a prior build's outputs like trainonly. |

Each recipe is a `build.yaml` + `parameters.yaml` pair; `parameters.yaml` is picked
up automatically from alongside `build.yaml`.

### Steps — `configurations/assets/environments/skypilot/steps/distill/`
The recipes reference SLURM-specific copies of the LSF distill steps (the LSF
originals declare `subtypes: [lsf]`, and gbserver rejects a build whose environment
subtype a step doesn't list). The copies differ from the LSF steps **only** as noted:

| Step (`space://steps/distill/...`) | Copied from | Difference |
|---|---|---|
| `tokenizer-align-slurm` | `tokenizer-align` | `subtypes: [slurm]` only (single-node CPU). |
| `corpus-prep-slurm` | `corpus-prep` | `subtypes: [slurm]` only (single-node CPU). |
| `gold-slurm` | `gold` | `subtypes: [slurm]`; per-node topology from `SKYPILOT_NODE_RANK`/`_NUM_NODES`/`_NODE_IPS`/`_NUM_GPUS_PER_NODE` instead of the LSF `RANK`/`TOTAL_NODES`/`MASTER_ADDR`/`NUM_GPUS_PER_NODE`; BlueVela IB NCCL tuning baked into the run block (the LSF env sourced it from a `nccl_tuning_file`). |
| `vllm-server-slurm` | `vllm-server` | `subtypes: [slurm]`; `NUM_GPUS_PER_NODE`→`SKYPILOT_NUM_GPUS_PER_NODE`; `LSB_JOBID`→`SLURM_JOB_ID`. |

`teardown` reuses the backend-agnostic `space://steps/skypilot-teardown` builtin (no
SLURM copy needed — it just runs `sky down`).

### Environment — `configurations/assets/environments/skypilot/slurm/ibm-bluevela/environment.yaml`
`space://environments/skypilot/slurm/ibm-bluevela` (name `sky-slurm-bluevela`,
subtype `slurm`): `cluster: bluevela`, `zone: gpu-mid`, env-wide `constraint: h200`,
SSH to the `login0/login1/login2` failover pool with `~/granite_build_id_ed25519`
(gbserver picks one at random per launch and fails over on a transient SSH error),
and the workdir-mount fix
(`cloud_config.slurm.cluster_configs.bluevela.workdir` is an **ancestor** of
`shared_workdir` so the per-run `$GB_BUILD_WORKDIR` on `/proj` is visible inside the
enroot container — the SLURM backend does not identity-mount all of `/proj` like LSF).

---

## Running

### Off-policy (default)
```
gb build start -f recipes/granite4-gold-distillation/slurm/distill-gold/build.yaml
```
Runs align → corpus → train-gold (3 targets) on 2× gpu-mid.

### Ramped on-policy (2× gpu-mid trainer + 1× gpu-high vLLM server)
```
gb build start -f recipes/granite4-gold-distillation/slurm/distill-gold/build.yaml \
  --param ON_POLICY=true --param LMBDA=1.0 --param LMBDA_SCHEDULE=linear \
  --param LMBDA_INIT=0.0 --param VLLM_QUEUE=gpu-high
```
Adds vllm-server (gpu-high) and teardown (5 targets). `lmbda` ramps 0 → 1 across
training; the trainer pushes student weights to the server over NCCL.

### Ramped on-policy, co-located (RECOMMENDED on-policy path)
```
gb build start -f recipes/granite4-gold-distillation/slurm/distill-gold-colocated/build.yaml
```
One SLURM allocation of `NUM_NODES` (default 2): the **last** `VLLM_NUM_SERVERS` (1)
node serves the student under vLLM, the rest train. `vllm_num_servers>0` with the
`vllm` input left **unbound** selects gold-slurm's in-allocation split. `lmbda` ramps
0 → 1 (`linear`) and the trainer syncs student weights to the serve node over NCCL —
all of it **inside one allocation**, so it does not hit the cross-allocation deadlock
the separate-server path does (see [Open risk](#open-risk--cross-allocation-nccl-on-policy-only)).
This is the path proven end-to-end on the cluster (loss decreasing, `n_sync_params>0`).

### Train-only (reuse a prior build's align + corpus)
```
gb build start -f recipes/granite4-gold-distillation/slurm/distill-gold-trainonly/build.yaml
# point at a specific prior build's outputs:
#   --param RETAGGED_STUDENT_URI=env:///proj/.../distill-gold-slurm/<BUILD>/align/retagged_student
#   --param CORPUS_URI=env:///proj/.../distill-gold-slurm/<BUILD>/corpus/train.jsonl
```
`RETAGGED_STUDENT_URI` and `CORPUS_URI` **must share a tokenizer** — gold.py reads the
tokenizer the corpus was built against; a mismatch trains on mis-segmented text silently.
`CORPUS_URI` may be **either** a corpus-prep `train.jsonl` (cleaned, length-filtered
`{messages}` text) **or** a pre-tokenized `save_to_disk` Arrow dir carrying `input_ids`
(see the 1b worked example below). It may **not** be a raw messages JSONL that still
carries bare integer literals > 2^64 (see the pitfall in the worked example). What the
`align` / `corpus` artifacts are and how those steps build them is documented in
[UPSTREAM_ARTIFACTS.md](UPSTREAM_ARTIFACTS.md).

#### Worked example — granite-4.0-1b ← 4.2-30b on the all-category ~1.1M set (pretokenized)
Validated 2026-10-21 (smoke build `45d1bde5`, full epoch `567da1c4`). The student is
**already** retagged onto the teacher's ChatML (`granite-4.0-1b-instruct_retagged_42`, a
hand-staged `model_copies/` dir — NOT a prior build's `align/retagged_student`), the
teacher is the `-slurm` overlay, and the corpus is a **pre-tokenized Arrow dir**
(`g42sft_allcat_rh_pretok_basehub`, 1,100,909 rows). ChatML student ⇒
`RESPONSE_TEMPLATE='<|im_start|>assistant'`.

> **Why the pretok Arrow dir, not the raw `..._messages.jsonl`.** gold.py's
> `_load_dataset` reads a `.jsonl` corpus with `pd.read_json`, whose ujson backend
> raises `ValueError: Value is too big!` on the bare >2^64 integer literals the
> all-category `messages` text contains (that 29 GB `g42sft_allcat_rh_messages.jsonl` is
> the **input to the `corpus` step**, not a trainer corpus). The fix is to feed a
> pre-tokenized dir: the trainonly recipe pins a trainer fork
> (`CODE_REPO`/`CODE_REF` in `parameters.yaml` → `thinkahead/gb-steps-distillation`)
> that (a) `load_from_disk`'s a `save_to_disk` dir when `CORPUS_URI` is one, and (b)
> short-circuits the tokenize pass when `input_ids` is already a column. The dir was
> built once with gbansible's `gold.py --prep_only` (see `gbansible/docs/distill-runs.md`
> "ALL-CATEGORY"); its tokenizer is byte-identical to the 1b retagged student, so it is
> reusable as-is. To re-pretokenize a different corpus, run `--prep_only` with the 1b
> tokenizer against the **original** `g42sft_allcat_rh.jsonl` (messages/tools as JSON
> strings — big ints safe inside quotes), NOT the decoded `_messages.jsonl`.

**Smoke (test) — 50 steps, ~1.5 h on 2 nodes:**
```bash
gb build start -f recipes/granite4-gold-distillation/slurm/distill-gold-trainonly/build.yaml \
  --param RETAGGED_STUDENT_URI=env:///proj/granite-build/g4os/kd-sandbox/model_copies/granite-4.0-1b-instruct_retagged_42 \
  --param CORPUS_URI=env:///proj/granite-build/g4os/kd-sandbox/data/g42sft_allcat_rh_pretok_basehub \
  --param TEACHER_MODEL_URI=env:///proj/granite-build/g4os/kd-sandbox/teacher_overlays/granite-4.2-30b-slurm \
  --param MAX_STEPS=50 --param SAVE_STEPS=25 \
  --param NUM_TRAIN_EPOCHS=1.0 --param NUM_NODES=2 \
  --param MAX_LENGTH=16384 --param MAX_COMPLETION_LENGTH=15872 \
  --param PER_DEVICE_TRAIN_BATCH_SIZE=4 --param GRADIENT_ACCUMULATION_STEPS=2 \
  --param USE_LIGER_FUSED_JSD=true \
  --param RESPONSE_TEMPLATE='<|im_start|>assistant' \
  --param ON_POLICY=false \
  --param LOG_RETRIEVAL_INTERVAL_SECONDS=30 --param POLL_INTERVAL_SECONDS=30 \
  --skip-version-check
```

**Full epoch — same corpus, `MAX_STEPS=-1`, checkpoint every 500:**
```bash
gb build start -f recipes/granite4-gold-distillation/slurm/distill-gold-trainonly/build.yaml \
  --param RETAGGED_STUDENT_URI=env:///proj/granite-build/g4os/kd-sandbox/model_copies/granite-4.0-1b-instruct_retagged_42 \
  --param CORPUS_URI=env:///proj/granite-build/g4os/kd-sandbox/data/g42sft_allcat_rh_pretok_basehub \
  --param TEACHER_MODEL_URI=env:///proj/granite-build/g4os/kd-sandbox/teacher_overlays/granite-4.2-30b-slurm \
  --param MAX_STEPS=-1 --param NUM_TRAIN_EPOCHS=1.0 --param SAVE_STEPS=500 \
  --param NUM_NODES=2 \
  --param MAX_LENGTH=16384 --param MAX_COMPLETION_LENGTH=15872 \
  --param PER_DEVICE_TRAIN_BATCH_SIZE=4 --param GRADIENT_ACCUMULATION_STEPS=2 \
  --param USE_LIGER_FUSED_JSD=true \
  --param RESPONSE_TEMPLATE='<|im_start|>assistant' \
  --param ON_POLICY=false \
  --param LOG_RETRIEVAL_INTERVAL_SECONDS=30 --param POLL_INTERVAL_SECONDS=30 \
  --skip-version-check
```
Off-policy (`ON_POLICY=false`), so only `train-gold` runs — no vllm-server/teardown.
Remaining values (LR, NCCL pins, image, queues, the fork pin) are the trainonly
`parameters.yaml` defaults. Notes that cost real runs to learn:
- **`MAX_COMPLETION_LENGTH` (15872) must stay strictly below `MAX_LENGTH` (16384)** —
  equal values trip `gold_config.__post_init__` (`max_completion_length must be smaller
  than max_length to leave room for the prompt`) at config validation, before any GPU work.
- **`PER_DEVICE_TRAIN_BATCH_SIZE=4` + `GRADIENT_ACCUMULATION_STEPS=2` + Liger** is the
  throughput sweet spot: ~76 s/step vs ~97 s/step at the `parameters.yaml` default 2×4
  (effective batch 128 either way). batch=8/ga=1 gives no further gain (compute-bound);
  memory is never the limit (ZeRO-3 shards the 30B teacher, ~7 GB/rank peak).
- **A full epoch is ~8,601 steps ≈ 7.5 days at ~76 s/step — over the gpu-mid 7-day
  `MaxTime`, and gpu-mid is preemptible (`PreemptMode=REQUEUE`).** The env sets
  `time: 7-00:00:00`; the run checkpoints every `SAVE_STEPS` and, on a wall-clock cap
  OR a preemption, is **continued with a resume submit** (see below). On a near-full
  cluster preemptions are frequent and early — keep `SAVE_STEPS` well under the typical
  preempt interval (we saw evictions at 2 h and 9.6 h; `SAVE_STEPS=250` ≈ 5.3 h bounded
  the loss, after `=500` lost a whole 9.6 h window with zero checkpoints).

##### Resuming after a preemption / wall-clock cap
`RESUME_FROM_CHECKPOINT_DIR` points the next build at the interrupted run's
`<run_name>_node<N>` checkpoint dir. Rank 0 hardlinks every COMPLETE `checkpoint-<N>`
into the new run, then HF resumes from the highest one (weights + optimizer + scheduler +
RNG + data position). **The schedule and world size MUST match the source** (`NUM_NODES`,
`MAX_STEPS`, batch sizes, LR) — a ZeRO-3 checkpoint has one optimizer shard per rank.
Find the dir via `find .../builds/builds/<BUILD>/runs/<run>/checkpoints -maxdepth 1 -name
'<run_name>_node*'`; verify the highest `checkpoint-<N>` is complete (`model.safetensors`,
`trainer_state.json`, `scheduler.pt`, `rng_state_*` for every rank).
```bash
# Same command as the full-epoch launch, plus ONE param (the source checkpoint dir):
gb build start -f recipes/granite4-gold-distillation/slurm/distill-gold-trainonly/build.yaml \
  --param RESUME_FROM_CHECKPOINT_DIR=/proj/granite-build/g4os/gbtest/builds/builds/<BUILD>/runs/<run>/checkpoints/distill-gold-trainonly_node2 \
  --param RETAGGED_STUDENT_URI=env:///proj/granite-build/g4os/kd-sandbox/model_copies/granite-4.0-1b-instruct_retagged_42 \
  --param CORPUS_URI=env:///proj/granite-build/g4os/kd-sandbox/data/g42sft_allcat_rh_pretok_basehub \
  --param TEACHER_MODEL_URI=env:///proj/granite-build/g4os/kd-sandbox/teacher_overlays/granite-4.2-30b-slurm \
  --param MAX_STEPS=-1 --param NUM_TRAIN_EPOCHS=1.0 --param SAVE_STEPS=250 --param NUM_NODES=2 \
  --param MAX_LENGTH=16384 --param MAX_COMPLETION_LENGTH=15872 \
  --param PER_DEVICE_TRAIN_BATCH_SIZE=4 --param GRADIENT_ACCUMULATION_STEPS=2 \
  --param USE_LIGER_FUSED_JSD=true --param RESPONSE_TEMPLATE='<|im_start|>assistant' \
  --param ON_POLICY=false \
  --param LOG_RETRIEVAL_INTERVAL_SECONDS=30 --param POLL_INTERVAL_SECONDS=30 --skip-version-check
```
Confirm the resume took: the trainer's FIRST step line should read `<N+1>/8601`, not
`1/8601` — e.g. seeding from `checkpoint-1250` must start at step 1251, not a fresh 0.
`RESUME_FROM_CHECKPOINT_DIR` is empty by default, so a normal launch is unaffected.
- The first 1b attempt (build `c3698cfd`, 2026-10-20) was scancelled after a transient
  `nvidia-container-cli` driver-rpc timeout wedged container-init on `p1-r10-n1`; that
  host is now in the env's `sbatch_options.exclude` list.

> **"Reuse" is text-level, not token-level — the trainer still tokenizes.**
> corpus-prep's `train.jsonl` is cleaned, length-filtered conversation **text**
> (`{messages, row_id}`), NOT token IDs. It uses the tokenizer only to enforce the
> `MAX_LENGTH` filter, carve the eval split, and stamp `row_id` — then writes text
> back. So train-gold **re-tokenizes** that text to build the `input_ids`/labels/
> assistant-mask tensors it trains on (the "Tokenizing train dataset" / "Truncating"
> passes in its log). This is expected, not a missed reuse: no step in this pipeline
> persists ready-to-train `input_ids`. (`logit-precompute`, the separate two-pass
> path, saves teacher top-K *logits* + `index.jsonl` — still not input_ids, so its
> trainer also re-tokenizes.) At smoke scale the tokenize is seconds-to-a-minute per
> rank and dwarfed by on-policy generation; on a large corpus it runs under a
> `main_process_first` barrier (rank 0 tokenizes, others wait) which can trip
> `ddp_timeout` — see `gbansible/docs/skypilot_kd_bluevela_slurm.md`. What
> `distill-gold-trainonly` reuses is the prepped **corpus + retagged tokenizer**
> (skipping the align + corpus-prep targets), not the trainer's own tokenization.

##### Step timing vs batch size (1b ← 30b, 2 nodes / 16× H200, off-policy, pretok corpus)
Measured 2026-10-21 on the pretok all-category corpus. Effective batch is held at **128**
(`per_device × grad_accum × 16 GPU`), so these isolate microbatch size + the Liger fused
JSD kernel — same training dynamics, different throughput. Settled s/step (steps ~5–20,
after warmup); `t_train` is the per-metric-window train time the trainer logs.

| per_device × grad_accum | Liger fused JSD | settled s/step | t_train | notes |
|---|---|---|---|---|
| 2 × 4 (`parameters.yaml` default) | off | **~97 s** | ~480 s | baseline |
| 4 × 2 | on | **~76 s** | ~419 s | **−22%; the sweet spot** |
| 8 × 1 | on | **~76 s** | ~398 s | no gain over 4×2 — compute-bound; don't bother |

Why it plateaus: fitting `step = grad_accum × allgather + compute` to the first two rows
gives per-microstep teacher all-gather ≈ 10.5 s and fixed-per-token compute ≈ 55 s. Going
2→1 microbatch removes one all-gather but doubles per-microstep compute, so it washes out.
Memory is never the gate (ZeRO-3 shards the 30B teacher; ~7 GB/rank peak even at batch 8),
so batch 8 does **not** OOM — it just isn't faster. **Use 4 × 2 + Liger.** At ~76 s/step a
full epoch (~8,601 steps) is ~7.5 days — see the resume note above.

The 97→76 s gain is **two independent levers**, not just the batch reshape: (a) the 4×2
microbatch (fewer, fatter microbatches amortize the fixed per-step teacher all-gather),
and (b) `USE_LIGER_FUSED_JSD=true` — a GPU-kernel change that fuses the JSD loss +
gradient into one Triton pass over the vocab without materializing the full student/teacher
logits tensors (cuts memory traffic + kernel launches, lowering `t_train`). This is
`use_liger_fused_jsd` (the distillation loss kernel), NOT `use_liger_kernel` /
`use_liger_swiglu_mlp` (HF's RMSNorm/SwiGLU/RoPE fusions), which stay off.

> **⚠️ Fused JSD + Granite can NaN — watch the loss.** The step default is
> `use_liger_fused_jsd: false` on purpose: Granite's `logits_scaling=10` can overflow the
> fused **bf16** JSD kernel and produce NaN loss. We force it on here for the −22% speedup
> and verified finite, healthy loss (~0.46, no NaN) on the 1b←30b all-category run — but
> this is config-specific, not a blanket guarantee. On any new model/precision/scaling,
> **watch the first few hundred steps**: if `loss`/`off_policy_loss` goes `nan`/`inf`,
> drop `--param USE_LIGER_FUSED_JSD=false` (back to ~97 s/step, the unfused path) rather
> than train a corrupted student. A NaN here is silent to the step's exit code until a
> downstream check catches it.

### First bring-up
Start single-node (`--param NUM_NODES=1`) to validate before scaling to 2.

### Worked example — granite-4.0-1b ← 4.2-30b, all-category ~1.1M (full pipeline)

Distill the **granite-4.0-1b-instruct** student from the **granite-4.2-30b** teacher on
the all-category set (`g42sft_allcat_rh`, 1,100,909 rows — the 758k reasoning+agentic
`both_think` slice + ~335k general/chat/IF/safety), via the full `distill-gold` pipeline
(`align` → `corpus` → `train-gold`). Mirrors the gbansible tracking-doc run
"Full suite — 4.1-3b-base distill (ALL-CATEGORY ~1.1M)", but into a 1b student.

Two one-time prep facts, because the inputs are NOT drop-in:

1. **Student needs retagging — `align` does it, no manual step.** `granite-4.0-1b-instruct`'s
   tokenizer does NOT match the teacher (shared-vocab JSD requires student==teacher;
   `1b-instruct-clean` sha `e2bad664` ≠ teacher `88397531`). The full recipe's `align`
   step (`tokenizer-align-slurm`) retags the 1b onto the teacher's ChatML using
   `CHAT_TEMPLATE` and emits `retagged_student` — so bind `STUDENT_MODEL_URI` to the
   **raw** `granite-4.0-1b-instruct-clean`, not a pre-retagged dir. (The train-only
   recipe, by contrast, needs an already-retagged student — which is why this uses the
   full pipeline.) The chat template emits `<|start_of_role|>assistant<|end_of_role|>`,
   which matches the recipe's default `RESPONSE_TEMPLATE`, so no override is needed.

2. **The all-category corpus needed a format conversion.** The source
   `/proj/granite-build/g4os/kd-sandbox/data/g42sft_allcat_rh.jsonl` stores `messages`
   and `tools` as **stringified JSON**, but `corpus-prep` (`prep_corpus.py`) requires
   `isinstance(messages, list)` and drops any row that fails it — so fed as-is it would
   drop ALL 1.1M rows and produce an empty corpus. A streaming converter decodes those
   two fields into native arrays (keeping `reasoning_content` for the thinking run):
   ```bash
   # on a BlueVela login node; streams, does not load 28G into RAM
   python3 /proj/granite-build/g4os/kd-sandbox/convert_allcat_messages.py \
     /proj/granite-build/g4os/kd-sandbox/data/g42sft_allcat_rh.jsonl \
     /proj/granite-build/g4os/kd-sandbox/data/g42sft_allcat_rh_messages.jsonl
   # → DONE in=1100909 out=1100909 skipped=0  (29G; all rows kept)
   ```
   `corpus-prep` then re-tokenizes this text at train time (the recipe consumes a
   `{messages}` JSONL, NOT a pre-tokenized HF dataset dir — so the old
   `g42sft_allcat_rh_pretok_basehub` 3b-tokenizer pretok is irrelevant here).

Launch (full one-epoch run; `MAX_STEPS=-1` lets `NUM_TRAIN_EPOCHS=1.0` define the epoch,
`MAX_LENGTH=16384` keeps the long agentic/reasoning rows as the tracking-doc run did):
```bash
gb build start -f recipes/granite4-gold-distillation/slurm/distill-gold/build.yaml \
  --param STUDENT_MODEL_URI=env:///proj/granite-build/g4os/kd-sandbox/student_overlays/granite-4.0-1b-instruct-clean \
  --param TEACHER_MODEL_URI=env:///proj/granite-build/g4os/kd-sandbox/teacher_overlays/granite-4.2-30b \
  --param DATASET_URI=env:///proj/granite-build/g4os/kd-sandbox/data/g42sft_allcat_rh_messages.jsonl \
  --param MAX_STEPS=-1 --param NUM_TRAIN_EPOCHS=1.0 \
  --param MAX_LENGTH=16384 --param MAX_COMPLETION_LENGTH=15872 \
  --param THINK_POLICY=keep --param SAVE_STEPS=500 --param NUM_NODES=2 \
  --param LOG_RETRIEVAL_INTERVAL_SECONDS=30 --param POLL_INTERVAL_SECONDS=30 \
  --skip-version-check
```
`MAX_COMPLETION_LENGTH` **must be strictly less than** `MAX_LENGTH` — the gold trainer
reserves `MAX_LENGTH − MAX_COMPLETION_LENGTH` tokens for the prompt and
`gold_config.__post_init__` raises `ValueError` if they are equal (or completion >
length). 15872 leaves a 512-token prompt reserve (the ratio the default
`parameters.yaml` ships, 4096/3584). Off-policy JSD (`ON_POLICY=false`, the default). At 2 nodes the effective batch is
`per_device(2) × grad_accum(4) × 16 GPU = 128`, so one epoch over 1.1M rows is ~8,600
steps (fewer if `MAX_LENGTH=16384` filtering drops long rows). **Smoke first:** rerun
with `--param MAX_STEPS=50 --param SAVE_STEPS=25` to validate align-retag → corpus
(keeps ~1.1M rows) → train → checkpoint → teardown before committing the multi-day run.

> The converter `convert_allcat_messages.py` lives on `/proj` next to the data (not in
> this repo); it is reproducible from the snippet above if deleted. The 28G source
> `g42sft_allcat_rh.jsonl` is itself rebuilt by `convert_g42sft.py` + concat — see
> `gbansible/docs/distill-runs.md` "ALL-CATEGORY".

### Scaling: more trainer nodes is NOT faster at smoke scale
Measured on BlueVela, external-server on-policy, identical config except trainer size:

| | 1-node trainer (8 GPU) | 2-node trainer (16 GPU) |
|---|---|---|
| `t_train` (compute/step) | ~120–155 s | ~225–273 s (**~1.8× slower**) |
| `t_sync` (weight sync/step) | ~16 s | ~23 s |
| `t_gen` (on-policy, per step) | ~1300–2500 s | ~1300–2500 s (**same**) |

Adding a trainer node makes the step **slower**, not faster, because at the smoke
config the run is **communication- and generation-bound, not compute-bound**:
- The global batch is tiny and fixed (`per_device_train_batch_size=2` ×
  `grad_accum=4` × GPUs → 64 on 1 node, 128 on 2). Going 8→16 GPUs doesn't shrink
  per-GPU compute much; instead it turns the ZeRO-3 all-gather of the 30B params from
  an intra-node NVLink collective into a **cross-node IB** one — so `t_train` and
  `t_sync` go **up**.
- Wall-clock is dominated by `t_gen` (student generation on the vLLM server), which is
  ~10× everything else and **identical** across topologies — both use one server.

Multi-node trainers only pay off if you (a) raise the global batch so each GPU does
real work, or (b) scale the vLLM server so `t_gen` drops. At fixed smoke batch + a
single server, more trainer nodes just add fabric overhead. The 2-node run's value is
**correctness** (it's the config that hung before the NCCL-pin fix), not speed.

> **(RESOLVED) `NUM_NODES>=2` trainer used to hang at process exit, so the build never
> tore down.** First seen on the 2-node external-server run (build `eecd113e`): training
> ran all 30 steps and wrote the final checkpoint, but the step wedged afterwards and
> `train-gold` stayed `RUNNING`, so `teardown` never fired and both allocations hung
> until a manual `scancel`. Root cause: `accelerate ... | tee` — a straggler worker that
> didn't exit on the multi-node shutdown barrier held the stdout **pipe** open, so the
> shell blocked forever on `PIPESTATUS` and never reached checkpoint-emit. (1-node runs
> have no cross-node shutdown barrier and never hit it.) **Fix (in `gold-slurm`):**
> (1) run accelerate with a **direct file redirect, not a pipe** (`> "$TRAIN_LOG" 2>&1 &`),
> so no straggler can wedge the shell on a pipe; a `tail -F` restores the live `sky logs`
> view, and the shell `wait`s accelerate for its true exit status. (2) A **watchdog
> backstop** that, once training completed (`train_runtime`, shared to peers via a
> `.train-exit` sentinel) and a grace window (`GB_TRAIN_EXIT_GRACE`, default 300s) passed,
> dumps the straggler's thread stacks to `exit_hang_rank*.debug` and `pkill`s the worker
> tree so `wait` returns. (3) An **unconditional rank-0 success-gate** that refuses to
> emit the checkpoint artifact unless `train_runtime` was logged AND model weights exist
> on disk (closes a false-SUCCESS hole for `emit_checkpoint_artifacts: false` recipes).
> Verified on build `5d195050`: 2-node run trained to step 30, wrote `checkpoint-30`,
> accelerate exited cleanly within grace (watchdog no-op, no `exit_hang` dump),
> `teardown` fired, allocations cleared — no manual intervention. The redirect removed
> the actual wedge; the watchdog is the backstop if accelerate's own `wait` ever hangs
> on a worker.

---

## Key parameters

| Param | Default | Notes |
|---|---|---|
| `NUM_NODES` / `NUM_GPUS_PER_NODE` | 2 / 8 | Trainer topology (16 H200). |
| `GPU_CPUS` | 64 | CPUs/node for GPU targets. **Load-bearing** — see Gotchas. |
| `QUEUE` / `VLLM_QUEUE` | gpu-mid / gpu-high | Trainer vs server partition. |
| `VLLM_QOS` | grp_granite_dot_build_high_1n | gpu-high denies `qos=normal`. |
| `ACCELERATORS` / `VLLM_ACCELERATORS` | H200:8 | Must match the env `constraint: h200`. |
| `ON_POLICY` / `LMBDA` / `LMBDA_SCHEDULE` / `LMBDA_INIT` | false / 0.0 / constant / 0.0 | On-policy ramp controls. |
| `MAX_STEPS` / `MAX_LENGTH` / `PER_DEVICE_TRAIN_BATCH_SIZE` / `GRADIENT_ACCUMULATION_STEPS` | 30 / 4096 / 2 / 4 | Smoke scale. |

---

## Gotchas discovered porting this (all handled in the recipe/env)

- **8-vs-4 GPU / socket affinity.** BlueVela H200 nodes are `gpu:h200:8(S:0-1)` — 4
  GPUs per socket. A task pinned to one socket's CPUs sees only 4 GPUs; an 8-rank
  `accelerate` then dies with `cuda:7 out of range ... available: 4`. Fixed by
  `GPU_CPUS: 64` (spans both sockets), mirroring the direct run's `cpus: 64+`.
- **H100 vs H200 capacity.** `gpu-mid`/`gpu-high` reject an H100 request when H100 is
  full (`resubmit with --constraint=h200`). The env sets `constraint: h200` and the
  recipe uses `H200:8` accelerators.
- **gpu-high QOS.** `gpu-high` denies `qos=normal` (the global `~/.sky` default
  gpu-mid uses). The vllm-server target overrides `sbatch_options.qos` to `VLLM_QOS`.
- **SSH config collision.** gbserver refuses to overwrite a pre-existing `~/.slurm/config`
  `bluevela` block that differs from the env's. The env's `cluster_ssh_configs` must
  match the on-host block field-for-field.
- **Per-build isolation / no cross-build reuse.** Each build writes under
  `WORKDIR_ROOT/RUN_NAME/<build_id>/`, so align+corpus re-run every build (that is
  what `distill-gold-trainonly` exists to short-circuit). Within a build, a retry
  reuses already-SUCCESS targets (`retries.target_reuse_enabled: true`).

### NCCL tuning: where it comes from (LSF vs SLURM)
The BlueVela IB fabric needs specific NCCL env vars or NCCL picks the wrong interface
and cross-node collectives hang. The two backends source these completely differently —
**this is the single biggest porting hazard** and caused a multi-run on-policy debug.

- **LSF:** the pins live in **one file outside every step** —
  `nccl_tuning_file: /proj/granite-build/g4os/bv-nccl-tuning.sh` (declared in
  `lsf/ibm-bluevela/environment.yaml`). LSF identity-mounts all of `/proj` into the
  container and sources this file for **every** step automatically, so the LSF `gold`
  and `vllm-server` step bodies contain **zero** NCCL exports. The file sets (with `:=`
  so they're overridable): `NCCL_SOCKET_IFNAME==ibp26s0,ibp60s0,ibp77s0,ibp94s0,
  ibp156s0,ibp188s0,ibp204s0,ibp220s0` (the 8 compute IB rails), `NCCL_IB_HCA=^=mlx5_1,
  mlx5_6` (exclude storage NICs), `NCCL_IB_DISABLE=0`, `NCCL_IB_QPS_PER_CONNECTION=2`,
  `NCCL_IB_SPLIT_DATA_ON_QPS=0`, `NCCL_IB_TIMEOUT=20`, `NCCL_IB_RETRY_CNT=16`,
  `NCCL_CHECKS_DISABLE=1`, `NCCL_CHECK_POINTERS=0`, `TORCH_NCCL_ENABLE_MONITORING=1`,
  `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, plus `TORCH_CUDA_ARCH_LIST=9.0`, `MPI_HOME`,
  `UCX_NET_DEVICES=mlx5_0:1`, and a few build-tooling vars.

- **SLURM:** the SkyPilot SLURM backend does **NOT** identity-mount `/proj` and has
  **NO** `nccl_tuning_file` mechanism. So each SLURM step **exports the pins inline in
  its own run block** — see the block near the top of
  [`gold-slurm/step.yaml`](../../../configurations/assets/environments/skypilot/steps/distill/gold-slurm/step.yaml)
  (search `NCCL_IB_DISABLE`) and the matching block in
  [`vllm-server-slurm/step.yaml`](../../../configurations/assets/environments/skypilot/steps/distill/vllm-server-slurm/step.yaml).
  The SLURM blocks hardcode the same interface pins as the file above (with minor
  deltas: `NCCL_IB_TIMEOUT=30`/`NCCL_IB_RETRY_CNT=20`, and GDR forced off via
  `NCCL_NET_GDR_LEVEL=0` — see the Co-located fixes section). The `envs:` block of each
  step additionally sets the monitoring/timeout vars (`TORCH_NCCL_*`, `NCCL_TIMEOUT`).

**The rule that follows:** on SLURM, **every** step that joins a NCCL group with a peer
on another node — including a cross-allocation peer like an external vLLM server — needs
these inline exports, and all members of a group must pin the **same** rails. Omitting
them on `vllm-server-slurm` (because the LSF original it was copied from had none) is
exactly what hung the separate-server on-policy path: the trainer pinned the IB rails,
the un-pinned server fell back to a container bridge, and the weight-sync collective
could not connect. If you add a new SLURM step that does cross-node NCCL, copy the pin
block. See the Known limitation section for the full write-up.

### On-policy vLLM server gotchas (`vllm-server-slurm`)
These only bite the on-policy path — the vllm-server step is the newest and runs
differently from the torch-only trainer. All are handled in the step.

- **vLLM port selection (`[Errno 98] address already in use`).** gpu-high nodes are
  shared (partial-node allocations), so the configured `VLLM_PORT` (8001) can already
  be bound by another job on the same host. vLLM loads the model and captures CUDA
  graphs fine, then its HTTP server fails to bind the port and exits — so it never
  answers `/health`, never publishes `vllm_url`, and the trainer waits out the full
  `VLLM_HEALTH_TIMEOUT_SECONDS` with no obvious cause. **Fix:** the step probes the
  configured port and, if it's in use, binds a free one instead (`socket.bind(('',0))`,
  same approach as `gbansible/skypilot/bfcl_eval_bluevela_slurm.yaml`). The chosen
  `$PORT` drives BOTH `--port` and the advertised `vllm_url`, so bind and advertise
  stay in sync — the trainer always reaches the port the server actually bound. A
  fixed `VLLM_PORT` is only safe on an exclusive whole-node allocation.
- **libcuda.so / Triton (`libcuda.so cannot found!`).** vLLM's Triton / `torch.compile`
  path finds the CUDA driver via the ldconfig cache (`/etc/ld.so.cache`), not the
  filesystem. SkyPilot's SLURM backend creates the enroot container in a separate srun,
  so the cache baked at create-time misses the enroot-injected `libcuda.so.1` → the
  engine-core worker aborts before loading the model. **Fix:** the step runs `ldconfig`
  (root via `--container-remap-root`) + adds the driver dir to `LD_LIBRARY_PATH`. The
  torch-only gold trainer is unaffected (it dlopen's `libcuda.so.1` by full path), which
  is why gold-slurm worked and only the server hit this.
- **`verify_trainer_accepts_keys` import (`No module named 'vllm_ascend'`).** The gold
  step's trainer-key preflight `import`s `gb_steps_post_training`, which chains into
  `vllm` → `vllm_ascend` (a Huawei NPU backend absent from this image) → ImportError →
  the step exits before rendering its config. Only triggers on-policy (the check runs
  only when non-default keys like `lmbda_schedule`/`lmbda_init` are emitted). **Fix:**
  the recipe sets `VERIFY_TRAINER_ACCEPTS_KEYS: false` — the keys are accepted by the
  pinned trainer; the verify just can't import to confirm it in this image.

### Co-located on-policy fixes (`distill-gold-colocated` + `gold-slurm`)
Two further fixes are specific to the in-allocation split (case 2 of `gold-slurm`).

- **(RESOLVED) lmbda `linear` ramp in the pinned trainer.** The `-slurm` steps clone
  the trainer from `github.com/laminair/gb-steps-distillation`. The original pin
  `a5d59bc4` had a broken `_get_current_lmbda()` — it implemented only `"constant"` and
  fell off the end for `"linear"` (returning `None`, so `lmbda_schedule=linear` crashed
  at step 0 with `TypeError: '<=' not supported between NoneType and float`; the ramp
  body was dead code below the early `return` in `_draw_on_policy()`). This was fixed
  upstream in [PR #1](https://github.com/laminair/gb-steps-distillation/pull/1) (merge
  commit `39b2790`), and the four `-slurm` steps now pin `39b2790`
  (`code_config.ref`/`expect_ref`). The earlier stop-gap — a `patch_lmbda_schedule.py`
  wired via `code_config.setup_command` in the colocated/trainonly recipes — has been
  removed. Nothing extra is needed; a `linear` ramp works on the pinned trainer.
  (The LSF steps + `steps/distill/**` templates still pin `a5d59bc4` under the
  `test_source_contract` byte-identical contract; bump them separately if an LSF
  on-policy linear run needs the fix.)
- **Serve-node address handoff (`vLLM server can't be reached at
  http://0.0.0.0:8001`).** In the in-allocation split the serve rank starts vLLM on
  *its* node, but the trainer ranks have no SkyPilot-provided `vllm_url` to point at
  (that only exists for the separate-server path). Pointing them at `0.0.0.0` fails.
  **Fix:** the serve rank resolves its own IP (`/etc/hosts` + `getent`) and writes
  `token\nIP:port` to `$GB_BUILD_WORKDIR/.vllm-server-addr` (write-tmp-then-`mv`, so
  readers never see a half-written file); every trainer rank waits (up to 600s) for
  that file and sets `--vllm_server_host`/`--vllm_server_port` from it. The address
  file lives on the shared `/proj` workdir, so the handoff needs no extra networking.
  Host **and** port travel together because the serve rank may fall back to a free
  port (same partial-node reason as the server gotcha above).
- **Serve-rank shutdown so the build self-completes (otherwise the allocation
  strands after a successful run).** `run_vllm_serve` is a forever HTTP server. In the
  in-allocation split the serve rank and the trainer share **one** SLURM allocation, so
  when the trainer finishes its 30 steps and `accelerate` exits, the serve rank's run
  script is still blocked in the server — SkyPilot waits for **all** nodes' scripts to
  return before the job completes, so the job stays `RUNNING`, the build never reaches a
  terminal status, and the per-step `sky down` never fires. The run succeeded (checkpoint
  written) but the 2× H200 allocation is held until someone `scancel`s it by hand — which
  is exactly what happened on the first successful colocated run (`87cccce3`). Unlike the
  separate-server path, this recipe has **no `teardown` target and no
  `VLLM_MAX_LIFETIME_SECONDS`** to fall back on. **Fix:** a sentinel handshake in
  `gold-slurm`. The serve rank now **backgrounds** `run_vllm_serve` (instead of `exec`)
  and polls for a `$GB_BUILD_WORKDIR/.train-done` file; trainer rank 0 writes that file
  (token-scoped, `rc` on line 2) the moment `accelerate` exits — on **both success and
  failure**, before its own early exit — so a crashed trainer still releases the serve
  node. On seeing the sentinel the serve rank stops vLLM and exits with the trainer's
  return code; it also exits (non-zero) if the server dies on its own first. All run
  scripts then return → the job completes → `sky down` fires automatically. No manual
  `scancel` needed.

### GPU monitoring caveat
`gbansible/scripts/gpu_watch_slurm.sh` (and the on-cluster `/u/granitebuild/`
copy) can show fewer GPUs than a job holds — a `srun --overlap` probe artifact, not
a shortage. The authoritative allocation is `scontrol show job -d <jid>` (the
`GRES=gpu:...(IDX:...)` line) or `scontrol show step <jid>.<N>` TRES.

## On-cluster filesystem & cleanup
Where every tree a build writes on BlueVela lives (SkyPilot control/runtime dirs,
per-build workdirs, step artifacts), what writes each, and what is safe to delete —
see [FILESYSTEM.md](FILESYSTEM.md).

---

## Resolved — separate-server on-policy works once the server pins NCCL interfaces

Both on-policy paths work on BlueVela SLURM: the **co-located** split
(`distill-gold-colocated`, server in the trainer's allocation) and the **separate-server**
path (`distill-gold` / `distill-gold-trainonly` with `ON_POLICY=true`, server in its own
allocation). The separate-server path hung for several runs; the cause was a **one-sided
NCCL config bug, NOT any inherent cross-allocation limit** (there is none on this cluster).

**The bug.** The on-policy weight sync builds a NCCL group spanning the trainer and the
vLLM server. The trainer (`gold-slurm`) pins NCCL to BlueVela's 8 IB rails
(`NCCL_SOCKET_IFNAME`/`NCCL_IB_HCA`), but `vllm-server-slurm` originally set **none** of
those (its LSF ancestor got them from the env's `nccl_tuning_file`, which the SLURM
backend doesn't provide — see the NCCL-tuning section above). So the un-pinned server
auto-selected a different interface (a container bridge / `lo`): the HTTP handshake
(`GET /health` → `GET /get_world_size` → `POST /init_communicator`, all `200 OK`)
succeeded on plain sockets, then the first NCCL *collective* hung on mismatched
interfaces and the watchdog SIGABRT'd it (`default_pg ... WorkNCCL(... ALLREDUCE ...)`).

**The fix.** Export the **same** NCCL interface pins in `vllm-server-slurm` as
`gold-slurm` uses, so both members of the sync group agree on the fabric. With that,
the cross-allocation sync completes: confirmed on build `77f2e205` — `n_sync_params: 226`,
`t_sync ~17s`, server logging thousands of `POST /update_named_param/ 200 OK` plus
`/generate/`, training advancing normally (loss 0.81→0.60, same as co-located).

Chasing this cost four cluster runs that varied the wrong things (all hung identically
because they held the real bug — the un-pinned server — constant): IB-vs-TCP
(`NCCL_IB_DISABLE`), 2-node-vs-1-node trainer, gpu-high-vs-gpu-mid partition, 8-vs-1-GPU
server. The lesson: **on SLURM, every step joining a cross-node NCCL group needs the
inline interface pins, matching all other members** (see the NCCL-tuning section).

Two diagnostic aids added while chasing this are retained:
- **vLLM server logs tee'd to `/proj`** (`$GB_BUILD_WORKDIR/vllm_server.log`, and
  `vllm_server_rank${RANK}.log` for the co-located serve rank) at `--log_level info` —
  this is what showed the HTTP handshake succeeding while NCCL hung (pinning it to the
  interface mismatch, not routing).
- **`nccl_ib_disable` knob** (`--param NCCL_IB_DISABLE=1`, or `GB_NCCL_IB_DISABLE` env)
  forces NCCL onto TCP sockets — the transport-vs-rendezvous test.

Note: `NCCL_TIMEOUT_MS` / `nccl_timeout_ms` does NOT shorten a weight-sync hang — the
trainer's `default_pg` is created by accelerate/DeepSpeed with its own 1800s watchdog,
which those env vars don't override. A misconfigured separate-server run self-aborts in
~30 min, not sooner.

**Which to use.** `distill-gold-colocated` keeps everything in one allocation (simplest,
no separate server lifetime to manage). The separate-server path decouples the server
(its own allocation/partition, e.g. gpu-high) from the trainer and matches the LSF
design — now viable on SLURM too.

---

## Prerequisites / environment notes

- **gbserver** must be the local source build (editable), not the public `stable`
  tag — the stable 0.3.x lacks multi-node (`num_nodes`) support and silently runs on
  1 node. Install with `pip install -e . --no-deps`. The editable build reports
  version `0.0.0`, which trips the CLI's version floor; append `--skip-version-check`
  to `gb` commands.
- **gbserver MUST stay alive + reachable for the WHOLE queue wait AND the entire run —
  run it on a persistent host, NOT a laptop.** The standalone gbserver is the
  orchestrator: after `sbatch`, it is what `sky exec`s the workload into the containers
  once the job gets GPUs, and what drives checkpoint-artifact emission and teardown
  throughout. If its host sleeps or drops off the network:
  - *during the queue wait* → the job gets nodes, the containers start, but nothing
    exec's the workload into them → an **orphan** allocation (ready containers, idle
    GPUs, zero training) that also burns the group GRES quota and blocks the next
    submit. This repeatedly bit a laptop-hosted gbserver: a ~7 h gpu-mid queue wait is
    far longer than a laptop stays awake, so the job scheduled while the laptop was
    closed and orphaned every time.
  - *during the run* → gbserver can't emit checkpoint artifacts or tear down; on
    reconnect it may consider the build failed.
  An orphan is NOT recoverable in-framework: once gbserver marks the build failed and
  cleans it up it will never re-adopt that allocation, and the rendered `run:` script
  (the real workload) is streamed over SSH in-memory, never written to disk, so it
  cannot be hand-re-exec'd without reconstructing the full ~900-line launch contract
  (resume-seed handshake, 2-node accelerate rendezvous, the `gold_config` yaml). The fix
  is to host gbserver somewhere that never sleeps — ideally a **BlueVela login node**
  (under tmux/systemd, co-located with SLURM, no SSH-latency and no laptop-sleep window),
  or an always-on VM that can reach the cluster. Keeping it on a laptop with
  `caffeinate` + power is only a band-aid — it still dies on reboot or network drop.
  `gb build restart <id>` re-drives a failed build cleanly but launches a FRESH cluster
  (new job, back in the queue) — it does NOT reuse an orphan's still-`UP` allocation.
- **Stale SkyPilot Pyxis cache (`docker_image is not supported ... Pyxis plugin is not
  installed`).** A build fails in ~80s **before any SLURM job** with that error even
  though Pyxis IS installed (every prior docker-image run used it). SkyPilot caches per-
  cluster capabilities in **`~/.sky/kv_cache.db`** (table `kv_cache`, keys
  `slurm:pyxis_enabled:<cluster>` and `slurm:fuse_enabled:<cluster>`) and, per its own
  "availability is unlikely to change" assumption, does **not** re-probe until the entry
  expires. When the cluster is near-full or a login node is briefly flaky the probe can
  fail once and cache `false`, poisoning every subsequent docker-image build. Fix — BOTH
  steps (a disk delete alone is not enough; the running gbserver holds the stale value in
  memory and fails identically until restarted):
  ```bash
  # 1. delete the stale false-negatives so the next submit re-probes + re-caches 'true'
  python -c "import sqlite3, os; c = sqlite3.connect(os.path.expanduser('~/.sky/kv_cache.db')); \
  c.execute(\"DELETE FROM kv_cache WHERE key IN ('slurm:pyxis_enabled:bluevela', 'slurm:fuse_enabled:bluevela')\"); \
  c.commit(); c.close()"
  # 2. restart gbserver so it drops the in-memory copy (GB_ENVIRONMENT set; see below)
  pkill -f 'gbserver standalone'
  GB_ENVIRONMENT=STANDALONE nohup .venv/bin/gbserver standalone \
    --space-dir configurations/spaces/local > /tmp/standalone.log 2>&1 &
  ```
  Inspect current values with the same sqlite path:
  `SELECT key, value FROM kv_cache WHERE key LIKE 'slurm:%:bluevela';`. Read
  `sinfo NODES(A/I/O/T)` as Allocated/Idle/Other/Total — the 2nd field is free nodes.
- **Reference / provenance.** The hand-run SkyPilot YAMLs these recipes replace, and
  the fork/cluster setup notes, are in
  `gbansible/docs/skypilot_kd_bluevela_slurm.md` and
  `gbansible/skypilot/kd_distill_bluevela_slurm*.yaml`.
- Docs for the SLURM environment knobs (zone/partition, `sbatch_options`, the
  workdir mount): `docs/environments/skypilot-slurm.md`.
