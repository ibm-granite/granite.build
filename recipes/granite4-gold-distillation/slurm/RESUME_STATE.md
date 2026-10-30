# RESUME STATE — granite-4.0-1b GOLD distillation (all-category ~1.1M)

Live handoff for resuming the paused 1b epoch. For the full recipe, tuning table, Pyxis
cleanup, and gbserver-persistence rationale see [README.md](README.md) ("Resuming after a
preemption / wall-clock cap", "Step timing vs batch size", and the Gotchas section).

## Current state (2026-10-30 11:17 UTC) — auto-retry queued
- Build `ff8dd0bd-3077-4eda-9c63-7ef0fb27d466` (resume from checkpoint-1250) is on its
  **last** retry: SLURM job **547766**, gpu-mid, `PENDING (Priority)`. It still resumes
  from checkpoint-1250 (the retry re-renders the same params). `MAX_RETRIES=1` is used up —
  one more preemption ends the build FAILED and needs a manual resubmit.
- Latest valid checkpoint is still **checkpoint-1250**; steps 1251–1353 of the first leg
  were lost (no save before preemption). See the 2026-10-30 section below.

### 2026-10-30 resume — resumed OK, preempted after 2 h, auto-retried
- Submitted 00:44 UTC; SLURM job 534539 queued ~8 h on gpu-mid, started 08:41 on
  p3-r01-n3,p3-r06-n3. **Resume verified:** first trainer step `1251/8601`, first loss
  0.3649 (grad_norm 0.87), LR 9.78e-6 — continued schedule, no NaN with Liger fused JSD.
  Measured ~67 s/step on this leg.
- **Preempted 10:42 UTC** (sacct `PREEMPTED`, ran 2:00:59) at step 1353 — 147 steps short of
  checkpoint-1500, so the leg banked nothing. That is preemption #6. `SAVE_STEPS=250`
  (~4.7 h between saves) is too coarse for gpu-mid's observed ~2 h legs → **next manual
  submit: `SAVE_STEPS=50` (~1 h) and `MAX_RETRIES` ≈ 5.**
- Timeline of the automatic recovery: 10:43–11:00 gbserver logs `Cluster ... does not
  exist` every poll (15-min grace) → 11:00 step marked FAILED → 11:04 `retrying in place
  (attempt 1 of 1)` → 11:05 new SLURM job 547766 PENDING. ~23 min from preemption to
  resubmit, plus the new queue wait. Do NOT cancel during the grace window thinking the
  build is wedged — it is about to retry.

### How preemption recovery actually works (verified in code, 2026-10-30)
- **SLURM never requeues our jobs.** Cluster is `PreemptMode=REQUEUE` / `JobRequeue=1`, but
  SkyPilot hardcodes `#SBATCH --no-requeue` in every provision script
  (`~/skypilot/sky/provision/slurm/instance.py:741`) and lists it in
  `_SBATCH_PROTECTED_OPTIONS` (`:47`), so `sbatch_options` cannot override it. Confirmed in
  the generated `/proj/granite-build/g4os/.sky_provision/gb-*.sh`. A requeue wouldn't help
  anyway: it would restore only the container allocation — gbserver `sky exec`s the
  workload separately — and without the resume param.
- **SkyPilot doesn't auto-recover either.** Only *managed jobs* (`sky.jobs.launch`, a
  controller that relaunches) do that; the BlueVela env is `type: Skypilot` (unmanaged
  `sky.launch`, `gbserver/environment/skypilot.py`). `Skypilot_managed` exists only for
  Kubernetes in this repo.
- **gbserver never looks for PREEMPTED.** It polls `sky.job_status()` every
  `POLL_INTERVAL_SECONDS`; no `squeue`/`sacct`/`scontrol` call on SLURM. After preemption
  SkyPilot's record goes away and polls fail with "does not exist". `_PollFailureTracker`
  (`skypilot.py` ~336–450) waits `poll_failure_grace_seconds` (default 900 s, ≥3 failures)
  for SSH-HPC clouds, then declares "gone (preempted or terminated)" → FAILED. (LSF gets an
  extra direct `bjobs` probe; SLURM has no equivalent — it's purely the timeout.)
- **The resubmit is a whole-build retry**, not preemption-aware: `buildrunner._should_retry`
  = status FAILED and `retry_count < max_retries`. It does not inspect the cause, so
  `MAX_RETRIES` is shared between preemptions and real crashes (a real bug is retried too).
  The per-workload `resilience/retry_handler.py` (failure classifiers) is not configured
  for this step/env and did not act.
- Practical consequences: ~15–20 min lost per preemption before resubmit; size
  `MAX_RETRIES` for the expected preemption count; keep the grace ≥ several minutes (shorter
  graces killed healthy BV runs on SSH blips).
- Visibility: `/u/granitebuild/gpu_watch_slurm.sh` now has a "Recently Preempted Jobs"
  section (sacct, last 24 h; `GW_PREEMPT_HOURS=N`, `0` hides). Backup of the old script:
  `gpu_watch_slurm.sh.bak-20261030`.

## Earlier state (paused 2026-10-28, resume attempt cancelled 2026-10-29)
- Recipe `distill-gold-trainonly`, student granite-4.0-1b-instruct ← teacher granite-4.2-30b,
  pretokenized all-category corpus. Off-policy, 2 nodes, batch=4 + Liger (~76 s/step).
- **Paused after 5 consecutive gpu-mid preemptions.** `MAX_STEPS=-1` (full epoch = ~8,601
  steps). Latest valid checkpoint banked: **checkpoint-1250** (~16% of the epoch).
- (As of 2026-10-29: no SLURM job queued, no orphan. Superseded by the 2026-10-30 state above.)

### 2026-10-29 resume attempt — cancelled before it started (deliberate)
- Build `e32bea93-8295-4804-898e-74a0087c5af5` submitted with the resume command below
  (pre-checks passed: Pyxis/fuse cache keys `true`, checkpoint-1250 complete). It reached
  gpu-mid as SLURM job 488005 and sat `PENDING (Priority)`; **no training step ran**.
- Cancelled on purpose with `gb build cancel` (restart later). gbserver's cancel thread timed
  out at 120 s, but the SkyPilot teardown completed ~4 min after the cancel and removed
  job 488005; no auto-resubmit followed. Teardown only cleaned that build's own run dir —
  the resume source below is untouched. Checkpoint-1250 is still the latest.
- Note: this attempt ran on gbserver from the `~/granite.build` checkout (`defense` branch),
  inside the sandbox pod — NOT a persistent host. Its gold-slurm step has the same resume
  logic; it differs only by the deprecated `NCCL_ASYNC_ERROR_HANDLING` alias (harmless
  warning). The persistent-host requirement below is still unmet.
- After a cancel, wait for the "Torn down SkyPilot cluster" log line / empty `squeue`
  before resubmitting; the build shows CANCELLED minutes before the SLURM job is gone.

## Latest checkpoint (resume source)
```
/proj/granite-build/g4os/gbtest/builds/builds/3aa2d92f-f171-4332-ba88-18edec3d1af4/runs/7628422a-f0ad-40aa-9265-01fbcb76a272/checkpoints/distill-gold-trainonly_node2
```
Holds checkpoint-250/500/750/1000/1250 (all complete: model.safetensors + trainer_state.json
+ scheduler.pt + 16 rng_state). HF resumes from the highest (1250).

## Cluster access
```
ssh -i ~/granite_build_id_ed25519 granitebuild@login0.bluevela.rmf.ibm.com
```
Run `gb`/gbserver from this repo root with `GB_ENVIRONMENT=STANDALONE` and
`--skip-version-check`. gbserver must be the editable source build (`pip install -e .
--no-deps`), reports version `0.0.0`.

## DO THIS BEFORE RESUBMITTING (in order)
1. **Host gbserver on a PERSISTENT machine, not a laptop.** gbserver is what `sky exec`s
   the workload into the containers once the job gets GPUs. A gpu-mid queue wait can be
   ~7 h; if the gbserver host sleeps/drops in that window the job orphans (ready
   containers, idle GPUs, no training) and blocks the GRES quota. This caused the repeated
   failures. Use a BlueVela login node under tmux/systemd, or an always-on VM.
2. **Pyxis cache check** — in `~/.sky/kv_cache.db`, keys `slurm:pyxis_enabled:bluevela` and
   `slurm:fuse_enabled:bluevela` must be `true`. A stale `false` fails the build in ~80 s
   with "Pyxis plugin is not installed" (it IS installed). Fix: delete those two keys +
   restart gbserver (see README Gotchas for the exact sqlite command).

## Resume command
```bash
GB_ENVIRONMENT=STANDALONE gb build start -f recipes/granite4-gold-distillation/slurm/distill-gold-trainonly/build.yaml \
  --param RESUME_FROM_CHECKPOINT_DIR=/proj/granite-build/g4os/gbtest/builds/builds/3aa2d92f-f171-4332-ba88-18edec3d1af4/runs/7628422a-f0ad-40aa-9265-01fbcb76a272/checkpoints/distill-gold-trainonly_node2 \
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

## After it runs — verify / watch
- **Resume took:** the trainer's first step line is `1251/8601` (or similar), NOT `1/8601`.
  Fresh-from-0 means the resume seed failed — stop and investigate.
- **No NaN loss:** `USE_LIGER_FUSED_JSD=true` on granite risks NaN (logits_scaling=10 can
  overflow the fused bf16 JSD kernel). Watch `train_rank0.log` for finite `loss`. If NaN,
  rerun with `--param USE_LIGER_FUSED_JSD=false`.
- **Checkpoints:** next new one is checkpoint-1500, then every 250 steps.
- **To cancel:** `gb build cancel <build_id>` (NOT scancel — gbserver retries a scancel'd
  job under MAX_RETRIES). scancel only an orphan gbserver has already abandoned.
- Trainer log: `.../builds/builds/<BUILD>/runs/<run>/train_rank0.log`.

## Constraints / gotchas (learned the hard way)
- `MAX_COMPLETION_LENGTH` (15872) MUST be strictly < `MAX_LENGTH` (16384) or config
  validation crashes.
- gpu-high is NOT an escape from preemption: 1-node quota, but the trainer needs 2 nodes.
  So the run stays on preemptible gpu-mid — a persistent gbserver + frequent-enough
  SAVE_STEPS is the mitigation. If preemptions recur faster than ~5 h, drop SAVE_STEPS to
  100 or 50 so each short leg still banks a checkpoint.
- GDR stays OFF (`NCCL_NET_GDR_LEVEL=0`); GDR-on hangs the first all-gather in-container.
