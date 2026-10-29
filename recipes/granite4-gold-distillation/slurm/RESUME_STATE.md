# RESUME STATE — granite-4.0-1b GOLD distillation (all-category ~1.1M)

Live handoff for resuming the paused 1b epoch. For the full recipe, tuning table, Pyxis
cleanup, and gbserver-persistence rationale see [README.md](README.md) ("Resuming after a
preemption / wall-clock cap", "Step timing vs batch size", and the Gotchas section).

## Current state (paused 2026-10-28)
- Recipe `distill-gold-trainonly`, student granite-4.0-1b-instruct ← teacher granite-4.2-30b,
  pretokenized all-category corpus. Off-policy, 2 nodes, batch=4 + Liger (~76 s/step).
- **Paused after 5 consecutive gpu-mid preemptions.** `MAX_STEPS=-1` (full epoch = ~8,601
  steps). Latest valid checkpoint banked: **checkpoint-1250** (~16% of the epoch).
- No SLURM job queued, no orphan. Nothing lost — resume continues from checkpoint-1250.

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
