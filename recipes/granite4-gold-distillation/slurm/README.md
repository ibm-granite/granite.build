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
SSH to `login1` with `~/granite_build_id_ed25519`, and the workdir-mount fix
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
`RETAGGED_STUDENT_URI` and `CORPUS_URI` **must be from the same build** — gold.py
reads the tokenizer the corpus was prepped against; a mismatch trains on
mis-segmented text silently. What these two artifacts are and how the `align` /
`corpus` steps build them is documented in
[UPSTREAM_ARTIFACTS.md](UPSTREAM_ARTIFACTS.md).

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

### First bring-up
Start single-node (`--param NUM_NODES=1`) to validate before scaling to 2.

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
Three further fixes are specific to the in-allocation split (case 2 of `gold-slurm`).

- **Broken lmbda `linear` ramp in the pinned trainer (`TypeError: '<=' not supported
  between NoneType and float`).** The step clones its code from the public
  `github.com/laminair/gb-steps-distillation` and checks out the pinned commit
  `a5d59bc4` (`code_config.repo`/`ref` in `gold-slurm/step.yaml`). At that commit
  `custom_gold_trainer.py`'s `_get_current_lmbda()` implements only the `"constant"`
  schedule and **falls off the end for `"linear"`** (returns `None`); the actual
  linear-interpolation body was misplaced as dead code below the early `return` in
  `_draw_on_policy()`. With `lmbda_schedule=linear` the trainer then does
  `current_lmbda = None` and crashes at the first step (`on_policy_draw.py: if
  current_lmbda <= 0.0`). The pin is the only commit that ever touched that file and
  the single commit after it is docs-only, so there is **no upstream commit to bump
  to** — the correct ramp lives only in the private `kd-sandbox` source the repo was
  packaged from, whose flat layout the step cannot consume. **Fix:** a stop-gap patch
  script, [`gold-slurm/src/patch_lmbda_schedule.py`](../../../configurations/assets/environments/skypilot/steps/distill/gold-slurm/src/patch_lmbda_schedule.py),
  rewrites `_get_current_lmbda` to compute the ramp inline
  (`lmbda_init + progress*(lmbda - lmbda_init)`, `progress = global_step/max_steps`).
  It is wired via `code_config.setup_command` in
  [`distill-gold-colocated/build.yaml`](distill-gold-colocated/build.yaml), so it runs
  on **every rank after the clone, before `accelerate`**. It is **idempotent** (no-op
  if the linear branch is already present) and **fails loudly** (exit non-zero) if the
  known-broken signature is absent — i.e. if the pin ever changes, the build stops
  rather than silently training on a differently-behaving ramp. Remove the
  `code_config` block once the pin is bumped to a trainer with a working ramp.
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
- **Reference / provenance.** The hand-run SkyPilot YAMLs these recipes replace, and
  the fork/cluster setup notes, are in
  `gbansible/docs/skypilot_kd_bluevela_slurm.md` and
  `gbansible/skypilot/kd_distill_bluevela_slurm*.yaml`.
- Docs for the SLURM environment knobs (zone/partition, `sbatch_options`, the
  workdir mount): `docs/environments/skypilot-slurm.md`.
