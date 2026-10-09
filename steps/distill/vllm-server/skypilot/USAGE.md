# vllm-server (SkyPilot / LSF)

Serves a model under vLLM as a long-lived **SERVICE** target and publishes its URL as a
`mem://` binding, so a training target in a *separate* allocation can reach it. The URL
is published only once the server answers `/health`, so a consumer that binds to it
cannot start early. The step also publishes its SkyPilot cluster name, for the teardown
target that releases the allocation.

> **Developing or testing this step?** See `steps/distill/vllm-server/skypilot/README.md`
> in the granite.build repository for how the step is constructed, tested, and published
> — including what the contract tests cannot check without a cluster.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/vllm-server` URI:

```yaml
steps:
  - step_uri: space://steps/distill/vllm-server
```

## Config contract (`vllm_config`)

The server's fields live under the step's `config.vllm_config`.

The model to serve is no longer a config field; it is the required `model` target input —
see [Inputs](#inputs) below. For on-policy GOLD this is the **student**, not the teacher
(see [Serving the teacher](#serving-the-teacher-instead-of-the-student)).

### Optional

| Field | Type | Purpose |
|---|---|---|
| `port` | integer | Port the server listens on and the URL advertises; the reference launcher's `VLLM_API_PORT`. Default: `8001`. |
| `max_model_len` | integer | vLLM's `--max-model-len`. Must be at least the trainer's `max_length`, or a prompt the trainer accepts is one the server refuses mid-run. Default: `16384`. |
| `gpu_memory_utilization` | number | vLLM's `--gpu-memory-utilization`. Default: `0.9`. |
| `tensor_parallel_size` | integer | vLLM's `--tensor-parallel-size`. Default: `1`. |
| `data_parallel_size` | integer | vLLM's `--data-parallel-size`. Empty means one rank per GPU on the node (`NUM_GPUS_PER_NODE`, 8 if unset), taken from the allocation rather than from a parameter so it always describes the node it got. Default: `""`. |
| `min_completion_length` | integer | Minimum tokens the student must generate before EOS, enforced in the server process. Exported as `GOLD_MIN_TOKENS`. `0` turns it off. A non-zero value is refused unless the delivered `run_vllm_serve.py` reads `GOLD_MIN_TOKENS`; at the default pin it does not (see [The generation floor](#the-generation-floor)). Default: `0`. |
| `health_timeout_seconds` | integer | How long to wait for `/health` before giving up. A cold multi-GB load plus CUDA graph capture takes minutes, and on a shared filesystem the first read is the slow one; the reference launcher allows 1800 s. Default: `1800`. |
| `health_poll_seconds` | integer | Sleep between `/health` probes. Default: `10`. |
| `max_lifetime_seconds` | integer | Ceiling on how long the server holds its allocation once healthy. `0` means no ceiling: hold until the teardown target downs the cluster, which multi-hour runs rely on. Counted from the moment the server is healthy. Set it as a backstop against a stranded server (see [Nothing reaps the cluster](#nothing-reaps-the-cluster-but-the-teardown-target)); it must outlast the consumer's run. Default: `0`. |

### Code delivery (`code_config`)

The server code (`gb_steps_post_training.distillation.run_vllm_serve`) comes from an
unauthenticated HTTPS clone of the public source repo at run time, the same mechanism
and pin as the other ported distillation steps. Only `repo`, `ref` and `workdir` are
read by this step's run script; the other keys are accepted for parity with those steps
and have no effect here.

| Field | Type | Purpose |
|---|---|---|
| `repo` | string | Git repository cloned at run time. Default: `https://github.com/laminair/gb-steps-distillation.git`. |
| `ref` | string | Commit checked out after the clone; skipped when empty. Default: `a5d59bc45524a8d75706e20d44ae1a254f273f23`. |
| `workdir` | string | Directory under the workdir root the repo is cloned into. Any existing directory there is removed first. Default: `distill-code`. |
| `code_dir` | string | Not read by this step: there is no pre-built-checkout path, so setting it does not replace the clone. Default: `""`. |
| `expect_ref` | string | Not read by this step. Default: `a5d59bc45524a8d75706e20d44ae1a254f273f23`. |
| `token_secret` | string | Not read by this step; the clone is unauthenticated. Default: `""`. |
| `python` | string | Not read by this step; the interpreter is fixed at `/stage/.venv/bin/python`. Default: `/stage/.venv/bin/python`. |
| `setup_command` | string | Not read by this step. Default: `""`. |

### Top-level config

| Field | Type | Purpose |
|---|---|---|
| `compute_config` | mapping | Node and GPU counts, e.g. `{num_nodes: 1, num_gpus_per_node: 8}`. The server runs on one node; its GPU count sets the default data-parallel size. |
| `launcher_config` | mapping | SkyPilot launcher overrides: `image_id` (default `docker:us.icr.io/cil15-shared-registry/kd-sandbox-distill:0.1.0-uv`) and `resources` (`accelerators`, `memory`, `cluster`, `zone`). |
| `poll_interval_seconds` | integer | Monitor status-poll interval. Default: `300`. |
| `log_retrieval_mode` | string | Monitor log-retrieval mode. Default: `periodic` (see [log_retrieval.mode](#log_retrievalmode-is-periodic-not-startup_window)). |
| `log_retrieval_interval_seconds` | integer | How often the monitor retrieves the log, and so how soon after `/health` the URL is seen. Default: `15`. |

## Inputs and outputs

### Inputs

The step declares one required input, read as `{{ bindings.model.binding.path }}`:

| Input | Required | Type | Typical source |
|---|---|---|---|
| `model` | yes | model | the student: `align.retagged_student`, or an SFT target's `checkpoint` |

```yaml
inputs:
  model: {binding: align.retagged_student}
```

`vllm_config.model_path` is no longer read; a target that still sets it fails at run time
with a message naming the input to bind instead.

### Outputs

Both outputs are optional in the step and are declared on the target as `mem://` URIs.
Both are bindings, not artifacts: nothing is transferred, so the target need not give
them a lineage type. Each is registered with the shipped skypilot monitor's generic
marker, on its own log line, so the step carries no scrape rule of its own.

| Output | Value | Marker | Consumer |
|---|---|---|---|
| `vllm_url` | `http://<addr>:<port>`, published **only once `/health` answers** | `GB_ARTIFACT_ID:vllm_url GB_ARTIFACT_STATE:http://<addr>:<port>` | the trainer, e.g. the `space://steps/distill/gold` step's optional `vllm` input, read as `{{ bindings.<name>.binding.state }}` |
| `cluster_name` | the SkyPilot cluster name (`gb-<id>`, from `GB_SKYPILOT_CLUSTER_NAME`; `unknown` if unset), published at start-up | `GB_ARTIFACT_ID:cluster_name GB_ARTIFACT_STATE:<name>` | the teardown target, via the `space://steps/skypilot-teardown` step's `teardown_config.cluster_names` |

`<addr>` is an IP resolved from `/etc/hosts`, then `getent ahostsv4`, falling back to the
bare hostname — the way the `space://steps/distill/gold` step resolves `MASTER_ADDR`.
The consumer is in a different allocation, so a bare short hostname need not resolve
there.

Use `mem://` and `.binding.state`, never `env://` and `.binding.path`: `env://` runs the
value through filesystem-path normalisation and mangles `http://host:8001` into
`/http:/host:8001`.

## Working directory and paths

The run script starts in the step's per-run workdir (`GB_BUILD_WORKDIR`, or the current
directory if that is unset). It:

- clones `code_config.repo` into `<workdir>/<code_config.workdir>` (default
  `distill-code/`), checks out `ref`, and puts its `src/` on `PYTHONPATH`;
- runs the server with the image's interpreter, `/stage/.venv/bin/python`, with
  `/stage/.venv/bin` prepended to `PATH`;
- sets `VLLM_RPC_BASE_PATH` and `TMPDIR` to `/tmp/vllm-<user>-<LSF job id>`, created on
  start (see [The RPC socket path](#the-rpc-socket-path));
- sets `HF_HOME=/opt/hf-cache`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `OMP_NUM_THREADS=8`, carried
  from the reference launcher, the only configuration that had served this model.

The resolved `model` input path must be visible from the server node. The step writes
nothing else; its results are the two bindings.

## Example build.yaml

Three targets: the server, a trainer that binds to its URL, and a teardown target gated
on the trainer's checkpoint. Ordering is implicit — there is no `depends_on` key. A
target starts when all its input bindings resolve:

```
vllm-server ──vllm_url──> train ──checkpoint──> teardown
     └────────────────cluster_name───────────────┘
```

```yaml
granite.build:
  name: gold-onpolicy-example
  targets:
    vllm-server:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        model:
          uri: "env:///proj/.../student"          # the STUDENT, for on-policy GOLD
          type: model
      outputs:
        vllm_url:     {uri: "mem://gold-onpolicy-vllm"}
        cluster_name: {uri: "mem://gold-onpolicy-vllm-cluster"}
      steps:
        - step_uri: space://steps/distill/vllm-server
          config:
            compute_config: {num_nodes: 1, num_gpus_per_node: 8}
            launcher_config:
              resources: {accelerators: "H100:8", cluster: "bluevela", zone: "normal"}
            vllm_config:
              max_model_len: 16384                    # >= the trainer's max_length
              health_timeout_seconds: 1800
              max_lifetime_seconds: 18000             # backstop; must outlast the run
    train:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        student:
          uri: "env:///proj/.../student"
          type: model
        teacher:
          uri: "env:///proj/.../teacher"
          type: model
        vllm: {binding: vllm-server.vllm_url}
      outputs:
        checkpoint:
          uri: "env://{{ binding.path }}"
          type: model
      steps:
        - step_uri: space://steps/distill/gold
          config:
            gold_config:
              max_length: 16384
              # ... the rest of the trainer's config
              vllm_num_servers: 1
    teardown:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        gate: {binding: train.checkpoint}            # ordering only
        vllm_cluster: {binding: vllm-server.cluster_name}
      steps:
        - step_uri: space://steps/skypilot-teardown
          config:
            teardown_config:
              cluster_names:                          # a list, even with one entry
                - "{{ bindings.vllm_cluster.binding.state }}"
```

`recipes/granite4-gold-distillation/lsf/gold-onpolicy-smoke/build.yaml` and
`recipes/granite4-350m/lsf/distill-onpolicy-v2/build.yaml` in the granite.build
repository are complete versions of this graph.

## Notes and limitations

### Status

The shape has run. In build `9973e766` (2026-09-15, the `gold-onpolicy-smoke` recipe)
the trainer synced all 362 parameter tensors to this server over **NCCL** across two
separate LSF allocations, on both of its steps, and the teardown target ended the
server. That was the open question this step left: the trainer pushes updated student
weights over NCCL, not HTTP — the reference launcher's `VLLM_NCCL_COORDINATOR_PORT` — so
a separate server target means a NCCL process group spanning two allocations. Two
steps measure plumbing, not distillation quality. The fallback, had it failed, was the
`space://steps/distill/gold` step's in-allocation role split.

An earlier build, `d77546a9`, allocated this server, health-checked it and wired its
address in, but the trainer generated locally (`--use_vllm` was not passed), leaving 8
H100s idle; the `space://steps/distill/gold` step now passes that flag positively in
both directions.

### What supplies what

| | |
|---|---|
| deps | the image (`/stage/.venv`), the same one the `space://steps/distill/gold` step uses |
| server code | `gb_steps_post_training.distillation.run_vllm_serve`, from `code_config`'s clone |
| bsub, enroot, node topology | SkyPilot's LSF provisioner |
| readiness, address publication, teardown hook | this step |

The step is LSF-only (`subtypes: [lsf]`): the topology contract and the SM90 image are
both LSF/BlueVela specific.

### Why it is a separate target

The `space://steps/distill/gold` step can already serve vLLM by carving the last N nodes
out of its own allocation, which is what the reference launcher does. Three things that
shape needs are things a single step cannot do: the server's address is only known at
run time, the trainer must wait on `/health` before starting, and nothing tears the
server down afterwards.

Splitting the server out dissolves all three:

- the address travels as a `mem://` binding;
- a consumer target does not dispatch until **every** input binding resolves, and this
  step publishes its URL only *after* `/health` passes — so the dependency graph is the
  health gate;
- a `teardown` target gated on the trainer's checkpoint downs the cluster. That is not
  optional on LSF: `idle_minutes_to_autostop` is rejected for SSH/HPC clouds, so a
  SERVICE cluster never autostops and never gets a terminal-status cleanup.

### Bring-up and failure messages

The server runs in the background. The step probes `http://127.0.0.1:<port>/health/`
every `health_poll_seconds`, checking before each probe that the server process is
still alive. Once healthy it logs `vllm-server: healthy after <N>s` — which the monitor
turns into a RUNNING workload status — then publishes `vllm_url` and waits on the
server. All failures exit non-zero with a `vllm-server: FATAL:` line on stderr:

- `server exited with <RC> before becoming healthy` — the server process died during
  bring-up (e.g. an unloadable `model` input);
- `not healthy after <N>s` — `health_timeout_seconds` elapsed; the server is killed;
- `lifetime cap of <N>s reached; releasing the allocation` — `max_lifetime_seconds` was
  set and nothing tore the server down in time; check whether the consumer target
  failed;
- `vllm_config.min_completion_length is <N>, but the delivered ... does not read
  GOLD_MIN_TOKENS` — see [The generation floor](#the-generation-floor).

The messages are reported even though the launcher runs the script under `set -eu`.
Without a lifetime cap, the step exits with the server's own exit code.

### Serving the teacher instead of the student

Not an error. On-policy GOLD has the student generate and the teacher score those
generations; serving the teacher gives a different algorithm that runs to completion
and reports a loss.

### The generation floor

The floor has to be set here as well as on the trainer, and that is not redundancy: the
trainer's `min_completion_length` only bounds what the trainer asks for, while the
sampling decision is taken in this process. trl's `vllm_serve` has no `min_tokens`
field, so `run_vllm_serve.py` is meant to patch `SamplingParams.min_tokens` from
`GOLD_MIN_TOKENS`, which this step exports from `min_completion_length`. Left at 0 while
the trainer sets a floor, the student is free to emit an immediate EOS and the run
collapses into empty completions the teacher then scores as if they were rollouts.

At the default pin (`a5d59bc4`) `run_vllm_serve.py` does not read `GOLD_MIN_TOKENS`, so
the export alone would be a no-op that looks configured. The step therefore greps the
delivered `run_vllm_serve.py` for `GOLD_MIN_TOKENS` and refuses to start when
`min_completion_length` is non-zero and the handling is absent. To have a floor, point
`code_config` at a revision whose `run_vllm_serve.py` honours it; otherwise keep `0`.

### `log_retrieval.mode` is `periodic`, not `startup_window`

A window has to be guessed, and once it closes the scrape never fires again — so a load
slower than the guess means the URL binding never publishes, the consumer target never
dispatches, and nothing errors. Load time depends on model size and shared-filesystem
weather, which is exactly what cannot be guessed.

### The RPC socket path

vLLM binds an AF_UNIX socket under `VLLM_RPC_BASE_PATH`, capped at 108 bytes with
ZeroMQ refusing over 107. A default `TMPDIR` measured 111 on this cluster and failed
*after* the engine started loading, so the step sets a short per-job path.

### Nothing reaps the cluster but the teardown target

Forget the teardown target and the allocation is held until someone runs `sky down` by
hand.

Teardown is gated on the consumer's output (`binding: <train>.checkpoint`), which a
failed consumer never emits, and gbserver's schema has no on-failure semantics — so a
crashed trainer strands this server. LSF SERVICE clusters never autostop either, so
nothing underneath reclaims it. Build `d77546a9` held 8 H100s that way until they were
downed by hand. `max_lifetime_seconds` is the backstop, not the normal path: the
`gold-onpolicy-smoke` recipe sets 3600 and `distill-onpolicy-v2` 5 h.

### `code_config.code_dir` does not apply

Unlike the other distillation steps, this step always clones `repo` at `ref`; it has no
filesystem-checkout fallback, and `code_dir`, `expect_ref`, `token_secret`, `python` and
`setup_command` are not read. Setting `code_dir` here does not change the server's
source.
