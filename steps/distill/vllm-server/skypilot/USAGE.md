# vllm-server (SkyPilot / LSF)

Serves a model under vLLM as a long-lived **SERVICE** target and publishes its URL as
a `mem://` binding, so a training target in a *separate* allocation can reach it.

> **Not yet run on a cluster.** The contract below is tested; the thing no test can
> settle is whether the trainer's NCCL weight-sync group works across two LSF
> allocations. See [Status](#status).

## What supplies what

| | |
|---|---|
| deps | the image (`/stage/.venv`), the same one `distill-gold` uses |
| server code | `gb_steps_post_training.distillation.run_vllm_serve`, from `code_config`'s clone |
| bsub, enroot, node topology | SkyPilot's LSF provisioner |
| readiness, address publication, teardown hook | this step |

## Why it is a separate target

`distill-gold` can already serve vLLM by carving the last N nodes out of its own
allocation, which is what the reference launcher does. Three things that shape needs
are things a single step cannot do: the server's address is only known at run time,
the trainer must wait on `/health` before starting, and nothing tears the server down
afterwards.

Splitting the server out dissolves all three:

- the address travels as a `mem://` binding;
- a consumer target does not dispatch until **every** input binding resolves, and this
  step publishes its URL only *after* `/health` passes — so the dependency graph is
  the health gate;
- a `teardown` target gated on the trainer's checkpoint downs the cluster. That is not
  optional on LSF: `idle_minutes_to_autostop` is rejected for SSH/HPC clouds, so a
  SERVICE cluster never autostops and never gets a terminal-status cleanup.

## Minimal use

```yaml
targets:
  vllm-server:
    environment_uri: space://environments/skypilot/lsf/ibm-bluevela
    outputs:
      vllm_url:     {uri: "mem://vllm-server"}
      cluster_name: {uri: "mem://vllm-server-cluster"}
    steps:
      - step_uri: space://steps/distill/vllm-server
        config:
          compute_config: {num_nodes: 1, num_gpus_per_node: 8}
          launcher_config:
            resources: {accelerators: "H100:8", cluster: "bluevela", zone: "normal"}
          vllm_config:
            model_path: /proj/.../student            # the STUDENT, for on-policy GOLD
            max_model_len: 16384
  train:
    inputs:
      vllm: {binding: vllm-server.vllm_url}
    steps:
      - step_uri: space://steps/distill/distill-gold
        config:
          gold_config:
            vllm_server_url: "{{ bindings.vllm.binding.state }}"
```

`mem://` and `.binding.state`, never `env://` and `.binding.path`: `env://` runs the
value through filesystem-path normalisation and mangles `http://host:8001` into
`/http:/host:8001`.

## Config

| key | default | notes |
|---|---|---|
| `model_path` | — | **required.** For on-policy GOLD the **student**, not the teacher |
| `port` | `8001` | the reference launcher's `VLLM_API_PORT` |
| `max_model_len` | `16384` | must be ≥ the trainer's `max_length` |
| `gpu_memory_utilization` | `0.9` | |
| `tensor_parallel_size` | `1` | |
| `data_parallel_size` | `""` → one rank per GPU | from the allocation, not a parameter |
| `health_timeout_seconds` | `1800` | a cold multi-GB load plus CUDA graph capture is minutes |
| `health_poll_seconds` | `10` | |

## Outputs

- **`vllm_url`** — `http://<addr>:<port>`, published **only once `/health` answers**.
  The address is an IP resolved from `/etc/hosts` then `getent`, the way `distill-gold`
  resolves `MASTER_ADDR`, because the consumer is in a different allocation and a bare
  short hostname need not resolve there.
- **`cluster_name`** — the SkyPilot cluster (`gb-<id>`), for the teardown target.

Both use the shipped monitor's generic `GB_ARTIFACT_ID:… GB_ARTIFACT_STATE:…` form, on
separate lines, so the step carries no scrape regex of its own.

## Things that will bite

**Serving the teacher instead of the student** is not an error. On-policy GOLD has the
student generate and the teacher score those generations; serving the teacher gives a
different algorithm that runs to completion and reports a loss.

**`log_retrieval.mode` is `periodic`, not the service default `startup_window`.** A
window has to be guessed, and once it closes the scrape never fires again — so a load
slower than the guess means the URL binding never publishes, the consumer target never
dispatches, and nothing errors.

**The RPC socket path.** vLLM binds an AF_UNIX socket under `VLLM_RPC_BASE_PATH`, capped
at 108 bytes with ZeroMQ refusing over 107. A default `TMPDIR` measured 111 on this
cluster and failed *after* the engine started loading, so the step sets a short
per-job path.

**Nothing reaps the cluster but the teardown target.** Forget it and the allocation is
held until someone runs `sky down` by hand.

## Status

Contract-tested (`make test`), never run. The open question is not in this step at all:
the trainer pushes updated student weights to the server over **NCCL**, not HTTP — the
reference launcher's `VLLM_NCCL_COORDINATOR_PORT` — so a separate target means a NCCL
process group spanning two LSF allocations. If that cannot be made to work, the fallback
is `distill-gold`'s existing in-allocation role split, and this step is the wrong answer
rather than a broken one.
