# `command` (built-in step)

Runs an arbitrary shell command as a step. It is the simplest built-in workload:
give it a command line and it executes it on the launcher node (or inside a
container image, where supported). The command's **exit status becomes the step's
status**, so `command: "exit 1"` hard-fails the target — which makes it useful for
tests and quick experiments as well as real one-shot workloads.

> **Audience:** users authoring builds who want the exact config contract and
> environment support of the `command` step. For step concepts (referencing steps,
> the `step.yaml` structure, custom steps) see the [steps overview](README.md); for
> how a step reports outputs see
> [Monitoring and artifact events](monitoring-and-artifact-events.md).

## Referencing the step

```yaml
steps:
  - step_uri: space://steps/command
    config:
      command_config:
        command: "python train.py --epochs 3"
```

## Config contract (`command_config`)

| Field | Type | Required | Purpose |
|---|---|---|---|
| `command` | string | **required** | Shell command executed on the launcher node. The step exits with this command's status. |
| `image` | string | optional (**Docker / Skypilot only**) | Container image to run the command inside. Empty (default) ⇒ run on the **bare launcher node**. Not supported on the Bash environment. |

`compute_config` is also honored where the environment uses it (see the matrix
below); on Skypilot, resource requests are otherwise supplied via
`config.launcher_config.resources`.

## Environment support

The `command` step is implemented for three environments, with small but important
differences:

| Environment | `image` supported? | Launcher | Monitor | Notes |
|---|---|---|---|---|
| **Bash** | no | `nohup` | [`space://monitors/bash`](../../src/gbserver/builtins/monitors/bash/monitor.yaml) | Always runs on the bare node; the command is exec'd so its exit status is the step's status. |
| **Docker** | yes | `docker` | [`space://monitors/docker`](../../src/gbserver/builtins/monitors/docker/monitor.yaml) | Runs `bash -c '<command>'` inside `image`. Defaults: `num_gpus_per_node: 0`, `total_memory_per_node: 4Gi`. |
| **Skypilot** | yes | `skypilot` | [`space://monitors/skypilot`](../../src/gbserver/builtins/monitors/skypilot/monitor.yaml) | Empty `image` runs on the bare launcher node (no Pyxis SPANK plugin required, e.g. the local Docker SLURM cluster); a non-empty value renders to `docker:<image>`. Resources are left cloud-agnostic. |

## Running inside a container image (the BYOI pattern)

On **Docker** and **Skypilot**, setting `command_config.image` runs the command
inside a pre-built, pullable container image — the "bring your own image" pattern.
Point `image` at any accessible public or private image and invoke its tooling from
`command`:

```yaml
steps:
  - step_uri: space://steps/command
    config:
      command_config:
        image: "nvcr.io/nvidia/pytorch:24.01-py3"
        command: "torchrun --nproc_per_node=4 /workspace/train.py"
```

> If instead you want to **clone code from a git repo into a public image at run
> time**, use the [`byoc` step](../../configurations/assets/environments/skypilot/steps/byoc/README.md),
> which is the maintained exemplar for that workflow on Skypilot.

## Reporting outputs

`command` has no I/O schema of its own; declare `inputs:`/`outputs:` on the
**target** and register produced artifacts from your command by printing marker
lines that the environment's monitor captures:

```
GB_ARTIFACT_ID:<output-id> GB_ARTIFACT_PATH:<abs-path>
```

See [Monitoring and artifact events](monitoring-and-artifact-events.md) for the full
marker convention (path vs. `mem://` state, anchoring, and the shipped monitors).

## Examples

**Bash — a controllable local workload (no image):**

```yaml
steps:
  - step_uri: space://steps/command
    config:
      command_config:
        command: "python -m my_smoke_test && echo done"
```

**Docker — run inside an image with a GPU:**

```yaml
steps:
  - step_uri: space://steps/command
    config:
      command_config:
        image: "python:3.12-slim"
        command: "pip install -e . && python run.py"
      compute_config:
        num_gpus_per_node: 1
```

**Skypilot — run on a cloud cluster inside an image:**

```yaml
steps:
  - step_uri: space://steps/command
    config:
      command_config:
        image: "python:3.12-slim"
        command: "python main.py --out $(pwd)/out"
      launcher_config:
        resources: { cpus: "4+", accelerators: "A100:1" }
```

## Notes and limitations

- **Exit status is the contract.** The step succeeds or fails with the command's
  own exit code; there is no separate success marker to print.
- **`image` is Docker/Skypilot only.** On Bash the command always runs on the bare
  launcher node.
- **No dependency management.** Whatever the chosen `image` (or bare node) provides,
  plus anything the command installs at run time, is all that is available.
