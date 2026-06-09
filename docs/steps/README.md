# Steps

A step is the unit of execution within a target. Each target runs one or more
steps in sequence. A step is defined by a `step.yaml` file that declares how
it launches work on a given environment.

> **Audience:** users authoring builds and anyone creating custom steps.

## How steps are referenced

In a `build.yaml`, each step entry has a `step_uri` field:

```yaml
steps:
  - step_uri: space://steps/bash
    config:
      workload:
        commands:
          - "python train.py --epochs 3"
```

### URI schemes

| Scheme | Example | Description |
|--------|---------|-------------|
| `space://steps/<name>` | `space://steps/hfpull` | Resolves to a step registered in the active space. |
| `file://<path>` | `file://./my-step` | Local directory containing a `step.yaml`. |
| `git+ssh://<repo>#subdirectory=<path>` | `git+ssh://github.com/org/repo.git#subdirectory=steps/custom` | Step from a Git repository. |

If `step_uri` is omitted, the built-in `gbstep` step is used.

## Built-in steps

These steps ship with gbserver in `src/gbserver/builtins/steps/`:

| Step | Description |
|------|-------------|
| `bash` | Execute shell commands directly. The simplest step — no container image required for local environments. |
| `gbstep` | Base step runner. Default when `step_uri` is omitted. Supports `setup_command`, `start_command`, and `cleanup_command`. |
| `hfpull` | Pull a model or dataset from HuggingFace Hub. |
| `hfpush` | Push artifacts to HuggingFace Hub. |
| `s3pull` | Pull files from an S3-compatible object store. |
| `s3push` | Push files to an S3-compatible object store. |
| `cosrclone` | Transfer files using rclone (supports COS, S3, and many backends). |
| `image` | Run a custom container image (BYOI). |

## `step.yaml` structure

A step definition lives in a directory with a `step.yaml`:

```yaml
name: my-step
launchers:
  bash:
    setup_command: "pip install -r requirements.txt"
    start_command: "python main.py"
    cleanup_command: "rm -rf /tmp/work"
  k8s:
    image: my-registry/my-image:latest
    start_command: "python main.py"
monitors:
  - type: log
    pattern: "STEP_COMPLETE"
config:
  retry_enabled: false
  retry_transparently: false
```

### Key fields

| Field | Description |
|-------|-------------|
| `name` | Step identifier. |
| `launchers` | Map of environment type → launch config. The environment selects which launcher to use. |
| `monitors` | How gbserver detects step completion (log patterns, exit codes). |
| `config` | Default configuration (overridable by the build.yaml `step.config`). |

Each launcher type matches an environment backend (bash, docker, k8s, lsf,
skypilot, runpod). A step can support multiple environments by declaring
multiple launchers.

## Step configuration in build.yaml

The `config` block in a step entry is merged with the step's own defaults:

```yaml
steps:
  - step_uri: space://steps/tuning
    config:
      compute_config:
        num_nodes: 1
        num_gpus_per_node: 4
      tuning_config:
        epochs: 3
        learning_rate: 2e-5
```

See the [`build.yaml` reference](../users/build-yaml-reference.md#steps)
for the full set of fields.

## Extending with custom steps

Three approaches for running custom code:

| Approach | When to use |
|----------|-------------|
| [Bring Your Own Step (BYOS)](../users/bring-your-own-step.md) | Your code lives in a Git repo; you provide setup/start commands. |
| [Custom code steps](../users/custom-code-steps.md) | You want inline commands without a separate step definition. |
| [Bring Your Own Image (BYOI)](../users/bring-your-own-image.md) | You have a pre-built container image. |

## See also

- [Templates](../templates/README.md) — reusable build.yaml patterns
- [`build.yaml` reference](../users/build-yaml-reference.md) — full schema
- [`environment.yaml` reference](../operators/environment-yaml-config.md) — environment definitions
