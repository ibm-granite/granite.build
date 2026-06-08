# Bring Your Own Image (BYOI)

Run a pre-built container image as a Granite.Build step. Use this when your
workload is packaged as a Docker image — either one you built yourself or a
third-party image from a registry.

> **Audience:** users with existing container images who want to integrate them
> into build pipelines.

## Prerequisites

1. **Granite.Build CLI** installed — see [Getting started](../getting-started.md).
2. **Container image** pushed to a registry accessible from your execution
   environment (Docker Hub, GHCR, a private registry, etc.).
3. **Image pull secret** (optional) — only needed if the registry requires
   authentication and is not already configured in your environment.

## Upload an image pull secret (optional)

If your image is in a private registry, create a secret containing Docker
credentials:

```json
{
  "auths": {
    "ghcr.io": {
      "username": "my-user",
      "password": "ghp_...",
      "auth": "<base64 of user:password>"
    }
  }
}
```

Upload it:

```bash
gb secret create --from-file registry-secret.json --personal my-registry-secret
```

Delete when no longer needed:

```bash
gb secret delete --personal my-registry-secret
```

## Build.yaml for BYOI

A BYOI build uses `config.k8s.image` to specify the container image and
`config.workload.commands` or `config.k8s.env` to control execution:

```yaml
granite.build:
  name: byoi-example
  targets:
    run_image:
      environment_uri: space://environments/kubernetes
      inputs:
        input_data:
          uri: file:workspace/input
          binding:
            path: input_data
      outputs:
        results:
          uri: file:workspace/output
      steps:
        - step_uri: space://steps/image
          config:
            k8s:
              image: ghcr.io/my-org/my-image:latest
              secrets:
                secret_names_to_use_as_pull_secret:
                  - my-registry-secret
              env:
                APP_COMMAND:
                  value: "python /app/run.py --input {{ bindings.input_data.binding.path }} --output /workspace/output"
                ECHO_COMMAND:
                  value: "echo LLMB_ARTIFACT_ID:results LLMB_ARTIFACT_PATH:/workspace/output"
            compute_config:
              num_gpus_per_node: 1
              num_nodes: 1
```

### Key configuration fields

| Field | Description |
|-------|-------------|
| `config.k8s.image` | Full image reference (registry/org/image:tag). |
| `config.k8s.secrets.secret_names_to_use_as_pull_secret` | List of secret names for pulling the image. |
| `config.k8s.env.APP_COMMAND` | Command to execute inside the container. Can reference bindings. |
| `config.k8s.env.ECHO_COMMAND` | Log line signalling completion. Triggers artifact upload. Must include `LLMB_ARTIFACT_ID:<output_name>` and `LLMB_ARTIFACT_PATH:<path>`. |

### Custom image (editable code)

If you built the image yourself and want to pass dynamic parameters:

```yaml
config:
  k8s:
    image: my-registry/my-tool:v1.2
    env:
      APP_COMMAND:
        value: "python /app/process.py --epochs 5 --data {{ bindings.training_data.binding.path }}"
```

### Third-party image (immutable)

For images you cannot modify, use `APP_COMMAND` to invoke the image's
entrypoint with your arguments:

```yaml
config:
  k8s:
    image: nvcr.io/nvidia/pytorch:24.01-py3
    env:
      APP_COMMAND:
        value: "torchrun --nproc_per_node=4 /workspace/train.py"
```

## Running

```bash
gb build start
gb build status <build-id>
gb build log <build-id> --all
```

## Debugging tips

- **Keep log output under 1000 lines** when running on Kubernetes — excessive
  logging can exhaust ephemeral storage.
- **Verify the image exists** and is pullable from a node in your cluster
  before submitting.
- **Check secrets** — if the pull secret is wrong or missing, the pod won't
  start and you'll see no logs.

## See also

- [Bring Your Own Step](bring-your-own-step.md) — custom code from a Git repo
- [Custom code steps](custom-code-steps.md) — inline commands without images
- [Steps overview](../steps/README.md) — all step types
- [`build.yaml` reference](build-yaml-reference.md) — full schema
