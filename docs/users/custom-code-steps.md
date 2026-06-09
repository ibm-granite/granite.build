# Custom code steps

Simplified features for running custom code without writing Helm charts or
step definitions. These work on Kubernetes and LSF environments.

> **Audience:** users who want to run arbitrary commands as build steps with
> minimal boilerplate.

## Workload commands

Specify commands directly in `config.workload.commands` — no `custom_code_config`
or Helm chart required:

```yaml
steps:
  - step_uri: space://steps/gbstep
    config:
      workload:
        commands:
          - "pip install -e '.[all]'"
          - "python -m my_module --input $INPUT_PATH --output $OUTPUT_PATH"
```

Each command runs sequentially in the step's execution environment.

## Copy step repo into execution environment

By default, the step's source repo is not copied into the pod/container. To
make your repo contents available at runtime:

```yaml
config:
  gb:
    step_contents_in_env: true
```

When enabled, the contents of the directory pointed to by `step_uri` are
copied into the execution environment. Your commands can reference files from
the repo directly without a `git clone`.

## Node selection (Kubernetes)

Select or avoid specific nodes using standard Kubernetes affinity:

```yaml
config:
  k8s:
    affinity:
      nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          nodeSelectorTerms:
            - matchExpressions:
                - key: kubernetes.io/hostname
                  operator: NotIn
                  values:
                    - problem-node-1
                    - problem-node-2
```

Or use `nodeSelector` for simpler cases:

```yaml
config:
  k8s:
    nodeSelector:
      "gpu-type": "a100"
```

## Empty step URI

For workloads that don't need any step definition (e.g. GPU smoke tests,
standalone scripts), omit `step_uri` entirely:

```yaml
steps:
  - config:
      workload:
        commands:
          - "git clone https://github.com/my-org/my-tests.git"
          - "cd my-tests && pip install -e . && pytest"
```

The base default step configuration is used automatically.

## Generate files from config

Create files at runtime from values in the build config — useful for
configuration files that your code expects on disk:

```yaml
steps:
  - step_uri: space://steps/gbstep
    config:
      gb:
        files_to_create:
          - training_config.yaml: training_config
          - data_config.yaml: data_config
      training_config:
        learning_rate: 2e-5
        epochs: 3
        batch_size: 4
      data_config:
        dataset_path: "{{ bindings.data.binding.path }}"
        format: jsonl
```

This creates `training_config.yaml` and `data_config.yaml` in the working
directory with the YAML-serialized contents of the corresponding config keys.

## Secrets as environment variables

Reference secrets from the space's secret manager as environment variables:

```yaml
config:
  k8s:
    secrets:
      secret_names_to_use_as_env_variable:
        - env_name: HF_TOKEN
          secret_name: huggingface-token
        - env_name: AWS_SECRET_ACCESS_KEY
          secret_name: aws-secret-key
```

For image pull secrets (when using a private container registry):

```yaml
config:
  k8s:
    secrets:
      secret_names_to_use_as_pull_secret:
        - my-registry-secret
```

## Working directory

By default, commands run in the container image's working directory. Override
with:

```yaml
config:
  workload:
    cwd: $LLMB_TARGETSTEPRUN_ASSET_DIR
```

`$LLMB_TARGETSTEPRUN_ASSET_DIR` points to the directory where step contents
are copied (when `step_contents_in_env: true`). You can also set an absolute
path.

## Optional step.yaml

If the directory pointed to by a non-empty `step_uri` does not contain a
`step.yaml`, the base default step configuration is used. This means you can
point `step_uri` at a Git repo containing just your code — no step definition
file needed.

## See also

- [Bring Your Own Step](bring-your-own-step.md) — the full BYOS pattern with `custom_code_config`
- [Bring Your Own Image](bring-your-own-image.md) — pre-built container images
- [Steps overview](../steps/README.md) — step.yaml structure and built-in steps
