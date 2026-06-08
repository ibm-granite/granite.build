# Bring Your Own Step (BYOS)

Run custom code as a Granite.Build step. Your code lives in a Git repository;
you provide setup and start commands, and gbserver handles execution, artifact
binding, and output capture.

> **Audience:** users who have custom training, evaluation, or data processing
> code and want to run it as part of a build pipeline.

## The 6 elements

A BYOS step is defined by these pieces in the `build.yaml`:

| # | Element | Description |
|---|---------|-------------|
| 1 | **Source code repo** | A Git repository containing your code. |
| 2 | **`setup_command`** | Install dependencies (runs in the cloned repo directory). |
| 3 | **`start_command`** | Execute your code. |
| 4 | **Input artifact URI** | Artifact(s) mounted before step execution. |
| 5 | **Output artifact URI** | Artifact(s) captured after step completion. |
| 6 | **`cleanup_command`** | Optional post-run cleanup. |

## Example build.yaml

```yaml
granite.build:
  name: byos-custom-eval
  targets:
    evaluate:
      environment_uri: space://environments/docker
      inputs:
        model:
          uri: hf://huggingface.co/ibm-granite/granite-3.3-2b-instruct
          binding:
            path: model
      outputs:
        results:
          uri: file:workspace/eval_results
      steps:
        - step_uri: space://steps/gbstep
          config:
            custom_code_config:
              github_url: "github.com/my-org/my-custom-eval"
              setup_command: "pip install -r requirements.txt"
              start_command: "python evaluate.py --model $INPUT_PATH --output $OUTPUT_PATH"
              dir_to_save: "."
```

## How `$INPUT_PATH` and `$OUTPUT_PATH` work

Two environment variables are set automatically at runtime:

| Variable | Value |
|----------|-------|
| `$INPUT_PATH` | Path to the artifact declared as `input_artifact_path` in your inputs. |
| `$OUTPUT_PATH` | Path where you should write output. Everything at this path (or the subdirectory specified by `dir_to_save`) is captured as the output artifact. |

These are convenience shortcuts for the common single-input, single-output
case. For multiple inputs/outputs, use the full binding syntax:

```yaml
start_command: >
  python process.py
  --input {{ bindings.training_data.binding.path }}
  --model {{ bindings.base_model.binding.path }}
  --output {{ bindings.results.binding.path }}
```

## `dir_to_save`

Controls what gets captured from `$OUTPUT_PATH`:

- `"."` — save the entire output directory tree.
- `"result.jsonl"` — save only that file.
- A relative subdirectory — save just that subtree.

## Running and monitoring

```bash
# Submit the build
gb build start

# Check status
gb build status <build-id>

# Tail logs
gb build log <build-id> --tail 1000
```

## Multiple steps in a target

Steps within the same target run **sequentially**. If your second step needs
output from the first, they can share the filesystem within the same target.

To run steps **in parallel**, put them in separate targets with no dependency
between them.

To create a **dependency chain** across targets, use bindings:

```yaml
targets:
  generate_data:
    outputs:
      dataset:
        uri: file:workspace/synthetic_data
    steps:
      - ...

  train_model:
    inputs:
      training_data:
        binding: generate_data.dataset
    steps:
      - ...
```

## See also

- [Custom code steps](custom-code-steps.md) — simplified inline commands
- [Bring Your Own Image](bring-your-own-image.md) — pre-built container images
- [`build.yaml` reference](build-yaml-reference.md) — full schema
- [Steps overview](../steps/README.md) — built-in steps and step.yaml structure
