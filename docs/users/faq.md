# FAQ

Common questions about writing and running Granite.Build builds.

> For CLI usage, see [`cli-reference.md`](cli-reference.md). For the full
> build.yaml schema, see [`build-yaml-reference.md`](build-yaml-reference.md).

## Build.yaml authoring

### My template uses `{% raw %}` — what does that mean?

Jinja syntax like `{{question}}` or `{{answer}}` in strings (e.g. a
`data_formatter_template` for fine-tuning) would be expanded by Granite.Build's
template engine. Wrap them in `{% raw %} ... {% endraw %}` to pass through
literally:

```yaml
data_formatter_template: >
  {% raw %}### Question: {{question}}
  ### Answer: {{answer}}
  {% endraw %}
```

### Do steps in the same target run in parallel or sequentially?

**Sequentially.** Multiple steps in one target execute in order. The second
step can access files written by the first step on shared storage.

To run work **in parallel**, put it in separate targets with no dependency
between them. They will execute concurrently.

### How do I create a dependency between targets?

Use a binding in the downstream target's input:

```yaml
targets:
  generate:
    outputs:
      dataset:
        uri: file:workspace/data
    steps: [...]

  train:
    inputs:
      data:
        binding: generate.dataset
    steps: [...]
```

The `train` target waits for `generate` to complete and receives its output.

### What's the difference between `$INPUT_PATH`, `$OUTPUT_PATH`, and bindings?

| Mechanism | When to use |
|-----------|-------------|
| `$INPUT_PATH` | Single-input step; set to the path of the `input_artifact_path` input. |
| `$OUTPUT_PATH` | Single-output step; where to write output (captured by `dir_to_save`). |
| `{{ bindings.<name>.binding.path }}` | Multiple inputs/outputs; reference any named artifact by its binding name. |

`$INPUT_PATH` is equivalent to `{{ bindings.input_artifact_path.binding.path }}`.
Use the binding syntax when you have more than one input or output.

### What does `dir_to_save` do?

It specifies which part of the step's output directory to capture as the
output artifact:

- `"."` — capture the entire `$OUTPUT_PATH` tree.
- `"result.jsonl"` — capture only that specific file.
- `"subdir/"` — capture only that subdirectory.

## Data and artifacts

### What format should my training data be in?

JSONL is preferred. Each line is a JSON object. For chat-style fine-tuning,
use the standard messages format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

For simpler instruction tuning, use `input`/`output` fields:

```json
{"input": "...", "output": "..."}
```

### How do I use a dataset stored as a table vs. a fileset?

- **Table** — structured data (JSONL, Parquet). Accessed via bindings as a
  file path. If using `dataset_mixer_list`, reference with sampling rate:
  ```yaml
  dataset_mixer_list: ["{{ bindings.tuning_data.binding.path }}", "1.0"]
  ```
- **Fileset** — arbitrary files (documents, configs, raw data). Mounted as a
  directory.

## Compute and resources

### My model runs out of memory. How do I request more GPUs?

Increase `num_gpus_per_node` in `compute_config`. Each GPU adds ~80GB VRAM
(for A100). A 2B model needs ~1 GPU; 8B needs ~2; 20B+ needs a full node (8):

```yaml
config:
  compute_config:
    num_gpus_per_node: 8
    num_nodes: 1
```

For models larger than one node, increase `num_nodes` for multi-node training.

### My fine-tuning creates hundreds of checkpoints and is very slow.

Change `save_strategy` to save per-epoch instead of per-step:

```yaml
tuning_config:
  save_strategy: "epoch"
```

Each checkpoint must be uploaded to the artifact store, so fewer checkpoints
means faster builds.

## Logs and debugging

### I'm not seeing all logs.

Use the `--all` flag to fetch complete logs (the command may issue multiple
queries):

```bash
gb build log <build-id> --all
```

For older builds, also specify `--start-date` to query earlier log entries.
Logs older than 14 days may be unavailable.

### The build failed but I can't see why.

Common causes of missing logs:

- **Secret not found** — if a required secret doesn't exist, the pod won't
  start and produces no logs. Verify secrets with `gb secret list`.
- **Out of memory** — the pod was killed before writing logs. Increase
  `compute_config` resources.
- **Log lines too large** — avoid printing very long lines (>2MB) to stdout;
  they can cause logging pipeline issues.

## Naming conventions

### What characters are valid in artifact names?

Lowercase letters, digits, and underscores only. No uppercase, hyphens, or dots.

Valid: `my_model_v2`, `training_data_001`
Invalid: `my-model`, `MyModel`, `model.v2`

### What characters are valid in build names?

Lowercase letters, digits, underscores, and hyphens.

Valid: `my-first-build`, `sft_full_run_01`

## See also

- [`build.yaml` reference](../build-yaml-reference.md) — full schema
- [CLI reference](cli-reference.md) — all `gb` commands
- [Bring Your Own Step](bring-your-own-step.md) — custom code steps
- [Glossary](../glossary.md) — term definitions
