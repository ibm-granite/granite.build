# data-designer (SkyPilot)

Generates a synthetic dataset with [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)
on a SkyPilot cluster. You give the step a Data Designer config (columns, samplers, LLM
columns, judges, validators) and the model endpoints it calls. The step installs the
library on the node, runs the generation and registers the dataset as an output artifact.

Data Designer only makes HTTP calls to an OpenAI-compatible (or Anthropic) endpoint and
never loads a model, so the step is **CPU-only**. The GPU belongs to whatever serves the
model: a hosted API, a model gateway, or a local server. It runs on the bare launcher node by default, so
it needs no container runtime and works on clusters without the Pyxis SPANK plugin.

> **Developing or testing this step?** See `steps/data/data-designer/skypilot/README.md`
> in the granite.build repository for how the step is generated, tested, and published.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/data/data-designer` URI:

```yaml
steps:
  - step_uri: space://steps/data/data-designer
```

## Config contract (`data_designer_config`)

All fields live under the step's `config.data_designer_config`.

| Field | Type | Required | Purpose |
|---|---|---|---|
| `config` | map | **yes** | The Data Designer config, inline, in the same form `data-designer validate/create` reads from a `.yaml` file. Use either the shorthand (`columns`, `model_configs`, `seed_config`, … at the top level) or the full form under `data_designer:`. See [Prompts and Jinja](#prompts-and-jinja). |
| `model_providers` | list | **yes**, unless `local_model` is set | The endpoints the config's `model_configs` refer to by `provider:` name. See [Model providers](#model-providers). |
| `local_model` | map | no | Serve a small model on the node with Ollama, so the build needs no API key or endpoint. See [A model on the node](#a-model-on-the-node). |
| `output` | string | **yes** | Name of a declared target `outputs:` entry. Used as the registered artifact's ID. |
| `num_records` | int | no | Rows to generate. Default `10`. |
| `output_path` | string | no | Directory the dataset is written to. Default `./output` in the step's working directory. Set it explicitly, on the shared filesystem, when the output's `uri` names a path or a downstream target reads it. The rules are the same as the dpk step's. |
| `resume` | string | no | `if_possible` (default), `always` or `never`. See [Resume](#resume). |
| `run_config` | map | no | Data Designer `RunConfig` fields, passed through: `buffer_size` (rows per checkpoint, default 1000), `max_concurrent_row_groups` (3), `shutdown_error_rate` (0.5), `disable_early_shutdown`, … |
| `export_format` | string | no | `jsonl`, `csv` or `parquet`: also write the whole dataset as one file, `<output_path>/<name>.<format>`. Default empty, which writes only the parquet batches. |
| `min_records` | int | no | Fail the target when fewer rows come back. Data Designer drops rows whose cells failed. Default `null`, meaning at least one row. |
| `seed_glob` | string | no | Appended to the staged `seed` path, e.g. `"*.parquet"`, when a `local` seed source should read a directory. |
| `dd_version` | string | no | data-designer release to install. Default `0.9.5`. Ignored when `dd_image` is set. |
| `dd_image` | string | no | A container image that already provides data-designer. Setting it skips the install. Default `""`, the bare launcher node. No public image is published for the open-source library. |
| `pip_index_url` | string | no | Index for the install. Default `https://pypi.org/simple`. |

## Model providers

Each entry is a Data Designer `ModelProvider` (`name`, `endpoint`, `provider_type`
(`openai` or `anthropic`, default `openai`), `extra_body`, `extra_headers`). On top of
those, the step adds two keys that read a credential from the node's environment at run
time. This keeps keys out of the build.yaml, the persisted step config, and any file on
the node:

| Key | Becomes |
|---|---|
| `api_key_env: NAME` | `api_key: $NAME` (bearer token) |
| `extra_headers_from_env: {Header: NAME}` | `extra_headers: {Header: $NAME}` |

A missing variable fails the target and names the variable. Avoid a plain `api_key`:
Data Designer treats it as an environment variable name first and falls back to the
literal string.

**Gateways that take the key in a custom header** rather than as a bearer token: name
the header and the variable that holds the key. Some API gateways also give each model
its own endpoint URL. Replace the placeholders with your gateway's values:

```yaml
model_providers:
  - name: gateway
    endpoint: https://<gateway-host>/<model-path>/v1
    extra_headers_from_env:
      <Header-Name>: GATEWAY_API_KEY    # header -> env var holding the key
```

**OpenAI-style bearer endpoints** (build.nvidia.com, OpenAI, vLLM with `--api-key`):

```yaml
model_providers:
  - name: nvidia
    endpoint: https://integrate.api.nvidia.com/v1
    api_key_env: NVIDIA_API_KEY
```

### A model on the node

For tests, demos and smoke runs, the step can serve a model on the node itself, with no
API key and no model endpoint:

```yaml
data_designer_config:
  local_model:
    runtime: ollama
    model: granite4:350m      # any Ollama model name
  config:
    model_configs:
      - alias: text
        model: granite4:350m  # same name
        provider: local       # local_model.name, default "local"
```

During setup the step installs Ollama and pulls the model; during the run it serves the
model on `127.0.0.1` for as long as the generation takes, then stops it. It adds the
provider itself, so `model_providers` can be empty.

| `local_model` field | Default | Purpose |
|---|---|---|
| `runtime` | `""` (off) | `ollama` to serve a model on the node. |
| `model` | — | Ollama model name. Required when `runtime` is set. |
| `version` | `0.40.7` | Ollama release. |
| `port` | `11434` | Local port the server listens on. |
| `name` | `local` | Provider name to reference from `model_configs`. |

Things to know:

- **CPU only.** The step requests no GPU, so stick to small models. `granite4:350m`
  generated 4 short rows in about 20s on 2 CPUs. It writes fluent text whose facts are
  often wrong.
- **Downloads, cached.** Ollama's Linux bundle is about 1.5 GB, almost all of it CUDA
  libraries. The step streams it once and keeps only the CPU parts (about 50 MB). The
  model is extra (`granite4:350m` is about 0.7 GB). Both are cached under the
  environment's shared workdir (`<shared_workdir>/.data-designer-cache`), so they
  download once per cluster. Measured on the local SLURM cluster: 4m39s cold, 3m04s warm.
- **Size the node for the model.** It runs alongside Data Designer, within the target's
  `compute_config`.

### Credentials

The variables named above reach the node as space secrets that the build declares. The
step injects only the secrets you list:

```yaml
config:
  skypilot:
    secrets:
      secret_names_to_use_as_env_variable:
        - env_name: GATEWAY_API_KEY     # what the provider reads
          secret_name: GATEWAY_API_KEY  # the space secret
```

In standalone mode, store the secret once per space with
`gb secret create GATEWAY_API_KEY --value <key> --space <space-name>`.

## Prompts and Jinja

The step config is rendered by the build before the step sees it, and Data Designer
prompts are Jinja too (`{{ topic }}`). **Wrap each prompt in `{% raw %}…{% endraw %}`**,
or wrap the whole `config` value. A bare `{{ name }}` happens to survive, because the
build preserves undefined names. A filter, a `{% if %}` or a name that collides with a
build variable (`config`, `bindings`, `run_metadata`) would be evaluated by the build and
silently changed. Build-time values you *do* want substituted, such as `$${PARAM}` build
parameters, go outside the raw block.

## Resume

Data Designer writes the dataset in row groups of `run_config.buffer_size` rows and
checkpoints each finished group to `parquet-files/`, together with a fingerprint of the
config in `metadata.json`. A later run on the same `output_path` reads that back:

| `resume` | Same config | Changed config |
|---|---|---|
| `if_possible` (default) | Continues after the last finished row group. Finished groups are kept as they are, and a finished dataset is not regenerated. | The step clears `output_path` and starts fresh in place. |
| `always` | As above. | Fails: `Cannot resume: the current config … does not match`. |
| `never` | Refuses a non-empty `output_path`. | Refuses a non-empty `output_path`. |

The fingerprint covers what changes the rows: columns, prompts, models, inference
parameters. It ignores `max_parallel_requests`, `timeout`, `skip_health_check` and
profilers, so you can retune concurrency and still resume.

Two things decide whether resume helps in practice:

- **`output_path` must survive between attempts.** The default `./output` is in the
  per-run workdir, which teardown removes, so a *new* build starts from nothing. Use a
  fixed path on the shared filesystem, e.g. `/shared/datasets/<name>`, to resume across
  builds or after a failed target.
- **Checkpoints are only as fine as `buffer_size`.** With the default of 1000, a run of
  under 1000 rows has a single checkpoint, at the end. For long runs against a
  rate-limited API, set something like `run_config: {buffer_size: 100}`.

Raising `num_records` on a finished dataset extends it. Lowering it below what is
already on disk is an error.

## Multimodal

Data Designer is multimodal on input and on output. Everything below lives inside
`config`; the step passes it through unchanged.

**Input.** Each LLM column (`llm-text`, `llm-code`, `llm-structured`, `llm-judge`)
accepts `multi_modal_context`, a list of entries naming the column that holds the media:

```yaml
- name: caption
  column_type: llm-text
  model_alias: vision          # a model_config pointing at a vision model
  prompt: "Describe the chart in one sentence."
  multi_modal_context:
    - modality: image          # image | audio | video
      column_name: chart_url   # column holding the media
      data_type: url           # url | base64; omit to auto-detect
```

The media usually comes from a seed dataset column: URLs, base64, or (for images) file
paths. The model must accept that modality: a vision model such as
`ibm-granite/granite-4.0-3b-vision` takes images, a text-only model does not.

**Output.** `column_type: image` generates images with an image model, and an
`image_context` on it edits existing images. Images are saved under the output
directory next to the parquet files.

Only text generation has been run through this step so far. The multimodal fields are
Data Designer's own (0.9.5) and need no step change, but they are untested here.

## Inputs

| Input | Required | Type | Purpose |
|---|---|---|---|
| `seed` | no | dataset | A seed dataset. Its staged path replaces `config.seed_config.source.path`. The config must have a `seed_config` with a path-based source (`local`, `directory`, `file_contents`, `agent_rollout`). |

A sampler-only or LLM-only config needs no input. Extra inputs are accepted
(`allow_unknown: true`), so a recipe can bind others purely to order targets.

## Outputs

The step writes Data Designer's artifact directory to `output_path` and registers it
under `output`. The directory holds:

- `parquet-files/batch_*.parquet`: the dataset;
- `metadata.json` and `builder_config.json`: the run record and the exact config;
- `<name>.<export_format>`, when `export_format` is set.

It also records three values as step metadata, visible in build lineage:

- `data_designer_version`;
- `records_generated`: rows in the final dataset;
- `resumed_records`: how many of those were reused from an earlier run's checkpoints
  rather than generated by this one (`0` for a fresh run).

## Example

```yaml
targets:
  generate:
    environment_uri: space://environments/skypilot/slurm
    outputs:
      questions:
        uri: env:///shared/dd/questions
        type: dataset
    steps:
      - step_uri: space://steps/data/data-designer
        config:
          skypilot:
            secrets:
              secret_names_to_use_as_env_variable:
                - {env_name: GATEWAY_API_KEY, secret_name: GATEWAY_API_KEY}
          data_designer_config:
            output: questions
            output_path: /shared/dd/questions
            num_records: 100
            export_format: jsonl
            model_providers:
              - name: gateway
                endpoint: https://<gateway-host>/<model-path>/v1
                extra_headers_from_env: {<Header-Name>: GATEWAY_API_KEY}
            config:
              model_configs:
                - alias: text
                  model: <model-id>        # e.g. ibm-granite/granite-4.1-8b
                  provider: gateway
                  inference_parameters: {temperature: 0.7, max_tokens: 256, max_parallel_requests: 4}
              columns:
                - name: topic
                  column_type: sampler
                  sampler_type: category
                  params: {values: [astronomy, cooking, geology]}
                - name: question
                  column_type: llm-text
                  model_alias: text
                  prompt: "{% raw %}Write one trivia question about {{ topic }}.{% endraw %}"
          compute_config:
            num_cpus_per_node: 2
            total_memory_per_node: 4Gi
```

## Cancelling on SLURM

With the currently pinned SkyPilot, cancelling a build on a `skypilot/slurm` environment
marks it cancelled but does **not** stop the generation on the node. The task keeps
running, and keeps calling the model endpoint, until it finishes. Tracked in
https://github.com/ibm-granite/granite.build/issues/533; until it's fixed, also
`scancel` the step's SLURM job if you need it to stop. Anything it finished stays
checkpointed, so a later build with `resume: if_possible` and the same `output_path`
picks up from there.

## Telemetry

Data Designer sends usage telemetry to NVIDIA by default. The step always sets
`NEMO_TELEMETRY_ENABLED=false`.
