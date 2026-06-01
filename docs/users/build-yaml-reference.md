# `build.yaml` reference

Complete schema for a Granite.Build build file. The source of truth for this
reference is [`src/gbserver/types/buildconfig.py`](../../src/gbserver/types/buildconfig.py)
— the Pydantic models there define what is and isn't valid. This page is the
human-readable companion.

> **Audience:** users authoring `build.yaml` files. If you've never run a
> build, start with [getting-started.md](../getting-started.md) first.

## Top-level shape

```yaml
llm.build:                  # or "granite.build" — both keys are accepted
  version: "0.0.1"          # optional; defaults to current schema version
  name: my-build            # optional human-readable label
  retries:                  # optional; defaults to no retries
    max_retries: 0
    target_reuse_enabled: true
  targets:                  # required; at least one target
    <target-name>:
      environment_uri: ...
      inputs: { ... }
      outputs: { ... }
      steps: [ ... ]
```

| Field      | Type                  | Required | Default                        | Notes |
|------------|-----------------------|----------|--------------------------------|-------|
| `version`  | string                | no       | `0.0.1`                        | Schema version. |
| `name`     | string                | no       | `""`                           | Human-readable name. |
| `retries`  | object                | no       | `{}` (no retries)              | See [Retries](#retries). |
| `targets`  | map of target name → target | **yes** | —                        | At least one target required. |

The top-level key may be `llm.build` (current) or `granite.build` (legacy).

## Targets

A target is a logical stage of the pipeline — "download", "fine-tune",
"evaluate". Each target runs on one environment and produces zero or more
output artifacts.

```yaml
targets:
  download:
    environment_uri: space://environments/docker
    inputs:
      model:
        uri: hf://huggingface.co/ibm-granite/granite-3.3-2b-instruct
    outputs:
      model:
        uri: file:workspace/model
    steps:
      - step_uri: space://steps/somestep
```

| Field            | Type                       | Required | Default | Notes |
|------------------|----------------------------|----------|---------|-------|
| `environment_uri`| string                     | **yes**  | —       | URI of the environment definition. Supports Jinja templating. |
| `inputs`         | map of name → input        | no       | `{}`    | See [Inputs](#inputs). |
| `outputs`        | map of name → output       | no       | `{}`    | See [Outputs](#outputs). |
| `steps`          | list of step               | **yes**  | —       | At least one step. See [Steps](#steps). |

`environment_uri` is the most common place to reference the space's
environment definitions (`space://environments/<name>`); see
[`operators/environment-yaml-config.md`](../operators/environment-yaml-config.md)
for what those definitions can contain.

## Inputs

An input names an artifact the target depends on. Each input is either
**direct** (carries a `uri`) or **bound** (carries a `binding`) — exactly one,
not both.

```yaml
inputs:
  base-model:
    uri: hf://huggingface.co/ibm-granite/granite-3.3-2b-instruct      # direct: from outside
  training-data:
    binding: synthdata.dataset                          # bound: another target's output
```

| Field            | Type    | Required           | Default | Notes |
|------------------|---------|--------------------|---------|-------|
| `uri`            | string  | one of uri/binding | —       | Direct artifact URI. Schemes: `hf://`, `file://`, `git://`, `cos://`, `lh://`, `s3://`. |
| `binding`        | string  | one of uri/binding | —       | `<target-name>.<output-name>` — points at another target's output in this build. |
| `wait_for_push`  | bool    | no                 | `false` | If `true`, wait for the upstream's push to complete before starting. |
| `event`          | string  | no                 | —       | Event-driven trigger (e.g. `evalresults.success`); fires this target only when an upstream emits the event. |
| `metadata`       | object  | no                 | —       | Runtime-populated; usually omitted in user YAML. |

A bound input must reference an existing target name and an output name on
that target — `BuildConfig.my_validate()` will reject the build otherwise.

## Outputs

An output names an artifact the target produces. Other targets bind to it by
name.

```yaml
outputs:
  checkpoint:
    uri: file:workspace/checkpoint
  model:
    uri: hf://huggingface.co/my-org/my-model
    store_push:
      mode: hf_push 
      config:
        hf:
          private: false
```

| Field             | Type    | Required | Default | Notes |
|-------------------|---------|----------|---------|-------|
| `uri`             | string  | no       | —       | Output URI. Supports Jinja templating, e.g. `hf://huggingface.co/datasets/.../{{ run_metadata.targetsteprun_id \| short_hash }}`. |
| `store_push`      | object  | no       | —       | Push to a remote store after the step writes the artifact. See [`hf-push.md`](hf-push.md) for the HF case. |
| `event_selectors` | list    | no       | `[]`    | Event-payload matchers used by downstream `inputs.event` triggers. |

### `store_push`

| Field    | Type   | Required | Notes |
|----------|--------|----------|-------|
| `mode`   | string | yes      | `hfstore`, `lhstore`, `cosstore`, etc. |
| `config` | object | no       | Mode-specific (e.g. `hf.private`, `hf.resource_group`). |

For HuggingFace specifically, see [`hf-push.md`](hf-push.md) — it documents
the full URI format, resource-group resolution, and the relationship between
`store_push` here and the environment-level `store_push` block.

### `event_selectors`

```yaml
outputs:
  evalresults:
    uri: file:workspace/eval.json
    event_selectors:
      - field_name: status
        field_value: success
```

| Field               | Type   | Required               | Notes |
|---------------------|--------|------------------------|-------|
| `field_name`        | string | yes                    | Event-payload field. |
| `field_value`       | string | one of value/value_regex | Exact-match value. |
| `field_value_regex` | string | one of value/value_regex | Regex-match value. |

## Steps

Each target runs one or more steps in order. A step is the unit of work the
environment dispatches.

```yaml
steps:
  - step_uri: space://steps/sft
    config:
      compute_config:
        num_nodes: 1
        num_gpus_per_node: 2
        total_memory_per_node: "32Gi"
      tuning-config:
        epochs: 3
    retry_enabled: true
    retry_transparently: true
```

| Field                 | Type   | Required | Default | Notes |
|-----------------------|--------|----------|---------|-------|
| `step_uri`            | string | no       | built-in `gbstep` | Empty/missing → defaults to the bundled `gbstep` runner. |
| `launcher`            | string | no       | —       | Optional launcher name (environment-specific). |
| `config`              | object | no       | `{}`    | Free-form step config, including `compute_config` and step-specific blocks. |
| `config_dir`          | string | no       | —       | Directory of additional config files. |
| `retry_enabled`       | bool   | no       | step-default | Override step.yaml's `retry_enabled`. See [step-retry-configuration](../features/step-retry-configuration.md). |
| `retry_transparently` | bool   | no       | step-default | If true, retried step output replaces the failed run's metadata; if false, both runs remain visible. |

### `compute_config`

`compute_config` lives inside `step.config` and is read by the environment to
size the compute it allocates. It's a free-form dict — fields depend on the
environment, not on a fixed schema.

| Field                    | Used by                  | Notes |
|--------------------------|--------------------------|-------|
| `num_nodes`              | k8s, lsf, skypilot, runpod | Number of nodes. |
| `num_gpus_per_node`      | k8s, lsf, skypilot, runpod | GPUs per node. |
| `num_cpus_per_node`      | docker, k8s, skypilot      | CPU cores per node. |
| `total_memory_per_node`  | docker, k8s, skypilot      | Memory per node, e.g. `"4Gi"`, `"32Gi"`. |

Per-environment specifics (k8s `affinity`, skypilot `cluster_name`, lsf
`queue`, etc.) are documented in
[`operators/environment-yaml-config.md`](../operators/environment-yaml-config.md).

### Kubernetes `env` quoting

If `step.config` contains a `k8s.env` block, all numeric values **must** be
strings — `BuildTargetStepConfig.validate_k8s_env_section` will reject the
build otherwise. Kubernetes itself rejects unquoted ints.

```yaml
config:
  k8s:
    env:
      NCCL_TIMEOUT:
        value: "10800000"   # quoted — required
```

## Retries

Top-level `retries` controls *build-level* retry — restarting a failed build,
optionally reusing the artifacts of targets that already succeeded.

```yaml
retries:
  max_retries: 1
  target_reuse_enabled: true
```

| Field                  | Type | Default | Notes |
|------------------------|------|---------|-------|
| `max_retries`          | int  | `0`     | Number of automatic retries on failure. |
| `target_reuse_enabled` | bool | `true`  | If true, successful targets are skipped on retry and their outputs are reused. |

For the full mechanism — what counts as a successful target, how artifacts
are matched on retry, and the relationship to *step-level* retry — see:

- [Retry overview](../features/retry.md)
- [Build retry](../features/build-retry.md)
- [Target reuse](../features/target-reuse.md)
- [Step retry configuration](../features/step-retry-configuration.md)

## Jinja templating

Template expressions are evaluated in:

- `environment_uri`
- input and output `uri` values
- `step.config` values

Available variables:

| Variable          | What it is |
|-------------------|------------|
| `run_metadata.*`  | Per-run identifiers (`targetsteprun_id`, `build_id`, etc.). |
| `space.variables.*` | Variables defined in the space's `space.yaml`. |
| `binding.*`       | Resolved upstream artifact info for a bound input. |
| `bindings.*`      | Map of all bound inputs (when more than one is needed). |

Filters provided in
[`src/gbcommon/utils/template.py`](../../src/gbcommon/utils/template.py):

| Filter           | Purpose |
|------------------|---------|
| `short_hash`     | Short alphanumeric hash of a string — useful in URIs. |
| `path_basename`  | Basename of a path. |
| `to_yaml`        | Render a value as YAML. |
| `json_dumps`     | Render a value as JSON. |
| `indent(N)`      | Indent text by N spaces. |

Example:

```yaml
environment_uri: space://environments/{{ space.variables.DEFAULT_CPU_ENVIRONMENT }}
outputs:
  data:
    uri: hf://huggingface.co/datasets/my-org/synth_data_{{ run_metadata.targetsteprun_id | short_hash }}
```

## Worked examples

These samples in the repo exercise the schema end to end:

- [`samples/standalone/standalone-quickstart/build.yaml`](../../samples/standalone/standalone-quickstart/) — single target, multiple environment options, basic `compute_config`.
- [`samples/templates/local_multi_stage/build.yaml`](../../samples/templates/local_multi_stage/) — four targets chained by binding, event-driven trigger.
- [`samples/tests/digit-tuning-fmeval/build.yaml`](../../samples/tests/digit-tuning-fmeval/) — multi-GPU pipeline with templated Lakehouse URIs.
- [`test-data/integration/ibm/buildrunner/k8s/retry/cpu/build.yaml`](../../test-data/integration/ibm/buildrunner/k8s/retry/cpu/) — top-level `retries` block with `max_retries`.

## Validation errors

Common rejections from `BuildConfig.my_validate()`:

- An input has neither `uri` nor `binding`, or has both.
- A `binding` references a target or output that doesn't exist.
- A target has zero steps.
- A `k8s.env` value is an unquoted integer.

Run `gb build start -f build.yaml` and read the error — it'll point at the
field.
