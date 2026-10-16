# Inputs, outputs, and bindings

How data moves between targets in a build, and how a step declares and reads it. This page
puts the build.yaml side and the step.yaml side together in one place. The field-by-field
schemas stay in the [`build.yaml` reference](build-yaml-reference.md#inputs) and the
[steps overview](../steps/README.md#key-fields).

> **Audience:** anyone writing a `build.yaml` that chains targets, or a `step.yaml` that
> reads inputs and produces outputs.

## The model in one picture

```
 build.yaml                                              build.yaml
 target "producer"                                       target "consumer"
 ┌────────────────────────────┐                          ┌──────────────────────────────────────┐
 │ outputs:                   │                          │ inputs:                              │
 │   checkpoint:              │◄──── binding ────────────│   model:                             │
 │     uri: file:workspace/ck │  producer.checkpoint     │     binding: producer.checkpoint     │
 └─────────────┬──────────────┘                          │ steps:                               │
               │                                         │   - step_uri: space://steps/eval     │
   step workload prints                                  └──────────────────┬───────────────────┘
   GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:/ws/ck                        │
               │                                                            ▼
   monitor turns it into an artifact event               step.yaml of "eval" declares input `model`
   → output `checkpoint` now has a value                 and reads {{ bindings.model.binding.path }}
```

Four pieces have to agree:

| Piece | Where | Names that must match |
|-------|-------|-----------------------|
| Target **output** | producer's `build.yaml` `outputs:` | the output name (`checkpoint`) |
| Artifact **marker** | producer step's workload stdout | `GB_ARTIFACT_ID:<output name>` |
| Target **input** | consumer's `build.yaml` `inputs:` | `binding: <producer target>.<output name>` |
| Step **input schema** and template | consumer step's `step.yaml` | the input name (`model`) |

## Terminology

- **Input**: a named artifact a target consumes. It is either **direct** (`uri:`, an artifact
  from outside the build) or **bound** (`binding:`, another target's output). Each input has
  exactly one of the two.
- **Output**: a named artifact a target produces. Other targets bind to it by name.
- **Binding**: the string `<target-name>.<output-name>` on a bound input. It also creates
  the dependency, so the consumer is dispatched only once that output exists.
- **`bindings`**: the template variable a step sees at run time. It maps each of the
  target's input names, direct or bound, to its resolved value.

## build.yaml: declaring inputs and outputs

### Outputs

```yaml
targets:
  producer:
    outputs:
      checkpoint:
        uri: file:workspace/checkpoint            # filesystem output → consumers get a path
      model:
        uri: hf://huggingface.co/my-org/my-model  # pushed to a remote store after the step writes it
      server_url:
        uri: "mem://server_url"                   # opaque value → consumers get a string
```

The URI scheme of an output decides how a consumer receives it:

| Output scheme | What the producer's workload prints | What a consumer reads |
|---------------|-------------------------------------|-----------------------|
| `file://`, `env://`, `hf://`, other filesystem/remote stores | `GB_ARTIFACT_ID:<output> GB_ARTIFACT_PATH:<path>` | `{{ bindings.<input>.binding.path }}` |
| `mem://` | `GB_ARTIFACT_ID:<output> GB_ARTIFACT_STATE:<value>` | `{{ bindings.<input>.binding.state }}` |

A `mem://` value is passed verbatim, with no copy and no path normalisation, so a URL like
`http://host:8000` arrives intact. Don't use `env://` for a value that isn't a path: path
normalisation turns `http://host:8000` into `/http:/host:8000`. A `mem://` output takes no
`type` and no `store_push`. Output URIs can be Jinja-templated (for example
`{{ run_metadata.targetsteprun_id | short_hash }}` or `{{ unique_hash }}`); see
[Jinja templating](build-yaml-reference.md#jinja-templating).

Full output schema (`public`, `store_push`, `event_selectors`):
[`build.yaml` reference — Outputs](build-yaml-reference.md#outputs).

### Inputs

```yaml
targets:
  consumer:
    inputs:
      base-model:
        uri: hf://huggingface.co/ibm-granite/granite-3.3-2b-instruct   # direct
      model:
        binding: producer.checkpoint                                    # bound
      rm_url:
        binding: producer.server_url                                    # bound to a mem:// output
```

| Field | Notes |
|-------|-------|
| `uri` | Direct artifact URI. It is checked to exist when the build is validated. |
| `binding` | `<target-name>.<output-name>`. The target and output must both exist in this build. |
| `wait_for_push` | `true`: don't start until the upstream output's `store_push` has finished, not just until it was produced. |
| `event` | Event-driven trigger (for example `evalresults.success`), paired with the producer output's `event_selectors`. |

Full input schema: [`build.yaml` reference — Inputs](build-yaml-reference.md#inputs).

### Chaining targets

Bindings are the only thing that orders targets. Without a binding between them, targets run
in parallel.

```yaml
granite.build:
  name: tune-and-eval
  targets:
    download:
      outputs:
        model: { uri: file:workspace/model }
      steps:
        - step_uri: space://steps/hfpull
          config:
            hf_uri: hf://huggingface.co/ibm-granite/granite-3.3-2b-instruct

    fine-tune:
      inputs:
        model: { binding: download.model }          # runs after download
      outputs:
        checkpoint: { uri: file:workspace/checkpoint }
      steps:
        - step_uri: space://steps/sft

    evaluate:
      inputs:
        model: { binding: fine-tune.checkpoint }    # runs after fine-tune
      outputs:
        results: { uri: file:workspace/eval }
      steps:
        - step_uri: space://steps/unitxt-eval
```

A step that declares `inputs: {allow_unknown: true}` lets a target bind extra inputs only to
order itself after another target, even when the step never reads them.

## step.yaml: declaring what a step consumes and produces

A step's `inputs` and `outputs` are an I/O **schema**. They don't create artifacts; they
state which target inputs and outputs the step expects, so a mismatch fails at build
validation and not on the cluster.

```yaml
inputs:
  allow_unknown: true            # default true; false rejects inputs not listed below
  required:
    model:
      type: model                # dataset | fileset | model | bucket (required, not checked)
      accept: [uri, binding]     # which kinds of target input are allowed
  optional:
    tokenizer:
      type: model
      accept: [uri, binding]

outputs:
  allow_unknown: true            # default true; false rejects target outputs not listed below
  required:
    checkpoint:
      type: model
  optional:
    server_url:                  # a mem:// value output
      type: dataset
```

| Field | Applies to | Meaning |
|-------|------------|---------|
| `required.<name>` | inputs, outputs | The target **must** declare an input/output with this name. |
| `optional.<name>` | inputs, outputs | Checked only if the target declares it. |
| `type` | inputs, outputs | Must be set to one of `dataset`, `fileset`, `model`, `bucket`, but validation doesn't compare it to anything. |
| `accept` | inputs only | `uri`, `binding`, or both. Always list what you accept: an input whose kind isn't in the list fails validation, so an empty list rejects both kinds. |
| `allow_unknown` | inputs, outputs | Default `true`. With `false`, any target input/output not named under `required` or `optional` is an error. |

A real example is
[`steps/distill/vllm-server/skypilot/step-template.yaml`](../../steps/distill/vllm-server/skypilot/step-template.yaml):
it requires a `model` input and optionally produces `vllm_url` and `cluster_name` as `mem://`
values.

### What validation checks

Before anything runs, each target is checked against the I/O schema of each of its steps:

- A required input or output that the target doesn't declare → **error** (`... is missing`).
- An input whose kind (`uri` vs `binding`) isn't in `accept` → **error**.
- With `allow_unknown: false`, an extra input or output on the target → **error**.
- A target output that none of the target's steps lists in its schema → **warning** (the step
  may simply have no I/O schema).
- `type` is not compared.

The build-level checks also apply: an input with neither or both of `uri`/`binding`, a
`binding` to a target or output that doesn't exist, or a direct `uri` that doesn't exist.

### Reading inputs in a step

Every input, direct or bound, is resolved before the step starts and is available by its
input name. Use these expressions in the launcher's `run`, `setup`, `envs`, or `file_mounts`,
and in `step.config` values in a build.yaml:

| Input | Expression |
|-------|------------|
| A path input (model, dataset, fileset) | `{{ bindings.model.binding.path }}` |
| A `mem://` value input (a server URL, a cluster name) | `{{ bindings.rm_url.binding.state }}` |
| An optional input | `{% if bindings.tokenizer is defined %}…{% endif %}`, or `bindings.tokenizer.binding.path if bindings.tokenizer is defined else ""` |

Rules:

- **Launcher config is rendered strictly.** An unbound optional input raises an error rather
  than rendering as empty, so always guard optional inputs.
- **Required inputs are safe to reference directly**, because validation guarantees the target
  provides them.
- **Declare inputs; don't have recipes pass paths through config keys.** A build.yaml that
  copies `{{ bindings.<name>.binding.path }}` into a config key for the step to read turns a
  misnamed binding into literal text that only fails on the cluster. A declared input that
  isn't bound fails validation up front.
- **Don't read `$LLMB_*_INPUT_<NAME>`.** Only the bash and LSF launchers export it.

More detail, including how to rename an input that has already shipped:
[Steps — Declaring inputs](../steps/README.md#declaring-inputs).

### Producing outputs from a step

A step registers an output by printing a marker on stdout. The step's monitor turns the line
into an artifact event whose `binding_id` is the output name:

```bash
# filesystem output named "checkpoint"
echo "GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:/workspace/checkpoint"

# mem:// value output named "server_url"
echo "GB_ARTIFACT_ID:server_url GB_ARTIFACT_STATE:http://${ADDR}:${PORT}"
```

- The ID after `GB_ARTIFACT_ID:` must equal an output name declared on the target in
  build.yaml.
- The shared `bash`, `skypilot`, `docker`, and `lsf` monitors (`ref: space://monitors/<name>`)
  recognise both `PATH` and `STATE` markers. The k8s and runpod monitors don't recognise
  `STATE` yet. The legacy `LLMB_` prefix is still accepted.
- On SLURM via SkyPilot, each step runs on its own node, so write outputs to paths relative to
  the per-run workdir or a later step won't see them.

Monitor schema, per-environment behaviour, and overriding a shared monitor:
[Monitoring and artifact events](../steps/monitoring-and-artifact-events.md).

## End to end: a value passed between targets

```yaml
targets:
  start-server:
    outputs:
      server_url:
        uri: "mem://server_url"
    steps:
      - step_uri: space://steps/vllm-server      # its workload prints
                                                 # GB_ARTIFACT_ID:server_url GB_ARTIFACT_STATE:http://host:8000
  train:
    inputs:
      rm_url:
        binding: start-server.server_url         # <producer-target>.<output-name>
    steps:
      - step_uri: space://steps/command
        config:
          command_config:
            command: "python train.py --reward-url {{ bindings.rm_url.binding.state }}"
```

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| ``Required input `x` ... is missing`` | The target doesn't declare input `x`, or binds it under a different name. |
| `... does not accept uri` / `does not accept binding` | The step's `accept` list doesn't include that kind of input. |
| `found extra inputs that are not allowed` | The step has `allow_unknown: false` and the target declares an input the step doesn't list. |
| The consumer never starts | The producer never printed `GB_ARTIFACT_ID:<output>`, the ID doesn't match the output name, or its monitor doesn't recognise the marker (for example `STATE` on k8s). |
| A URL arrives as `/http:/host:8000` | The output uses `env://` or a `PATH` marker. Use `mem://` with a `STATE` marker. |
| Literal `{{ bindings... }}` text in a command | The template references an input name that isn't bound. Declare it in the step's schema so validation catches it. |
| An optional input fails to render | It isn't guarded with `bindings.<name> is defined`. |

## See also

- [`build.yaml` reference](build-yaml-reference.md): full input/output schema and Jinja variables
- [Builds overview](README.md#bindings-chaining-targets)
- [Steps overview](../steps/README.md#declaring-inputs): step.yaml structure and declaring inputs
- [Monitoring and artifact events](../steps/monitoring-and-artifact-events.md): how outputs are registered
- [Asset stores](../asset-stores/README.md): URI schemes and how artifacts are pulled and pushed
- [HuggingFace push](hf-push.md): `store_push` for `hf://` outputs
