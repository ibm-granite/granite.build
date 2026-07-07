# Builds

> **Audience:** anyone authoring a `build.yaml`. If you've never run a build, start with the
> [getting-started guide](../getting-started.md); for the complete field-by-field schema see the
> [`build.yaml` reference](build-yaml-reference.md).

A **build** is one execution of a `build.yaml`. The file declares a set of **targets** wired together
into a dependency graph; running it produces **artifacts** (models, datasets, filesets). This page
explains the pieces — targets, steps, environments, and artifacts — and how they fit together, then
points at the detailed references.

## What is a build definition?

A `build.yaml` has a small top-level shape: a `name`, optional `retries`, and a map of `targets`.
Everything else hangs off the targets.

```yaml
granite.build:                 # alias: llm.build — both keys are accepted
  name: tune-and-eval
  targets:
    <target-name>:
      environment_uri: ...     # where this target runs
      inputs: { ... }          # artifacts it consumes
      outputs: { ... }         # artifacts it produces
      steps: [ ... ]           # what it executes
```

- **Build** — an execution of the `build.yaml`, identified by a build id, producing artifacts. See the
  [glossary](../glossary.md) for formal term definitions.
- **Target** — a named unit of work (a pipeline stage like `download`, `fine-tune`, `evaluate`). Each
  target names an environment, declares inputs and outputs, and runs one or more steps.
- **Step** — the unit of execution inside a target. A target runs its steps in sequence.
- **Environment** — the compute backend a target runs on (bash, Docker, Kubernetes, LSF, RunPod,
  SkyPilot).
- **Artifact** — an input or output of a target, referenced by URI.
- **Binding** — a reference connecting one target's input to another target's output, forming the
  build graph.

## Targets and the build graph

Targets are the nodes of a build. A target declares:

| Part | Purpose |
|------|---------|
| `environment_uri` | Which environment runs the target (e.g. `space://environments/docker`). |
| `inputs` | Artifacts the target consumes — either a direct `uri` or a `binding` to another target's output. |
| `outputs` | Artifacts the target produces — an output `uri` (often templated), with optional `store_push`. |
| `steps` | The steps to run, in order. |

Targets form a **DAG**: a target that binds one of its inputs to another target's output depends on
that target, and gbserver runs them in dependency order. When an upstream output is produced, the
downstream targets that reference it are dispatched automatically. See
[target reuse](target-reuse.md) for how unchanged targets are skipped across builds and
retries.

## Steps: what a target executes

Each target runs a sequence of **steps**. A step is referenced by `step_uri` (e.g.
`space://steps/hfpull`, or omitted to use the built-in `gbstep`) and carries a `config` block merged
over the step's own defaults. Steps are where the actual work happens — pulling a model, fine-tuning,
evaluating, pushing results.

See [steps](../steps/README.md) for the built-in steps, the `step.yaml` structure, and the ways to run
custom code ([bring your own step](../steps/bring-your-own-step.md),
[custom code steps](../steps/custom-code-steps.md), [bring your own image](../steps/bring-your-own-image.md)).

## Environments: where a target runs

`environment_uri` binds a target to an **environment** — the compute backend that launches its steps.
The same step can run on very different backends depending on the target's environment. gbserver
resolves `space://environments/<name>` through the active space (see [spaces](../spaces/README.md)).

See [environments](../environments/README.md) for the compute-endpoint map and the per-type
`environment.yaml` reference (bash, Docker, Kubernetes, LSF, RunPod, and SkyPilot's clouds).

## Artifacts: inputs and outputs

Artifacts are the models, datasets, and filesets that flow through a build, referenced by **URI**.
Common schemes: `hf://` (HuggingFace), `file://`, `git://`, `cos://` / `s3://` (object storage),
`env://` (already on a shared filesystem). Each scheme is served by an
[asset store](../asset-stores/README.md).

- **Inputs** are either **direct** (`uri:` — an external artifact) or **bound** (`binding:` — an output
  of another target; see below).
- **Outputs** declare a `uri` (often Jinja-templated) and may set `store_push` to push the result to a
  remote store after the step writes it (e.g. HuggingFace — see [hf-push](hf-push.md)).

The full input/output schema — `wait_for_push`, `event`, `event_selectors`, `store_push`, artifact
`type` — is in the [`build.yaml` reference](build-yaml-reference.md#inputs).

## Bindings: chaining targets

A **binding** connects a downstream target's input to an upstream target's output:

```yaml
targets:
  fine-tune:
    outputs:
      checkpoint: { uri: file:workspace/checkpoint }
  evaluate:
    inputs:
      model:
        binding: fine-tune.checkpoint     # <upstream-target>.<output-name>
```

`evaluate` now depends on `fine-tune`; when `fine-tune.checkpoint` is produced, `evaluate` is
dispatched with its `model` input resolved to that artifact. In step config and templates the resolved
path is available as `{{ bindings.<name>.binding.path }}`.

## Spaces: the runtime context

Every `space://` URI in a build (`space://environments/...`, `space://steps/...`) is resolved against
the active **space**, which provides the environments, steps, asset stores, secrets, and template
variables. See [spaces](../spaces/README.md).

## A complete example

A three-stage pipeline — download a base model, fine-tune it, evaluate the result — chained through
bindings:

```yaml
granite.build:
  name: tune-and-eval
  targets:
    download:
      environment_uri: space://environments/docker
      outputs:
        model: { uri: file:workspace/model }
      steps:
        - step_uri: space://steps/hfpull
          config:
            hf_uri: hf://huggingface.co/ibm-granite/granite-3.3-2b-instruct

    fine-tune:
      environment_uri: space://environments/docker
      inputs:
        model: { binding: download.model }        # depends on `download`
      outputs:
        checkpoint: { uri: file:workspace/checkpoint }
      steps:
        - step_uri: space://steps/sft
          config:
            compute_config: { num_gpus_per_node: 4 }

    evaluate:
      environment_uri: space://environments/docker
      inputs:
        model: { binding: fine-tune.checkpoint }   # depends on `fine-tune`
      outputs:
        results: { uri: file:workspace/eval }
      steps:
        - step_uri: space://steps/unitxt-eval
```

gbserver runs `download` → `fine-tune` → `evaluate` in dependency order. To see a build run end to
end, try the [demos](../demos/README.md).

## Advanced

- [Retry overview](retry.md) — how build- and step-level retry fit together.
- [Build retry](build-retry.md) — re-run a failed build as a new attempt.
- [Step retry](step-retry-configuration.md) — re-launch a single step on a transient error.
- [Target reuse](target-reuse.md) — skip unchanged targets across builds.
- [Lineage tracking](lineage.md) — record build/target/artifact provenance.
- [Event notifications](event-notifications.md) — subscribe to a build's real-time events.

## See also

- [`build.yaml` reference](build-yaml-reference.md) — the complete, canonical schema.
- [Getting started](../getting-started.md) · [CLI reference](../cli/gb-cli-reference.md) · [Templates](../templates/README.md)
- [Glossary](../glossary.md)
