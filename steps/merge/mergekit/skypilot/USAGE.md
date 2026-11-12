# `mergekit` step

Merges two or more models with [MergeKit](https://github.com/arcee-ai/mergekit)
(`mergekit-yaml`, v0.1.5, LGPL-3.0) on any SkyPilot endpoint (slurm, kubernetes, aws), and
registers the merged model as a standard Hugging Face model directory. The merge recipe is
MergeKit's own YAML, passed through unchanged except that a model may be named by one of
the step's input names.

## Referencing the step

```yaml
steps:
  - step_uri: space://steps/merge/mergekit
```

## Inputs and outputs

| Name | Direction | Type | Required | Meaning |
|---|---|---|---|---|
| `model_1` | input | `model` | yes | A model to merge |
| `model_2`, `model_3`, `model_4` | input | `model` | no | Further models to merge |
| `base_model` | input | `model` | no | Base for base-relative methods (`ties`, `dare_*`, `task_arithmetic`, `slerp`, ...) |
| `model` | output | `model` | yes | The merged model directory |

## Configuration (`config.mergekit_config`)

| Key | Type | Default | Purpose |
|---|---|---|---|
| `merge_config` | mapping | `{}` | The MergeKit recipe (required). Set precision here with `dtype` and `out_dtype`. |
| `args` | mapping | `{}` | Extra `mergekit-yaml` flags without dashes: `true` → `--k`, `false` → `--no-k`, `null` → omitted, other → `--k 'v'`; an empty value is refused. Not allowed, because the step owns them: `transformers-cache`, `lora-merge-cache`, `cuda`, `device`, `gpu-rich`, `allow-crimes`, `trust-remote-code`, `copy-tokenizer`, `lazy-unpickle` (use the named keys) and `safe-serialization` (the step checks the output's `.safetensors` files). `-v` has no long form, so it can't be set. |
| `copy_tokenizer` | bool | `true` | `--copy-tokenizer` (MergeKit's default). When true, or when the recipe builds a tokenizer, the merged model must have one, or the step fails rather than register it without one. |
| `lazy_unpickle` | bool | `true` | `--lazy-unpickle`: load pickled `.bin` checkpoints lazily (MergeKit defaults it off). No effect on `.safetensors` inputs, which MergeKit always memory-maps. |
| `allow_crimes` | bool | `false` | `--allow-crimes`: allow mixing architectures |
| `trust_remote_code` | bool | `false` | `--trust-remote-code`: run a model repo's own code |
| `output_path` | string | `""` (`./model`) | Where the merged model is written. Set it to a shared path, matching the output's `env:///` uri, when another target reads the model. |
| `mergekit_version` | string | `"0.1.5"` | MergeKit release installed with uv |
| `pip_index_url` | string | `https://pypi.org/simple` | Package index for the install |
| `mk_image` | string | `""` | A container image that already has `mergekit-yaml` and `python3` with PyYAML; skips the install. Not usable on the local Docker SLURM cluster (no Pyxis). |

`config.compute_config` sizes the node (see [Memory](#memory)). `config.skypilot.secrets`
passes secrets such as `HF_TOKEN` to the merge as environment variables.

## Naming models in the recipe

A model is given to the step once, as a target input. The target's `inputs` say *which*
models the build stages; the recipe says *how* they are merged (which is the base, which
layers come from which model, the weights), so it names each of them too, by its **input
name**: `model: model_1`. The step swaps in the input's staged path before MergeKit runs.

Use the input name rather than `{{ bindings.model_1.binding.path }}`. The bindings form
works for a bound input (it renders to the staged path, which the step counts as naming
that input), but for one the target doesn't bind it arrives as literal `{{ ... }}` text,
which the step refuses. With input names, a recipe copied from MergeKit's docs keeps its
shape, and the step checks every name against the target's bindings (below).

Every model reference in `merge_config` keeps MergeKit's syntax: `<model>`,
`<model>@<revision>`, `<model>+<lora>`. `<model>` and `<lora>` can each be:

- **An input name** (`model_1` … `model_4`, `base_model`) of a **bound** input. The step
  swaps in the staged path before MergeKit runs, so the model is a build input and shows in
  lineage. Naming an input that the target doesn't bind is refused rather than sent to
  MergeKit as a Hub id. A string reference in the top-level `base_model`, `models[].model`
  or `slices[].sources[].model` is refused when the step starts, before the merge node is
  provisioned. Any other reference (the mapping form, a slice's `base_model`,
  `tokenizer_source`, `tokenizer`, or a `modules` recipe) is refused on the node, before
  MergeKit runs.

  Every bound input must also be named in the recipe, in one of its model references. A
  bound input the recipe never names, most often a typo such as `model: model1` for
  `model_1`, is refused, rather than the merge going ahead without it (and the typo going
  to the Hub as a repo id). Usually that is caught when the step starts; a name that
  appears in the recipe only outside a model reference (e.g. inside a Hub id, as in
  `org/model_1-x`) is caught on the node. The reference fields are MergeKit 0.1.5's: with
  a newer `mergekit_version`, an input named only in a field 0.1.5 doesn't have counts as
  unused.
- **A Hugging Face repo id** or a path on the node. Passed through untouched: MergeKit
  downloads it itself, exactly as with `mergekit-yaml` on its own. Such models are not build
  inputs and are not in lineage. A gated repo needs `HF_TOKEN` (see `skypilot.secrets` in
  [Configuration](#configuration-configmergekit_config)).

MergeKit's mapping form of a reference (`{model: ..., lora: ..., override_architecture:
...}`) is substituted too. Where `model` or `lora` is a string, MergeKit splits it on `@`
only (the `+` LoRA syntax belongs to the plain string form), and the step does the same.
Where it is a `{path, revision}` mapping, MergeKit takes `path` literally,
so the step replaces it only when the whole `path` is an input name, and never changes
`revision`. A bound input whose staged path contains `@` or `+` is refused, because
MergeKit would read those as a revision or a LoRA.

A recipe copied from MergeKit's docs works as is, and inputs can be mixed in:

```yaml
merge_config:
  merge_method: ties
  base_model: base_model                     # a bound input
  models:
    - model: model_1                         # a bound input
      parameters: {density: 0.5, weight: 0.5}
    - model: Qwen/Qwen2.5-0.5B-Instruct      # downloaded by MergeKit
      parameters: {density: 0.5, weight: 0.5}
  dtype: bfloat16
```

## Inline chat templates

One exception to passing the recipe through unchanged: **an inline Jinja
`chat_template`**. gbserver renders every string in a step's config as Jinja before the
step runs. That drops its `{% ... %}` blocks, so an
unwrapped template arrives empty or as stray fragments such as
`{{ bos_token }}{{ eos_token }}`. MergeKit would accept some of those and ship a broken
template. So wrap an inline template in `{% raw %}...{% endraw %}` in the build.yaml.

The step accepts only:
- `null` (no template, MergeKit's default);
- a template *name* MergeKit ships: `auto`, `alpaca`, `chatml`, `exaone`, `llama3` or
  `mistral` in 0.1.5. The step reads the list from the installed mergekit. Names need no
  wrapping;
- a template that still contains `{%` blocks, which survive only inside `{% raw %}`.

Anything else is refused before the merge runs, including a template made only of
`{{ }}` expressions, which after the fill can't be told apart from a mangled one. The
error lists the valid names.

```yaml
merge_config:
  chat_template: "{% raw %}{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% endraw %}"
```

## Output checks

The step checks the merged model before registering it, because MergeKit doesn't always
report a tokenizer problem as a failure:
- **Copying** a tokenizer (`copy_tokenizer: true`) happens inside a `try` that logs
  "Failed to copy tokenizer. The merge was still successful" on any error: a template
  MergeKit rejects, a donor model with no tokenizer, or a file that fails to copy
  partway.
- **Building** a tokenizer (`tokenizer_source` or a `tokenizer` block, even an empty
  one) is not inside that `try`. A failure there fails the merge outright.
- `chat_template: auto` copies the most common template among the input models. If
  none of them has one, MergeKit silently sets none.

So the step fails, and registers nothing, when:
- MergeKit logs "Failed to copy tokenizer", whichever files had been copied by then;
- a tokenizer was copied or built, but the output has no `tokenizer_config.json` plus a
  vocabulary file (`tokenizer.json`, or a slow tokenizer's own file such as
  `tokenizer.model`, `vocab.json` or `vocab.txt`);
- `chat_template` is set, on any tokenizer path, but the output has no chat template.

It also refuses `chat_template` with `copy_tokenizer: false` up front, unless the recipe
sets `tokenizer_source` or a `tokenizer` block. Without either, MergeKit has no tokenizer
to put the template in: it only warns and drops it.

## Worked example

MergeKit's `examples/gradient-slerp.yml`, scaled to a public 135M base / instruct pair. This is the step's own build-test fixture:

```yaml
granite.build:
  name: mergekit-example
  targets:
    merge:
      environment_uri: space://environments/skypilot/slurm/local
      inputs:
        model_1:
          uri: hf:///HuggingFaceTB/SmolLM2-135M-Instruct
        base_model:
          uri: hf:///HuggingFaceTB/SmolLM2-135M
      outputs:
        model:
          uri: "env:///shared/mergekit-example/model"
          type: model
      steps:
        - step_uri: space://steps/merge/mergekit
          config:
            mergekit_config:
              output_path: /shared/mergekit-example/model
              merge_config:
                slices:
                  - sources:
                      - model: base_model
                        layer_range: [0, 30]
                      - model: model_1
                        layer_range: [0, 30]
                merge_method: slerp
                base_model: base_model
                parameters:
                  t:
                    - filter: self_attn
                      value: [0, 0.5, 0.3, 0.7, 1]
                    - filter: mlp
                      value: [1, 0.5, 0.7, 0.3, 0]
                    - value: 0.5
                dtype: bfloat16
```

## Memory

Two measurements with mergekit 0.1.5 on a CPU: a peak of 1.25 GB for the example above (two 135M models in bf16), and 6.4 GB for two 0.5B models merged in float32. Peak RAM grows with the models' size and the merge `dtype`; size the node with headroom over the combined size of the models.

- **Kubernetes and cloud:** `compute_config.total_memory_per_node` is the request.
- **SLURM and LSF:** gbserver does not send `compute_config` memory. The job gets the partition default, and an explicit `launcher_config.resources.memory` applies instead.
  - On a cluster that enforces per-job memory, set `launcher_config.resources.memory` (a bare GiB number as a string, e.g. `"16"`).
  - On the local Docker SLURM cluster, do not set it. Memory is neither scheduled nor enforced there, so a job that asks for nothing can use the node's free RAM. A request above the nodes' advertised 1024 MB is refused at scheduling.

## Limits

- **Hugging Face ids in the recipe are cached on the environment's shared root**
  (`$GB_SHARED_WORKDIR/.mergekit-hf-cache`, as for the uv cache), so a cluster with a
  shared workdir (local SLURM and kind) downloads each once. Where there is no shared
  workdir (aws), the cache is the run's own workdir and every build downloads them again.
  Either way they are not build inputs and not in lineage; to track a model, bind it as
  an input (`hf:///...`). The shared cache is never pruned, and is created with the
  first user's umask: on a workdir several users share (e.g. a team project directory),
  make it group-writable, or a second user's download fails with a permission error.
- **The model card doesn't list staged inputs as base models.** MergeKit writes a
  `README.md` model card into the output, and lists Hugging Face ids in its `base_model:`
  front matter. A model bound as a step input reaches MergeKit as a node path, so it isn't
  listed, and the `mergekit_config.yml` that MergeKit writes beside it records the node
  path. The build's own lineage still records every input. If the merged model is pushed
  to the Hub, edit the card's `base_model:` first.
- **A retry re-registers, not re-merges.** If an attempt finished the merge and moved it
  into place but the job was then lost, gbserver's step retry (which relaunches the same
  step run) registers that output: a `.<output name>.mergekit-done` record beside it names
  the step run. Any other run still refuses an existing output, including a retry of the
  whole build, which creates new step runs: give it a new `output_path`, or remove the
  old output first.
- **A killed job leaves its staging dir.** The merge is written to a
  `.mergekit-stage.*` directory beside `output_path` and moved into place only on
  success. A normal failure removes it, but a job killed outright (SIGKILL, a preempted
  node) does not, and the staged model can be as large as the output. Remove leftover
  `.mergekit-stage.*` directories by hand.

## Running on BlueVela

The step runs on the CPU by default. BlueVela's environment sends jobs to the `gpu-mid` partition with an `h200` constraint, and that partition's submit filter rejects a job with no GPUs. For a CPU merge, send the step to the `cpu` partition and clear the constraint, as the environment does for its own hfpull and hfpush steps (`configurations/assets/environments/skypilot/slurm/bluevela/environment.yaml`):

```yaml
- step_uri: space://steps/merge/mergekit
  config:
    zone: cpu
    launcher_config:
      sbatch_options: {constraint: null}
      resources: {memory: "16"}
```

Not yet verified by a BlueVela build test.

For the step framework, see `steps/README.md` in the granite.build repository.
