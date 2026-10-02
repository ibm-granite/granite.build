# bfcl-eval (SkyPilot)

Scores a model's **tool calling** against ground truth with the [Berkeley Function Calling
Leaderboard](https://github.com/ShishirPatil/gorilla) harness, and registers the resulting
score file as the step's output.

The harness and its runtime are baked into a container image built from
[`Dockerfile`](Dockerfile); the `run` block invokes the baked entrypoint
([`src/run-bfcl.sh`](src/run-bfcl.sh)) with parameters from `config.bfcl_config`.

> **This step replaces an IBM-internal image.** A `bfcl-eval` step asset already exists
> under `configurations/assets/environments/skypilot/lsf/ibm-bluevela/steps/bfcl-eval/`,
> and it runs `docker:us.icr.io/cil15-shared-registry/bfcl-py311:0.02` — not pullable
> outside IBM. This step keeps that config surface and builds its image from public bases
> with the harness pinned in [`uv.lock`](uv.lock), so the version that produced a score is
> recoverable from the repository. **The image has not been built or pushed yet**; see
> [README.md](README.md) before using it on a cluster.

## What this measures, and what it does not

BFCL scores the student against **ground truth** — *did it call the right function with the
right arguments?* It cannot say whether a distilled student moved toward its **teacher**,
which is a distance between two distributions; that is what
[`distill-eval`](../../distill-eval/skypilot/USAGE.md) measures. A distillation recipe wires
**both**, and they are not substitutes in either direction: a student can track its teacher
closely and still call functions badly, or score well while having learned nothing from the
teacher.

## Who emits the artifact line?

The **step**, not the workload — the preferred pattern for a fixed, known output path.

`run-bfcl.sh` prints no Granite.build marker and knows nothing of the artifact convention
(it is also run standalone, outside any build). The `run:` block selects the score file the
harness wrote and registers it:

```sh
RESULT_FILE=$(head -n 1 "$CANDIDATES_FILE")
echo "GB_ARTIFACT_ID:bfcl_results GB_ARTIFACT_PATH:${RESULT_FILE}"
```

The harness writes a `score/` tree rather than one fixed filename, so the step picks the
**newest** `.json` under it and **prints the list whenever there is more than one
candidate** — the choice is in the log rather than implicit. If the tree holds no score file
at all, the step prints the directory contents and **exits 1** instead of completing green
with nothing registered.

## Referencing the step

```yaml
steps:
  - step_uri: space://steps/bfcl-eval
```

### That URI does not resolve here yet — two older copies shadow it

**Authoring this step is not the same as switching the pipeline over to it, and the
difference is invisible in a build.yaml.** `space://steps/<name>` resolves by walking *up*
from the active environment directory, **nearest-wins**
(`gbcommon/uri/space.py::_walk_colocated_steps`). `make publish-step` writes this step
env-nested at `configurations/assets/environments/skypilot/steps/bfcl-eval` — but two
cluster-nested copies of the same name already sit *nearer* to their environments:

| Published asset | Image it names |
|---|---|
| `environments/skypilot/lsf/ibm-bluevela/steps/bfcl-eval` | `us.icr.io/cil15-shared-registry/bfcl-py311:0.02` |
| `environments/skypilot/aws/steps/bfcl-eval` | `022767362696.dkr.ecr.us-east-2.amazonaws.com/granite-build/bfcl-py311:0.02` |

So on **`ibm-bluevela`** — the cluster this step is for — `space://steps/bfcl-eval` keeps
resolving to the ICR image, published or not. Both shadowing copies name a **private**
registry, which is the reason this step exists.

Switching over takes two steps in this order, and the first one needs credentials nobody
authoring this step had:

1. `make image publish-image REGISTRY=<public-registry>/<org>`, then `make publish-step`
   (which **refuses** until that image is actually pullable — `PUBLISH_REQUIRE_IMAGE`).
2. Retire the shadowing copy for whichever cluster you are switching: delete its directory,
   or repoint its `image_id` at the now-public image.

Doing 2 before 1 does not degrade gracefully — it resolves the target to an image that does
not exist, which fails at run time rather than at render time. Until both are done, treat
this directory as the reviewable public **source** of the step, not as what the recipe runs.

## Config contract (`bfcl_config`)

All fields live under `config.bfcl_config` and are templated into the `run` block.

| Field | Type | Required | Purpose |
|---|---|---|---|
| `model_path` | string | **required** | Checkpoint to score, as a **local path** — vLLM serves it from disk, so a hub id would need a fetch this step does not perform. Usually set from a bound input: `"{{ bindings.model.binding.path }}"`. |
| `model_id` | string | **required** | Which harness **handler** formats prompts and parses replies out of raw output. Not cosmetic: a wrong id scores the right model with the wrong parser and reports a plausible number. [`src/bfcl-shim`](src/bfcl-shim) registers real Granite-4 checkpoint ids that upstream's `MODEL_CONFIG_MAPPING` lacks, so this can name the actual checkpoint instead of borrowing an unrelated size's entry. |
| `experiment` | string | optional | Label; a path component of the output directory only. |
| `eval_name` | string | optional (default `bfclv4`) | Label; a path component of the output directory only. **Does not select a benchmark version** — the harness version is whatever `uv.lock` pins. |
| `test_categories` | string | optional (default `all`) | Comma-separated BFCL category names, or `all`. Passed as `--test-categories`. |
| `num_gpus_generate` | int | optional (default `1`) | Tensor-parallel size of the vLLM server. Must not exceed the accelerators the target requests in `launcher_config.resources`. |
| `num_gpus_evaluate` | int | optional (default `1`) | Accepted for compatibility with the legacy asset and **currently unused**: today's harness scores already-generated responses on CPU and its `evaluate` subcommand takes no GPU flag. |
| `gpu_memory_utilization` | string | optional (default `"0.9"`) | Fraction of GPU memory vLLM may reserve. |
| `output_dir` | string | optional (default `output`) | Root of the output tree. Relative values resolve inside the per-run working directory; the step absolutises the path before registering the artifact. The full tree is `<output_dir>/<experiment>/<eval_name>`. |
| `sample_fraction` | string | optional (default empty ⇒ **full corpus**) | **Cost knob.** Run this fraction (0–1) of each category's ids for a directional read. Memory categories are sampled by whole scenario, and `evaluate` then runs with `--partial-eval` so the run scores what it generated instead of raising on every id it skipped. |
| `sample_seed` | string | optional (default `"42"`) | Determines *which* ids are sampled, so two runs with the same fraction and seed are comparable. Only meaningful with `sample_fraction`. |
| `hf_home` | string | optional | Sets `HF_HOME` for anything the harness resolves from the Hub. Empty leaves the image's default alone. Deliberately **not** under `config.workload`, whose schema is typed and would silently drop it. |

Two monitor keys are also read from `config`, matching the sibling distill steps:
`poll_interval_seconds` and `log_retrieval_interval_seconds` (both default `900`).

> **`sample_fraction` and `test_categories` do not compose the way they look.** Sampling
> works through the harness's only sub-corpus lever, `--run-ids`, which reads an explicit id
> file and **ignores `--test-category` entirely**; the id file is built across *every*
> scoring category. So with `sample_fraction` set, `test_categories` no longer limits what is
> **generated** — only what is **scored**. To evaluate one category cheaply, set
> `test_categories` and leave `sample_fraction` empty; to get a cheap read across the whole
> benchmark, set `sample_fraction` and leave `test_categories` at `all`. Setting both
> narrowly generates far more than it scores.

### Flags that are not in the config surface

`run-bfcl.sh` also accepts `--num-shards`/`--shard-index`, `--exclude-categories`,
`--skip-evaluate`, `--evaluate-only` and `--vllm-port`. These are **deliberately not
exposed** as step config: sharding partitions one corpus across concurrent jobs and forces
scoring to be skipped, so a target that set it would register no score; `--evaluate-only`
re-scores a finished tree, which is a human operation; and a recipe-settable port is how
two targets on one node collide silently. Use them by running the script directly —
[`src/merge_shard_results.py`](src/merge_shard_results.py) merges a sharded tree for a
single later `--evaluate-only` pass. A contract test asserts they stay out of the template.

## Inputs and outputs

- **Inputs** — the model is a target `inputs.model` artifact, usually bound to an upstream
  export target. Its resolved path is templated into `bfcl_config.model_path` via
  `"{{ bindings.model.binding.path }}"`, so the step needs no change to consume it.
- **Outputs** — `outputs.required.bfcl_results` (`type: dataset`), a single score `.json`.
  **Required, not optional**: the step fails when the harness produces no score, so a
  successful run always has one. Bind a matching `outputs.bfcl_results` on the target.

## Working directory and paths

`run` starts in the step's per-run working directory. A relative `output_dir` resolves
there, giving per-run isolation; the step absolutises it before registering the artifact,
because a relative `env://` URI is rejected at config load and the monitor may hand the path
to the store from another host.

> The per-run working directory is **removed at teardown**. Anything to keep beyond the run
> must be bound as an output (or written to an absolute `output_dir`).

## Example build.yaml

```yaml
granite.build:
  name: bfcl-eval-example
  version: 0.0.1
  targets:
    eval-bfcl:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        model:
          uri: hf:///models/ibm-granite/granite-4.0-h-350m
      outputs:
        bfcl_results:
          uri: "env://{{ binding.path }}"
          type: dataset
      steps:
        - step_uri: space://steps/bfcl-eval
          config:
            poll_interval_seconds: 900
            log_retrieval_interval_seconds: 900
            bfcl_config:
              # The resolved path of the target's `model` input.
              model_path: "{{ bindings.model.binding.path }}"
              model_id: "ibm-granite/granite-4.0-h-350m"
              experiment: "smoke"
              eval_name: "bfclv4"
              test_categories: "simple,parallel"
              num_gpus_generate: 1
              num_gpus_evaluate: 1
              gpu_memory_utilization: "0.9"
              output_dir: "output"
              # Empty on purpose: `test_categories` above is already the cost knob here,
              # and setting both would generate the whole benchmark to score two
              # categories of it -- see the note under the config table.
              sample_fraction: ""
              sample_seed: "42"
            launcher_config:
              resources:
                accelerators: "H100:1"
```

In a distillation pipeline this target binds to the export step instead of declaring a
`uri`, e.g. `model: { binding: export.hf_model }`.

## Notes and limitations

- **The image must exist before a build runs.** The cluster *pulls* the reference frozen
  into the step at `make space` time. No image has been published for this step yet — see
  [README.md](README.md).
- **Publishing this step does not make `space://steps/bfcl-eval` mean this step.** Two
  cluster-nested copies naming private registries resolve first, `ibm-bluevela`'s among them
  — see [That URI does not resolve here yet](#that-uri-does-not-resolve-here-yet--two-older-copies-shadow-it).
- **`num_gpus_generate` is not a request for GPUs.** It sets vLLM's tensor-parallel size;
  the GPUs themselves come from the target's `launcher_config.resources`. Setting it higher
  than the target requests fails at server start.
- **A sampled score is not a leaderboard score.** `sample_fraction` exists to answer
  "did this pipeline produce something plausible" cheaply. Report full-corpus runs.
- **`eval_name` does not pin the benchmark.** The harness version comes from `uv.lock`;
  rebuilding the image after bumping it changes what "bfclv4" means.
- **Rank variables are cleared before the harness starts.** vLLM refuses to start inside an
  initialised `torch.distributed` environment, which is the state this target inherits when
  it follows a training target.
