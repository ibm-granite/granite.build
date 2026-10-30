# nl2repo (SkyPilot)

Scores a checkpoint on the granite-evals **nl2repo** family's benchmark:
NL2RepoBench, one OpenHands 0.56 agent run per task in an enroot sandbox, graded in
a fresh sandbox from that task's own GHCR test image.

The code is the vendored granite-evals runtime (`steps/eval/src`), plus this family's
adapter (`steps/eval/nl2repo/src/nl2repo.py`, our port of the upstream harness
logic), baked into this family's image by `Dockerfile.gpu` / `Dockerfile.cpu` one
directory up (built by `steps/eval/Makefile`). Nothing is built from this step at
run time.

```yaml
steps:
  - step_uri: space://steps/eval/nl2repo
    config:
      benchmark: nl2repo
```

## LICENSE BLOCKER — read before touching this family

The upstream task repository,
[multimodal-art-projection/NL2RepoBench](https://github.com/multimodal-art-projection/NL2RepoBench)
at commit `781a1da1ee41fb8edb0bed22f586d69111610edf`, has **no license**. No file
from it may be vendored, baked into an image, mirrored or pushed.

### Why this shape is safe

- **The pushed image contains only open-licensed content**: the granite-evals
  runtime (Apache-2.0) and its dependencies, each verified open — see the
  enumeration below. This family adds **zero** pip dependencies beyond the core
  granite-evals set every family image already ships.
- **The unlicensed upstream repository is fetched at launch, never baked.** The
  adapter, at run start — before any task is processed — downloads the pinned
  commit's tarball, extracts only `test_files/` (the task data; the harness logic
  is our port, so nothing else is needed), verifies it against a pinned sha256 of
  the extracted tree plus the 104-task count, and caches it under
  `~/.cache/granite/nl2repobench/<commit>` inside the job. A fetch that does not
  match the pin refuses to run and never replaces a good cache. Nothing from the
  repository is baked, mirrored or pushed by us. `GRANITE_EVALS_NL2REPO_DIR` (or
  `--dataset`) points at an existing `test_files/` tree fetched by other means.
- **The OpenHands app image (`docker.all-hands.dev/all-hands-ai/openhands:0.56`)
  and the per-task GHCR test images
  (`ghcr.io/multimodal-art-projection/nl2repobench/<task>:1.0`) are pulled from
  their upstream registries at run time via enroot** (into `sandbox_cache`) and are
  never re-distributed by us.

**Pending OSPO sign-off.** The launch-time fetch reduces exposure — nothing
unlicensed is stored, baked or redistributed, and only the minimum (task data) is
read — but it does not make unlicensed use licensed. Until OSPO clears it, treat
this benchmark as blocked for any scored, reported run.

### Dependency / license enumeration

Everything this family's image installs is already in the granite-evals core venv
(no `nl2repo` extra exists; the adapter is stdlib + `httpx`):

| Package | License | Where |
|---|---|---|
| granite-evals (this repo) | Apache-2.0 | the vendored runtime |
| datasets | Apache-2.0 | core dependency |
| httpx | BSD-3-Clause | core dependency |
| huggingface-hub | Apache-2.0 | core dependency |
| pyyaml | MIT | core dependency |
| vllm | Apache-2.0 | `serve` extra — GPU image only |

Transitives are as resolved by `src/core/uv.lock`, the same set every other family
image ships. No package was found that lacks a clear open license, so none was
held back.

## Benchmarks in this family

| `benchmark` | Size | Notes |
|---|---|---|
| `nl2repo` | 104 tasks (25,640 test cases) | implement a whole PyPI project from its `start.md` spec; graded by the project's own tests inside its own test image; metric: **test pass rate** (mean over tasks of `min(passed / test_case_count, 1)`) |

See `granite-evals list` for the current, authoritative set and sizes.

## Config

| Field | Purpose |
|---|---|
| `benchmark` | Required. `nl2repo`. |
| `image` | Required. The granite-evals `nl2repo` image, pinned by tag (`GRANITE_EVALS_IMAGE_NL2REPO`). |
| `model_path` | Required. Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | Name vLLM serves under. Empty = basename of `model_path`. |
| `output_dir` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | **Smoke knob:** score only the first N tasks by name. Empty = the full 104. |
| `repeats` | Independent agent runs per task (avg-of-k). Empty = benchmark default (1). |
| `workers` | Concurrent tasks (agent sandboxes + requests to vLLM). |
| `dataset` / `dataset_revision` | Override the pinned task data with a LOCAL PATH to an existing `test_files/` tree (trusted, not digest-checked — the pin's sha256 is then recorded as absent in results.json). |
| `options` | Space-separated `key=value` benchmark options; see below. |
| `phase` | `all` generates and scores in one job. `generate` serves the model and writes `generation.json` (`granite_generation`); `score` grades that output dir without a GPU (same `output_dir`, `limit`, `repeats`) and writes `results.json`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | vLLM. |
| `sandbox_cache` | Shared squashfs cache for the OpenHands and per-task test images. Every benchmark in this family sandboxes, so recipes always set this. |
| `hf_home` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. No HF token is needed; the task data comes from the launch-time
fetch above.

### Options (`options`)

| Option | Default | Purpose |
|---|---|---|
| `agent` | `model` | `gold` grades each task image with an untouched workspace (see below). |
| `tasks` | (all) | Regex on the task name, e.g. `^six$`. |
| `openhands_image` | `docker.all-hands.dev/all-hands-ai/openhands:0.56` | The agent app image. |
| `image_prefix` / `image_tag` | `ghcr.io/multimodal-art-projection/nl2repobench` / `1.0` | The per-task grade images. |
| `max_iterations` | `500` | OpenHands agent iterations (upstream's setting). |
| `enable_browsing` | `true` | Upstream's default; OpenHands 0.56's browsing runs a local browser, no API key. |
| `temperature` / `top_p` / `max_tokens` | (vLLM defaults) | Sampling for the agent. |
| `agent_timeout` | `14400` | Wall-clock bound per agent run, seconds; `0` = none. Upstream has none. |
| `eval_timeout` | `1800` | Wall-clock bound per grade command, seconds. Upstream has none. |
| `ws_max_mb` | `512` | A larger agent workspace grades as 0 (`workspace_too_large`). |
| `max_failed_frac` | `0.05` | Infrastructure-failed tasks above this fraction fail the run. |
| `sandbox` | `enroot` | Sandbox backend. |

## Gold mode

`options: "agent=gold"` (with `model_path: none`) grades each task image with an
untouched workspace — `start.md` only — so the fetch, the images, the sandbox, the
strip, the staging and the parser all run without a model. Upstream has no
equivalent; how much passes is whatever the task images themselves ship (expect a
low score — it is a plumbing check, not a baseline).

## Output

`granite_results` (dataset): the `results.json` file (phase `all` or `score`).
`granite_generation` (dataset): the `generation.json` file (phase `generate`). A split run
is two targets on the same `output_dir`: the score target binds the generate target's
output, so it runs after it:

```yaml
nl2repo-generate:            # GPU
  outputs:
    granite_generation:
      uri: "env://{{ binding.path }}"
nl2repo:                      # CPU only, phase: score
  inputs:
    generation:
      binding: nl2repo-generate.granite_generation
  outputs:
    granite_results:
      uri: "env://{{ binding.path }}"
```

Per-task work is kept next to it under `output/repeat-<k>/<task>/`:
`workspace.tar.gz` (the agent's workspace, what the score phase grades),
`openhands.log`, `trajectory.tar.gz` (the OpenHands trajectory), `meta.json`, and
once graded `report.json` and `test_output.txt`. Finished tasks are skipped on
restart; infrastructure failures are retried, never recorded as model losses.

## Developing

`make unit-tests` runs the template contract tests. The runtime's own tests live in
`steps/eval/src` (`make test-src` from `steps/eval/`). Every granite-evals family step
under `steps/eval/` follows this template; the test checks this.
