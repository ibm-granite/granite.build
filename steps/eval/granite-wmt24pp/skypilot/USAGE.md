# granite-wmt24pp (SkyPilot)

Scores a checkpoint on **WMT24++ (en→xx)** (Granite metric: en→xx COMET (XCOMET-XXL)).
The job serves the model with vLLM and runs, locally:
NeMo-Skills' WMT24++ on `google/wmt24pp`, en→xx for its default languages de_DE, es_MX, fr_FR, it_IT
and ja_JP (998 segments each; `languages=de_DE,ja_JP` chooses), with its segment-translation prompt.
The value is NeMo-Skills' `comet`: XCOMET-XXL (`Unbabel/XCOMET-XXL`, ~10.7B parameters, gated,
CC-BY-NC-SA-4.0) as NeMo-Skills runs it, in the image's separate COMET env. **Phase `score` needs a
GPU** (~22 GB in bf16). BLEU (sacrebleu, ja-mecab for Japanese) is always reported;
`score_metric=bleu` makes it the value and skips COMET (no GPU). Values are fractions.
It writes one `results.json`.

The code is the external [granite-evals](https://github.com/laminair/granite-evals) runtime,
shipped as a prebuilt image (the `nemoskills` family). Nothing is built from this step.

```yaml
steps:
  - step_uri: space://steps/eval/granite-wmt24pp
```

## Config (`granite_config`)

| Field | Default | Purpose |
|---|---|---|
| `image` | `""` (required) | granite-evals nemoskills image, pinned by tag (`GRANITE_EVALS_IMAGE_NEMOSKILLS`). |
| `model_path` | `""` (required) | Local HF checkpoint dir, usually `{{ bindings.model.binding.path }}`. |
| `served_model_name` | basename of `model_path` | Name vLLM serves under. |
| `output_dir` | `output` | Relative to `$GB_BUILD_WORKDIR`. |
| `limit` | `""` (all 998 segments per language) | **Smoke knob:** first N items of the pinned data. |
| `repeats` | `""` (1) | Independent generations per item; k > 1 reports pass@1[avg-of-k] and `details.pass_at_k`. |
| `workers` | `32` | Concurrent requests to vLLM. |
| `dataset` / `dataset_revision` | `""` | Override the data pinned in granite-evals (google/wmt24pp, pinned in granite-evals). |
| `options` | `""` | Space-separated `key=value` benchmark options: `languages`, `score_metric` (`comet` or `bleu`), `comet_model`, `comet_encoder`, `temperature`, `top_p`, `top_k`, `max_tokens`, `ns.<key>=<value>` (any NeMo-Skills generation override), `answers=gold`. |
| `tensor_parallel_size` / `gpu_memory_utilization` / `max_model_len` | `1` / `0.9` / `""` | vLLM. |
| `sandbox_cache` | `""` | Unused: this benchmark runs no sandbox. |
| `hf_home` | `""` | Overrides `HF_HOME`. |

GPUs, queue and memory come from the target's `launcher_config.resources`. The step
declares none. The data is public, but the scoring model is gated: `HF_TOKEN` must have access to `Unbabel/XCOMET-XXL` (and `facebook/xlm-roberta-xxl`).

Sampling defaults to the checkpoint's `generation_config` (NeMo-Skills' own default is
greedy); `results.json` records the sampling used. Only the content after vLLM's reasoning
parser is scored.

`answers=gold` serves the reference answers through the same NeMo-Skills generation and
scoring without a model (set `model_path: none`); it should score about 100%. Use it to validate
the data and grading on a new cluster.

## Output

`granite_results` (dataset): the `results.json` file (phase `all` or `score`).
`granite_generation` (dataset): the `generation.json` file (phase `generate`). A split run
is two targets on the same `output_dir`: the score target binds the generate target's
output, so it runs after it:

```yaml
<bench>-generate:            # GPU
  outputs:
    granite_generation:
      uri: "env://{{ binding.path }}"
<bench>:                     # CPU only, phase: score
  inputs:
    generation:
      binding: <bench>-generate.granite_generation
  outputs:
    granite_results:
      uri: "env://{{ binding.path }}"
```

Per-item outputs and logs are kept next to it.
