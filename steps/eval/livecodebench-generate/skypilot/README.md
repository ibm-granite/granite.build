# eval/livecodebench-generate (SkyPilot): development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `eval/livecodebench-generate` in a `build.yaml`. This file covers how the step is
> built, tested and published.

GPU step that generates LiveCodeBench completions with vLLM, for any chat model,
scored by [`eval/livecodebench-score`](../../livecodebench-score/skypilot/README.md).
Like [byoc](../../../byoc/skypilot/README.md) it is a **public-image** step: there is
no `Dockerfile`; it runs in `lcb_generate_config.image` (`python:3.12-slim` by
default; `""` runs on the bare node). `setup` fetches a pinned LiveCodeBench commit,
then installs `vllm_package` and the few packages in `src/requirements-generate.txt`,
and LiveCodeBench itself `--no-deps`, into a `uv` venv. It is generated from the
sources here by the shared Makefile conventions; see
[steps/README.md](../../../README.md).

## Why LiveCodeBench is used as a library

- **Its CLI accepts only listed models.** `lcb_runner.runner.main --model` must name
  an entry in its `lm_styles.py`, which also picks the prompt format; any other name
  is a `KeyError`. So `src/lcb_generate.py` calls LiveCodeBench's prompt text, code
  extractor and dataset loader directly, and formats each prompt with the model's
  own chat template.
- **Its lock pins an old GPU stack** (vLLM 0.8.4, torch 2.6). This step installs
  LiveCodeBench `--no-deps` and its own vLLM (`vllm_package`), so the torch build can
  match the node's driver. The library pieces it calls import without vLLM.

## Layout

- `step-template.yaml`: the step. `run` refuses an empty `release_version` or
  `max_tokens`, then runs `src/lcb_generate.py` from inside the LiveCodeBench
  checkout, offline, and registers `output_path` as the `completions` artifact.
- `src/lcb_generate.py`: prompts, vLLM generation, code extraction and the three
  output files. It detects whether the chat template itself opens a `<think>`
  block, so an output cut off mid-reasoning counts as having no code rather than
  having a draft.
- `src/lcb_dataset.py`: the same file as the score step's; see its README.
- `src/requirements-generate.txt`: the light packages LiveCodeBench's library
  modules import.
- `test/test_lcb_generate.py`: unit tests for `lcb_generate.py`, with LiveCodeBench's
  two library functions stubbed, under the repo venv.
- `test/test_livecodebench_generate_step_render.py`: Mode-1 render tests. They
  execute the rendered `run` block with a stub `python` on `PATH`.

## Building, testing and publishing

```sh
make -C steps/eval/livecodebench-generate/skypilot space
make -C steps/eval/livecodebench-generate/skypilot test
make -C steps/eval/livecodebench-generate/skypilot publish-step
make -C steps/eval/livecodebench-generate/skypilot check-published
```

## Status

New. Not yet published. There is no local build test: vLLM needs a GPU, and the
local Docker SLURM cluster has none. Planned: a GPU end-to-end run with the score
step.
