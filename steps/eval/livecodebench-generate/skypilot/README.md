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
- `test/test_skypilot_eval_livecodebench_generate_render.py`: Mode-1 render tests. They
  execute the rendered `run` block with a stub `python` on `PATH`.
- `test/slurm-bluevela/` + `test-data/slurm-bluevela/`: the GPU build test, for this
  step and the score step together, on an internal GPU SLURM cluster through the
  in-repo environment `space://environments/skypilot/slurm/ibm-bluevela`. `generate`
  runs `ibm-granite/granite-4.2-3b` on one H200 on one contest day (6 problems);
  `score` scores the completions on a CPU partition. Both outputs go to `test_`
  Hugging Face datasets that `HfPushCleanupMixin` checks and deletes. The score isn't
  asserted, since sampling varies; the score step's `slurm-cpu` test checks an exact
  one. It is marked `ibm` and skipped in SPS CI/CD: the environment logs in with a
  key file, `~/.ssh/ibm-bluevela.key`, so it needs no secret manager, but CI has no
  key. Run it by hand from the IBM venv (`make ibm-venv`), with an `HF_TOKEN` that can
  write to `GBTEST_HF_PUSH_ORG`:

  ```sh
  make -C steps/eval/livecodebench-generate/skypilot space
  cd steps/eval/livecodebench-generate/skypilot
  GB_ENVIRONMENT=STANDALONE GBTEST_MODE=live GBTEST_STANDALONE_ENVIRONMENT=STAGING \
    PYTHONPATH=src ../../../../.venv/bin/python -m pytest -s -v test/slurm-bluevela
  ```

  The generate job goes to the environment's default GPU partition. When your group's
  quota there is full (`squeue` shows the pending reason `QOSGrpGRES`), point a local
  copy of the build at a partition and QOS that have room, with `zone:` and
  `launcher_config.sbatch_options.qos:`. Don't commit those: they are specific to one
  group.

## Building, testing and publishing

```sh
make -C steps/eval/livecodebench-generate/skypilot space
make -C steps/eval/livecodebench-generate/skypilot test
make -C steps/eval/livecodebench-generate/skypilot publish-step
make -C steps/eval/livecodebench-generate/skypilot check-published
```

## Status

Published to `configurations/assets/environments/skypilot/steps/eval/livecodebench-generate/`.
There is no local build test: vLLM needs a GPU, and the local Docker SLURM cluster
has none. The `slurm-bluevela` GPU build test, which runs this step and the score
step end to end, passes on the internal GPU SLURM cluster (one H200, about 33
minutes, most of it job launches).
