# eval/livecodebench-score (SkyPilot): development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `eval/livecodebench-score` in a `build.yaml`. This file covers how the step is
> built, tested and published.

CPU step that scores LiveCodeBench completions written by
[`eval/livecodebench-generate`](../../livecodebench-generate/skypilot/README.md). Like
[byoc](../../../byoc/skypilot/README.md) it is a **public-image** step: there is no
`Dockerfile`; it runs in `lcb_score_config.image` (`python:3.12-slim` by default; `""`
runs on the bare node). `setup` fetches a pinned LiveCodeBench commit, then installs
CPU torch, the pinned packages in `src/requirements-score.txt`, and LiveCodeBench
itself `--no-deps`, into a `uv` venv. It is generated from the sources here by the
shared Makefile conventions; see [steps/README.md](../../../README.md).

## Layout

- `step-template.yaml`: the step. `run` refuses to start with `HF_TOKEN` set, then
  runs `src/lcb_score.py` from inside the LiveCodeBench checkout, offline, and
  registers `output_path` as the `results` artifact.
- `src/lcb_score.py`: checks the completions against `generation_meta.json` and the
  dataset's window, runs LiveCodeBench's `custom_evaluator` unchanged (through
  `runpy`, under an `if __name__ == "__main__"` guard, because the scorer uses
  `multiprocessing`), and writes `summary.json`.
- `src/lcb_dataset.py`: points LiveCodeBench's hard-coded dataset name at the
  downloaded `dataset` input, and fixes two memory problems in loading (see its
  docstring). **The generate step ships an identical copy;** a unit test keeps the
  two in step.
- `src/requirements-score.txt`: the packages scoring imports, pinned to
  LiveCodeBench's `uv.lock` at `lcb_commit`.
- `test/test_lcb_score.py`: unit tests for `lcb_score.py`'s own logic. They run
  under the repo venv, which has no LiveCodeBench.
- `test/test_skypilot_eval_livecodebench_score_render.py`: Mode-1 render tests. They execute
  the rendered `run` block with a stub `python` on `PATH`.

## Regenerating the pinned packages

`src/requirements-score.txt` is the subset of LiveCodeBench's lock that scoring
imports, not the whole lock (which also carries vLLM and CUDA). To regenerate it for
a new `lcb_commit`:

1. Check out that commit and export its lock without the GPU stack:
   `uv export --frozen --no-hashes --no-emit-project`, minus `vllm`, `torch*`,
   `nvidia-*`, `triton`, `xformers`, `xgrammar`, `flashinfer*`, `ray` and friends.
2. Install that, CPU torch, and LiveCodeBench `--no-deps` into a fresh venv; run a
   score; list the distributions it imported (`importlib.metadata.packages_distributions()`
   over `sys.modules`).
3. Keep only those, at the lock's versions.

## Building, testing and publishing

```sh
make -C steps/eval/livecodebench-score/skypilot space         # render space/
make -C steps/eval/livecodebench-score/skypilot test          # render, then run test/
make -C steps/eval/livecodebench-score/skypilot publish-step  # promote into configurations/ + test/steps/
make -C steps/eval/livecodebench-score/skypilot check-published
```

## Status

New. Not yet published. Planned: a build test on the local Docker SLURM cluster
against a tiny fixture dataset, and a GPU-to-CPU end-to-end run with the generate
step.
