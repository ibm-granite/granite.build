# eval/unitxt (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `eval/unitxt` in a `build.yaml`. This file covers how the step is built, tested
> and published.

Evaluation step that runs the [unitxt](https://www.unitxt.ai) `unitxt-evaluate`
CLI as a black box, in `hf` mode, against the target's declared `model` input. Like
[dpk](../../../dpk/skypilot/README.md) it is a **public-image** step: there is no
`Dockerfile`. `setup` installs torch (`torch_package` from `torch_index_url`), then a
pinned unitxt plus `hf_packages` (transformers, accelerate, tabulate), into a `uv` venv
on the bare node. `unitxt_config.unitxt_image` optionally switches to a prebuilt image
instead. It is generated from the sources here by the shared Makefile conventions; see
[steps/README.md](../../../README.md).

## Layout

- `step-template.yaml`: the step. `setup` installs torch, then unitxt and
  `hf_packages`; `run`
  calls `unitxt-evaluate --model hf` with `pretrained=` set to
  `bindings.model.binding.path` and the rest of `unitxt_config`, then registers
  `output_path` as the `results` artifact.
- `test/test_unitxt_step_render.py`: Mode-1 render tests. They execute the
  rendered `run` block with a stub `unitxt-evaluate` on `PATH`, so they need no
  unitxt install, network or cluster.
- `test/slurm-cpu/` + `test-data/slurm-cpu/`: a build test on the local Docker
  SLURM cluster, modelled on dpk's `slurm-tok`. It evaluates the public
  `SmolLM2-135M-Instruct` (the bound `model` input) on 2 instances with the CPU
  torch build: a plumbing check, not a meaningful score. It skips itself unless
  `make slurm-setup` has brought the cluster up. Like other build tests it needs
  `GB_ENVIRONMENT=STANDALONE GBTEST_MODE=live` (which `make extended-tests` sets; the
  step's own `make test` doesn't, so the storage fixtures refuse to run):

  ```sh
  cd steps/eval/unitxt/skypilot
  GB_ENVIRONMENT=STANDALONE GBTEST_MODE=live PYTHONPATH=src \
    ../../../../.venv/bin/python -m pytest -v test/slurm-cpu
  ```

  `make publish-step` copies it to `test/steps/eval/unitxt/skypilot/slurm-cpu/`,
  where the nightly extended suite runs it against the published step.

## Building, testing and publishing

```sh
make -C steps/eval/unitxt/skypilot space         # render space/
make -C steps/eval/unitxt/skypilot test          # render, then run test/
make -C steps/eval/unitxt/skypilot publish-step  # promote into configurations/ + test/steps/
make -C steps/eval/unitxt/skypilot check-published  # diff the published trees against a fresh render
```

## Status

First cut for granite.build issue #235. Published to
`configurations/assets/environments/skypilot/steps/eval/unitxt/`. The only
per-cluster build test is the CPU `slurm-cpu` one, which passes on the local Docker
SLURM cluster. Planned follow-ups:

- A GPU build test on an internal GPU SLURM cluster, under `test/integration/ibm/`.
  Then Kubernetes, LSF and AWS.
- Remote models (`--model cross_provider`), probably as a separate value input.
- Handling for reasoning models such as granite-4.2 (`--chat_template_kwargs`).
