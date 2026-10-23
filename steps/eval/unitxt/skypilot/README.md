# eval/unitxt (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `eval/unitxt` in a `build.yaml`. This file covers how the step is built, tested
> and published.

Evaluation step that runs the [unitxt](https://www.unitxt.ai) `unitxt-evaluate`
CLI as a black box, in `hf` mode, against the target's declared `model` input. Like
[byoc](../../../byoc/skypilot/README.md) it is a **public-image** step: there is no
`Dockerfile`, and it runs in a public image, `unitxt_config.image`
(`python:3.12-slim` by default; `""` runs on the bare node). `setup` installs torch
(`torch_package` from `torch_index_url`), then a pinned unitxt plus `hf_packages`
(transformers, accelerate, tabulate), into a `uv` venv inside that image, following
dpk's install pattern. `install: false` skips the install for an image that already
provides everything. It is generated from the sources here by the shared Makefile conventions; see
[steps/README.md](../../../README.md).

## Layout

- `step-template.yaml`: the step. `setup` installs torch, then unitxt and
  `hf_packages`; `run`
  calls `unitxt-evaluate --model hf` with `pretrained=` set to
  `bindings.model.binding.path` and the rest of `unitxt_config`, then registers
  `output_path` as the `results` artifact.
- `test/test_skypilot_eval_unitxt_render.py`: Mode-1 render tests. They execute the
  rendered `run` block with a stub `unitxt-evaluate` on `PATH`, so they need no
  unitxt install, network or cluster.
- `test/slurm-cpu/` + `test-data/slurm-cpu/`: a build test on the local Docker
  SLURM cluster, modelled on dpk's `slurm-tok`. It evaluates the public
  `SmolLM2-135M-Instruct` (the bound `model` input) on 2 instances with the CPU
  torch build: a plumbing check, not a meaningful score. It sets `image: ""`: the local
  cluster has no Pyxis plugin, so it runs on the bare node, as byoc's slurm fixture
  does. Its `results` output goes to
  a `test_`-prefixed Hugging Face dataset in `GBTEST_HF_PUSH_ORG` (default
  ibm-research). `HfPushCleanupMixin` (`test/libgbtest/hfpush.py`) names it per run,
  checks the push landed, and deletes it, so the queued hfpush needs an `HF_TOKEN`
  that can write (and delete in) that org; the nightly extended suite has one. It skips itself unless
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
- `test/slurm-bluevela/` + `test-data/slurm-bluevela/`: the GPU build test, on an
  internal GPU SLURM cluster through the in-repo environment
  `space://environments/skypilot/slurm/ibm-bluevela`. It evaluates
  `ibm-granite/granite-4.2-3b` on one H200 (`device=cuda`, PyPI's default CUDA torch)
  on 10 instances, and pushes `results` the same way as `slurm-cpu`. It is marked
  `ibm` and skipped in SPS CI/CD: the environment logs in with a key file,
  `~/.ssh/ibm-bluevela.key`, so it needs no secret manager, but CI has no key. Run it
  by hand from the IBM venv (`make ibm-venv`), with an `HF_TOKEN` that has write
  access through the `gbspace-public-staging` resource group (a read-only token fails
  the push with `Could not resolve resource group id`):

  ```sh
  make -C steps/eval/unitxt/skypilot space
  cd steps/eval/unitxt/skypilot
  GB_ENVIRONMENT=STANDALONE GBTEST_MODE=live GBTEST_STANDALONE_ENVIRONMENT=STAGING \
    PYTHONPATH=src ../../../../.venv/bin/python -m pytest -s -v test/slurm-bluevela
  ```

  The step job goes to the environment's default GPU partition. When your group's
  quota there is full (`squeue` shows the pending reason `QOSGrpGRES`), point a local
  copy of the build at a partition and QOS that have room, with `zone:` and
  `launcher_config.sbatch_options.qos:`. Don't commit those: they are specific to
  one group.

## Building, testing and publishing

```sh
make -C steps/eval/unitxt/skypilot space         # render space/
make -C steps/eval/unitxt/skypilot test          # render, then run test/
make -C steps/eval/unitxt/skypilot publish-step  # promote into configurations/ + test/steps/
make -C steps/eval/unitxt/skypilot check-published  # diff the published trees against a fresh render
```

## Status

First cut for granite.build issue #235. Published to
`configurations/assets/environments/skypilot/steps/eval/unitxt/`. Two build tests:
the CPU `slurm-cpu` one, which passes on the local Docker SLURM cluster, and the GPU
`slurm-bluevela` one, which passes on the internal GPU SLURM cluster (one H200, 13
minutes end to end). Planned follow-ups:

- Kubernetes, LSF and AWS build tests.
- Remote models (`--model cross_provider`), probably as a separate value input.
- Handling for reasoning models such as granite-4.2 (`--chat_template_kwargs`).
