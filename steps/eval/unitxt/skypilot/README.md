# eval/unitxt (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `eval/unitxt` in a `build.yaml`. This file covers how the step is built, tested
> and published.

Evaluation step that runs the [unitxt](https://www.unitxt.ai) `unitxt-evaluate`
CLI as a black box, in `hf` mode, against the target's declared `model` input. Like
[dpk](../../../dpk/skypilot/README.md) it is a **public-image** step: there is no
`Dockerfile`, `setup` installs a pinned unitxt plus `hf_packages` (torch,
transformers, accelerate, tabulate) into a `uv` venv on the bare node, and
`unitxt_config.unitxt_image` optionally switches to a prebuilt image instead. It is generated from the sources here by the
shared Makefile conventions; see [steps/README.md](../../../README.md).

## Layout

- `step-template.yaml`: the step. `setup` installs unitxt and `hf_packages`; `run`
  calls `unitxt-evaluate --model hf` with `pretrained=` set to
  `bindings.model.binding.path` and the rest of `unitxt_config`, then registers
  `output_path` as the `results` artifact.
- `test/test_unitxt_step_render.py`: Mode-1 render tests. They execute the
  rendered `run` block with a stub `unitxt-evaluate` on `PATH`, so they need no
  unitxt install, network or cluster.

## Building, testing and publishing

```sh
make -C steps/eval/unitxt/skypilot space         # render space/
make -C steps/eval/unitxt/skypilot test          # render, then run test/
make -C steps/eval/unitxt/skypilot publish-step  # promote into configurations/ (not yet done)
```

## Status

First cut for granite.build issue #235. Not yet published to `configurations/`,
and there is no per-cluster build test yet. Planned follow-ups:

- An AWS build test (`test/aws-*/`), then SLURM, Kubernetes and LSF.
- Remote models (`--model cross_provider`), probably as a separate value input.
- Handling for reasoning models such as granite-4.2 (`--chat_template_kwargs`).
