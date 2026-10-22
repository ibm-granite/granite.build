# nemo-rl (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `nemo-rl` in a `build.yaml` (config contract, inputs/outputs, examples). This file
> covers how the step is *built, tested, and published*, and why it is shaped the
> way it is.

A reusable NeMo-RL training step: one step serves every NeMo-RL algorithm (SFT, DPO,
GRPO, PPO, RM), selected by `nemo_rl_config.algorithm` + `config_name`. It runs
inside the public NGC `nemo-rl` release image, so it has no `Dockerfile`. It is
referenced as `space://steps/train/nemo-rl` (the URI follows the nested
`steps/train/nemo-rl` authoring path) and is tested end to end on two clusters, AWS
and IBM BlueVela SLURM. The framework it follows is described in
[steps/README.md](../../../README.md).

## Layout

```
step-template.yaml   the step; renders `${IMAGE_REF}`-free (public-image step)
.gbignore            files gbserver ships without Jinja-rendering (docs, src/)
src/                 shipped with the step (file_mounts), run on the node
  nemo_rl_guard.sh   pre-flight checks on the staged binding paths
  nemo_rl_setup.sh   verify the image carries the NeMo-RL tree; prep dirs
  nemo_rl_run.sh     assemble Hydra argv, train, convert, push, emit marker
  clearml_launch.py  run the driver in-process under ClearML Task.init()
test/                unit tests at the root; one dir per cluster e2e test
  aws-sft/           SFT on a real EC2 GPU instance
  slurm-sft/         SFT on a real BlueVela SLURM GPU node
test-data/           build.yaml exemplar + one fixture dir per cluster test
USAGE.md             user docs; published as the released step's README.md
```

**Why the shell lives in `src/`, not inline in `step-template.yaml`.** The template
only computes *values* with Jinja (algorithm, paths, the overrides map) and passes
them as argv; the scripts do the shell. That keeps every line of shell `bash -n`-able
and unit-testable instead of only renderable. The step's `.gbignore` must list
`src/**` because gbserver runs a strict Jinja pass over the whole merged step
directory, and `nemo_rl_guard.sh` contains literal `{{`/`{%` (its
unrendered-binding detector).

## Generating and publishing the step

No `Dockerfile`, so `image`/`publish-image` are no-ops. The targets that matter:

- `make space` — render a self-contained Space into `space/` (references the step
  as `space://steps/train/nemo-rl`). This is what the cluster tests run against.
- `make publish-step` — promote the step into the committed release trees:
  `configurations/assets/environments/skypilot/steps/train/nemo-rl/` (rendered
  `step.yaml`, `src/`, and `USAGE.md` as `README.md`), plus the cluster tests into
  `test/steps/train/nemo-rl/skypilot/` and their fixtures into `test-data/steps/...`.
- `make check-published` — re-render and diff against the committed trees. CI does
  not run it, so run it after any change to the template, `src/`, `USAGE.md`, or a
  cluster test.

Publishing also requires the step to be listed in the step-inputs inventory in
`test/unit/asset/test_step_inputs_declared.py` (`base_model` required, `dataset`
optional).

## Running the tests

```
make -C steps/train/nemo-rl/skypilot test   # unit tests + both cluster tests
```

The **unit tests** at `test/`'s root need no cluster. They fake `uv` (the node's
only entrypoint into the image) so they can pin the exact argv the step emits:

| Test | Pins |
|---|---|
| `test_nemo_rl_run_sh.py` | Hydra override order, a fresh checkpoint dir per run, `output_path` and the staged, re-run-safe conversion, converter argv, `hf upload` argv (private by default), and the artifact marker against the shipped monitor's regexes |
| `test_v2_save_format.py` | the V2-only `torch_save` override, resolved in the driver's process through NeMo-RL's own loader (config + overrides) |
| `test_nemo_rl_guard_sh.py` | binding-path rejections on the node (empty or unrendered `base_model`/`dataset`, a dataset without its key) |
| `test_nemo_rl_step_render.py` | the rendered template: render-time config validation (missing keys, bad `output`, `num_nodes != 1`), accelerators, image ref (including `.sqsh`), guard/run argv |
| `test_exemplar_build.py` | every shipped build.yaml (exemplar and both cluster fixtures) against the step contract |
| `test_clearml_launch.py` | ClearML enablement (`auto`/`on`/`off`), and that the driver runs as `python examples/run_x.py` would (argv, `sys.path[0]`) |
| `test_clearml_rule_parity.py` | the bash install rule (`uv --with clearml`) and the Python import rule agree on every mode × credential combination |
| `test_aws_sft_skip_gate.py` | aws-sft skips unless a Hub-writable `HF_TOKEN` will reach the build |

The **cluster tests** are marked `extended` and skip unless their backend is
available. `make test` and a plain `pytest test/` do **not** deselect `extended`, so
with AWS credentials (or the BlueVela key) present they provision real resources.
To run only the unit tests, use `pytest test/ -m "not extended"`.

### aws-sft (real EC2)

Skips unless `AWS_PROFILE` or an AWS key pair is exported, and unless the Hub push
can succeed (see the last bullet). To run it:

- `AWS_PROFILE=gb-skypilot`, with the `GB_AWS_*` secrets stored for **the generated
  space**: `~/.granite.build/space_secrets/public/` (`make space` names it `public`,
  common.mk's `SPACE_NAME` default). Files placed directly in `space_secrets/` are
  ignored. Runbook: `docs/environments/skypilot-aws.md`.
- `GB_ENVIRONMENT=STANDALONE`, as `make extended-tests` sets. Under the default
  `DEV`, gbserver records lineage to WandB, which is not configured here, and the
  test fails with "created 0 JobStats".
- For the Hub push: a write-capable `HF_TOKEN` plus
  `GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE=true`. Without both the test skips: the test
  setup would replace the token with a read-only one, and the push would fail with
  403 after the whole training run. The skip cannot check write access itself (no
  network at collection time), so a read-only token still fails at the push.

### slurm-sft (real BlueVela SLURM)

Marked `ibm`. Skips unless the BlueVela key for the shared `granitebuild` functional
ID is at `~/.ssh/ibm-bluevela.key` (mode 600, no passphrase), and skips on SPS CI. It
runs against `space://environments/skypilot/slurm/ibm-bluevela` and needs:

- The prebuilt image at the path in `test-data/slurm-sft/build.yaml`, readable by
  `granitebuild` (see "SLURM needs a prebuilt image" below).
- `GB_ENVIRONMENT=STANDALONE`, for the same lineage reason as aws-sft.

## Why the step looks the way it does

Each of these was found by a real-cluster run; the tests that pin them are named so a
change that undoes one fails fast instead of on a GPU node.

- **`torch_save` only for V2 workers.** NeMo-RL's V2 DTensor worker (`_v2: true`, the
  default for SFT/GRPO/PPO) saves sharded HF safetensors with no DCP `.metadata`, and
  `convert_dcp_to_hf.py` requires one. `nemo_rl_run.sh` adds
  `++checkpointing.model_save_format=torch_save` only for a V2 run; V1 workers
  (DPO/RM) reject a non-null `model_save_format` and already write DCP. `++` is
  needed because the key is absent and Hydra runs in struct mode. V2 vs V1 is
  decided by `clearml_launch.py --v2-save-format`, which resolves the config with
  NeMo-RL's own loader (`defaults:` inheritance plus the Hydra overrides) in the
  driver's own process, just before running it: recipe configs inherit `_v2` from a
  parent and a user may override it, and a text grep of the named file misses both.
  Resolving in-process costs no extra interpreter start (slow on SLURM's `.sqsh`
  container fs), and a bad override fails there with the loader's own traceback.
- **Convert into a staging dir, then move.** The converter raises `FileExistsError`
  if `--hf-ckpt-path` exists, so it writes `<output_path>.partial` (cleared first),
  which replaces `output_path` once non-empty. A re-run of the run phase therefore
  replaces the previous model instead of failing on it.
- **A fresh checkpoint dir per run.** `results/<algorithm>/` is removed before
  training. The working dir survives on a warm cluster, and both NeMo-RL's
  auto-resume and checkpoint discovery pick the highest `step_N`, so an earlier
  run's `step_20` would otherwise be resumed from, or registered instead of a later
  run's `step_3`.
- **Config errors fail at step creation.** The launcher prepends the inline
  `hf download` to `setup:`, so a script there runs only after the node is up and
  the base model is downloaded. Config-only checks are `raise_error`s in the
  template's `validate` macro instead, which fail the strict launcher-config fill.
  `nemo_rl_guard.sh` keeps only what is known on the node: the staged binding paths.
- **`hf upload`, not `huggingface-cli`.** The image's `huggingface_hub` removed the old
  CLI.
- **wandb off by default.** The shipped configs default `logger.wandb_enabled: true`,
  which aborts with "No API key configured". The step injects `false` *before* the
  user's overrides, so a user who brings a key can turn it back on.
- **The artifact marker is pinned to the monitor.** A unit test loads
  `src/gbserver/builtins/monitors/skypilot/monitor.yaml` and checks its regexes
  extract the id and path from the script's real output. A prefix the monitor does
  not match registers zero artifacts while the build still reports SUCCESS.
- **GPUs come from the step.** gbserver's SkyPilot launcher does not source GPUs from
  `compute_config`, so the template renders `accelerators: <gpu_model>:<count>` and
  pins NeMo-RL's `cluster.gpus_per_node` to the same count.
- **SLURM needs a prebuilt image.** Pulling the NGC image onto a cold BlueVela node
  took 73 minutes, longer than a gpu-mid job's 2-hour preemption protection can
  safely absorb on top of training. `slurm-sft` therefore points `image_id` at a
  prebuilt `.sqsh`. It stays a plain path: the template adds `docker:`, which
  SkyPilot's SLURM cloud requires before it will read an image, and SkyPilot then
  passes a `.sqsh` path to Pyxis as-is (`test_nemo_rl_step_render.py` pins both
  halves).

## The two cluster tests, side by side

`aws-sft` and `slurm-sft` run the same SFT smoke test, so their fixtures differ only
where the cluster forces it:

| | aws-sft | slurm-sft |
|---|---|---|
| environment | `skypilot/aws` | `skypilot/slurm/ibm-bluevela` |
| image | NGC registry ref (pulled at launch) | prebuilt `.sqsh` on shared /proj |
| GPU | `A10G:1` | `H200:1` (the environment pins `h200`) |
| `step_count` | 1 (aws stages `hf://` inline) | 2 (separate hfpull step) |
| Hub push | yes (`hf_output_repo`) | no |
