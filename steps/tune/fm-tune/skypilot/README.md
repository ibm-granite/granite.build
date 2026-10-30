# fm-tune (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `fm-tune` in a `build.yaml` (config, inputs/outputs, examples). This file covers how
> the SkyPilot env of the step is *tested and published*.

The SkyPilot env (slurm subtype only) of `space://steps/tune/fm-tune`: one fm-tune
(`autotunex/src/fm-tune/main.py`) run on a bare Slurm node, in a venv the run script builds
itself. No `Dockerfile`. The bash env is in [`../bash/`](../bash/README.md). What both envs
share (the run script, the render-time checks, the `FMTUNE_*` contract) is in the step's
[developer notes](../README.md). The framework is described in
[steps/README.md](../../../README.md).

## Layout

```
step-template.yaml   the step: maps fm_tune_config and the bindings to FMTUNE_* env
.gbignore            files gbserver ships without Jinja-rendering (docs, src/)
src/fmtune_run.sh    the run script, byte-identical to ../bash/src/fmtune_run.sh
test/                Mode-1 render and shared-check tests at the root (see below)
  slurm/             the real build test
test-data/slurm/     its build.yaml + buildtest.yaml
USAGE.md             user docs; published as the released step's README.md
```

## What differs on SkyPilot

- **Validation in `run:`.** Each launcher-config string is filled on its own, and `run:` is
  also what a warm re-run renders, so the `validate` macro sits in `run:`. After it come the
  SkyPilot-only checks: `num_nodes` must be 1, and a GPU count needs
  `launcher_config.resources.accelerators`, since gbserver doesn't request GPUs from
  `compute_config` (issue #536). See [Render-time checks](../README.md#render-time-checks).
- **Cache path.** The template sets
  `FMTUNE_CACHE_DIR="${GB_SHARED_WORKDIR:-$PWD}/.fm-tune-cache-$(id -u)"`: per user, with no
  shared parent directory, so a second user never hits the first user's permissions.
- **Logs.** SkyPilot builds write no local `job.log`; read
  `~/sky_logs/<cluster>/1-<cluster>/run.log`.

## Generating and publishing

- `make space` renders a self-contained Space into `space/`. The build test runs against it.
- `make publish-step` promotes the step into
  `configurations/assets/environments/skypilot/steps/tune/fm-tune/` (`step.yaml`, `src/`, and
  `USAGE.md` as `README.md`) and copies the build test into
  `test/steps/tune/fm-tune/skypilot/` and its fixtures into
  `test-data/steps/tune/fm-tune/skypilot/`.
- `make check-published` re-renders and diffs against the committed trees. CI does not run
  it, so run it after any change to the template, `src/`, `USAGE.md` or the build test.

`src/fmtune_run.sh` is shared with the bash env: edit it in both, then publish both.

## Running the tests

- **Mode 1 (fast, no build):** `test_skypilot_tune_fm_tune_render.py`, and
  `test_fm_tune_script_copies_match.py`, which checks what both envs share: the script copies
  match, the `validate` macro is the same text, and one table of bad values fails both
  renders and the script. The `fmtune_run.sh` behaviour tests are in `../bash/test/`. Run
  them by file:

  ```
  GB_ENVIRONMENT=STANDALONE .venv/bin/python -m pytest steps/tune/fm-tune/skypilot/test/test_fm_tune_script_copies_match.py
  ```

- **Build test:** `test/slurm/test_skypilot_slurm_tune_fm_tune.py`, a real 1-epoch LoRA run on
  the local Docker/podman Slurm cluster (CPU-only arm64 nodes; it proves the wiring, not GPU
  training). It is `@extended_testing_only` and `skypilot_integration`, and it skips unless
  the cluster is reachable. Bring the cluster up once with `make test-setup` (the repo-root
  `slurm-setup`), then run `make -C steps/tune/fm-tune/skypilot test`. It needs a venv with
  the `thirdparty` extra (the SkyPilot fork).

`make test` and `pytest` on the `test/` directory both collect the build test. Name test files
to run only the Mode-1 tests. See the [RAM warning](../README.md#ram-warning) and
[test-setup notes](../README.md#test-setup-notes) before running the build test, and stop the
Slurm VM when you're done.
