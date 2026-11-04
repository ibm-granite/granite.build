# fm-tune (bash) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `fm-tune` in a `build.yaml` (config, inputs/outputs, examples). This file covers how
> the bash env of the step is *tested and published*.

The bash env of `space://steps/tune/fm-tune`: one fm-tune (`autotunex/src/fm-tune/main.py`)
run on the gbserver host, in a venv the run script builds itself. No `Dockerfile`. The
SkyPilot env is in [`../skypilot/`](../skypilot/README.md). What both envs share (the run
script, the render-time checks, the `FMTUNE_*` contract) is in the step's
[developer notes](../README.md). The framework is described in
[steps/README.md](../../../README.md).

## Layout

```
step-template.yaml   the step: maps fm_tune_config and the bindings to FMTUNE_* env
.gbignore            files gbserver ships without Jinja-rendering (docs, src/)
src/fmtune_run.sh    the run script, byte-identical to ../skypilot/src/fmtune_run.sh
test/                Mode-1 unit and render tests at the root (see below)
  bash/              the real build test
test-data/bash/      its build.yaml + buildtest.yaml
USAGE.md             user docs; published as the released step's README.md
```

## What differs on bash

- **Validation key.** Bash env values are separate strings, so the `validate` macro has a
  launcher-config key of its own, `validate:`, which renders to `""` and is read by nothing
  (see [Render-time checks](../README.md#render-time-checks)).
- **Output and cache paths.** Bash launcher env values reach the child unexpanded, so the
  template leaves `FMTUNE_OUTPUT_DIR` and `FMTUNE_CACHE_DIR` empty and the script derives them
  from the `LLMB_BASH_*` builtins: the model goes to `$LLMB_BASH_OUTPUT_DIR/model/`, the cache
  to `<GB_HOME>/.fm-tune-cache`.
- **Artifact scan off.** The template sets `config.bash.skip_finding_output_artifacts: true`,
  so only the script prints `GB_ARTIFACT_ID:model`, and only on success (see
  [Bash output-subdir caveat](../README.md#bash-output-subdir-caveat)).

## Generating and publishing

- `make space` renders a self-contained Space into `space/`. The build test runs against it.
- `make publish-step` promotes the step into
  `configurations/assets/environments/bash/steps/tune/fm-tune/` (`step.yaml`, `src/`, and
  `USAGE.md` as `README.md`) and copies the build test into `test/steps/tune/fm-tune/bash/`
  and its fixtures into `test-data/steps/tune/fm-tune/bash/`.
- `make check-published` re-renders and diffs against the committed trees. CI does not run
  it, so run it after any change to the template, `src/`, `USAGE.md` or the build test.

`src/fmtune_run.sh` is shared with the SkyPilot env: edit it in both, then publish both.

## Running the tests

- **Mode 1 (fast, no build):** the `fmtune_run.sh` tests (`test_fmtune_*_sh.py`, against a stub
  `main.py`) and `test_bash_tune_fm_tune_render.py`. The checks both envs share are in
  `../skypilot/test/test_fm_tune_script_copies_match.py`. Run them by file:

  ```
  GB_ENVIRONMENT=STANDALONE .venv/bin/python -m pytest steps/tune/fm-tune/bash/test/test_fmtune_run_sh.py
  ```

- **Build test:** `test/bash/test_bash_bash_tune_fm_tune.py`, a real 1-epoch LoRA run on a tiny
  random Llama with `alpaca_100_test`, fm-tune cloned from upstream at a pinned SHA. It is
  `@extended_testing_only`, so the quick/PR suites skip it; the nightly extended suite runs
  its published copy under `test/steps/`. Run it locally with
  `make -C steps/tune/fm-tune/bash test`, which renders the Space first. Without `make space`
  the build goes INVALID with "No space.yaml found".

`make test` and `pytest` on the `test/` directory both collect the build test and start a real
build. Name test files to run only the Mode-1 tests. See the
[RAM warning](../README.md#ram-warning) and
[test-setup notes](../README.md#test-setup-notes) before running the build test.
