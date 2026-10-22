# eval/lm-eval (SkyPilot): development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `eval/lm-eval` in a `build.yaml` (config contract, inputs and outputs, an example).
> This file covers how the step is *generated, tested, and published*.

A public-image step: it runs EleutherAI's `lm_eval` in `python:3.12-slim` (or on the
bare launcher node), installing `lm-eval` at setup time. There's no `Dockerfile`, so the
`image`/`publish-image` targets are no-ops. For the shared targets and variables, see
[steps/README.md](../../../README.md).

| | |
|---|---|
| `STEP_NAME` | `lm-eval` |
| `STEP_PATH` / URI | `eval/lm-eval` / `space://steps/eval/lm-eval` |
| Config key | `lm_eval_config` |
| Launcher | `lm_eval` |
| Published to | `configurations/assets/environments/skypilot/steps/eval/lm-eval/` |

## Layout

| Path | What it is |
|---|---|
| `step-template.yaml` | The step. `setup` and `run` compute values in Jinja and call the scripts below. |
| `src/lm_eval_guard.sh` | Config refusals, run at the top of both `setup` and `run`, so a bad config fails before the install. |
| `src/lm_eval_setup.sh` | uv bootstrap, `./venv`, then two install phases: torch from `torch_index_url`, then `lm-eval[extras]` from `pip_index_url`. |
| `src/lm_eval_run.sh` | Runs `lm_eval run`, checks a results file exists, then emits the metadata and artifact markers. |
| `test/test_skypilot_eval_lm_eval_render.py` | Render tests (Mode 1 only, no cluster, no network). |
| `test/slurm-cpu/`, `test-data/slurm-cpu/` | CPU build test on the local Docker SLURM cluster. |

## Running the tests

```sh
make -C steps/eval/lm-eval/skypilot test-setup     # once: the local Docker SLURM cluster
make -C steps/eval/lm-eval/skypilot space          # render space/
```

**Render tests.** They execute the rendered `setup` and `run` blocks in a scratch dir with
the real `src/` scripts and stubs for `lm_eval`, `pip` and `uv`, and assert on the exact
argv, the refusals, quoting, the two install phases and the marker:

```sh
cd steps/eval/lm-eval/skypilot
../../../../.venv/bin/python -m pytest test/test_skypilot_eval_lm_eval_render.py
```

**SLURM build test (`slurm-cpu`).** Uses `aime25` with `--limit 1` against
`hf-internal-testing/tiny-random-gpt2`, on the bare node (the local cluster has no Pyxis),
with CPU torch. The results go to an `hf://` dataset,
`<org>/test_lm-eval-results_<run>`: the test class uses `HfPushCleanupMixin`
(`test/libgbtest/hfpush.py`), which fills the build.yaml's `__GBTEST_HF_PUSH_ORG__`
(default `ibm-research`, or `GBTEST_HF_PUSH_ORG`) and `__GBTEST_HF_PUSH_RUN__`
placeholders, asserts the dataset reached the Hub, then deletes it. So it needs an
`HF_TOKEN` that can write, and delete, in that org. In Mode 1 the push needs two more
settings that Mode 2 gets for free:

- **Space name `public`.** The push's HF resource group is derived from the space name
  (`gbspace-<space>`), and `make space` names the space after `STEP_NAME` by default,
  giving a group that doesn't exist.
- **`GBTEST_STANDALONE_ENVIRONMENT=STAGING`,** which points a standalone push at the
  test group (`gbspace-public-staging`). `test/conftest.py` sets it, but pytest only
  loads that conftest for tests under the repo's `test/`, not for this directory.

`make test` doesn't set `GB_ENVIRONMENT` either, so run it like this:

```sh
make -C steps/eval/lm-eval/skypilot space SPACE_NAME=public
cd steps/eval/lm-eval/skypilot
GB_ENVIRONMENT=STANDALONE GBTEST_MODE=live GBTEST_STANDALONE_ENVIRONMENT=STAGING \
  PYTHONPATH=src ../../../../.venv/bin/python -m pytest -v -s test/slurm-cpu
```

In Mode 2 (`test/steps/…`) the space is `configurations/spaces/local`, named `public`, and
the conftest loads, so only `GB_ENVIRONMENT` and `GBTEST_MODE` are needed.

It expects 3 steps (hfpull, lm-eval, hfpush), one input and one output artifact, checks
the `lm_eval_version` metadata, and checks that the results dataset exists on the Hub. The step's own log (score table, marker) is in
`~/sky_logs/<sky-cluster>/.../tasks/run.log` on the host running gbserver.

## Publishing

```sh
make -C steps/eval/lm-eval/skypilot publish-step
make -C steps/eval/lm-eval/skypilot check-published
```

Then run the published test from `test/steps/eval/lm-eval/skypilot/slurm-cpu/` with the
same environment variables, plus `test/unit/asset/` (the step is in its `EXPECTED`
inventory) and `make quick-tests`. Commit the generated trees with the source.

## Known gaps

- **No committed GPU fixture.** The step was proven on one H200 on an internal SkyPilot
  SLURM cluster with a plain build, not a build test: aime25 at `limit: 5`,
  exact_match 0.6. The build is the example in USAGE.md. It needed host memory in
  `launcher_config.resources`: the default 16 GB was OOM-killed on long generations,
  and the run that passed asked for 128 GB and peaked at about 28 GB (`sacct` MaxRSS).
- **No image-mode cluster run.** The local SLURM cluster has no Pyxis, and the GPU proof
  ran on the bare node, so the default `python:3.12-slim` path is covered only by the
  render tests.
