# lm-engine-pretrain (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `lm-engine-pretrain` in a `build.yaml`. This file covers how the step is
> *developed, tested and published*.

The SkyPilot env (kubernetes subtype: Vela) of `space://steps/train/lm-engine/pretrain`:
one lm-engine pretraining job from scratch. The framework is described in
[steps/README.md](../../../../README.md).

The launcher's `run:` block exports `config` as the environment
[`src/run_pretrain.sh`](src/run_pretrain.sh) documents and hands it the job; the
script owns every decision from there. Its handshake files (the per-attempt resume
decision, the per-build progress baseline, the watcher's stop file) live under
`config.train_dir`, and the reasons for each lifetime are in the script's comments.

## Layout

```
step-template.yaml   the step: the config schema and the SkyPilot launcher
.gbignore            files gbserver ships without Jinja-rendering (docs, src/)
src/                 the step's Python helpers and entry point (run_pretrain.sh),
                     file-mounted into the run
test/                Mode-1 tests: the helpers, the template's contract, the
                     rendered run block (golden_run_scripts.json) and the entry
                     point's behaviour under stubbed python/torchrun
USAGE.md             user docs; published as the released step's README.md
```

## No image is built here

No `Dockerfile` sits next to the `Makefile`, so `common.mk` treats this as a
non-image step and `make image` / `make publish-image` are no-ops. lm-engine and its
dependencies live in the family image built one level up by
`steps/train/lm-engine/Makefile`, shared with the sibling `lm-engine-export` step. A
run picks it at launch from `config.image`, the way the `steps/eval/` family steps do.

## Nesting

This is a four-level step (`steps/train/lm-engine/pretrain/skypilot`), so the
`Makefile` includes `../../../../common.mk`. `common.mk` derives
`STEP_PATH=train/lm-engine/pretrain` from `CURDIR`, which is what `space://steps/...`
and the publish destinations use; `STEP_NAME` is only the logical name in log
messages.

## Generating and publishing

- `make space` renders a self-contained Space into `space/`.
- `make publish-step` promotes the step into
  `configurations/assets/environments/skypilot/steps/train/lm-engine/pretrain/`.
- `make check-published` re-renders and diffs against the committed trees. CI does not
  run it, so run it after any change to the template, `src/` or `USAGE.md`.

## Running the tests

```
cd steps/train/lm-engine/pretrain/skypilot
GB_ENVIRONMENT=STANDALONE GB_HOME_DIR=$(mktemp -d) ../../../../../.venv/bin/pytest test -q
```

Regenerate the rendered-`run:` goldens after a deliberate template change:

```
GB_ENVIRONMENT=STANDALONE ../../../../../.venv/bin/python test/golden_run_scripts.py
```

`GB_ENVIRONMENT` must be set (unset means `PROD`, and the suite refuses to run), and a
throwaway `GB_HOME_DIR` keeps the tests from touching the live space. Do not collect
two steps' `test/` directories in one pytest run: their `test_step_template.py`
basenames collide.
