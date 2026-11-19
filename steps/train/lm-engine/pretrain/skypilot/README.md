# lm-engine-pretrain (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `lm-engine-pretrain` in a `build.yaml`. This file covers how the step is
> *developed, tested and published*.

The SkyPilot env (kubernetes subtype: Vela) of `space://steps/train/lm-engine/pretrain`:
one lm-engine pretraining job from scratch. The framework is described in
[steps/README.md](../../../../README.md).

**This directory is a scaffold.** The config schema in `step-template.yaml` is final;
the launcher's `run:` block is a stub that exits non-zero, and `src/` does not exist
yet. Do not publish the step until both are real.

## Layout

```
step-template.yaml   the step: the config schema and the SkyPilot launcher
.gbignore            files gbserver ships without Jinja-rendering (docs, src/)
src/                 (not yet) the step's Python helpers and entry point,
                     file-mounted into the run
test/                Mode-1 tests of the template's contract
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

`GB_ENVIRONMENT` must be set (unset means `PROD`, and the suite refuses to run), and a
throwaway `GB_HOME_DIR` keeps the tests from touching the live space. Do not collect
two steps' `test/` directories in one pytest run: their `test_step_template.py`
basenames collide.
