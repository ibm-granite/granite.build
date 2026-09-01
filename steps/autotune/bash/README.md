# autotune / bash — development notes

Authoring home for the `autotune` step's **bash** implementation. This file is
dev-oriented and is **not** published; `USAGE.md` is what ships as the released
step's `README.md`.

Released to `configurations/assets/environments/bash/steps/autotune/` by
`make publish-step`. That copy is a release artifact — edit it here, never there.

```
step-template.yaml            -> published step.yaml (only ${IMAGE_REF} is substituted)
USAGE.md                      -> published README.md (user docs)
bash_scripts/autotune/*.sh|py -> published verbatim path-wise (see SRC_DIR below)
test/local/                   -> test/steps/autotune/bash/local/
test-data/local/              -> test-data/steps/autotune/bash/local/
```

## `SRC_DIR := bash_scripts` — the one framework deviation

`common.mk` defaults `SRC_DIR` to `src`, publishing to `<step>/src/`. The bash and
docker launchers do not look there: `llmb_bash_jobsub.sh` resolves a launcher's
`script_path` relative to

```
LLMB_BASH_ASSET_BASH_SCRIPTS_DIR="${LLMB_BASH_ASSET_DIR}/bash_scripts/${STEP_FOLDER}"
```

so a bash/docker step's scripts *must* land at `<step>/bash_scripts/<name>/`.
autotune is the framework's first non-skypilot step, so this is the first time the
`src/` default has met that contract. The Makefile therefore points `SRC_DIR` at
`bash_scripts/` (assigned after the `include`, because `common.mk` uses a plain `=`).

**Consequence, and it is load-bearing:** the `.gbignore` that `publish-step` writes
lists `src/**`, which no longer matches, so these scripts *are* Jinja-rendered at
build time. autotune needs that — `command.sh` materializes the inline
`config.autotune-config` block via
`{{ config['autotune-config'] | to_yaml | b64encode }}`. `run.py` deliberately
contains no Jinja, and a unit test enforces it.

A docker variant of this step is deferred to the `feat/autotune-step-docker`
branch; the same `SRC_DIR` reasoning applies there.

If the framework would rather keep every step's sources verbatim, the alternative is
to move materialization out of `command.sh` into `step-template.yaml`'s launcher
`env` (as a base64 blob — the bash launcher does read `launcher_config.env`, see
`bash.py:120`) and teach `common.mk` the `bash_scripts/` layout. Worth deciding
before more bash/docker steps arrive.

## Loop

```sh
make space            # render a local Space into space/ (offline)
make publish-step     # promote into configurations/assets/...
make check-published  # verify the committed release matches this source
make test             # end-to-end: builds a venv, installs torch+ray, one LoRA pass
                      # (~2.5 min once the venv is warm; extended suite only)
```

Fast feedback lives in the repo's unit suite, which asserts this step's structure
and `run.py`'s behavior without any fm-tune install:

```sh
GB_ENVIRONMENT=STANDALONE GBTEST_MODE=mock \
  python -m pytest test/unit/builtins/steps/test_autotune.py -q
```
