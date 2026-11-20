# lm-engine-export (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure
> `lm-engine-export` in a `build.yaml`. This file covers how the step is *developed,
> tested and published*.

The SkyPilot env (kubernetes subtype: Vela) of `space://steps/train/lm-engine/export`:
one lm-engine checkpoint rung converted to a Hugging Face model directory. The framework
is described in [steps/README.md](../../../../README.md).

## Layout

```
step-template.yaml   the step: the config schema and the SkyPilot launcher
.gbignore            files gbserver ships without Jinja-rendering (docs, src/)
src/run_export.sh    the entry point: config as environment -> one export.py invocation
src/export.py        the export driver: the five stages, the manifest, the artifact line
src/export_checks.py the post-export verification: config and tokenizer checks
test/                Mode-1 tests of the template, the entry point and the driver
USAGE.md             user docs; published as the released step's README.md
```

The launcher holds no logic beyond turning `config` into the environment
`src/run_export.sh` documents and two render-time refusals a recipe can only get wrong
once (a missing required path, an unsupported `model_type`). The script assembles the
`export.py` argv, extends `PYTHONPATH` with its own directory so the mounted
`export.py`/`export_checks.py` and the image's `tools.model_conversion` at `/app/src` are
both importable, and hands back the driver's exit code unchanged. It runs no pipeline
around the driver: `export.py` prints the one `GB_ARTIFACT_ID:hf_model` line itself, and a
`tee` could duplicate it while a `pipefail` slip could swallow the non-zero that keeps an
unfaithful export unpublished.

**An optional vLLM serve-and-score check is deferred.** The design has it as off-by-default
future work — load the exported directory in vLLM and compare a greedy sample — and an
earlier draft of the schema declared a `vllm_check` key for it. Nothing read that key, so
it was removed rather than left as a flag a recipe could set and believe: a green target
would then claim a vLLM load that never happened. Reintroduce the key with the stage, not
before.

## What the export does

Two stages of upstream code, both in the family image:

1. `python -m lm_engine.training.unshard --config <cfg>` turns the sharded checkpoint
   into one unsharded lm-engine `gpt_base` model. transformers and vLLM cannot load it.
2. `tools.model_conversion.export_to_huggingface(unsharded, model_type, save_path)`
   rewrites it as a `GraniteConfig` / `LlamaConfig` / `GraniteMoeHybridConfig` model.

Both stages can succeed and still produce a model that is not the one that was trained,
which is what `src/export_checks.py` is for. Its module docstring names the incident
behind each check — a dropped `rope_theta`, a re-saved tokenizer, a collapsed muP
multiplier — and is the place to read before changing a check.

The tokenizer is **copied** (`shutil.copy2`), never re-saved: upstream's exporter calls
`tokenizer.save_pretrained(...)`, which has leaked the trainer's runtime truncation
settings into `tokenizer.json`, and when it cannot load a tokenizer it writes none at
all.

## No image is built here

No `Dockerfile` sits next to the `Makefile`, so `common.mk` treats this as a non-image
step and `make image` / `make publish-image` are no-ops. lm-engine and its dependencies
live in the family image built one level up by `steps/train/lm-engine/Makefile`, shared
with the sibling `lm-engine-pretrain` step. A run picks it at launch from `config.image`.

## Nesting

This is a four-level step (`steps/train/lm-engine/export/skypilot`), so the `Makefile`
includes `../../../../common.mk`. `common.mk` derives `STEP_PATH=train/lm-engine/export`
from `CURDIR`, which is what `space://steps/...` and the publish destinations use;
`STEP_NAME` is only the logical name in log messages.

## Generating and publishing

- `make space` renders a self-contained Space into `space/`.
- `make publish-step` promotes the step into
  `configurations/assets/environments/skypilot/steps/train/lm-engine/export/`.
- `make check-published` re-renders and diffs against the committed trees. CI does not
  run it, so run it after any change to the template, `src/` or `USAGE.md`.

## Running the tests

```
cd steps/train/lm-engine/export/skypilot
GB_ENVIRONMENT=STANDALONE GB_HOME_DIR=$(mktemp -d) ../../../../../.venv/bin/pytest test -q
```

`GB_ENVIRONMENT` must be set (unset means `PROD`, and the suite refuses to run), and a
throwaway `GB_HOME_DIR` keeps the tests from touching the live space. Do not collect two
steps' `test/` directories in one pytest run: their `test_step_template.py` basenames
collide.
