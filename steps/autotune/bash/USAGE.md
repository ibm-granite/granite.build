# autotune (bash) — AutoTune / fm-tune HPO + training step

Runs fm-tune's `main.py` on the standalone **bash** environment. Materializes an
inline `config.autotune-config` block (or the `hpo_config` input) to a YAML file,
then runs the tuning pipeline against a local fm-tune checkout.

## Inputs
- `model` (model, uri|binding) — HF id / Local / PVC; resolved to `$LLMB_BASH_INPUT_MODEL`.
- `dataset_files` (dataset, uri|binding) — a fileset with `*_train.jsonl` + `*_validation.jsonl`.
- `hpo_config` (fileset, optional) — used only if no inline `config.autotune-config`.
  May bind either a single YAML file or a directory containing exactly one
  `.yaml`/`.yml`.

## Output
- `custom` (model) — the tuned-model output dir, registered via the `GB_ARTIFACT_ID`
  marker that `run.py` prints on success only.

Note: fm-tune's single-GPU drivers save under `<output_dir>/models/`, while the
multi-GPU drivers save directly to `<output_dir>`; the marker registers
`$LLMB_BASH_OUTPUT_DIR` either way, so a consumer binding this as a model may need
to descend one level. The materialized tuning config is written *outside* this dir
so it is not published with the model.

## Key params (`config.bash.env`)
| Param | Meaning | Default |
|---|---|---|
| `FM_TUNE_ROOT` | fm-tune source: a local checkout path **or** a git remote (`ssh`/`https`/`*.git`). A remote is cloned once into the venv-base dir and used as the checkout (private repos need git creds on the runner). Required. | — |
| `FM_TUNE_REF` | Branch or tag to clone when `FM_TUNE_ROOT` is a git remote (`git clone --depth 1 --branch`). The clone dir is keyed on this value and refreshed on reuse, so changing it does not silently reuse an older ref. | default branch |
| `FM_TUNE_EXTRA` | pip extra installed into the venv — `core` (ray+datasets) or `full` (adds verl/vllm/flash-attn); empty = base only. `main.py` needs `ray`, which is only in these extras. | `core` |
| `BACKEND` | Runtime `main.py --backend`: `mlx` (Apple Silicon) or `torch`. Note: this is a runtime flag, not a pip extra — deps come from `FM_TUNE_EXTRA`. | `torch` |
| `NO_AUTOTUNE` | Skip HPO, single training run | `false` |
| `CLEANUP` / `SAVE_HISTORY` | Pass-through flags | `false` |
| `RUN_NAME` / `OUTPUT_MODEL_NAME` | Run / output-model name | `$JOB_ID` |
| `TUNING_ALGO` / `RL_ALGO` | Override; else read from `training_config.tuning_algorithm`/`rl_algorithm` in the tuning config. Those keys exist in AutoTuneX-generated configs; a raw fm-tune config has neither (fm-tune takes both from the CLI), so set these explicitly with one of those. | config / `lora` / `none` |
| `TRAIN_FILE` / `VAL_FILE` | Override split filenames | globbed |
| `SETUP_COMMAND` | Optional `git checkout … && pip install -e .` hook, run in `$FM_TUNE_ROOT` with the step's own interpreter first on `PATH` | — |
| `BASH_BUILD_VENV` | Build a venv (bash) vs use image python | `true` |
| `JOB_ID` | the build id, e.g. `{{ run_metadata.build_id }}` | — |

See `samples/autotune/` for reference `build.yaml`s (bash and k8s).
