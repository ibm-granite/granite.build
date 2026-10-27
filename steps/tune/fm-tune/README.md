# fm-tune step (developer notes)

Source for `space://steps/tune/fm-tune`, in two environment directories:

- `bash/` publishes to `configurations/assets/environments/bash/steps/tune/fm-tune/`
- `skypilot/` publishes to `configurations/assets/environments/skypilot/steps/tune/fm-tune/` (slurm subtype)

User documentation is `USAGE.md` in each directory (same text, different "Running"
section). `make publish-step` copies it into the published step as `README.md`.
Never hand-edit `configurations/`, `test/steps/` or `test-data/steps/`; edit the source here and
re-publish.

## Shared script rule

`src/fmtune_run.sh` is byte-identical in `bash/` and `skypilot/`. All behaviour lives
there; the two `step-template.yaml` files only map `fm_tune_config` and the input bindings
to `FMTUNE_*` variables. Edit one copy and copy it to the other.
`skypilot/test/test_fm_tune_script_copies_match.py` fails if they differ.

## Render-time checks

Config-only checks are `raise_error`s in a `validate` macro, so they fail gbserver's strict
launcher-config fill (`TargetStepRun.__init__`, once the step's inputs are bound): the step
fails before it is launched or provisioned, though in a multi-target build only after its
upstream targets have run. On skypilot the macro is in `run:` (each launcher-config string
is filled on its own, and `run:` is also what a warm re-run renders), followed by the
skypilot-only checks: `config.compute_config` and `config.launcher_config` must be mappings
(null or absent is unset); `num_nodes` must be 1 in both, the two sources gbserver's
`_num_nodes_from_configs` reads from the build, and a whole number (`""` fails: gbserver's
`int("")` would raise at launch); and the GPU check below. On
bash the env values are separate strings, so the macro has a launcher-config key of its own,
`validate:`, which renders to `""` and is read by nothing. The macro text is the same in both
templates, and `skypilot/test/test_fm_tune_script_copies_match.py` checks that, and that one
table of bad values is rejected by both the render and `fmtune_run.sh`. Change a rule in the
macro (both templates) and in the script together.

The macro checks, in order: `source` and `autotunex` are mappings and `hpo_config` a mapping
or null (first, so no later check reads an attribute of a non-mapping); no list or mapping
for any other key; then the rules USAGE lists, including the two from fm-tune's
`autotune/pipeline.py` that otherwise fail after the clone and venv build: `tuning_algo` and
`rl_algo` not both empty or `none`, and an offline `rl_algo` (`AUTOTUNE_OFFLINE_RL`: `dpo`,
`kto`) only with a `tuning_algo`. Both templates render a YAML `null` as `""`.

GPUs: the template cannot render `resources.accelerators` conditionally (only string values
are filled, so the key is always emitted, and `""` reads as a blank-named accelerator). It
keeps `resources: {}` and fails a `num_gpus_per_node` > 0 that has no
`launcher_config.resources.accelerators` (issue #536).

## `FMTUNE_*` contract

Set by the launcher; defaults are applied by the script.

| Variable | Default | Meaning |
|---|---|---|
| `FMTUNE_DATASET_DIR` | required | Dataset input directory |
| `FMTUNE_OUTPUT_DIR` | bash: `$LLMB_BASH_OUTPUT_DIR/model` | Where `main.py` writes the model; required elsewhere. Must be new or empty, and not an input dir. main.py trains in `<FMTUNE_OUTPUT_DIR>.partial.<random>`, renamed onto it on success (see USAGE) |
| `FMTUNE_CACHE_DIR` | bash: `<GB_HOME>/.fm-tune-cache` (beside `workdir/`; `<LLMB_BASH_OUTPUT_DIR>/../../.fm-tune-cache` when that path has no `/workdir/`); the skypilot template sets `${GB_SHARED_WORKDIR:-$PWD}/.fm-tune-cache-$(id -u)` (no shared parent: see USAGE) | Clone, tree and venv cache; required elsewhere. Created `0700`; the run fails if it is not the user's own or is group- or world-writable |
| `FMTUNE_PYTHON` | `$LLMB_BASH_PYTHON_DIR/python3`, else `python3` | Interpreter that creates the venv (>= 3.10) |
| `FMTUNE_SOURCE_PATH` / `FMTUNE_SOURCE_REPO` | empty | Exactly one must be set |
| `FMTUNE_SOURCE_REF` | `main` | Branch, tag (`git clone --branch`), or full 40-character commit SHA (fetched by id) |
| `FMTUNE_SOURCE_SUBDIR` | `autotunex/src/fm-tune` | fm-tune dir inside the repo |
| `FMTUNE_EXTRA` | `core` | pip extra |
| `FMTUNE_BASE_MODEL_DIR` | empty | Bound `base_model`; wins over `FMTUNE_MODEL` |
| `FMTUNE_MODEL` | empty | `--model_name_or_path` fallback |
| `FMTUNE_TRAIN_FILE` / `FMTUNE_VALIDATION_FILE` | required | Relative to the dataset dir; no leading `/`, no `..` component |
| `FMTUNE_TUNING_ALGO` | empty | `--tuning_algo` (templates default to `lora`) |
| `FMTUNE_RL_ALGO` | empty | `--rl_algo`, omitted when empty |
| `FMTUNE_RUN_NAME` | `fm-tune` | `--run_name`, `--output_model_name`; must match `^[A-Za-z0-9._-]+$` |
| `FMTUNE_AUTOTUNE` | `true` | `true` or `false`; `false` adds `--no_autotune` |
| `FMTUNE_BACKEND` / `FMTUNE_SEED` | empty | Flags omitted when empty |
| `FMTUNE_CONFIG_PRESET` | `autotune.yaml` | Bundled preset used when the HPO config is empty |
| `FMTUNE_HPO_CONFIG_JSON` | empty (`{}` = empty) | Inline HPO config as JSON; the script converts it to `<FMTUNE_CONFIG_NAME>.yaml` |
| `FMTUNE_CONFIG_NAME` | `fm_tune_hpo` | File name for the inline HPO config (also its name in AutoTuneX, when reporting); must match `^[A-Za-z0-9._-]+$` |
| `FMTUNE_EXTRA_ARGS` | empty | Whitespace-split and appended to argv |
| `FMTUNE_SERVER_URL` / `FMTUNE_JOB_ID` | empty | AutoTuneX bridge flags; with `FMTUNE_SERVER_URL` set, an empty `FMTUNE_JOB_ID` defaults to `GB_BUILD_ID` |
| `FMTUNE_LOCK_TIMEOUT` | `1800` | At most 9 digits. Seconds to wait on any cache lock held by another build before failing; a lock is never taken over (see USAGE) |
| `LOG_PATH` | `<FMTUNE_OUTPUT_DIR>.logs` | Where fm-tune's AutoTuneX bridge writes `logs/`; kept if already set. A sibling, not the staging dir (`--model_at_root` clears it) or the output dir's parent (shared by builds with sibling output dirs) |
| `FMTUNE_SKIP_INSTALL` | unset | `1` uses `FMTUNE_PYTHON` as is (unit tests only) |

Before anything uses them, the script makes `FMTUNE_OUTPUT_DIR`, `FMTUNE_CACHE_DIR`,
`FMTUNE_DATASET_DIR`, `FMTUNE_SOURCE_PATH` and `FMTUNE_BASE_MODEL_DIR` absolute against its
starting directory (it later runs `main.py` from the fm-tune root, and the marker is read
off-node), and drops every trailing `/` from `FMTUNE_OUTPUT_DIR` and `FMTUNE_CACHE_DIR`. `FMTUNE_OUTPUT_DIR` must not
contain `"`, `\` or a control character: the monitors build the binding as
`{ "path": "<path>" }` and parse it as JSON. It must not end in a space either: the bash
monitor's `LocalFileStream` strips trailing whitespace from each log line.

Bash launcher env values reach the child unexpanded, so the bash template leaves
`FMTUNE_OUTPUT_DIR` and `FMTUNE_CACHE_DIR` empty and the script derives them from the
`LLMB_BASH_*` builtins.

## Bash output-subdir caveat

On bash the model goes to `$LLMB_BASH_OUTPUT_DIR/model/`, not the output dir itself:
`--model_at_root` clears its output directory and `job.log` lives in `LLMB_BASH_OUTPUT_DIR`.
The template sets `config.bash.skip_finding_output_artifacts: true`, so only the script prints
`GB_ARTIFACT_ID:model`, and only on success (the wrapper's own scan printed it before checking
the exit code).

## Tests

- Mode 1 (no cluster, fast):
  - `fmtune_run.sh`, against a stub `main.py`, in `bash/test/`: `test_fmtune_run_sh.py`
    (argv, HPO config, validation), `test_fmtune_source_sh.py` (git source and cache),
    `test_fmtune_lock_sh.py` (the cache lock) and `test_fmtune_paths_sh.py` (cache and output
    dirs, the venv). Their shared helpers are `_fmtune_helpers.py` and the `env` fixture in
    `conftest.py`.
  - Each template's own mapping: `bash/test/test_bash_tune_fm_tune_render.py` and
    `skypilot/test/test_skypilot_tune_fm_tune_render.py`.
  - `skypilot/test/test_fm_tune_script_copies_match.py`: the shared parts, once for both
    templates. The script copies match, the `validate` macro is the same text, and every
    `validate` rule is checked against both renders (and the script, where it can check it).
    Add a `validate` rule's tests here, not in a render test.
- Build tests (Mode 2): `bash/test/bash/` and `skypilot/test/slurm/`. Run with
  `make -C steps/tune/fm-tune/<env> test`, which renders the Space first.

### RAM warning

The build tests train for real (Ray, torch, a venv install). Real builds have run a laptop out
of RAM. Run one build test at a time, never in parallel. Both fixtures inline a 1-epoch
`hpo_config` and set `FMTUNE_OBJECT_STORE_BYTES=268435456` to stay small. The slurm test
needs the local podman Slurm VM: stop the VM when you are not testing slurm.

### Test-setup notes

- `make test` with a local `.env` that sets SQL metadata storage without the psycopg driver
  fails at startup; set `GBSERVER_METADATA_STORAGE=sqlite`.
- The slurm test needs a venv with the `thirdparty` extra (the SkyPilot fork); the repo's main
  `.venv` may not have it.
- The node venv needs python3.10 (fm-tune needs >= 3.10). On the local Slurm cluster the cache
  is `/shared/.fm-tune-cache-<uid>` and the model goes to `/shared/fm-tune-test/model`. The nodes are
  CPU-only arm64, so bf16 is downgraded to fp32.
- SkyPilot builds write no local `job.log`; read `~/sky_logs/<cluster>/1-<cluster>/run.log`.
- `eval_loss` is `nan` in both fixtures (1 epoch, 4 steps, random weights); the tests check
  plumbing, `train_loss` is finite.
- `hpo_config` is written as YAML because fm-tune's `yaml.safe_load` (YAML 1.1) reads JSON's
  `5e-06` as a string (found in build c9d46858).
- A cached clone of a branch or tag is fetched on every run, and each trained commit adds a
  `tree-<sha>` and a `venv-<key>` that nothing prunes; see `USAGE.md`, "Caches and cleanup".

## Verified

- bash: build cfdb6814 succeeded on macOS/MPS with the 1-epoch fixture (train_runtime 4.2 s).
- slurm: build 22aab68c succeeded on the local podman Slurm cluster (CPU-only arm64, node venv
  on python3.10) with the 1-epoch fixture.
- Build c9d46858 found the YAML-float bug (see above).
- Not verified: GPU training (including the `accelerators` request on a GPU cluster), a commit SHA as `source.ref`, outputs other than `env://`,
  `autotune: true`, RL algorithms.
