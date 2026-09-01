# dpk (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure `dpk`
> in a `build.yaml` (config contract, inputs/outputs, examples). This file covers how the
> step is *generated, tested, and published*.

A general [Data Prep Kit](https://github.com/data-prep-kit/data-prep-kit) transform runner
for SkyPilot clusters. Runs on the **bare launcher node** by default (or in a public
container image), installing the transform's dependencies with `uv` during `setup` — no
custom image is built or published for this step.

It is the public-image counterpart of the custom-image
[eval](../../eval/skypilot/README.md) step, and is closely modelled on
[byoc](../../byoc/skypilot/README.md). It is *generated* from the sources in this directory
by the shared Makefile conventions — see the framework overview:
[steps/README.md](../../README.md).

## Design: byoc with three deltas

`byoc` already puts the build.yaml in control of the work. `dpk` keeps that shape and
changes three things:

| | byoc | dpk |
|---|---|---|
| **Dependencies** | `git clone` + `setup_command` | `transform`/`packages` → `uv pip install` into `./venv` |
| **Invocation** | verbatim `command` only | `transform:` derives the module, flags, and data config; `command:` remains as an escape hatch |
| **Dependency set** | whatever the image/repo provides | derived from `transform` via DPK's per-transform pip extras |

The point of the derivation is that **one step serves every DPK transform**: adding
tokenization, PII redaction, or dedup is a build.yaml change, never a step change. Two rules
carry it, and both hold for any transform DPK ships (verified across all ~30):

- module — every transform exposes `dpk_<name>.runtime`, a `PythonTransformLauncher`
  accepting `--data_local_config`.
- pip extra — `data-prep-toolkit-transforms` declares one extra per transform, so
  `<name>` with `_`→`-` names it.

**Flag prefixes are deliberately *not* derived.** DPK's own prefix is an arbitrary
abbreviation for roughly 40% of transforms (`dpk_tokenization` → `tkn_`,
`gopher_repetition_annotator` → `gra_`, `doc_quality` → `docq_`), so `args` keys are the
full flag name and the step passes them through verbatim. Anything else would need a
per-transform table in the step — exactly what stops it being general.

## Generating and deploying the step

`dpk` has no `Dockerfile`, so the `image`/`publish-image` targets are no-ops; only
`make space` (render the Space + bundle `src/`), `make test`, `make clean`, and `make help`
do anything here. For the full target list and variables, see the shared
[Makefile target conventions](../../README.md#makefile-target-conventions).

`make space` renders a self-contained Space into `space/` (see `SPACE_DIR` in the framework
overview). Point a build's Space at that directory to reference the step by
`space://steps/dpk`.

To promote the step into the repo's committed assets tree
(`configurations/assets/environments/skypilot/steps/dpk/`) and copy its per-cluster build
tests into `test/steps/dpk/skypilot/` so they are runnable from VSCode against the published
step, run `make publish-step`. Publishing also copies [USAGE.md](USAGE.md) to `README.md`
beside the published `step.yaml`, so the released step ships user-facing docs. See
[Two test modes](../../README.md#two-test-modes) for how the same test runs both against the
locally rendered `space/` (Mode 1, `make test`) and against the published step (Mode 2,
under `test/steps/`).

### Where the step's shell lives: `src/*.sh`

The `setup` and `run` blocks in `step-template.yaml` do **not** contain the step's shell.
They compute *values* with Jinja and hand them to two bundled scripts as arguments:

| Script | Invoked from | Does |
|---|---|---|
| `src/dpk_setup.sh` | `setup` (bare-node only) | bootstraps `uv`, anchors `UV_CACHE_DIR`, creates the venv, installs the requirements |
| `src/dpk_run.sh` | `run` (transform mode only) | creates + absolutizes the output dir, builds `--data_local_config`, runs `python -m <module>`, emits the artifact marker |

The reason is testability. Shell embedded in a YAML scalar behind Jinja can only be
*rendered and pattern matched*; in a file it can be executed, `shellcheck`ed, and
`bash -n`'d. That matters for this step specifically: every bug it has had was a
shell/quoting bug invisible until a cluster run failed — a trailing `\` that swallowed the
artifact marker, and single-quote escaping that broke a transform's `ast.literal_eval`'d
value. It also collapses the escaping: flags now arrive as **real argv** after a `--`
separator, so nothing re-quotes them (the old inline form needed
`replace("'", "'\"'\"'")` to survive Jinja → YAML → shell).

Calling a file-mounted script from `setup` is safe because SkyPilot syncs file mounts
first: `sky/execution.py`'s stage order is `PROVISION → SYNC_WORKDIR → SYNC_FILE_MOUNTS →
SETUP → PRE_EXEC → EXEC`, and `_execute` calls `sync_file_mounts()` before `setup()`
unconditionally.

**Command mode stays inline.** `config.dpk_config.command` is user-supplied shell injected
verbatim; routing it through a script argument would add exactly the quoting layer this
removes.

### A note on `src/` and `__pycache__`

`src/` holds python as well as shell. `make test` imports
`src/validate_tokenization2arrow.py`, which leaves a `src/__pycache__/` behind, and both `space` and
`publish-step` copy `src/` verbatim — so the Makefile drops that cache first, keeping the
tree shipped to a cluster to just the source. (It would never be *committed*:
`__pycache__/` is in the repo-root `.gitignore`, which also covers the published copy.)

## Running the tests

`make test` runs the whole `test/` tree against the locally rendered `space/`:

```
make -C steps/dpk/skypilot test    # uses the repo-root .venv; runs `make space` first
```

Two kinds of test live here, following the framework's split:

**Cluster-agnostic unit tests** at the root of `test/` — these need no infrastructure and
run everywhere. They are Mode-1 only (not copied by `publish-step`), the same placement as
`eval/skypilot/test/test_eval.py`:

- `test_dpk_step_render.py` — pins what the *template* computes: that `transform:` derives
  the right module and pip extra for a range of transforms, that `args` become the right
  argv words in order (with `0`/`true`/`false` handled correctly), that `dpk_image` switches
  between bare-node and `docker:` mode, that command mode injects verbatim and skips the
  transform path, and that the rendered shell parses under `bash -n`. Because the blocks now
  invoke the bundled scripts, most assertions run the rendered block with a stub script on
  `PATH` and check **the argv bash actually built** (`_script_argv`) rather than matching
  rendered text — bash is what splits and unquotes these words on the node, so a quoting
  slip surfaces here exactly as it would in production.
- `test_dpk_run_sh.py` — executes `src/dpk_run.sh` with a stub `python`: the
  `--data_local_config` literal, the output dir being created and absolutized before the
  marker, flags (including quote-bearing python literals) reaching `python` untouched, the
  `--` separator shielding a transform flag that collides with one of the script's own
  options, failure propagation under `set -e`, and the marker being exactly one
  line-initial command.
- `test_dpk_setup_sh.py` — executes `src/dpk_setup.sh` with stub `pip`/`uv` that record
  their argv: that `uv` (not `pip`) installs and `pip` only bootstraps `uv`, that
  `UV_CACHE_DIR` is exported before `uv venv` and anchors at `$GB_SHARED_WORKDIR` when
  present, that the install keeps its cache, that a `[extra]` specifier arrives as one
  argument, and that zero requirements still yields a venv.
- `test_dpk_validate_tokenization2arrow.py` — unit tests for the bundled `src/validate_tokenization2arrow.py`,
  covering both of its axes: consistency (the Arrow token stream vs its `meta/` sidecars)
  and completeness (`--input`: every non-empty source Parquet produced output). Each
  corruption case asserts a *failure*, since a validator that only ever passes is worthless.

**Per-cluster build tests** under `test/<cluster>/` with fixtures in
`test-data/<cluster>/` — real-infra, extended-suite only, each self-skipping unless its
backend is reachable:

- **slurm** — needs the local Docker SLURM cluster (+ MinIO). Bring them up once with
  `make test-setup` (delegates to the repo-root `slurm-setup` / `minio-setup`). Runs the
  transform → `env:///shared/…` → validate handoff, so it exercises both modes and the
  cross-node path with no credentials.
- **slurm-pii** — the same step with `transform: pii_redactor`, proving generality: only
  `transform`/`args`/artifact names differ from the tokenization fixture. Slow (the
  `[pii-redactor]` extra is ~125 packages), hence `timeout_minutes: 60`.
- **slurm-default-output** — tokenization with `dpk_config.output_path` **omitted**, the only
  fixture that exercises the `./output` default (both siblings must set it: `slurm` needs a
  shared path for its handoff, `slurm-pii` declares an `env:///tmp` uri whose path must
  match). Verifies on real infra what a render test cannot: that the relative default
  resolves inside SkyPilot's working directory on a compute node and that the resulting
  absolute path is one the monitor accepts and registers.
- **aws** — needs AWS credentials in the environment; provisions a real EC2 instance.

> Container images require the Pyxis SPANK plugin on SLURM/LSF, which the local Docker
> SLURM cluster does not have — so the slurm fixtures leave `dpk_image` empty and run on the
> bare node.

## Inputs, outputs, and bundled scripts

Moved here from USAGE.md: a build author needs the short rules (which USAGE.md keeps), but
the full mechanics below — which URI schemes stage where, why `output_path` cannot be
derived, and what the bundled scripts do — are implementation detail.

### Inputs and outputs

#### Inputs

Declare each input on the **target** (a direct `uri:`, or a `binding:` to an upstream
target's output). The step exports every declared input as an environment variable holding
its staged local path:

```
inputs.<name>  →  $GB_INPUT_<name>
```

In transform mode, `input: <name>` selects which one feeds the transform. In command mode,
reference any of them directly from your `command`. Filesystem-backed schemes (`hf://`,
`env://`, `file://`, `s3://`, `lh://`) are staged by the assetstore before `run`; an
`hf://` input is downloaded during `setup` automatically.

#### Outputs

Declare each output on the target, then make sure the bytes land at the path in its `uri`:

- **Transform mode** — the step creates the directory, points the transform's
  `output_folder` at it, and emits the artifact marker for you. `output_path` defaults to
  `./output` in the step's working directory; set it explicitly when the output's `uri` names
  a path, or when a **downstream target** reads the output — see
  [When the default is not enough](#when-the-default-is-not-enough).
- **Command mode** — your command writes wherever it likes and prints the marker itself:

  ```
  GB_ARTIFACT_ID:<output-id> GB_ARTIFACT_PATH:<abs-path>
  ```

  `<output-id>` must match a declared `outputs.<id>`. Repeat the line to register several
  artifacts under one output. For `mem://` outputs use `GB_ARTIFACT_PATH` →
  `GB_ARTIFACT_STATE:<value>`.

##### When the default is not enough

`output_path: ""` (the default) writes to `./output` in the step's working directory, and the
step absolutizes it before emitting the marker. That is right for two common cases:

- a **terminal** output nothing downstream consumes;
- an output an **assetstore pushes** (e.g. `s3://…`), where the local dir is only a staging
  area.

Set it explicitly in either of these cases:

1. **The output's `uri` names a path.** The step cannot derive it — declared output URIs are
   not in the runtime render context — so keeping `uri` and `output_path` in agreement is
   the build author's job. A mismatch fails at run time, not at submit time.
2. **A downstream target reads the output.** The working directory is the *per-run* workdir
   (`${shared_workdir}/builds/<build_id>/runs/<targetrun_id>`), which is keyed **per target**
   and removed when that target finishes. A consumer in another target would read a deleted
   directory, so give the output an explicit shared path instead — see
   [Choosing the output URI per endpoint](#choosing-the-output-uri-per-endpoint).

> **Why is there no `input_path`?** Because inputs and outputs reach the step differently, and
> this is the asymmetry the default only partly hides. An input is **staged before `run`** and
> its resolved path is handed to the step as `$GB_INPUT_<name>`, so there is nothing to
> specify. A declared **output does not exist yet** at render time: the runtime context is
> `bindings` + `run_metadata` + `setup_config` only, and declared output URIs reach *static
> validation* alone. Plumbing them into the runtime context would let the step derive
> `output_path` and drop the field — a gbserver change, tracked separately.

#### Choosing the output URI per endpoint

`validate`-style downstream targets read what an upstream target wrote, and the two may run
on **different nodes**. What works depends on the endpoint:

| Endpoint | Output URI | Why |
|---|---|---|
| `skypilot/slurm`, `skypilot/lsf` | `env:///shared/…` + an **explicit** matching `output_path` | `/shared` is the environment's `shared_workdir`, mounted on every node — cross-node safe. The `./output` default will not do here: it lives in the per-target workdir, which is deleted at target teardown. |
| `skypilot/aws`, `skypilot/kubernetes` | `s3://bucket/…`, `output_path` may be **left default** | Each target gets its own instance with no shared filesystem. The S3 assetstore pushes the staged dir and the consumer pulls it, so a node-local `./output` is fine. |

A node-local `env:///tmp/…` is **not** safe for a cross-target handoff: if the two targets
land on different nodes the consumer reads an absent directory.

### Bundled scripts (`src/`)

The step ships `src/` to `./src` on the node. Two of the three scripts are the step's own
machinery — you do not invoke them, but they are where the step's shell actually lives:

- `src/dpk_setup.sh` — the bare-node dependency install (`uv venv` + `uv pip install`),
  invoked from the step's `setup` phase. Skipped entirely when `dpk_image` is set.
- `src/dpk_run.sh` — the transform-mode invocation: creates and absolutizes the output
  directory, builds DPK's `--data_local_config`, runs `python -m <module>`, and emits the
  artifact marker. Not used in command mode, where your `command` runs instead.

The step.yaml computes the *values* (module, requirements, flags) and passes them to these
as arguments; the scripts do the shell. That keeps the shell in real files — checkable with
`shellcheck` and testable directly — rather than embedded in YAML behind Jinja.

The remaining script is a helper you *do* call, from a command-mode `command`:

- [`src/validate_tokenization2arrow.py`](src/validate_tokenization2arrow.py) — validates `tokenization2arrow`
  output. Checks that the Arrow token stream agrees with its `meta/*.docs` /
  `meta/*.docs.ids` sidecars (a truncated or duplicated write is caught, where an
  exists-and-non-empty check would pass), and with `--input <parquet_dir>` also checks
  **completeness**: that every non-empty source Parquet actually produced output. Exits
  non-zero with a per-file report, failing the build.

  ```
  python src/validate_tokenization2arrow.py <tokenize_output_dir> <report_dir> [--input <source_parquet_dir>]
  ```

  It is read-only — it never deletes or rewrites anything.
#### Adding a validator for another transform

`validate: true` resolves the validator by **rule**: `src/validate_<transform>.py`, the same
derive-don't-tabulate shape as the module and pip-extra derivations. So adding coverage for a
second transform is:

1. Write `src/validate_<transform>.py` taking `<output_dir> <report_dir> [--input <src_dir>]`
   and exiting non-zero on a problem (mirror `validate_tokenization2arrow.py`).
2. Add `test_dpk_validate_<transform>.py` beside it. Make each corruption case assert a
   *failure* — a validator that only ever passes is worthless.

No change to `step-template.yaml`, `dpk_run.sh`, or this README's tables: builds that already
say `validate: true` for that transform start validating on the next publish. Until such a
file exists, `validate: true` for that transform is a loud no-op (see USAGE.md).

Keep the report interface identical — `<report_dir>` receives `validation.json` — because
`dpk_run.sh` passes the output dir for both arguments so the record ships inside the
registered artifact.

## Known gaps

- **`output_path` is unchecked against the output's `uri`.** `output_path` is optional and
  defaults to `./output` in the step's working directory, but when the declared output `uri`
  names a path the build still supplies both and nothing validates that they agree; a
  mismatch fails at run time. The step cannot derive the path because declared output URIs
  are not in the runtime render context — `step_outputs` exists
  (`src/gbserver/build/targetstep.py`, `_get_validation_context`) but is built for static
  validation only, while the runtime render passes `bindings` / `run_metadata` /
  `setup_config` (`build/targetsteprun.py`). Plumbing declared outputs into the runtime
  context would let the step derive `output_path` and auto-emit the artifact marker for every
  declared output, removing the field entirely; that is a gbserver change, tracked
  separately.
- **The default `output_path` cannot serve a cross-target handoff.** `./output` resolves
  inside the per-run workdir (`${shared_workdir}/builds/<build_id>/runs/<targetrun_id>`),
  which is minted per **target** by `setup_skypilot` and `rm -rf`'d by `teardown_skypilot`
  when that target completes. A downstream target binding such an output would read a deleted
  directory, so an output another target consumes needs an explicit shared path — the
  tokenization fixture's `env:///shared/…` is the worked example. Nothing detects the
  mistake at submit time; it surfaces as a missing input on the consumer.
- **A step default cannot be asserted via `expected_steps`.** `expected_steps` compares
  against the **persisted** step config, which records only the keys the `build.yaml`
  supplied — `step-template.yaml`'s own defaults are not merged into it. So an omitted
  `dpk_config.output_path` is *absent* from the stored config rather than present as `""`,
  and `_assert_contains_subset` requires every expected key to exist (it fails with
  `missing key 'output_path'`). The `slurm-default-output` fixture therefore asserts the
  default's *effect* (`output_artifact_count: 1` — the artifact could not register without
  it) rather than the field's value.
- **Heavy transforms pay a per-cluster install.** `pii_redactor` pulls presidio + flair
  (hundreds of MB of models). `dpk_image` is the escape hatch, but note DPK publishes only
  `.devN` snapshot images — there is no `1.1.8` image on quay.io; the newest
  `tokenization-ray` tag carrying `dpk_tokenization2arrow` is `1.1.7.dev0`.
