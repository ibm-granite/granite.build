# dpk (SkyPilot) — development

> **Using this step?** See [USAGE.md](USAGE.md) for how to reference and configure `dpk`
> in a `build.yaml` (config contract, inputs/outputs, examples). This file covers how the
> step is *generated, tested, and published*.

A general [Data Prep Kit](https://github.com/data-prep-kit/data-prep-kit) transform runner
for SkyPilot clusters. Runs on the **bare launcher node** by default (or in a public
container image), installing the transform's dependencies with pip during `setup` — no
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
| **Dependencies** | `git clone` + `setup_command` | `transform`/`packages` → pip into `./venv` |
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

### A note on `src/` and `__pycache__`

This is the first step whose `src/` is python rather than shell. `make test` imports
`src/validate_tokens.py`, which leaves a `src/__pycache__/` behind, and both `space` and
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

- `test_dpk_step_render.py` — renders `step-template.yaml`'s `setup`/`run` blocks and pins
  the contract: that `transform:` derives the right module and pip extra for a range of
  transforms, that `args` render as full flag names in order (with `0`/`true`/`false`
  handled correctly), that `image` switches between bare-node and `docker:` mode, that
  command mode injects verbatim and skips the transform path, and that the rendered shell
  parses under `bash -n`. The step emits shell, so a templating slip is otherwise invisible
  until a cluster run fails — one regression guard covers a real bug where a trailing line
  continuation swallowed the artifact marker.
- `test_dpk_validate_tokens.py` — unit tests for the bundled `src/validate_tokens.py`,
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
- **aws** — needs AWS credentials in the environment; provisions a real EC2 instance.

> Container images require the Pyxis SPANK plugin on SLURM/LSF, which the local Docker
> SLURM cluster does not have — so the slurm fixtures leave `image` empty and run on the
> bare node.

## Known gaps

- **`output_path` is unchecked against the output's `uri`.** In transform mode the build
  supplies both, and nothing validates that they agree; a mismatch fails at run time. The
  step cannot derive the path because declared output URIs are not in the runtime render
  context — `step_outputs` exists (`src/gbserver/build/targetstep.py`) but is built for
  static validation only, and the runtime render passes `bindings` / `run_metadata`.
  Plumbing declared outputs into the runtime context would let the step derive
  `output_path` and auto-emit the artifact marker for every declared output; that is a
  gbserver change, tracked separately.
- **Heavy transforms pay a per-cluster install.** `pii_redactor` pulls presidio + flair
  (hundreds of MB of models). `image` is the escape hatch, but note DPK publishes only
  `.devN` snapshot images — there is no `1.1.8` image on quay.io; the newest
  `tokenization-ray` tag carrying `dpk_tokenization2arrow` is `1.1.7.dev0`.
