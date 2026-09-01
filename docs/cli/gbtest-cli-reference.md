# `gbtest` — run a single buildtest.yaml

`gbtest` is a console-script wrapper around the YAML-driven build-test harness.
It lets you point at a `buildtest.yaml` file directly and run it through pytest
without having to author or remember the path of a concrete test class.

```shell
# Run a buildtest.yaml (assumes a sibling build.yaml)
gbtest path/to/buildtest.yaml [extra pytest args...]

# Run against an explicit executable build.yaml (need not be named build.yaml)
gbtest path/to/buildtest.yaml -f path/to/exec-build.yaml

# Generate a skeleton buildtest.yaml from an executable build.yaml
gbtest render path/to/build.yaml [-o buildtest.yaml]
```

It is installed automatically by `make venv` (via the `[project.scripts]`
entry in [pyproject.toml](../../pyproject.toml)). You can also invoke it without
the venv-installed shim:

```shell
python -m libgbtest.buildrunner.gbtest path/to/buildtest.yaml
```

## What a `buildtest.yaml` looks like

A `buildtest.yaml` lives next to the `build.yaml` it drives. Together they form
a self-contained fixture: move the directory and both files still resolve
correctly.

```yaml
# Required: the assertions to make against the finished build.
target_expectations:
  - target_name: download_file
    step_count: 5
    input_artifact_count: 1
    output_artifact_count: 1
    jobstats_count: 3

# Optional (defaults shown).
build_yaml: ./build.yaml          # path relative to this YAML; defaults to sibling
expected_status: SUCCESS          # SUCCESS | INVALID | FAILED | ...
space_name: public                # space the build runs under
targets: null                     # list of targets to run; null = run all
timeout_minutes: 30
simulate_step_failure: true            # inject one environment failure to exercise retry path
space_uri: null                   # if set, takes precedence over the space's
                                  #   git_repo_uri in the gb_spaces table
                                  #   (relative file:// or filesystem paths resolve against this YAML's dir)
tests:                            # which test methods to run (see below)
  - runner
  - runner_cancellation
```

### Field reference

| Field                  | Type            | Default                            | Notes |
|------------------------|-----------------|------------------------------------|-------|
| `target_expectations`  | list[ExpectedTarget] | **required**                  | Per-target assertions; see below. |
| `build_yaml`           | str             | `./build.yaml`                     | Relative paths resolve against this YAML's directory. |
| `expected_status`      | str (Status)    | `SUCCESS`                          | Case-insensitive (`SUCCESS`/`success` both accepted). |
| `space_name`           | str             | `GBTEST_SPACE_NAME` (`public`)     | Space the build runs in. |
| `targets`              | list[str] \| null | `null` (run all)                 | Subset of targets in `build.yaml` to run. |
| `timeout_minutes`      | int             | `30`                               | Wall-clock cap for the build. |
| `simulate_step_failure`     | bool            | `true`                             | If `true`, signals the environment to inject one simulated failure to exercise the retry path. |
| `space_uri`            | str \| null     | `null`                             | When set, takes precedence over the `git_repo_uri` recorded for `space_name` in the `gb_spaces` table — the BuildRunner resolves `space://` URIs from this value instead of cloning the registered git repo. Relative `file://` URIs and bare relative filesystem paths resolve against this YAML's directory. PR creation and verification are skipped automatically (no GitHub repo). |
| `tests`                | list[str]       | `["runner", "runner_cancellation"]` | Which test methods opt in for this spec. Unknown values fail at load time. |

`ExpectedTarget` fields (all required):

| Field                    | Notes |
|--------------------------|-------|
| `target_name`            | Must match a target in `build.yaml`. |
| `step_count`             | Expected number of step records (use `-1` to skip checking). |
| `input_artifact_count`   | Expected number of input artifacts on the recorded target run. |
| `output_artifact_count`  | Expected number of output artifacts on the recorded target run. |
| `jobstats_count`         | Expected number of jobstats (lineage) entries. |

### The `tests:` list

Each value `<key>` in `tests:` maps to a `test_<key>` method on
`AbstractYamlBuildRunnerTest`. The default list runs both `test_runner`
(basic build) and `test_runner_cancellation` (sends a cancel mid-build).
Override it to opt out:

```yaml
# Skip the cancellation variant for this fixture.
tests:
  - runner
```

Unknown keys fail at YAML-load time with a clear error rather than silently
collecting zero tests — the valid keys are tracked in
[`BuildTestSpecification.KNOWN_TEST_KEYS`](../../test/libgbtest/buildrunner/buildtest.py).

## Examples

```shell
# Run both the basic and cancellation tests for this fixture.
gbtest test-data/integration/ibm/buildrunner/k8s/1step/cpu/buildtest.yaml

# Pass extra pytest args after the YAML path — for example, pick a single method:
gbtest test-data/integration/ibm/buildrunner/k8s/1step/cpu/buildtest.yaml \
       -k test_runner_cancellation

# Verbose output:
gbtest test-data/integration/ibm/buildrunner/k8s/1step/cpu/buildtest.yaml -vv

# Collect only — see what would run without actually running it.
gbtest test-data/integration/ibm/buildrunner/k8s/1step/cpu/buildtest.yaml \
       --collect-only -q
```

## Testing a parameterized sample/template

Samples and templates often use the `gb` CLI's `$${...}` parameterization, but a
`buildtest.yaml` needs a concrete, executable `build.yaml`. The convention
([issue #278](https://github.com/ibm-granite/granite.build/issues/278)) is:

```shell
# 1. Render the executable build from parameters.
#    (Skip this step if the build file is NOT parameterized.)
gb build describe -f template/build.yaml --raw --param ENVIRONMENT=skypilot/aws > exec-build.yaml

# 2. Generate the initial buildtest.yaml from the executable build.
gbtest render exec-build.yaml -o buildtest.yaml

# 3. Edit buildtest.yaml to confirm the verification values — replace the
#    step_count FIXME. See "gbtest render" below.

# 4. Run it — -f points gbtest at the executable build.
gbtest buildtest.yaml -f exec-build.yaml
```

`gb build describe --raw` (or `gb build start --dry-run`) is the entry point for
parameter substitution — both reuse the same engine as `gb build start`, so `gb`
stays the single source of truth; `gbtest` only *consumes* an executable
`build.yaml`.

Both write the resolved `build.yaml` verbatim to **stdout** (no banner, so
`… > exec-build.yaml` is pipe-safe); status/confirmation messages go to
**stderr**. `gb build start --dry-run --save-build-file <file>` writes the
resolved build to `<file>` instead and prints only a `✅ wrote …` note (to
stderr).

### `gbtest render`

`gbtest render <build.yaml> [-o <out>]` reads an executable `build.yaml` and emits
a **skeleton** `buildtest.yaml` (to stdout, or to `-o`). It:

- derives `input_artifact_count` / `output_artifact_count` from each target's
  declared inputs/outputs;
- emits a `FIXME` for `step_count`, the one value it cannot determine statically
  (it is environment-dependent), which you must replace;
- defaults `jobstats_count` to `-1` (skip) — jobstats are not asserted at run
  time yet, so it is not forced to a value;
- pre-sets `simulate_step_failure: false` (no step-retry testing) and
  `tests: [runner]` (no cancellation run).

The `step_count` `FIXME` **fails validation** if left unreplaced: loading the spec
(on any `gbtest` run) raises a clear error naming the field, so you cannot
accidentally run a half-filled skeleton. Replace `step_count` with the observed
step count (or `-1` to skip that assertion).

> `output_artifact_count` counts *declared* outputs; a declared output whose
> command emits no `GB_ARTIFACT` marker will over-count — the first real run
> catches it.

### The `-f` override

`gbtest <buildtest.yaml> -f <build.yaml>` runs the test against the given
`build.yaml` instead of the sibling default. The path resolves against the current
directory (where you ran `gbtest`), so a rendered build kept under a distinct name
(e.g. `exec-build.yaml`) works without renaming.

## How it works

`gbtest` invokes `pytest.main([...])` against the runner module
[gbtest_runner.py](../../test/libgbtest/buildrunner/gbtest_runner.py), passing
`--buildtest-yaml=<path>` as a custom pytest option (registered in
[test/conftest.py](../../test/conftest.py)).

`gbtest_runner.py` defines `TestYamlRunnerCli`, an `AbstractYamlBuildRunnerTest`
subclass whose `_get_yaml_spec_dir` returns the parent directory of the path
passed to the flag. The inherited `test_runner` and `test_runner_cancellation`
methods then load the YAML, consult its `tests:` list, and either run or skip.

Without the flag, the runner module's tests skip cleanly with a clear reason —
so collecting the whole tree (`pytest --collect-only test/`) is unaffected.

## Authoring a permanent test class instead

`gbtest` is for ad-hoc invocation. To anchor a fixture to a permanent named
test class (so it shows up in the IDE Test Explorer and runs as part of the
normal pytest run), drop a small concrete subclass next to the existing tests:

```python
# test/integration/ibm/buildrunner/k8s/test_buildrunner_1step_cpu.py
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = pytest.mark.ibm


@extended_testing_only
@pytest.mark.xdist_group(name="buildtest_cpu")
class TestBuildRunner1StepCPU(AbstractYamlBuildRunnerTest):
    def _get_yaml_spec_dir(self) -> Path:
        return get_test_data_dir_for(__file__) / "1step/cpu"
```

The single override (`_get_yaml_spec_dir`) is the entire body — everything
else is inherited from the abstract base. `get_test_data_dir_for(__file__)`
implements the `test/<...>` ↔ `test-data/<...>` parallel-tree convention so
the fixture path is greppable directly on the class.

## Exit codes

`gbtest` returns pytest's exit code, plus two of its own:

| Code | Meaning                                             |
|------|-----------------------------------------------------|
| `0`  | All selected tests passed.                          |
| `1`  | The path argument is not a file.                    |
| `2`  | No path argument was supplied.                      |
| ≥3   | Forwarded from pytest (test failure, collect error, etc.). |

## See also

- [Testing](../testing/README.md) — running the test suites, markers, and environment variables
- [CLIs overview](README.md) — how `gb`, `gbserver`, and `gbtest` relate
