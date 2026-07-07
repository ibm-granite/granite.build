# Testing

> **Audience:** contributors running gbserver's test suites, and build authors validating a build with
> `gbtest`. Basic commands also live in [CONTRIBUTING.md](../../CONTRIBUTING.md).

Tests are driven through the **Makefile** (which wraps `pytest`) and, for build-level assertions, the
**`gbtest`** CLI. Markers and a mock/live mode select which tests run and whether they hit real services.

## Set up a virtual environment

| Target | Use |
|--------|-----|
| `make venv` | Full/IBM development venv (needs `ARTIFACTORY_USER` / `ARTIFACTORY_API_KEY`). |
| `make standalone-venv` | Open-source venv — no IBM Artifactory (`SKIP_ARTIFACTORY_CHECK=1`). |
| `make g4os-skypilot-venv` | Venv used by the quick/extended suites (standalone + third-party + dev). |

Then `source .venv/bin/activate`. Loading test secrets from IBM Secrets Manager requires
`GBTEST_SPS_IBMCLOUD_API_KEY` (see [env vars](#test-modes-and-environment-variables)).

## Run the suites (Makefile)

| Command | What it runs |
|---------|--------------|
| `make quick-tests-setup quick-tests` | Fast suite — `GBTEST_MODE=mock`, `-m "not ibm and not extended"`, no infrastructure. |
| `make extended-tests-setup extended-tests` | Full suite — `GBTEST_MODE=live`, `-m "not ibm"` (includes `extended`); setup also brings up [MinIO + SLURM](../environments/setup/skypilot-slurm-setup.md). |
| `make test-standalone` | Open-source CI suite — `test/unit`, no IBM infra. |
| `make cicd-pr-test` / `make cicd-merge-test` | CI suites (abbreviated / extended), with coverage + parallelism. |
| `make py-test ARGS="…"` | Quick local `pytest -s` with the default markers; pass extra pytest args via `ARGS`. |

Under the hood these call `pytest` with coverage and `pytest-xdist` (`-n auto --dist=loadgroup`).

### Overriding test macros

Pass any of these as `make VAR=value` to control a test run (they're `?=` defaults in the Makefile):

| Macro | Default | Effect |
|-------|---------|--------|
| `PYTEST_TEST_TARGETS` | `test` | The path(s) pytest runs (used by `.test`-based suites like `quick-tests`/`extended-tests`). |
| `PYTEST_MARKERS` | per-suite | The `-m` marker expression (e.g. `"not ibm and not extended"`). |
| `GBTEST_MODE` | per-suite | `mock` or `live`. |
| `PYTEST_NUM_TEST_PROC` | `auto` | xdist worker count (`-n`); use `0` or `1` to disable parallelism when debugging. |
| `PYTEST_DIST_MODE` | `loadgroup` | xdist distribution (`--dist`). |
| `PYTEST_COV` | `--cov --cov-report=xml` | Coverage flags; override empty (`PYTEST_COV=`) to skip coverage. |
| `PYTEST_CAPTURE` | `-s` | Output capture; override empty (`PYTEST_CAPTURE=`) to let pytest capture. |
| `ARGS` | `test` | For `make py-test` only — passed straight to `pytest` (paths, `-k`, etc.). |

The marker-set variables `DEFAULT_PYTEST_MARKERS`, `STANDALONE_PYTEST_MARKERS`, `PR_PYTEST_MARKERS`, and
`MERGE_PYTEST_MARKERS` can likewise be overridden.

**Scope a suite to specific tests** with `PYTEST_TEST_TARGETS` — this keeps the suite's mode/markers/infra
while narrowing the paths. For example, to run just the SkyPilot-SLURM standalone build-integration tests
through the extended suite (live mode):

```bash
# one-time: provision the venv + MinIO + SLURM the extended suite needs
make extended-tests-setup

# then run just those tests (repeat as needed; the setup is reused)
make extended-tests PYTEST_TEST_TARGETS=test/integration/standalone/buildrunner/skypilot_slurm
```

For a quick **single file or test**, `make py-test` with `ARGS` (passed straight to `pytest`) is simplest:

```bash
# a whole standalone build-integration file
make py-test ARGS="test/integration/standalone/buildrunner/local/test_buildrunner_1step_local.py"
# narrow to one test with -k
make py-test ARGS="test/integration/standalone/buildrunner/local -k retry"
```

`py-test` uses the default markers and `GBTEST_MODE=live`; add `GBTEST_MODE=mock` for mocked services.

## Run pytest directly

```bash
# a whole directory
pytest -s test/unit/space
# a single file, class, or method (fastest feedback)
pytest -s test/unit/space/test_space_config.py::TestSpaceConfig::test_load
# select by marker
pytest -s -m "not ibm and not extended" test/unit
```

`--dist=loadgroup` keeps tests tagged `@pytest.mark.xdist_group(name=…)` on the same worker (so a group
that shares state doesn't split across processes).

## Markers

Tests are tagged with markers (declared in `pyproject.toml`); suites select tests with `-m` expressions.
The two you'll reach for most:

- **`ibm`** — requires **IBM infrastructure** (cloud, cluster, RabbitMQ, …). Open-source and
  standalone runs exclude these with `-m "not ibm"`. Running them needs the
  [IBM-infrastructure env vars](#test-modes-and-environment-variables) below.
- **`extended`** — slow / real-infrastructure tests (defined as `extended_testing_only = pytest.mark.extended`
  in [`test/libgbtest/constants.py`](../../test/libgbtest/constants.py)). The fast suites run `-m "not extended"`;
  `make extended-tests` (or `pytest -m extended`) includes them. There is **no enable-flag environment
  variable** — extended tests are gated purely by this marker.

Other markers: `standalone` (only standalone deps), `thirdparty` (open-source deps, CI-runnable),
`secret_manager` (IBM Cloud Secrets Manager), `nats_server` (running NATS), `docker_required`
(Docker/Podman daemon), `skypilot_integration` (local SLURM + MinIO — `make integration-test`),
`hf_integration` (real HuggingFace Hub), `slow`, and `live` (opt specific services into live mode).

## Test modes and environment variables

`GBTEST_MODE` controls whether external services are mocked or real: `mock` (the quick suite's default —
placeholder credentials, services stubbed) or `live` (real connections). You can opt individual services
into live mode with `GBTEST_LIVE_<SERVICE>=true` (e.g. `GBTEST_LIVE_STORAGE=true`) or per test with
`@pytest.mark.live("storage", "github")`.

The run-relevant environment variables:

| Variable | Purpose |
|----------|---------|
| `GB_ENVIRONMENT` | `DEV` / `STAGING` / `PROD` / `STANDALONE` — selects cluster, namespace, storage, and standalone defaults. |
| `GBTEST_MODE` | `mock` (quick suite) or `live` (extended/CI). |
| `GBSERVER_DEFAULT_BUILDRUNNER_TYPE` | `job` (k8s), `process`, or `thread`. Use `thread` for local test runs. |
| `GBTEST_SPS_IBMCLOUD_API_KEY` | Loads test secrets from IBM Cloud Secrets Manager (SPS). |
| **IBM infrastructure** | For `ibm`-marked / live-cluster tests: |
| `GBSERVER_IMAGE_TAG` | The gbserver build-runner image tag to run against. |
| `GBSERVER_SIDECAR_MONITORING_IMAGE_TAG` | The monitoring sidecar image tag. |

For the IBM-infra tags, the Makefile prints ready-to-`source` lines (`export GBSERVER_IMAGE_TAG=…` /
`export GBSERVER_SIDECAR_MONITORING_IMAGE_TAG=…`) derived from the current git commit — use those so tests
run against the image you built.

## Running tests in VS Code

The repo ships a `.vscode/settings.json` that enables pytest discovery:

```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["test"]
}
```

so the Test Explorer lists and runs tests out of the box (against the activated `.venv`). Set the
environment variables above through the Python test configuration, or via an `env` block in a
`launch.json` config — the repo's `launch.json` includes a `gbserver build-runner` example that pins
`GB_ENVIRONMENT=STANDALONE`.

> **Caveat:** the Test Explorer's **Debug** button hangs on tests that make blocking `subprocess.wait()`
> calls. Debug via a "pytest as module" `launch.json` config (`python -m pytest <file>::<test>`) instead.
> See [troubleshooting](../help/troubleshooting.md#vscode-pytest-debugger-hangs-on-a-subprocess).

## Formatting and linting

```bash
make format        # isort + black
make staticcheck   # pylint + mypy on src/gbserver/
make lint          # full check-mode lint across src/gbserver, src/gbcli, src/gbcommon
```

`make xformat` / `make xcheck` limit formatting/typecheck to files changed against the `dev` branch.

## Test layout

```
test/
├── conftest.py            # session fixtures (secret loading, failure-state dumps)
├── libgbtest/             # shared harness (incl. buildrunner/{buildtest,gbtest,gbtest_runner}.py)
├── unit/                  # unit tests
├── integration/
│   ├── ibm/               # @pytest.mark.ibm — IBM infrastructure
│   └── standalone/        # standalone/open-source
└── e2e/                   # end-to-end
```

## Build-level tests: `gbtest`

For asserting on a *build* (expected status, per-target step/artifact counts, cancellation behaviour),
write a `buildtest.yaml` and run it through the harness:

```bash
gbtest path/to/buildtest.yaml
```

`gbtest` wraps pytest against the YAML-driven build-test runner — no need to author a test class. See the
[`gbtest` CLI reference](../cli/gbtest-cli-reference.md) for the `buildtest.yaml` schema and options.

## See also

- [CONTRIBUTING.md](../../CONTRIBUTING.md) — contributor quickstart
- [`gbtest` CLI reference](../cli/gbtest-cli-reference.md) — the build-test harness
- [SkyPilot SLURM setup](../environments/setup/skypilot-slurm-setup.md) — local MinIO + SLURM for extended tests
