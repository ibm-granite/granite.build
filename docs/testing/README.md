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
| `make extended-tests-setup extended-tests` | Full suite — `GBTEST_MODE=live`, `-m "not ibm"` (includes `extended`); setup also brings up [SLURM](../environments/setup/skypilot-slurm-setup.md). |
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
# one-time: provision the venv + SLURM the extended suite needs
make extended-tests-setup

# then run just those tests (repeat as needed; the setup is reused)
make extended-tests PYTEST_TEST_TARGETS=test/integration/standalone/buildrunner/skypilot_slurm
```

For a quick **single file or test**, `make py-test` with `ARGS` (passed straight to `pytest`) is simplest:

```bash
# a whole standalone build-integration file
make py-test ARGS="test/integration/standalone/buildrunner/bash/test_buildrunner_1step_local.py"
# narrow to one test with -k
make py-test ARGS="test/integration/standalone/buildrunner/bash -k retry"
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
(Docker/Podman daemon), `skypilot_integration` (local SLURM — `make integration-test`),
`hf_integration` (real HuggingFace Hub), `slow`, and `live` (opt specific services into live mode).

## Test modes and environment variables

`GBTEST_MODE` controls whether external services are mocked or real: `mock` (the quick suite's default —
placeholder credentials, services stubbed) or `live` (real connections). You can opt individual services
into live mode with `GBTEST_LIVE_<SERVICE>=true` (e.g. `GBTEST_LIVE_STORAGE=true`) or per test with
`@pytest.mark.live("storage", "github")`. The services are listed in `SERVICES` in
`test/libgbtest/mode.py`. `"space_access"` lifts only the mock-mode bypass of space-access checks
(`is_super_admin`, write access) and keeps the other GitHub mocks; `"github"` lifts all of them.

The run-relevant environment variables:

| Variable | Purpose |
|----------|---------|
| `GB_ENVIRONMENT` | `DEV` / `STAGING` / `PROD` / `STANDALONE` — selects cluster, namespace, storage, and standalone defaults. |
| `GBTEST_MODE` | `mock` (quick suite) or `live` (extended/CI). |
| `GBSERVER_DEFAULT_BUILDRUNNER_TYPE` | `job` (k8s), `process`, or `thread`. Use `thread` for local test runs. |
| `GBTEST_SPS_IBMCLOUD_API_KEY` | Loads test secrets from IBM Cloud Secrets Manager (SPS). |
| `GBTEST_STANDALONE_ENVIRONMENT` | Under `GB_ENVIRONMENT=STANDALONE`, which environment's HF resource group pushes target. The **source** default is empty, i.e. the production `gbspace-public` a real standalone user must get; `test/conftest.py` defaults it to `STAGING` for any pytest run, so tests push to a group the CI token can write. Set it explicitly (including to empty) to override. |
| `GBTEST_ENABLE_MANUAL_TESTS` | Set to `1` to run tests marked `@manual_testing_only` (from `libgbtest.constants`) — manual-only tests that depend on external, possibly-unpushed resources (private repos, unpushed assets steps, cluster SSH, real secrets). Skipped otherwise, including in CI. See the test's docstring for its specific prerequisites. |
| `GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE` | When `true`, a locally-set env var supersedes the value `conftest.set_test_env` fetches from SPS (default `false`, so the SPS secret wins). Set it alongside a local write `HF_TOKEN` to push with your own token instead of the shared `hf-token` secret (see the hf:// push note below). |
| **IBM infrastructure** | For `ibm`-marked / live-cluster tests: |
| `GBSERVER_IMAGE_TAG` | The gbserver build-runner image tag to run against. K8s build tests (`@requires_k8s_image_tags`) are skipped unless it is set. |
| `GBSERVER_SIDECAR_MONITORING_IMAGE_TAG` | The monitoring sidecar image tag. K8s build tests (`@requires_k8s_image_tags`) are skipped unless it is set. |

For the IBM-infra tags, the Makefile prints ready-to-`source` lines (`export GBSERVER_IMAGE_TAG=…` /
`export GBSERVER_SIDECAR_MONITORING_IMAGE_TAG=…`) derived from the current git commit — use those so tests
run against the image you built.

> **`hf://` push tests need a write token.** Tests that push an `hf://` output (e.g. the gated
> `test/integration/ibm/buildrunner/skypilot/aws/test_shared_fs.py`) resolve `HF_TOKEN` through
> `conftest.set_test_env` — from the shared SPS `hf-token` secret unless overridden. That secret should be a
> **write**-capable token; otherwise run with `GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE=true` and a local write
> `HF_TOKEN`. Note that an ambient `HF_TOKEN` env var also takes precedence over the token from
> `hf auth login`, so a read-only `HF_TOKEN` in your shell will shadow a write token you logged in with.

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

### REST API tests: standalone and IBM copies

The API tests (artifacts, builds, lineage, spaces) run twice from one set of test bodies:

- **`test/integration/standalone/api/`** holds the bodies, as mixin classes (`ArtifactAPITests`,
  `BuildAPITests`, …), and runs them on **apikey auth** through `AbstractStandaloneAPITest`
  (`libgbtest/api/utils.py`). They need no GitHub, IBM, or WandB access, in mock or live mode:
  storage is SQLite, the `public` space is `configurations/spaces/local`, and tests needing two users
  (admin vs. non-admin) get them by switching `GBSERVER_API_USER` per request. The lineage tests use the
  real WandB lineage store with its WandB service stubbed out.
- **`test/integration/ibm/api/`** subclasses the same mixins with `AbstractAPITest` and runs them on
  **GitHub-token auth** (`GBSERVER_GITHUB_TOKEN`, plus `GBTEST_ADMIN_GITHUB_TOKEN` /
  `GBTEST_NON_ADMIN_GITHUB_TOKEN` for the two-user tests), against the real WandB lineage service.

Write new API tests in the standalone mixins so both copies pick them up. Tests name their callers
through the identity hooks (`get_test_client(identity)`, `get_caller_username(identity)`,
`admin_identity()`, `non_admin_identity()`, `grant_super_admin(identity)`), never a token or apikey
user directly.

### SQL storage tests: local PostgreSQL

The SQL storage tests follow the same pattern. Their bodies are mixins (`SQL*Tests`) in
`test/unit/storage/postgres/`, next to the SQLite tests in `test/unit/storage/sqlite/`, and run
against a **local, TLS-enabled PostgreSQL**.
`test/integration/ibm/storage/sql/` runs the same tests against the hosted PostgreSQL, plus the
`*Legacy*` tests. Those read existing rows in the hosted database's real tables, so they exist only
in the IBM copy.

```bash
make postgres-setup      # start it (also run, best-effort, by `make extended-tests-setup`)
make extended-tests PYTEST_TEST_TARGETS=test/unit/storage/postgres
make postgres-teardown   # stop it and delete its data, certs and credentials
                         # (`make extended-tests-teardown` also stops SLURM)
```

`postgres-setup` (`scripts/postgres/setup-postgres.sh`) does the following:
- Starts `quay.io/sclorg/postgresql-16-c9s` on `127.0.0.1:55432`, with a random password. The server
  requires TLS 1.3 and a password for every TCP login and rejects plaintext. Its `pg_hba.conf` replaces
  the image's, which trusts `127.0.0.1` with no password. The tests connect with `sslmode=verify-full`
  against a CA the script generates.
- Creates the `gbtest` schema.
- Writes the `GBSERVER_SQL_*` settings and `GBTEST_LOCAL_POSTGRES=true` to
  `.cache/gbtest-postgres/postgres.env`, readable only by you.

The tests load that file themselves (`test/libgbtest/local_postgres.py`, imported first by
`test/conftest.py`) in `GB_ENVIRONMENT=STANDALONE` runs, which `make extended-tests` sets. That
covers plain `pytest` and VS Code too, as long as `GB_ENVIRONMENT=STANDALONE`. Without the file, or
outside STANDALONE, the standalone SQL tests skip (`requires_local_postgres`). They also skip when the
file is loaded but its server isn't reachable (a stopped container, or a reboot), so a stale file can't
turn into connection failures. If setup fails, it leaves no env file. While it's loaded, its
`GBSERVER_SQL_*` values take precedence over the SPS secrets and over any you set yourself. That points
everything using SQL storage in the run, subprocesses included, at the local server. To keep your own
`GBSERVER_SQL_*` in a STANDALONE run while the file exists, set `GBTEST_LOCAL_POSTGRES=false`. If the
file, which holds the password, is readable by group or others, loading it warns.

The IBM SQL storage tests target the hosted database, so they skip in every STANDALONE run
(`requires_hosted_postgres`), whether or not the local server is set up: the STANDALONE schema
(`standalone`) doesn't exist in the hosted database, and with the env file loaded `GBSERVER_SQL_*` point
at localhost. Run them with `GB_ENVIRONMENT=STAGING` or `DEV`. They also skip when storage is mocked
(`GBTEST_MODE=mock`, unless `GBTEST_LIVE_STORAGE=true`), where they would otherwise run on SQLite; SQLite
has its own tests in `test/unit/storage/sqlite/`. Both copies get the SQL storage factory from
`PostgresStorageTestMixin` (`test/libgbtest/storage/postgres_storage.py`).

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
- [SkyPilot SLURM setup](../environments/setup/skypilot-slurm-setup.md) — local SLURM for extended tests
