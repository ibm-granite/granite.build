# Contributing to Granite.Build

Thank you for your interest in contributing to Granite.Build! This guide covers development setup, code style, testing, and the pull request process.

## Prerequisites

- Python 3.11 or later (3.13 recommended; 3.14 is not yet supported)
- Git
- GNU Make

## Development Setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/<your-username>/granite.build.git
   cd granite.build
   ```

2. Create the virtual environment:

   ```bash
   make venv
   ```

   This installs the project in editable mode with the standalone, third-party (SkyPilot, Kubernetes, Docker) and development dependencies. IBM developers who need the IBM-only dependencies use `make ibm-venv` instead (requires `ARTIFACTORY_USER` / `ARTIFACTORY_API_KEY`). `make venv` needs no IBM Artifactory access, but it does download from PyPI.

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

## Running Tests

Run the standalone test suite:

```bash
make test-standalone
```

This runs unit tests, skipping any that require IBM-internal infrastructure, a NATS server, or Docker.

To run a specific test file or method directly:

```bash
pytest -s test/unit/space/test_space_config.py
pytest -s test/unit/space/test_space_config.py::TestSpaceConfig::test_load
```

## Project Structure

The repository is a monorepo with several source packages:

| Package | Location | Description |
|---------|----------|-------------|
| gbserver | `src/gbserver/` | Build orchestration server |
| gbcli | `src/gbcli/` | CLI client (`gb`, `gbcli`, `llmbuild`, `llmb`, `lamb`) |
| gbcommon | `src/gbcommon/` | Shared types and utilities |
| gb_ui_backend | `src/gb_ui_backend/` | Analytics service (charts, AI analysis, Agent Chat) backing the frontend, included directly into gbserver (`analytics`/`standalone`/`chat` extras). Also independently pip-installable via its own nested `pyproject.toml` — see `AGENTS.md`'s "Frontend (gb-ui)" section — so the IBM-internal `gb-ui` deployment can depend on it without pulling in the rest of this repo. `gbcommon` is bundled into that nested distribution rather than declared as its own dependency (see the same section for why) |

All packages follow the same code style rules and are linted together.

The frontend lives at `frontend/`, a yarn workspace: `frontend/packages/ui-core` is the generic, standalone-capable dashboard library (including the Agent Chat widget) and `frontend/apps/standalone` is the thin Next.js app shell that builds it into a static export served by gbserver. See `AGENTS.md`'s "Frontend (gb-ui)" section for build/dev commands.

## Code Style

The project uses **black** for formatting and **isort** for import sorting, with **pylint** and **mypy** for linting.

Format and lint only files changed relative to `main` (recommended before PRs):

```bash
make xformat    # isort + black on changed files
make xcheck     # pylint + mypy on changed python files (all but autotunex/)
```

Format or check the entire codebase:

```bash
make format       # isort + black on all files
make lint         # isort --check + black --check + license headers + pylint + mypy
```

### License headers

Every source file — Python, shell, JavaScript/TypeScript, Svelte, Dockerfiles and
Makefiles — at the repo root or under `src/`, `test/`, `steps/`, `configurations/`,
`scripts/`, `frontend/`, `autotunex/` and `.claude/` starts with the Apache-2.0 header, after
any shebang, Python encoding line or Dockerfile parser directive:

```python
# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

Use the file type's comment syntax (`//` for JavaScript/TypeScript, `<!-- -->` for
Svelte). The copyright line carries no year. The short
`SPDX-License-Identifier: Apache-2.0` form is also accepted, so existing headers in
that style can stay. Empty files (such as an empty `__init__.py`) need no header.

Ported code — a file copied from another repository — carries a
`PORTED, not authored here` provenance note naming the upstream repo, path and
commit, and the check skips it. Whether it also gets our header depends on who
owns the upstream code: ports from IBM's internal GitHub (`github.ibm.com`) carry
the header above the note; code from a third party keeps that party's notices
instead of ours. Third-party code vendored as whole directories is listed in
`EXCLUDE_RES` in `scripts/check_license_headers.py` (the granite-evals runtime
under `steps/eval/` is one; see [steps/eval/VENDORED.md](steps/eval/VENDORED.md)).

`scripts/check_license_headers.py` checks this. The pre-commit hook runs it with
`--fix`, which inserts a missing header and fails the commit so you can review and
stage the change. CI runs it in check-only mode. To fix files by hand:

```bash
python scripts/check_license_headers.py --fix            # every tracked file
python scripts/check_license_headers.py --fix FILE...    # just these files
```

## Pull Request Process

1. Create a feature branch from `dev`:

   ```bash
   git checkout -b my-feature dev
   ```

2. Make your changes. Write tests for new functionality.

3. Format and lint:

   ```bash
   make xformat
   make xcheck
   ```

4. Run the test suite:

   ```bash
   make test-standalone
   ```

5. Commit with a clear message:

   ```bash
   git commit -m "feat: add support for new environment backend"
   ```

6. Push to your fork and open a pull request against `dev`.

## Commit Messages

Use a short imperative summary (50 characters or less), optionally followed by a blank line and longer description. Common prefixes:

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation only
- `test:` — adding or updating tests
- `chore:` — maintenance (formatting, dependencies, CI)

## Reporting Issues

Use the [GitHub issue tracker](../../issues). Bug reports and feature requests have templates to guide you.

## Questions?

Open a [discussion](../../discussions) or file an issue.
