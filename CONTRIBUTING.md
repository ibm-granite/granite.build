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

   This installs the project in editable mode with the standalone, third-party (SkyPilot, Kubernetes, Docker) and development dependencies, and registers the git [pre-commit hooks](#pre-commit-hooks). IBM developers who need the IBM-only dependencies use `make ibm-venv` instead (requires `ARTIFACTORY_USER` / `ARTIFACTORY_API_KEY`). `make venv` needs no IBM Artifactory access, but it does download from PyPI.

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
make lint         # isort --check + black --check + pylint + mypy
```

### Pre-commit hooks

[`.pre-commit-config.yaml`](.pre-commit-config.yaml) defines hooks that run on every `git commit`: **black**, **isort** (`--profile black`) and **gitleaks** (secret scanning). `make venv` / `make ibm-venv` install the `pre-commit` tool and run `pre-commit install` for you. The first commit downloads and builds the hook environments, so it needs network access and is slower than later ones.

If you created your environment another way, or the hooks are not firing, register them manually:

```bash
pre-commit install
```

To run the hooks against the whole tree:

```bash
pre-commit run --all-files
```

Hook versions are pinned by `rev` in `.pre-commit-config.yaml`; change them deliberately in their own PR rather than running `pre-commit autoupdate`. When bumping black or isort, change the matching `black==` / `isort==` pins in the `dev` extra of `pyproject.toml` in the same PR, so `make format`, CI and the hook all use one version.

The installed git hook runs `.venv/bin/python3` by absolute path, so it works without activating the venv. If `.venv` is later deleted or a `make venv` rebuild fails part-way, commits fail with "`pre-commit` not found" until the venv is rebuilt (`make venv`) or the hook is removed (`pre-commit uninstall`, from an environment that has it).

If a hook reformats files (the output says "files were modified by this hook"), the commit is aborted but the fixes are already written to your working tree. Review them with `git diff`, then `git add` the modified files and run the same `git commit` again. Always re-stage before retrying: committing again with the fixes left unstaged can make pre-commit print "Stashed changes conflicted with hook auto-fixes... Rolling back fixes", which is resolved the same way (`git add` the files, then commit). Please do not bypass the hooks with `--no-verify`.

The gitleaks hook only scans your staged changes. The "Secret scan" workflow ([`.github/workflows/gitleaks.yml`](.github/workflows/gitleaks.yml)) also runs gitleaks over the full git history on every pull request, so a secret that slips past the local hook still fails CI. If a real secret is found, rotate or revoke it: deleting it from the PR does not undo the exposure. If the finding is a placeholder or public value, add its fingerprint to [`.gitleaksignore`](.gitleaksignore) as `<path>:<rule>:<line>`, **without** a leading commit hash (pull requests are squash-merged, so the hash would not survive the merge), with a comment explaining why it is safe. See the header of that file for details.

When bumping the gitleaks version, change the `rev` in `.pre-commit-config.yaml` and `GITLEAKS_VERSION` / `GITLEAKS_SHA256` in `gitleaks.yml` together; CI fails if the two versions differ.

## Pull Request Process

1. Create a feature branch from `main`:

   ```bash
   git checkout -b my-feature main
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

6. Push to your fork and open a pull request against `main`.

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
