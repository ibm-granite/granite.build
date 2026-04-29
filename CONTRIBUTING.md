# Contributing to gbserver

Thank you for your interest in contributing to gbserver! This guide covers development setup, code style, testing, and the pull request process.

## Prerequisites

- Python 3.11 or later (3.13 recommended; 3.14 is not yet supported)
- Git

## Development Setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/<your-username>/gbserver.git
   cd gbserver
   ```

2. Create the virtual environment:

   ```bash
   make standalone-venv
   ```

   This installs gbserver in editable mode with development dependencies. No external package registry access is needed.

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

## Running Tests

Run the open-source test suite:

```bash
pytest -m "not ibm" -s test
```

This skips tests that require IBM-internal infrastructure. All other tests run by default.

To run a specific test file:

```bash
pytest -m "not ibm" -s test/gbserver_test/api/test_artifacts.py
```

To run a single test:

```bash
pytest -m "not ibm" -s test/gbserver_test/api/test_artifacts.py::TestArtifactAPI::test_artifact_get
```

## Project Structure

The repository is a monorepo with three source packages:

| Package | Location | Description |
|---------|----------|-------------|
| gbserver | `src/gbserver/` | Build orchestration server |
| gbcli | `src/gbcli/` | CLI client (`gb`, `gbcli`, `llmbuild`, `llmb`, `lamb`) |
| gbcommon | `src/gbcommon/` | Shared types and utilities |

All packages follow the same code style rules and are linted together.

### gbcli Development

The gbcli source lives in `src/gbcli/`. It uses Click for the CLI framework and shares types with gbserver via `src/gbcommon/`.

Run the standalone test suite (covers both gbserver and gbcli):

```bash
make test-standalone
```

Lint both server and CLI:

```bash
make lint
```

## Code Style

gbserver uses **black** for formatting and **isort** for import sorting, with **pylint** and **mypy** for linting.

Format your changed files:

```bash
make xformat
```

Run lint checks on your changed files:

```bash
make xcheck
```

To format or check the entire codebase:

```bash
make format       # format all files
make staticcheck  # lint all files
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
   pytest -m "not ibm" -s test
   ```

5. Commit with a clear message:

   ```bash
   git commit -m "feat: add support for new environment backend"
   ```

6. Push and open a pull request against `dev`.

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
