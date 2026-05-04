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
   make standalone-venv
   ```

   This installs the project in editable mode with standalone and development dependencies. No external package registry access is needed.

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

The repository is a monorepo with three source packages:

| Package | Location | Description |
|---------|----------|-------------|
| gbserver | `src/gbserver/` | Build orchestration server |
| gbcli | `src/gbcli/` | CLI client (`gb`, `gbcli`, `llmbuild`, `llmb`, `lamb`) |
| gbcommon | `src/gbcommon/` | Shared types and utilities |

All packages follow the same code style rules and are linted together.

## Code Style

The project uses **black** for formatting and **isort** for import sorting, with **pylint** and **mypy** for linting.

Format and lint only files changed relative to `dev` (recommended before PRs):

```bash
make xformat    # isort + black on changed files
make xcheck     # pylint + mypy on changed files
```

Format or check the entire codebase:

```bash
make format       # isort + black on all files
make lint         # isort --check + black --check + pylint + mypy
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
