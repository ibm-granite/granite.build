# AutoTuneX documentation

AutoTuneX is a service for **automated fine-tuning and hyperparameter optimization (HPO)
of large language models**. You describe what to optimize as a *job*; AutoTuneX searches a
hyperparameter space by running one training *trial* per candidate and reports which
configuration performed best.

This directory is the full documentation. The project [`README`](../README.md) is the
quick front door; the pages here go deeper.

## Start here

| If you want to… | Read |
| --- | --- |
| Understand what AutoTuneX is and how the pieces fit together | [Concepts](concepts.md) |
| Install it and run your first tuning job | [Getting started](getting-started.md) |
| Call the REST API directly | [API overview](api/overview.md) → the resource references |
| Deploy and operate the service | [Operations](#operations) |

## Contents

### Understand

- **[Concepts](concepts.md)** — fine-tuning and HPO in plain language, the domain model
  (configuration, job, trial, result, dataset, user, task), and the job lifecycle.

### Use

- **[Getting started](getting-started.md)** — install, run the server, and walk through
  creating a configuration, registering a dataset, submitting a job, and reading results.

### API reference

- **[API overview](api/overview.md)** — conventions shared by every endpoint: the
  `/api/v1` base path, pagination, error format, and ownership scoping.
- **[Authentication](api/authentication.md)** — standalone mode, API keys, OIDC bearer
  tokens, and browser sessions.
- **[Authentication testing](authentication-testing.md)** — a runbook for exercising each
  auth provider against a running server.
- **[Jobs API](api/jobs.md)** — submit, list, read, cancel, delete and reconcile jobs, page
  their trials, read their logs and per-step training metrics, and download result reports.
- **[Reward functions API](api/reward-functions.md)** — validate a user-supplied
  online-RL reward function: static analysis plus an optional sandboxed test run.
- **[Configurations API](api/configurations.md)** — full CRUD for reusable tuning
  configurations.
- **[Datasets API](api/datasets.md)** — dataset CRUD, file upload, and the
  LLM-backed dataset-intelligence helpers.
- **[Users API](api/users.md)** — admin-only user management and self-service metadata.
- **[Chat API](api/chat.md)** — the conversational chat endpoints.
- **[MCP server](api/mcp.md)** — the Model Context Protocol server reference.

### Operations

- **[Configuration reference](operations/configuration.md)** — every `AUTOTUNEX_*`
  environment variable, grouped and defaulted.
- **[Job backends](operations/job-backends.md)** — how a submitted job actually executes:
  the `none`, `local`, and `llmb` backends.
- **[Database & migrations](operations/database.md)** — SQLite / PostgreSQL / MySQL,
  schema creation, and adopting an existing database.
- **[Production deployment](operations/deployment.md)** — the checklist for running
  AutoTuneX safely in production.

## Project governance

- [Contributing guide](../../CONTRIBUTING.md)
- [Code of conduct](../../CODE_OF_CONDUCT.md)
- [Security policy](../../SECURITY.md)
- [License](../../LICENSE)

## Maintainer docs

Contributor and maintainer references — not user-facing API documentation.

- **[Schema review](schema-review.md)** — a recorded correctness and normalization
  review of `resources/autotunex_schema.sql`; its recommendations are deliberately not
  applied.
