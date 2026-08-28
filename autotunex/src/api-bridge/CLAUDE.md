# api-bridge — subsystem notes

A standalone, **synchronous** FastAPI service (distribution `autotunex-api-bridge`, import
package `api_bridge`) that is the MySQL write-path/logging bridge the tuning pipeline uses
to record jobs, trials, results, and logs. It is not part of `src/autotunex` and is not
imported by it.

## Deliberately different stack

This subproject is a **synchronous** service — synchronous FastAPI routes, synchronous service
methods, `requests`, `PyJWT`, and a synchronous tuning-pipeline client. That has not changed
and should not be "fixed" to the async architecture of the main `src/autotunex` service; the
two services are independent by design.

The database layer uses **sync SQLAlchemy Core** — a sync `create_engine` over the metadata
in `tables.py`, executing Core constructs (no raw `text()` SQL) — over `pymysql` / `psycopg` /
stdlib `sqlite3`, configured by a single `AUTOTUNEX_DATABASE_URL` (the same connection string
the main service reads; SQLite, MySQL, and PostgreSQL are all supported, and an async driver
such as `mysql+asyncmy` or `postgresql+asyncpg` is coerced to its synchronous equivalent
internally). This replaced the original raw `pymysql` + `DBUtils` pooling and consciously
supersedes the DB-layer guidance in
`docs/superpowers/specs/2026-08-13-api-bridge-integration-design.md`. It remains independent of
`src/autotunex`: nothing is imported from it — the small amount of shared logic (URL driver
coercion, the TLS-context builder) is copied, not imported. See
`docs/superpowers/specs/2026-08-18-api-bridge-sqlalchemy-url-unification-design.md` for the
original synchronous-SQLAlchemy/URL-unification rationale. **This supersedes the MySQL-only
stance of the 2026-08-18 spec**; see
`docs/superpowers/specs/2026-08-20-api-bridge-multi-dialect-design.md` for the multi-dialect
design.

## Layout

`src/`-layout package at `src/api_bridge/`, with a `services/` subpackage
(`config_service.py`, `dataset_service.py`, `job_service.py`, `user_service.py`,
`github_service.py`). Has its own `pyproject.toml` and `Dockerfile` — a separate,
independently versioned artifact from the main service.

## Running it

```bash
pip install .
python -m api_bridge.server
```

Or, from the repo root, `make dev-api-bridge` (port 8001; override with `API_BRIDGE_PORT=...`),
which runs the module from source with `PYTHONPATH=src`.

Running as a module (not `uvicorn api_bridge.server:app` directly) preserves the
`if __name__ == "__main__"` block's startup behavior: `log_config=None` (so uvicorn doesn't
clobber this service's own logging setup) and env-driven `host`/`port`/`reload` via
`API_BRIDGE_SERVER_PORT` and `DEV_MODE`.

It requires a single database connection string — `AUTOTUNEX_DATABASE_URL` (the same variable
the main service reads; SQLite, MySQL, and PostgreSQL are all supported, and an async driver
value such as `mysql+asyncmy://...` or `postgresql+asyncpg://...` is accepted and coerced to
its sync equivalent — `pymysql` / `psycopg` / stdlib `sqlite3`), with optional
`AUTOTUNEX_DATABASE_SSL_CA` / `AUTOTUNEX_DATABASE_SSL_MODE` for TLS to MySQL — see
`.env.example`. PostgreSQL needs the optional driver: `pip install "autotunex-api-bridge[postgres]"`.
`database.Database()` raises at startup if `AUTOTUNEX_DATABASE_URL` is unset.

## Lint/format only — not part of `make check`

This subproject is linted and formatted from the **root** `make lint` / `make format`, which
lint it by explicit path (`ruff check src/api-bridge`) under this directory's own nested
`[tool.ruff]` config (relaxed vs. the main service — no `ANN`/`D`, plus a few legacy-code
codes ignored). The root config's `extend-exclude` skips `src/api-bridge` so the strict main
config never governs it. It
is deliberately **outside** the main service's `make test`, `make typecheck` (mypy strict),
and coverage targets — this package is not typed to that bar and has its own test setup
under `src/api-bridge/tests/`.

## More detail

`docs/superpowers/specs/2026-08-13-api-bridge-integration-design.md` has the full design and
rationale for this subsystem's existence and its boundary with `src/autotunex`.
