# Database & migrations

AutoTuneX stores job, trial, result, and related state in a relational database accessed
through SQLAlchemy's async ORM. This page covers the supported databases, how the schema
is created, and how to run and adopt migrations.

The database is selected entirely by [`AUTOTUNEX_DATABASE_URL`](configuration.md#persistence).

## Supported databases

| Database | Driver | Install extra | Role |
| --- | --- | --- | --- |
| SQLite | `aiosqlite` | *(none — bundled)* | Dev and test default; zero setup, no server. |
| PostgreSQL | `asyncpg` | `pip install -e ".[postgres]"` | Supported for production. |
| MySQL | `asyncmy` | `pip install -e ".[mysql]"` | The production database. |

The ORM mirrors a single schema across all three dialects, so never assume MySQL-only or
PostgreSQL-only SQL. Migrations are written to run on every supported dialect.

### Connection-string examples

```bash
# SQLite — the default; creates ./autotunex.db on first run, no server needed
AUTOTUNEX_DATABASE_URL=sqlite+aiosqlite:///./autotunex.db

# PostgreSQL — install the `postgres` extra first
AUTOTUNEX_DATABASE_URL=postgresql+asyncpg://user:pass@host/db

# MySQL — install the `mysql` extra first (production)
AUTOTUNEX_DATABASE_URL=mysql+asyncmy://user:pass@host/db
```

> **MySQL note.** The `mysql` extra also pulls in `cryptography`. MySQL 8.4 defaults new
> accounts to `caching_sha2_password`, and `asyncmy` fails a cold connection without
> `cryptography` present, so it is installed as part of the extra rather than left
> optional.

## Connecting to a managed MySQL database

Against a managed remote MySQL (IBM Cloud Databases), every *new* connection pays a full
TLS + `caching_sha2_password` handshake worth seconds. The two settings groups below keep
that cost off the request path.

### Connection pooling

The async engine keeps a warm pool of reused connections and recycles them before the
server's `wait_timeout` closes them, so a request that reuses a pooled connection skips the
handshake entirely. The pool is tuned by:

- `AUTOTUNEX_DATABASE_POOL_SIZE` (`10`) — warm, reused connections per worker; size it to
  cover request handlers **plus** the background reconcile loop.
- `AUTOTUNEX_DATABASE_MAX_OVERFLOW` (`5`) — extra connections opened on demand; the hard
  ceiling is `POOL_SIZE + MAX_OVERFLOW`.
- `AUTOTUNEX_DATABASE_POOL_TIMEOUT_SECONDS` (`30.0`) — how long a request waits for a free
  connection before failing.
- `AUTOTUNEX_DATABASE_POOL_RECYCLE_SECONDS` (`1800`) — recycle a connection older than this,
  comfortably under the DB's `wait_timeout` (IBM Cloud default `3600`); `-1` disables it.
- `AUTOTUNEX_DATABASE_POOL_PRE_PING` (`true`) — liveness-check a connection on checkout,
  reconnecting if it is dead.
- `AUTOTUNEX_DATABASE_POOL_USE_LIFO` (`true`) — hand out the most-recently-used connection
  first, keeping a small subset hot under bursty traffic.

The sizing knobs apply only to server databases (SQLAlchemy's `QueuePool`); SQLite ignores
them.

### TLS to a managed database

Managed MySQL such as IBM Cloud Databases refuses plaintext auth, so a production connection
must negotiate TLS:

- `AUTOTUNEX_DATABASE_SSL_MODE` (`disable` | `require` | `verify`) — `require` encrypts
  without verifying the server's certificate; `verify` also checks it against the CA below,
  and **hard-requires** it: `verify` with no `AUTOTUNEX_DATABASE_SSL_CA` raises at engine
  construction.
- `AUTOTUNEX_DATABASE_SSL_CA` — path to the CA-certificate PEM used when verifying.

That pairing is *not* one of the settings-level fail-fast checks, so it surfaces on the
first engine build rather than at import — including under `alembic`, which resolves its TLS
context through the same helper.

See the [Persistence](configuration.md#persistence) table in configuration.md for the full
setting reference.

## Schema creation

There are two ways the schema comes into existence, and which one you use depends on the
environment.

`AUTOTUNEX_AUTO_CREATE_SCHEMA` (default `true`) creates any missing tables on startup.
This is a **development convenience** — it lets a fresh checkout run against SQLite with no
setup. It is idempotent against an existing schema, so it does not corrupt anything, but it
also creates tables without recording any Alembic version, which hides the fact that
migrations were never applied.

**Production sets `AUTOTUNEX_AUTO_CREATE_SCHEMA=false` and uses Alembic migrations** so the
schema version is tracked explicitly.

## Migrations

AutoTuneX uses Alembic (async environment). Four Make targets wrap it:

```bash
make migrate                    # alembic upgrade head — apply all pending migrations
make migration m="add trials"   # autogenerate a new revision from ORM changes
make migrations-check           # alembic check — fail if db/tables/ has drifted
make downgrade                  # alembic downgrade -1 — roll back one revision
```

Migrations are verified against SQLite, PostgreSQL 16, and MySQL 8.4 in CI, and
`alembic check` — what `make migrations-check` runs — must stay clean on all three. Any
change to the ORM tables ships with an accompanying Alembic revision in the same commit.

### Revision history

The revisions form a single linear chain from the baseline to the current head:

| Revision | Summary |
| --- | --- |
| `1fb645a87b48` | **Baseline** — creates the full schema, including `jobs.precision`. |
| `78f6bb7de0df` | Backfills `jobs.precision` into `config_snapshot`, then drops the column (**modifies data**; reversible). |
| `7f175ebf55ad` | Adds dataset `status` and `status_detail`. |
| `b27a008ed0cf` | Makes `datasets.description` nullable. |
| `c628b830e8a3` | Adds the `jobs` reward-function columns. |
| `0a2caef2a185` | Widens `trials.id` and `results.trial_id` to `VARCHAR(16)` (**current head**). |

## Fresh or empty database (local dev, tests, CI)

For a brand-new database with no tables, nothing special is required. A single upgrade
builds the entire schema from the baseline forward:

```bash
alembic upgrade head
```

This is what tests and CI use, and it works on SQLite, PostgreSQL, and MySQL alike.
(In development you can instead rely on `AUTOTUNEX_AUTO_CREATE_SCHEMA=true` to create the
tables on startup, but running the migration is what a production-shaped setup does.)

## Adopting an existing database that already has the schema

> **Do not run `make migrate` / `alembic upgrade head` blindly against a database that
> already contains the AutoTuneX schema.** The baseline revision `1fb645a87b48` would try
> to *create* tables that already exist and fail.

The baseline revision reproduces the schema as it already exists in an established
deployment. When adopting such a database, you **stamp** the baseline as already-applied
(which runs no DDL) and then upgrade only the revisions that come after it:

```bash
# 1. Do not let startup create tables underneath you
export AUTOTUNEX_AUTO_CREATE_SCHEMA=false

# 2. Record the baseline as applied — runs no DDL, just writes the version
alembic stamp 1fb645a87b48

# 3. Apply only the revisions after the baseline
alembic upgrade head
```

Two things to know about these steps:

- **`alembic stamp 1fb645a87b48`** records the baseline as the current version without
  executing its `upgrade()`. This is the whole point: the tables it would create already
  exist, so it must be marked applied, never run.
- **`alembic upgrade head` then applies the later revisions** — beginning with
  `78f6bb7de0df` and continuing through the current head (`0a2caef2a185`). The first of
  these, `78f6bb7de0df`, **modifies data**: it copies every existing `jobs.precision`
  value into `config_snapshot['precision']` (a JSON field) and then drops the `precision`
  column. Nothing is lost — its `downgrade()` re-adds the column (as nullable) and restores
  the values out of the snapshot, so the change is reversible in both directions.

> **Caveat — trial-id widths.** This stamp-then-upgrade path holds only for a deployment
> whose trial ids match the baseline's `VARCHAR(10)`. `0a2caef2a185` widens `trials.id` and
> `results.trial_id` from `VARCHAR(10)` to `VARCHAR(16)`, but
> `resources/autotunex_schema.sql` already declares both columns as `VARCHAR(16)` — so a
> database built from that raw schema file and stamped at the baseline re-widens columns that
> are already wide, across the `results.trial_id` foreign key. Adopting such a database needs
> that foreign key dropped and recreated around the retype instead.

## Related

- [configuration.md](configuration.md) — `AUTOTUNEX_DATABASE_URL`,
  `AUTOTUNEX_DATABASE_ECHO`, and `AUTOTUNEX_AUTO_CREATE_SCHEMA`.
- [deployment.md](deployment.md) — deploying the service against a real database.
