# Schema review: `resources/autotunex_schema.sql`

Date: 2026-07-29
Status: recommendations only — **apart from the five deviations in section F, none of
section A–D is applied to the ORM**

## How to read this

The ORM in `src/autotunex/db/tables/` deliberately **mirrors
`resources/autotunex_schema.sql` as-is**, bugs included, because the MySQL database is live
and other systems may depend on its current columns and types. This document records what
could be corrected and normalized, so the decision to change any of it is explicit and
separate.

Five deviations from the file are applied. See section F.

Severity: **high** — risks data loss or wrong answers. **medium** — will bite at scale or
during a migration. **low** — hygiene.

---

## A. Integrity and data-loss risks

### A1. `trials.id` is a global primary key holding per-job identifiers — high

`resources/autotunex_schema.sql:73` — `id VARCHAR(16) PRIMARY KEY`.

Trial identifiers of the Ray Tune variety are unique *within a run*, not across all runs. As
a global primary key this will eventually reject a legitimate insert, or worse, attach a
trial to the wrong job. `results.trial_id UNIQUE` inherits the same assumption.

Recommended: composite primary key `(job_id, id)`, or a surrogate key plus a unique
constraint on `(job_id, number)`. The scaffold's original `TrialTable` used the latter.

### A2. `log_entries.trial_id` cannot join to `trials` — high

`resources/autotunex_schema.sql:86` declares `CHAR(36)`, but `trials.id` is `VARCHAR(16)`.
No value can ever match, and there is no foreign key to catch it. `results.trial_id`
correctly uses `varchar(16)`, which confirms the intended type.

Recommended: `VARCHAR(16)` with a foreign key to `trials(id) ON DELETE CASCADE`.

### A3. Deleting a user destroys all tuning history — high

`resources/autotunex_schema.sql:22`, `:43`, `:66` cascade from `users` into
`configurations`, `datasets` and `jobs`. Meanwhile `:67` and `:68` use `RESTRICT`, so a
configuration cannot be deleted while a job references it.

The two rules contradict each other: the schema protects a configuration from deletion, yet
removing its owner deletes the configuration *and* every job that `RESTRICT` was protecting.

Recommended: `RESTRICT` on the user foreign keys, or soft-delete `users` with a
`deleted_at` column.

### A4. Every timestamp stops working in 2038 — high

The `created_at` / `updated_at` pairs on the six tables that carry them — `users`,
`configurations`, `datasets`, `jobs`, `trials` and `results` — are `TIMESTAMP`, which MySQL
stores as a 32-bit value spanning `1970-01-01` to `2038-01-19 03:14:07`. `DATETIME` spans
years 1000–9999. `gb_tasks` is the exception, and a worse one: its `started_at` / `updated_at`
are `VARCHAR(255)` and it has no `created_at` at all — see B5.

`TIMESTAMP` also converts to the session timezone on read while `DATETIME` does not, so
every reader has to pin its session timezone to get consistent values back. Switching to
`DATETIME` removes both the 2038 cliff and that requirement.

Recommended: `DATETIME` throughout. This is a type change on a live table — batch it with
other work.

### A5. Integer columns that overflow — high

| Column | Line | Limit | Recommended |
| --- | --- | --- | --- |
| `datasets.train_file_size` | 33 | 2 GB as signed `INT` | `BIGINT` |
| `datasets.validation_file_size` | 36 | 2 GB | `BIGINT` |
| `log_entries.id` | 84 | 2.1 B rows | `BIGINT UNSIGNED` |

`log_entries` is by far the highest-volume table here; an `INT` surrogate key is the wrong
default for it.

---

## B. Query and view defects

### B1. `autotunex_jobs` returns one row per task, not per job — high

`resources/autotunex_schema.sql:147` left-joins `gb_tasks` without bound. `type` is an
`ENUM('RITS','TUNING','DOWNLOAD')`, so a job that has all three produces **three rows**.
For a view named `autotunex_jobs` backing a job list this silently breaks pagination: a
`LIMIT 20` returns fewer than 20 jobs, and `COUNT(*)` counts pairs rather than jobs.

Recommended: join only the latest task per job, or drop `gb_tasks` from the view and expose
tasks separately. Section E gives a corrected definition.

### B2. `SELECT j.*` inside a view is frozen at creation — high

MySQL expands `*` when the view is created and stores the resulting column list. Adding a
column to `jobs` later will **not** appear in `autotunex_jobs` until the view is recreated,
so a migration can ship a column that silently never reaches consumers.

Recommended: enumerate columns explicitly.

### B3. The `GROUP BY` is unnecessary and not portable — medium

`gt.type` is selected at `:134` but absent from the `GROUP BY` at `:148`, so the view relies
on MySQL detecting functional dependency through an outer join. Postgres enforces grouping
strictly with no such escape hatch and **rejects this view outright**.

The grouping exists only to compute `num_trials`. A scalar subquery removes it entirely:

```sql
(SELECT COUNT(*) FROM trials t WHERE t.job_id = j.id) AS num_trials
```

That recommendation is for **direct-MySQL consumers of the view only**. The API no longer
counts trial rows at all: its `num_trials` is the configured trial budget
(`config_data.tune_config.num_samples`), not a row count — read from the job's own
`config_snapshot` by `resolve_planned_trials` (`src/autotunex/services/mappers.py:300`), with
the actual row count served as the `GET /jobs/{id}/trials` page's `total`. So the view's
`COUNT(DISTINCT t.id)` and the API's `num_trials` now intentionally differ.

### B4. `ORDER BY` in a view blocks predicate pushdown — medium

`:150` forces MySQL to materialize the view (temptable algorithm) instead of merging it, so
`WHERE` and `LIMIT` from the calling query cannot be pushed down. Every query against the
view sorts the full result set first.

Recommended: remove it and order at the call site.

### B5. `gb_tasks` timestamps are strings — high

`:118-119` — `started_at` and `updated_at` are `VARCHAR(255)`. They cannot be compared or
sorted correctly, are not auto-maintained, and the view exposes them as strings
(`task_started_at`, `task_updated_at`), so every consumer inherits the problem. Sorting
tasks by recency works only accidentally, when the strings happen to be zero-padded ISO.

Recommended: `DATETIME`, with `created_at` added and `ON UPDATE CURRENT_TIMESTAMP` on
`updated_at`.

---

## C. Consistency and hygiene

| # | Finding | Line(s) | Recommendation | Sev |
| --- | --- | --- | --- | --- |
| C1 | `user_id VARCHAR(255)` references `users.id VARCHAR(36)` | 14, 28, 49 | `CHAR(36)`; the extra 219 bytes are indexed on every table | med |
| C2 | `VARCHAR(36)` and `CHAR(36)` used interchangeably for the same UUIDs | 48, 52, 74, 98 | Pick one — `CHAR(36)`, or `BINARY(16)` for ~2.25× smaller indexes | low |
| C3 | No index on `log_entries(job_id, timestamp)`, `log_entries(trial_id)`, `jobs(status)`, `jobs(user_id, created_at)` | — | Add. Foreign keys get implicit indexes; these are not covered, and log retrieval is the hottest read path | med |
| C4 | No `started_at` / `finished_at` on `jobs` or `trials` | — | Add. Duration is only derivable indirectly today: the job list computes `finished_at` as `MAX(gb_tasks.updated_at)` and a client renders `finished_at - created_at`. That is `NULL` for a job with no build task (the `local` backend), sorts `VARCHAR` timestamps lexicographically (B5), treats submit time as the start, and gives no per-trial duration at all — real columns fix all four. `updated_at` remains a poor proxy | med |
| C5 | `gb_tasks` has no `created_at`, and `updated_at` has no `ON UPDATE` | 108-122 | Add both | low |
| C6 | `SET GLOBAL time_zone` in a schema script | — | **Resolved** — no longer present in the schema file. If it returns: it needs `SYSTEM_VARIABLES_ADMIN` and affects every other database on the server, so set the timezone per connection instead | med |
| C7 | Database is `autotune`; everything else says `autotunex` | 1-2 | Align | low |
| C8 | `datasets.description TEXT NOT NULL` with no default | 30 | Nullable, or `DEFAULT ''` — `TEXT` cannot take a literal default in MySQL. **Applied** as nullable; see section F | low |
| C9 | `datasets.artifact_url` vs `gb_tasks.artifact_uri` | 39, 116 | Pick one spelling | low |
| C10 | The identical six-value status `ENUM` is declared three times | 50, 75, 112 | Unavoidable in MySQL, but note that adding a state requires three `ALTER TABLE`s on large tables | low |
| C11 | `u.email AS user` shadows MySQL's `USER()` conceptually | 126 | `user_email` | low |
| C12 | `users.role` is free-text `VARCHAR(50)` | 7 | `ENUM`, or a `roles` table if the set is managed | low |

---

## D. Normalization

### D1. `results` is a one-to-one extension of `trials` — highest-value change

`results.trial_id` is `UNIQUE` (`:99`), so there is exactly one results row per trial. The
table adds a join, a second surrogate key, and a second pair of timestamps for no
additional cardinality.

Recommended: fold `metrics` into `trials` as a JSON column and drop `results`.

**Caveat:** keep it separate if metrics are meant to become a time series (many samples per
trial). In that case drop the `UNIQUE`, rename it `trial_metrics`, and add a
`(trial_id, recorded_at)` key — the current `UNIQUE` actively prevents that use.

### D2. `results.job_id` is transitively dependent — 3NF violation

`:98`. The parent job is reachable via `trials.job_id`. Storing it again means two sources
of truth that can disagree, with nothing to stop `results.job_id` naming a job that is not
the trial's actual parent.

Recommended: drop; join through `trials`.

### D3. `results.metric` duplicates the objective

`:100`. This appears to hold the objective metric name, which is a property of the *job*
(and already inside `config_data` / `config_snapshot`), copied onto every result row.

Recommended: drop, and read the objective from the job's snapshot.

### D4. `datasets` contains a repeating group — 1NF smell

Four columns describe two files:

```
train_records       train_file_size
validation_records  validation_file_size
```

Adding a test split means another `ALTER TABLE`, and `data_format` (`:37`) is stored once
for what are really per-file properties.

Recommended:

```sql
dataset_files (dataset_id, split, filename, records, size_bytes, format)
```

A new split then costs a row.

### D5. `train_file` and `validation_file` are derived data

`:31`, `:34` — `GENERATED ALWAYS AS (CONCAT(name, '_train')) STORED`. Pure functions of
`name`, materialized to disk, encoding a filename convention in the database.

Recommended: drop and derive in application code, or at minimum switch `STORED` to
`VIRTUAL` so they cost no storage. They are also the least portable construct in the file
(see section E).

### D6. `jobs` mixes identity with configuration

A field earns a column if something filters, sorts, joins or aggregates on it. Otherwise it
belongs in `config_snapshot`, which already exists for exactly this purpose.

| Column | Line | Recommendation |
| --- | --- | --- |
| `cleanup`, `autotune` | 61-62 | Runner-only booleans; nothing filters on them → snapshot |
| `seed` | 51 | Reproducibility metadata → snapshot, unless it is reported per job |
| `precision` | 59 | Training hyperparameter → snapshot. **Applied**; see section F |
| `ray_address` | 60 | Per-execution infrastructure placement, not job definition → `gb_tasks`, or drop. `VARCHAR(50)` is also tight for `ray://host:port` |

Keep as columns: `status`, `model`, `experiment_name`, `user_id`, `created_at` — these are
what lists and filters are built on.

### D7. Three overlapping "kind of tuning" fields

`jobs.tuning_type` (`:58`), `configurations.tuner_type` (`:16`), and
`configurations.rl_tuner_type` (`:17`) all describe the tuning strategy, across two tables,
with no constraint keeping them consistent. The view's `COALESCE` at `:127` exists to paper
over the ambiguity.

Recommended: one authoritative field. If reinforcement-learning tuners genuinely need a
sub-type, model it as `(tuner_type, tuner_subtype)` in one place.

### D8. Artifacts are represented three different ways

`datasets.artifact_id` + `artifact_url` (`:38-39`), `gb_tasks.artifact_id` +
`artifact_uri` (`:115-116`), and `jobs.output_artifacts` JSON (`:63`).

Recommended: one `artifacts (id, owner_type, owner_id, kind, uri, created_at)` table.
`CLAUDE.md` already lists artifact storage as an open decision, so this may be premature —
but the `url`/`uri` naming should be unified regardless.

### D9. `log_entries` mixes logging with training progress

`iteration` (`:90`) and `epoch` (`:91`) are training-progress metrics attached to log rows,
and `filename` (`:88`) is debug provenance.

Recommended: if these are logs, drop the progress fields. If they are progress events, that
is a `trial_progress (trial_id, iteration, epoch, recorded_at)` table with its own
retention policy — log rows and progress samples have very different lifetimes.

### D10. `config_id` and `config_snapshot` — keep both

This *looks* like redundancy and is not. `config_snapshot` is the immutable reproducibility
record; it must not change when the referenced configuration is edited. Documented here so
that nobody normalizes it away.

Two improvements: make `config_snapshot` `NOT NULL` and always populate it at submit time,
which removes the `COALESCE` fallbacks in the view; `config_id` can then relax from
`RESTRICT` to `SET NULL`.

### Summary — columns that can go

| Column / table | Reason |
| --- | --- |
| `results` (table) | 1:1 with `trials` |
| `results.job_id` | Transitive dependency |
| `results.metric` | Duplicates the job's objective |
| `datasets.train_file`, `validation_file` | Derived from `name` |
| `datasets.{train,validation}_{records,file_size}`, `data_format` | Repeating group → `dataset_files` |
| `jobs.cleanup`, `autotune`, `seed`, `precision` | Not filtered on → `config_snapshot` |
| `jobs.ray_address` | Per-execution, not job definition |
| `jobs.tuning_type` | Overlaps `configurations.tuner_type` |
| `log_entries.iteration`, `epoch`, `filename` | Mixes progress metrics into logs |

---

## E. Cross-dialect portability

The ORM targets MySQL in production and SQLite in tests, with Postgres supported. Two
constructs in this file have no portable form.

**Generated columns (`:31`, `:34`).** No single literal SQL expression works everywhere:

- SQLite gained a `CONCAT()` function only in 3.44; before that it is `||` only.
- MySQL reads `||` as logical **OR** by default, so `||` cannot be the shared form.
- Postgres requires generation expressions to be immutable, and `concat()` is only
  *stable* — it rejects `CONCAT(...)`. `name || '_train'` is immutable and works.

One ORM expression covers all three regardless: SQLAlchemy's concatenation operator renders
`concat()` on MySQL and `||` on SQLite and Postgres, so `_suffixed()`
(`src/autotunex/db/tables/datasets.py:33-41`, used at `:61-68`) mirrors the schema everywhere
with no dialect branch. Dropping these columns (D5) removes the problem entirely.

**The view.** Postgres rejects it for the grouping reason in B3, and
`JSON_UNQUOTE(JSON_EXTRACT(...))` is MySQL-only (`->>` on Postgres, `json_extract` on
SQLite). The API composes the equivalent query in SQLAlchemy instead, so this costs nothing.
Equivalent in *shape*, not column for column: `num_trials` needs no `trials` query there at
all, because the API reports the configured trial budget
(`config_data.tune_config.num_samples`), not a row count
(`src/autotunex/db/repositories/sqlalchemy.py:202`). The corrected view below keeps the
row-count semantics, for the direct-MySQL consumers it is written for.

Three behavioural differences remain that no abstraction removes:

1. MySQL's default collation is case-**insensitive**, so `users.email UNIQUE` (`:6`) rejects
   `A@x.com` when `a@x.com` exists. Postgres and SQLite accept both as distinct.
2. MySQL `TIMESTAMP` spans 1970–2038 (A4).
3. MySQL normalizes and reorders JSON object keys on storage; SQLite preserves the input
   text. Compare parsed JSON, never raw strings.

One operational note: SQLite disables foreign keys unless `PRAGMA foreign_keys=ON` is issued
per connection. The test fixtures do issue it (`tests/conftest.py:254-258`), so the `CASCADE`
and `RESTRICT` rules in this schema are enforced under test. `src/autotunex/db/session.py`
does not, so it is a SQLite *runtime* — a dev or standalone deployment pointed at a SQLite
URL — that leaves them silently unenforced.

### Corrected view

Provided for anyone querying MySQL directly. Not created by Alembic — the API does not read
it.

```sql
CREATE OR REPLACE VIEW autotunex_jobs_v2 AS
SELECT
    j.id, j.user_id, j.status, j.seed, j.config_id, j.dataset_id,
    j.model, j.model_source, j.experiment_name, j.tuning_type,
    j.ray_address, j.cleanup, j.autotune, j.created_at, j.updated_at,
    u.email AS user_email,
    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(j.config_snapshot, '$.name')),
             c.name) AS config_name,
    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(j.config_snapshot, '$.rl_tuner_type')),
             c.rl_tuner_type) AS rl_tuner_type,
    d.name AS dataset,
    (SELECT COUNT(*) FROM trials t WHERE t.job_id = j.id) AS num_trials,
    gt.id         AS task_id,
    gt.build_id,
    gt.status     AS task_status,
    gt.type       AS task_type,
    gt.pr_url     AS github_pr_url,
    gt.artifact_id,
    gt.artifact_uri,
    gt.build_status,
    gt.started_at AS task_started_at,
    gt.updated_at AS task_updated_at,
    gt.rits_url
FROM jobs j
INNER JOIN users          u ON j.user_id    = u.id
INNER JOIN configurations c ON j.config_id  = c.id
INNER JOIN datasets       d ON j.dataset_id = d.id
LEFT JOIN gb_tasks gt
       ON gt.id = (SELECT gt2.id
                     FROM gb_tasks gt2
                    WHERE gt2.job_id = j.id
                 ORDER BY gt2.started_at DESC, gt2.id DESC
                    LIMIT 1);
```

Changes: columns enumerated instead of `j.*` (B2); `num_trials` via scalar subquery, so no
`GROUP BY` (B3); no `ORDER BY` (B4); latest task only, so exactly one row per job (B1);
`config_snapshot` and `output_artifacts` omitted as unsuitable for list queries; `user_email`
instead of `user` (C11).

The `num_trials` subquery here deliberately preserves the view's own row-count meaning — it is
not what the API computes. The API's `num_trials` is the configured trial budget
(`config_data.tune_config.num_samples`), not a row count.

The latest-task subquery orders by `started_at`, which is a `VARCHAR` (B5) — it sorts
lexicographically and is correct only for zero-padded ISO strings. Fixing B5 makes it
reliable.

### Dialect verification (2026-07-29)

The baseline migration's three commands (`upgrade head`, `check`, `downgrade base`) were run
locally against real SQLite, PostgreSQL 16, and MySQL 8.4 servers — not just SQLite — and a
CI job (`migrations`, matrixed over `dialect: [sqlite, postgres, mysql]`) now runs the same
three commands on every push and PR. All nine command/dialect combinations pass.

What the process found that this report did not already cover:

- **`alembic check` does not see the `TIMESTAMP`-vs-`DATETIME` drift from A4.** `check`
  compares the live database against `Base.metadata` — both built from the same
  `UtcDateTime` type — so on a database the migration itself created, the two sides always
  agree on `DATETIME NOT NULL` regardless of what the real schema file says. The drift
  against `resources/autotunex_schema.sql`'s `TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP` is
  real (confirmed by loading the raw schema file into a scratch MySQL database and comparing
  `SHOW CREATE TABLE`) but is invisible to any tool that only diffs the ORM against a
  database the ORM itself produced. Nothing currently diffs against the schema file. Adding
  `with_variant(mysql.TIMESTAMP(), "mysql")` to `UtcDateTime` was evaluated and **not
  applied**: it would not change what CI's `alembic check` reports (both sides would just
  agree on `TIMESTAMP` instead), and it reintroduces the 2038 cutoff on all 13 timestamp
  columns for no CI-visible benefit — exactly the tradeoff A4 already documents. Mirroring
  the defect, bugs included, remains the deliberate choice here.
- **The `mysql` extra was missing a real runtime dependency: `cryptography`.** MySQL 8.4
  defaults new users to `caching_sha2_password`. `asyncmy==0.2.11` declares zero
  dependencies of its own, but its auth handshake raises
  `RuntimeError: 'cryptography' package is required for sha256_password or
  caching_sha2_password auth methods` the moment a connection is opened before the server has
  cached a faster path for that session — i.e., on the very first connection to a freshly
  started server. That is the state of every CI service container on every run, so
  `mysql+asyncmy` connections failed in CI until `cryptography` was added alongside
  `asyncmy` wherever the `mysql` extra is declared. This did not surface in earlier isolated
  testing because a warm connection to an already-authenticated server skips the RSA
  exchange and masks the gap. **Resolved** — this is why the pin exists:
  `cryptography==48.0.1` is declared both as a base runtime dependency
  (`pyproject.toml:58`) and inside the `mysql` extra (`pyproject.toml:90`).
- **Everything else the plan anticipated did not reproduce.** Postgres created and dropped
  the `run_status` / `gb_task_type` enum types correctly via plain `create_table`/
  `drop_table` with no explicit `sa.Enum(...).create()`/`.drop()` needed; the `"precision"`
  reserved-word quoting and MySQL foreign-key downgrade ordering were not exercised because
  Task 7's revision (which touches `jobs.precision`) had not landed yet — only the baseline
  revision existed at verification time. That revision has since landed (`78f6bb7de0df`) and
  the `migrations` matrix runs the whole chain on all three dialects on every push
  (`.github/workflows/ci.yml:85-161`), so **the precision-backfill `COALESCE` check (seed a
  `precision` value with a null `config_snapshot`, upgrade, confirm the snapshot is not
  `NULL`) is done** — see section F. The generated columns (`concat(name, '_train')`) built
  without incident on MySQL, Postgres, and SQLite.

---

## F. Deviations applied to the ORM

Five changes are applied, at the maintainer's request. Each has an Alembic revision, and that
migration history — not this list — is the authority on how the ORM diverges from the file.

**`jobs.precision` is removed** (`:59`, revision `78f6bb7de0df`). It was `VARCHAR(50) NOT
NULL` with no default, so every writer had to supply it. This aligns with D6 — it is a
training hyperparameter nothing filters on.

Because the column holds live data, the migration backfills before dropping. On MySQL and
SQLite:

```sql
UPDATE jobs
   SET config_snapshot = json_set(COALESCE(config_snapshot, JSON_OBJECT()),
                                  '$.precision', precision)
 WHERE precision IS NOT NULL;
```

PostgreSQL has no `json_set`, so that branch composes with `jsonb` instead:

```sql
UPDATE jobs
   SET config_snapshot = COALESCE(config_snapshot::jsonb, '{}'::jsonb)
                      || jsonb_build_object('precision', "precision")
 WHERE "precision" IS NOT NULL;
```

Every branch is guarded by `WHERE precision IS NOT NULL`, and the drop goes through
`op.batch_alter_table("jobs")` rather than a bare `ALTER TABLE`, so SQLite gets its table
rebuild.

`COALESCE` is required: `json_set(NULL, ...)` returns `NULL` on **both MySQL and SQLite**
(verified against SQLite 3.50.2 — this is not a MySQL-only quirk, as an earlier draft of this
report implied), which would discard every value for rows with no snapshot. Because the values
are preserved in the snapshot, `downgrade()` restores them and the change round-trips.
Verified end to end on SQLite, MySQL 8.4 and PostgreSQL 16 with a seeded null-snapshot row.

Consumers reading `precision` from `autotunex_jobs` (via `j.*`) will break. `JobSummary` and
`JobRead` no longer expose the field.

**`datasets.description` is relaxed to nullable** (`:30`, revision `b27a008ed0cf`). The file
declares it `TEXT NOT NULL` with no default — and `TEXT` cannot take a literal default in
MySQL — so every writer had to supply a description. This applies C8.

**`datasets.status` and `datasets.status_detail` are added** (revision `7f175ebf55ad`,
`db/tables/datasets.py:74-77`). Both are absent from the schema file; they carry the
dataset-upload lifecycle the API needs.

**`jobs.reward_function_code` and `jobs.reward_function_name` are added** (revision
`c628b830e8a3`, `db/tables/jobs.py:72-73`). Also absent from the schema file. An online-RL
job persists its reward function in these dedicated columns rather than inside
`config_snapshot`.

**The `training_metrics` table is added** (revision `f09bd54b61b7`,
`db/tables/training_metrics.py`). The whole table is absent from the schema file. It backs the
per-step training-metrics read path, and is created with two `id`-trailing indexes —
`(job_id, id)` and `(job_id, trial_id, id)` — so that path's keyset page is served by the
index alone rather than falling back to a sort. Dual declaration is not what makes it special
— `src/api-bridge` mirrors all nine shared tables in its own metadata
(`src/api-bridge/src/api_bridge/tables.py:133-276`) and writes most of them. What is unique is
that `training_metrics` is the only one of those nine with no definition in the schema file to
anchor the two, so its two metadata declarations must stay in step by convention alone.

### The one non-baseline revision that is not a deviation — `0a2caef2a185`

`0a2caef2a185` widens `trials.id` and its foreign key `results.trial_id` from `VARCHAR(10)` to
`VARCHAR(16)` (`alembic/versions/0a2caef2a185_widen_trial_id_to_16.py:31-45`). It is recorded
here for completeness rather than as a sixth deviation: it moves the ORM *toward* the file, not
away from it. The baseline creates both columns at `sa.String(length=10)`
(`alembic/versions/1fb645a87b48_baseline_real_schema.py:195`, `:220`) — narrower than Ray's
trial ids, which would truncate — while the file already declares the wider type at
`resources/autotunex_schema.sql:73` and `:99`. A1 cites that `:73` correctly for the file, but
the migrations do not start there.

That asymmetry has a consequence for the live-MySQL procedure. A database built from
`resources/autotunex_schema.sql` directly starts *past* the baseline for those two columns, so
`alembic stamp 1fb645a87b48` followed by `upgrade head` would re-apply a widen already in
place — a straight stamp-then-upgrade does not describe such a database. The divergence needs
reconciling; `CLAUDE.md`'s migrations section records the same gap.
