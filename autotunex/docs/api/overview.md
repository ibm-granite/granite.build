# API overview

AutoTuneX is a FastAPI service for automated fine-tuning and hyperparameter
optimization of large language models. This page documents the conventions that
apply to every endpoint: where routes live, how requests are authenticated and
paginated, how results are scoped to their owner, and the single shape every
error takes.

For endpoint-by-endpoint detail, use the interactive docs (below). For the
credential modes referenced throughout, see [authentication.md](authentication.md).

## Base path

All resource endpoints live under a common prefix, `/api/v1` by default:

```
GET /api/v1/jobs
GET /api/v1/configurations
GET /api/v1/datasets
```

The prefix is configurable with the `AUTOTUNEX_API_PREFIX` environment variable.
If you change it, every resource path changes with it — the examples in this
documentation assume the `/api/v1` default.

Three health endpoints are mounted at the **root**, outside the prefix:
`GET /health`, `GET /health/live` (liveness alias), and `GET /health/ready`
(DB-gated readiness; `503` when the database is unreachable).

```bash
curl https://api.example.com/health
```

```json
{ "status": "ok", "service": "AutoTuneX API", "version": "0.3.5" }
```

`GET /health` never touches the database — it is a liveness probe, not a
readiness one — and it does not require authentication.

`GET /health/live` is an alias of `GET /health`: same payload, no database call,
no credential. It exists so an orchestrator can configure an explicit live/ready
split with two symmetrical paths.

```bash
curl https://api.example.com/health/live
```

```json
{ "status": "ok", "service": "AutoTuneX API", "version": "0.3.5" }
```

`GET /health/ready` is the readiness half, and the one health endpoint that
*does* touch the database: it runs a trivial `SELECT 1` and answers `200` only
once that succeeds. It needs no credential either.

```bash
curl https://api.example.com/health/ready
```

```json
{ "status": "ready", "database": "ok" }
```

A `SQLAlchemyError` — a dead pool connection, an unreachable host — comes back
as a `503` problem detail instead of the generic `500` an uncaught error would
produce, so an orchestrator can gate traffic on database reachability rather than
mere liveness. Any *other* failure still surfaces as a `500`.

```json
{
  "type": "about:blank",
  "title": "Service Unavailable",
  "status": 503,
  "detail": "The database is not reachable."
}
```

The bare service root (`/`) issues a temporary redirect to `/autotune`, where
the SPA mounts by default, so it 404s unless the SPA is built and
`AUTOTUNEX_FRONTEND_DIR` is configured. That target is hard-coded: changing
`AUTOTUNEX_FRONTEND_BASE_PATH` moves the SPA but not this redirect. The
interactive docs are at `/docs`.

`GET /api/v1/app-config` is the one *prefixed* endpoint that likewise needs no
credential: it returns non-sensitive, backend-defined values the web UI reads at
boot (the dataset upload size cap and the client-side gzip/preview thresholds),
none of which are user data.

## Interactive documentation

The service publishes its own OpenAPI schema and two browsable UIs:

| Path            | What it is                          |
| --------------- | ----------------------------------- |
| `/docs`         | Swagger UI (try requests in-browser)|
| `/redoc`        | ReDoc (reference-style rendering)   |
| `/openapi.json` | The raw OpenAPI 3.1 schema          |

These are the authoritative, always-current description of every endpoint,
request body, and response model. Generate client code from `/openapi.json`
rather than hand-transcribing schemas.

## Authentication

**Every data route requires authentication.** A request that presents no usable
credential is rejected with `401 Unauthorized` and a `WWW-Authenticate` header.

In the default **standalone** mode no credential is needed — every caller
resolves to the same principal, which is intended for local development and
single-user deployments. Production deployments configure a real credential
provider (API key, OIDC bearer token, or browser session). See
[authentication.md](authentication.md) for how to send each kind of credential
and how to configure the accepted providers.

## Pagination

The resource-collection endpoints — `GET /jobs`, `GET /configurations`,
`GET /datasets`, and `GET /users` — return a page, not a bare array. Two query
parameters control the window:

| Parameter | Type | Range        | Default | Meaning                          |
| --------- | ---- | ------------ | ------- | -------------------------------- |
| `limit`   | int  | 1–100        | 20      | Maximum items to return          |
| `offset`  | int  | ≥ 0          | 0       | Number of items to skip          |

```bash
curl "https://api.example.com/api/v1/jobs?limit=50&offset=100" \
  -H "X-API-Key: <your-key>"
```

The response is a `Page` object:

```json
{
  "items": [ /* ... up to `limit` records ... */ ],
  "total": 237,
  "limit": 50,
  "offset": 100
}
```

- `items` holds this page of records, **newest first**.
- `total` is the count of all matching records, **ignoring** pagination — use it
  to compute how many pages exist, not `len(items)`.
- `limit` and `offset` echo back the effective window.

`GET /jobs/{id}/trials` returns the same `Page` object with two differences worth
reading for: its `limit` defaults to **50** (range 1–100), and its `items` are
**oldest first** — chronological, the order the search evaluated them in — rather
than newest first like the resource collections above.

Three kinds of endpoint deliberately do not follow that shape:

- **Log endpoints** (`GET /jobs/{id}/logs` and
  `GET /jobs/{id}/trials/{trial_id}/logs`) use **keyset** pagination and return a
  `LogPage`, not a `Page`. They take `before_id` (int, ≥ 0, default `0` — the
  newest page) and their own `limit` (int, **1–500**, default **50**), and answer
  with `logs`, `has_more`, and `next_before_id` — the `before_id` to send for the
  next, older page.
- **Metrics endpoints** (`GET /jobs/{id}/metrics` and
  `GET /jobs/{id}/trials/{trial_id}/metrics`) also use **keyset** pagination, but
  ascending: they take `after_id` (int, ≥ 0, default `0` — the oldest page) and
  walk forward from it, rather than the log endpoints' descending `before_id`.
  Their `limit` is int, **1–2000**, default **500**, and they answer with a
  `MetricPage`, not a `Page` or a `LogPage` — `metrics`, `has_more`, and
  `next_after_id`, the `after_id` to send for the next, newer page.
- **Some sub-resources return a bare array**, with no window at all:
  `GET /jobs/{id}/result-report` returns a list of output-asset records, and
  `GET /jobs/{id}/gb-logs` returns a list of log lines (oldest-first).

## Ownership scoping

Reads, updates, and deletes on **jobs**, **configurations**, and **datasets** are
scoped to the caller's own rows by default. You see and act on what you own; you
do not see other owners' data — with one exception, the shared system tier
described below.

A `scope` query parameter widens that view:

| Value        | Effect                                                        |
| ------------ | ------------------------------------------------------------- |
| `own`        | The caller's own rows, plus the shared system tier (default). |
| `all`        | All owners' rows.                                             |

```bash
# Default — your own jobs only
curl "https://api.example.com/api/v1/jobs" -H "X-API-Key: <your-key>"

# Cross-user view — admin only
curl "https://api.example.com/api/v1/jobs?scope=all" -H "X-API-Key: <admin-key>"
```

`scope=all` is honored **only for an admin**. A non-admin who requests it gets
`403 Forbidden`. Being an admin grants the *ability* to ask for the cross-user
view — it is not an automatic all-tenants result: an admin who omits `scope`
still sees only their own rows.

**Configurations** and **datasets** carry one documented exception to that rule:
a shared system tier. A reserved *system user* — the fixed id
`00000000-0000-0000-0000-000000000001` — owns a curated set of starter
configurations and datasets, and **every** caller reads those in addition to its
own rows, including under the default `scope=own`. System-owned rows appear
inline in `GET /configurations` and `GET /datasets` alongside your own,
distinguishable only by that `user_id`, and a job may reference one at submit
time. **Jobs themselves have no shared tier.**

The tier is read-only. Mutations never widen to it, so `PUT` on a system-owned row
returns `404`, exactly as another owner's row would; only an admin passing
`?scope=all` may edit one. `POST /datasets/{id}/upload` takes no `scope` parameter
at all: it is strictly own-only, so not even an admin can upload into a shared
dataset, or into another owner's.

`DELETE` is stricter still, and is the one place a `404` is *not* what you get.
Deleting a system-owned row would remove starter content from every account at
once, so it is refused for **every** caller — a normal user, an admin passing
`?scope=all`, and a caller whose own identity happens to resolve to the system
user alike — with a `403` whose `title` is `System Resource Protected`. A `403`
rather than a `404` because the row is already readable by the caller, so naming
it leaks nothing, and a client needs to tell "protected" apart from "gone" to
explain itself. The single exemption is an admin with an active impersonation
overlay onto the system user (`POST /auth/assume/{id}`; see `authentication.md`),
which is the sanctioned, audited way to curate shared content.

> User-management endpoints are gated differently. They are admin-only in whole
> (there is no per-row "own" view of an identity) and take no `scope` parameter.
> See `users.md`.

## Errors

Every `4xx` and `5xx` response the API's own handlers emit — including
request-validation failures — is an
[RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) problem detail, served with
content type `application/problem+json`. Clients therefore parse exactly one
error shape:

| Field    | Type     | Present            | Meaning                                       |
| -------- | -------- | ------------------ | --------------------------------------------- |
| `type`   | string   | always             | A URI identifying the problem (`about:blank`).|
| `title`  | string   | always             | Short, human-readable summary of the class.   |
| `status` | int      | always             | The HTTP status code, repeated in the body.   |
| `detail` | string   | always             | Human-readable explanation of this instance.  |
| `errors` | array    | validation (422)   | Per-field details (see below).                |

A typical domain error (here, requesting `scope=all` as a non-admin):

```json
{
  "type": "about:blank",
  "title": "Forbidden",
  "status": 403,
  "detail": "Only an administrator may request the cross-user (scope=all) view."
}
```

On a request-validation failure the `errors` array is present, one entry per
offending field:

```json
{
  "type": "about:blank",
  "title": "Unprocessable Entity",
  "status": 422,
  "detail": "Request validation failed.",
  "errors": [
    {
      "type": "greater_than_equal",
      "loc": ["query", "limit"],
      "msg": "Input should be greater than or equal to 1",
      "input": "0"
    }
  ]
}
```

Error bodies never echo credentials or other caller-supplied secrets, and `5xx`
responses carry only a fixed, safe message — the underlying cause is logged
server-side, never returned.

> **The one exception.** Three handlers are registered — for domain errors, for
> request validation, and for any otherwise-unhandled exception — so a response
> Starlette raises before reaching them is served by Starlette's own default
> handler: an unrouted path (`404`) and a wrong method on a real path (`405`) come
> back as `{"detail": "..."}` with content type `application/json`, not as a
> problem detail.

## Status codes

| Code  | Name                  | When you see it                                                                 |
| ----- | --------------------- | ------------------------------------------------------------------------------- |
| `200` | OK                    | A successful `GET`, or a `PATCH`/`PUT` that returns the updated resource.        |
| `201` | Created               | A `POST` that creates a resource (a job, configuration, or dataset record).      |
| `202` | Accepted              | A dataset upload — the file is processed off the request, so success is polled.  |
| `204` | No Content            | A successful `DELETE`.                                                            |
| `400` | Bad Request           | A malformed request, e.g. presenting two credentials at once.                    |
| `401` | Unauthorized          | No credential, or a credential that failed to verify.                            |
| `403` | Forbidden             | Authenticated but not permitted — e.g. a non-admin requesting `scope=all`.       |
| `404` | Not Found             | The resource does not exist, or belongs to another owner (existence never leaks).|
| `409` | Conflict              | Duplicate name, a resource still referenced by a job, or a role/state guard.     |
| `413` | Payload Too Large     | An uploaded dataset file exceeds the server's size limit.                        |
| `415` | Unsupported Media Type| An uploaded file's format is not one of the accepted types.                      |
| `422` | Unprocessable Entity  | Request validation failed; see the `errors` array in the body.                   |
| `502` | Bad Gateway           | An upstream dependency (LLM gateway, build server) returned an invalid response. |
| `503` | Service Unavailable   | A required upstream dependency or optional feature is not currently available.   |

Notes:

- A `404` is deliberately returned when a resource exists but is owned by someone
  else, so a scoped caller cannot distinguish "yours, gone" from "someone
  else's". `403` is reserved for verdicts that reveal nothing about what data
  exists (for example, refusing `scope=all` to a non-admin).
- `409 Conflict` covers several distinct guards: a duplicate `(owner, name)`
  pair, deleting a configuration or dataset a job still references, a local
  in-process run still stopping on delete or cancel
  (`JobCancellationInProgressError`), and role guards on user management (you
  cannot change your own role, and the last administrator cannot be demoted).
