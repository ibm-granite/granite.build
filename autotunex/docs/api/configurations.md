# Configurations API

A **configuration** is a named, reusable set of tuning settings, stored in a schema-less
JSON column (`config_data`). Configurations are the resource with full CRUD: you create,
read, update, and delete them through this API, and a job references one at submit time.
This page documents the endpoints under the `/api/v1/configurations` prefix.

See [overview.md](overview.md) for shared conventions, [authentication.md](authentication.md)
for owner resolution, and [../concepts.md](../concepts.md) for the domain model.

## Ownership and scope

Reads and mutations are owner-scoped. By default a caller — admin included — sees its own
configurations **plus** the shared system tier: rows owned by the reserved system user
(`00000000-0000-0000-0000-000000000001`), the curated starter configurations every caller
can read and launch a job from. An admin widens to all owners per request with `scope=all`
(`own` | `all`, default `own`); a non-admin passing `scope=all` gets a **403**. `POST` is
always own-scoped; a caller with no resolvable owner gets a **403** on create.

Mutations never widen to the shared tier, so `PUT` against a system-owned configuration
returns **404** — only an admin via `scope=all` may edit one.

`DELETE` against a system-owned configuration is refused outright, for **every** caller:
a normal user, an admin passing `scope=all`, and a caller whose own identity resolves to
the system user all get a **403** (`title`: `System Resource Protected`). Starter content
is shared by the whole deployment, so a delete is not a per-tenant loss. The single
exemption is an admin with an active impersonation overlay onto the system user
(`POST /auth/assume/{id}`) — the sanctioned way to curate shared content. To keep an
editable copy of a shared configuration instead, `GET` it and `POST` its `name`,
`tuner_type`, `rl_tuner_type` and `config_data` back as a new configuration, which is
created under your own ownership.

`config_data` shape is **not** validated. The tuning pipeline writes a rich, evolving
structure, so the API requires only that `config_data` be a non-empty JSON object.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/configurations` | Create a configuration |
| `GET` | `/api/v1/configurations` | List configurations, newest first |
| `GET` | `/api/v1/configurations/template` | Get a starter-template `config_data` object |
| `GET` | `/api/v1/configurations/{configuration_id}` | Get one configuration |
| `PUT` | `/api/v1/configurations/{configuration_id}` | Fully replace a configuration |
| `DELETE` | `/api/v1/configurations/{configuration_id}` | Delete a configuration |

There is no `PATCH`; updates are `PUT`-only full replacements.

---

## POST /api/v1/configurations

Create a configuration owned by the calling principal. Returns `201` with the full
`ConfigurationRead`. Unknown fields are rejected (`extra="forbid"`).

### Request body — `ConfigurationCreate`

| Field | Type | Required | Default | Constraints |
| --- | --- | --- | --- | --- |
| `name` | string | yes | — | 1–255 chars |
| `tuner_type` | string \| null | no | `null` | ≤ 50 chars |
| `rl_tuner_type` | string \| null | no | `null` | ≤ 50 chars |
| `config_data` | object | yes | — | Non-empty JSON object; shape not enforced |

```bash
curl -X POST https://example.com/api/v1/configurations \
  -H "Content-Type: application/json" \
  -d '{
    "name": "granite-sft-sweep",
    "tuner_type": "sft",
    "config_data": { "training_config": { "learning_rate": { "min_val": 1e-5, "max_val": 1e-3 } } }
  }'
```

### Response `201` — `ConfigurationRead`

See [the `ConfigurationRead` shape](#the-configurationread-shape). On create,
`associated_jobs` is always `[]` — a brand-new configuration cannot yet be referenced.

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Caller has no resolvable owner (unprovisioned) |
| `409` | The caller already owns a configuration with this `name` (unique per owner) |
| `422` | `config_data` is empty or not a JSON object, or the body otherwise fails validation |

---

## GET /api/v1/configurations

List the caller's configurations, newest first. Returns a `Page<ConfigurationSummary>`.

### Query parameters

| Name | Type | Default | Constraints |
| --- | --- | --- | --- |
| `limit` | int | `20` | 1–100 |
| `offset` | int | `0` | ≥ 0 |
| `scope` | string | `own` | `own` \| `all` (admin only for `all`) |
| `q` | string | `none` | Case-insensitive substring filter. Matches the configuration name. |

### Response `200` — `Page<ConfigurationSummary>`

`{ "items": ConfigurationSummary[], "total": int, "limit": int, "offset": int }`. Here
`associated_jobs` is populated (owner-scoped) for each configuration.

`ConfigurationSummary` is the lean list shape: every field of `ConfigurationRead` except
`config_data`, which is *absent* rather than `null` — a `null` could not be told apart from a
row that genuinely has no `config_data` stored. See
[the `ConfigurationRead` shape](#the-configurationread-shape) below for those fields; a client
that needs a configuration's search space reads
`GET /api/v1/configurations/{configuration_id}`.

### Notable statuses

`403` if a non-admin requests `scope=all`.

---

## GET /api/v1/configurations/template

Return the starter-template `config_data` object for a new configuration, provided by the
tuning backend. The response is a plain JSON object. This route is declared before
`/{configuration_id}`, so the literal path `template` is matched here rather than parsed
as an id.

### Response `200` — `object`

An opaque JSON object suitable as a starting `config_data`.

### Notable statuses

`503` if the template provider is unavailable.

---

## GET /api/v1/configurations/{configuration_id}

Return one configuration. Returns `ConfigurationRead`.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `configuration_id` | path | UUID | — | Configuration id |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### The `ConfigurationRead` shape

| Field | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Configuration id |
| `user_id` | string | Owner's id |
| `name` | string | Configuration name (unique per owner) |
| `tuner_type` | string \| null | Tuner type, if set |
| `rl_tuner_type` | string \| null | RL tuner type, if set |
| `config_data` | object \| null | The tuning settings; **nullable on read** (legacy rows may hold `null`) |
| `associated_jobs` | `ConfigurationJobRef[]` | Compact refs to jobs launched from this configuration; `[]` on create |
| `created_at` | datetime | ISO 8601 |
| `updated_at` | datetime | ISO 8601 |

**`ConfigurationJobRef`** (owner-scoped):

| Field | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Job id |
| `experiment_name` | string \| null | The job's run label |
| `status` | string | One of the six run states |

```json
{
  "id": "d4e1...",
  "user_id": "a2c9...",
  "name": "granite-sft-sweep",
  "tuner_type": "sft",
  "rl_tuner_type": null,
  "config_data": { "training_config": { "...": "..." } },
  "associated_jobs": [
    { "id": "6b1f...", "experiment_name": "sweep-2026-08", "status": "running" }
  ],
  "created_at": "2026-08-10T12:00:00Z",
  "updated_at": "2026-08-10T12:00:00Z"
}
```

### Notable statuses

`403` (non-admin requesting `scope=all`), `404` (no such configuration, or not the caller's).

---

## PUT /api/v1/configurations/{configuration_id}

Fully replace a configuration's mutable fields. Same body as create
(`ConfigurationCreate`). Returns `ConfigurationRead`.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `configuration_id` | path | UUID | — | Configuration id |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Non-admin requesting `scope=all` |
| `404` | No such configuration, or not the caller's |
| `409` | The new `name` collides with another of the caller's configurations |
| `422` | `config_data` is empty/not an object, or the body otherwise fails validation |

---

## DELETE /api/v1/configurations/{configuration_id}

Delete a configuration. Returns `204` with an empty body.

A configuration owned by the reserved system user cannot be deleted by anyone — see
*Ownership and scope* above. `scope=all` does not override it.

### Path & query parameters

| Name | In | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `configuration_id` | path | UUID | — | Configuration id |
| `scope` | query | string | `own` | `own` \| `all` (admin only for `all`) |

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Non-admin requesting `scope=all` |
| `403` | The configuration belongs to the shared system tier (`title`: `System Resource Protected`); refused for every caller except an admin impersonating the system user |
| `404` | No such configuration, or not the caller's |
| `409` | A job still references this configuration (delete is restricted) |

## See also

- [overview.md](overview.md) — base URL, pagination, error shape
- [authentication.md](authentication.md) — owners, admin, and `scope`
- [jobs.md](jobs.md) — jobs reference a configuration at submit time
- [../concepts.md](../concepts.md) — the configuration concept and `config_data`
- [../operations/configuration.md](../operations/configuration.md) — service settings
