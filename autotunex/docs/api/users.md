# Users API

A **user** is an identity that owns configurations, datasets, and jobs — not an owned
resource itself. Because a user is an identity, management of users is gated by an
**admin** check rather than the `?scope=all` model the owned resources use: there is no
"own user" view to widen. This page documents the endpoints under `/api/v1/users`.

See [overview.md](overview.md) for shared conventions, [authentication.md](authentication.md)
for how a caller is resolved to a `Principal` and role, and [../concepts.md](../concepts.md)
for the domain model.

## Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/v1/users` | admin only | List all users, newest first |
| `GET` | `/api/v1/users/me/metadata` | any authenticated caller | The caller's own usage counts |
| `GET` | `/api/v1/users/{user_id}` | admin only | Get one user |
| `PATCH` | `/api/v1/users/{user_id}` | admin only | Change a user's role |

There is no user create/delete or email-edit endpoint. Email is an identity key linked to
the auth provider and is not editable here.

---

## GET /api/v1/users

List all users, newest first. **Admin only.** Returns a `Page<UserRead>`.

### Query parameters

| Name | Type | Default | Constraints |
| --- | --- | --- | --- |
| `limit` | int | `20` | 1–100 |
| `offset` | int | `0` | ≥ 0 |

### Response `200` — `Page<UserRead>`

`{ "items": UserRead[], "total": int, "limit": int, "offset": int }`.

### Notable statuses

`403` for a non-admin caller.

---

## GET /api/v1/users/me/metadata

Return the calling user's own job / configuration / dataset counts. Open to **any
authenticated caller** (not admin-gated). This route is declared before `/{user_id}`, so
the literal path is matched here. Returns a `UserMetadata`.

### Response `200` — `UserMetadata`

| Field | Type | Notes |
| --- | --- | --- |
| `number_of_jobs` | int | Jobs the caller owns |
| `number_of_configurations` | int | Configurations the caller owns |
| `number_of_datasets` | int | Datasets the caller owns |

```json
{ "number_of_jobs": 3, "number_of_configurations": 2, "number_of_datasets": 5 }
```

### Notable statuses

`401` if unauthenticated. A caller with no `users` row — unprovisioned under a real
provider, or unrestricted standalone — gets all-zero counts rather than an error.

---

## GET /api/v1/users/{user_id}

Return one user. **Admin only.** Returns a `UserRead`.

### Path parameter

| Name | Type | Notes |
| --- | --- | --- |
| `user_id` | UUID | User id |

### The `UserRead` shape

| Field | Type | Notes |
| --- | --- | --- |
| `id` | UUID | User id |
| `email` | string | Identity key, linked to the auth provider |
| `role` | string \| null | `admin` or `user`; nullable on read (may hold a legacy/pipeline value) |
| `created_at` | datetime | ISO 8601 |
| `updated_at` | datetime | ISO 8601 |

```json
{
  "id": "a2c9...",
  "email": "you@example.com",
  "role": "user",
  "created_at": "2026-08-01T09:00:00Z",
  "updated_at": "2026-08-10T12:00:00Z"
}
```

### Notable statuses

`403` (non-admin), `404` (no such user).

---

## PATCH /api/v1/users/{user_id}

Change a user's role. **Admin only.** Returns the updated `UserRead`. This is the only
mutable user field — email is not editable here.

### Path parameter

| Name | Type | Notes |
| --- | --- | --- |
| `user_id` | UUID | User id |

### Request body — `UserRoleUpdate`

| Field | Type | Required | Constraints |
| --- | --- | --- | --- |
| `role` | string | yes | `admin` \| `user` (any other value is a `422`) |

```bash
curl -X PATCH https://example.com/api/v1/users/a2c9... \
  -H "Content-Type: application/json" \
  -d '{ "role": "admin" }'
```

### Guards

- You cannot change **your own** role → `409`.
- You cannot demote the **last remaining admin** → `409`.

### Notable statuses

| Status | When |
| --- | --- |
| `403` | Caller is not an admin |
| `404` | No such user |
| `409` | Changing your own role, or demoting the last admin |
| `422` | `role` is not `admin` or `user` |

## See also

- [overview.md](overview.md) — base URL, pagination, error shape
- [authentication.md](authentication.md) — how roles and admin status are assigned
- [../concepts.md](../concepts.md) — the user concept and ownership
- [../operations/configuration.md](../operations/configuration.md) — auth provider and role settings
