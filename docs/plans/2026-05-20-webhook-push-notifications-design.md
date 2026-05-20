# Design: Push Notifications for Build Events (Webhooks)

**Issue:** [#8 — Feature request: Push notifications of build events](https://github.com/ibm-granite/granite.build/issues/8)
**Branch:** `feat/8-push-notifications-build-events`
**Date:** 2026-05-20

## Problem

Services using granite.build programmatically create many builds and must poll for status changes. A push mechanism (webhooks) allows subscriptions to events for specific builds, eliminating polling loops.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Event granularity | Build + step-level with filter | ~15% more work than build-only; gives step progress visibility |
| Subscription scope | Per-build first, space-wide later | Directly solves polling; space-wide is natural extension |
| Delivery | Retry with exponential backoff | Handles transient failures without full queue infrastructure |
| Execution model | Background asyncio.Task | Decouples webhook delivery from event processing |
| Authentication | HMAC-SHA256 signature | Industry standard (GitHub/Stripe), secret never in transit |

## Architecture

```
Build Step -> dispatch_event() -> asyncio.Queue -> BuildRunner.__process_event()
                                                        |
                                                   persist event
                                                        |
                                                   lookup subscriptions
                                                        |
                                                   spawn asyncio.Task --> sign payload (HMAC-SHA256)
                                                                                |
                                                                          POST to webhook_url
                                                                                |
                                                                     on failure: retry w/ backoff
```

## Storage Model

New table: `gb_webhook_subscriptions`

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID | PK |
| `space_name` | str | Required — scopes access control |
| `build_id` | UUID | Nullable — NULL means space-wide (phase 2) |
| `webhook_url` | str | Delivery endpoint |
| `secret` | str | HMAC signing key |
| `event_types` | JSON array | Filter: `["STATUS_EVENT"]` or `["STATUS_EVENT", "ARTIFACT_EVENT"]` etc. |
| `created_by` | str | Username who created it |
| `active` | bool | Soft-disable without deletion |
| `created_time` / `updated_time` | datetime | Auto-managed |

## REST API

### Phase 1 — Per-build subscriptions

```
POST   /api/v1/builds/{build_id}/webhooks    — Create subscription
GET    /api/v1/builds/{build_id}/webhooks    — List subscriptions for a build
DELETE /api/v1/webhooks/{webhook_id}          — Remove subscription
```

### Phase 2 — Space-wide subscriptions

```
POST   /api/v1/spaces/{space_name}/webhooks  — Create space-wide subscription
GET    /api/v1/spaces/{space_name}/webhooks  — List space subscriptions
```

### Create Subscription Request

```json
{
  "webhook_url": "https://example.com/hooks/gb",
  "secret": "my-signing-secret",
  "event_types": ["STATUS_EVENT", "ARTIFACT_EVENT"]
}
```

### Create Subscription Response

```json
{
  "id": "uuid",
  "build_id": "uuid",
  "space_name": "my-space",
  "webhook_url": "https://example.com/hooks/gb",
  "event_types": ["STATUS_EVENT", "ARTIFACT_EVENT"],
  "active": true,
  "created_time": "2026-05-20T12:00:00Z"
}
```

## Webhook Delivery Format

### Payload

```json
{
  "event_id": "uuid",
  "event_type": "STATUS_EVENT",
  "timestamp": "2026-05-20T12:00:00Z",
  "build_id": "uuid",
  "build_name": "my-build",
  "space_name": "my-space",
  "status": "SUCCESS",
  "target_name": "finetune",
  "step_name": "hfpull",
  "message": "Step completed successfully",
  "failure_reason": null
}
```

### Headers

```
Content-Type: application/json
X-GB-Event: STATUS_EVENT
X-GB-Delivery: <unique-delivery-uuid>
X-GB-Signature-256: sha256=<hmac-hex-digest>
```

### Signature Verification (client-side)

```python
import hmac, hashlib

def verify(payload_bytes: bytes, signature_header: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), payload_bytes, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)
```

## Retry Policy

- Max attempts: 5
- Backoff: exponential — 1s, 2s, 4s, 8s, 16s
- Timeout per attempt: 10s
- Trigger: non-2xx response or connection error
- Failures are logged but do not affect build execution

## Auto-Cleanup

Per-build subscriptions are automatically deactivated when the build reaches a terminal state (SUCCESS, FAILED, CANCELLED) after delivering the final status event.

## New Module: `src/gbserver/webhooks/`

| File | Responsibility |
|------|---------------|
| `__init__.py` | Package init |
| `models.py` | Pydantic models for subscription and delivery payload |
| `storage.py` | Subscription CRUD (SQLAlchemy) |
| `dispatcher.py` | Match events to subscriptions, spawn delivery tasks |
| `delivery.py` | HMAC signing, HTTP POST, retry with tenacity |
| `api.py` | FastAPI router for webhook subscription endpoints |

## Integration Points

1. **`BuildRunner.__process_event()`** — After event persistence, call `dispatcher.dispatch(event)` to look up matching subscriptions and fire background tasks.
2. **`api/root_api.py`** — Mount webhook API router.
3. **`storage/singleton_storage.py`** — Register webhook subscription storage.
4. **`types/constants.py`** — Add `GBSERVER_WEBHOOKS_ENABLED` env var (default: True).

## Existing Code to Leverage

- `src/gbserver/resilience/alert_handlers.py` — `WebhookAlertHandler` pattern for HTTP POST delivery
- `src/gbserver/resilience/retry_handler.py` — Tenacity-based retry infrastructure
- `src/gbserver/storage/sql/` — SQLAlchemy model/storage patterns to follow
- `src/gbserver/api/` — FastAPI route patterns and auth middleware

## Phase Summary

| Phase | Scope |
|-------|-------|
| 1 | Per-build subscriptions, STATUS + step-level events, HMAC auth, retry delivery |
| 2 | Space-wide subscriptions (nullable build_id), delivery logging/history endpoint |
