# Dashboard Webhook Integration Design

**Date:** 2026-05-21
**Status:** Approved
**Related:** Issue #8 (Push Notifications for Build Events)

## Overview

Add a webhook receiver to the GB Dashboard web application so it can receive
real-time build status events from gbserver, reducing latency compared to the
current K8s polling approach (5s sync daemon cycles).

## Architecture

The webhook receiver lives in the **Web Dashboard** FastAPI app (`main.py`).
On startup, it registers space-wide subscriptions with gbserver. When events
arrive, it updates `gbd_builds` status directly in PostgreSQL.

```
gbserver (build events)
    │
    │  POST /api/webhooks/build-events
    │  (batched every 15s, HMAC-signed)
    ▼
┌──────────────────────────────┐
│  Web Dashboard (FastAPI)     │
│  api/webhooks.py             │
│    - verify HMAC signature   │
│    - upsert gbd_builds       │
│    - store in audit table    │
└──────────────┬───────────────┘
               │
               ▼
       PostgreSQL (gbd_builds)
               │
        ┌──────┴──────┐
        │             │
   Frontend        AI Daemon
   (reads DB)    (triggers faster)
```

## Components

### 1. New file: `src/gb_dashboard/api/webhooks.py`

FastAPI router with a single endpoint:

```
POST /api/webhooks/build-events
```

- Exempt from AuthMiddleware (uses HMAC verification instead)
- Reads raw body, computes HMAC-SHA256 with configured secret
- Compares against `X-GB-Signature-256` header
- Returns 401 on mismatch, 200 on success
- For each event in the batch:
  - `status_event` with `type=Build` → upsert `gbd_builds.status`
  - `status_event` with `type=Target/TargetStep` → update target/step timestamps
  - Other events → store in audit table only

### 2. New file: `src/gb_dashboard/services/webhook_subscriber.py`

Startup service that ensures space-wide subscriptions exist:

- Called during `lifespan()` startup
- For each space the dashboard monitors, calls:
  `POST {gbserver_url}/api/v1/webhooks/spaces/{space}/subscriptions`
- Stores returned subscription IDs in `gbd_webhook_subscriptions`
- Idempotent: skips if active subscription already exists in DB
- Uses `httpx` (already a dependency) for the registration call

### 3. Config additions (`config.py`)

```python
# Webhook receiver
webhook_enabled: bool       # GB_DASHBOARD_WEBHOOK_ENABLED (default: False)
webhook_secret: str         # GB_DASHBOARD_WEBHOOK_SECRET (required if enabled)

# gbserver connection (for subscription registration)
gbserver_api_url: str       # GB_DASHBOARD_GBSERVER_URL
gbserver_token: str         # GB_DASHBOARD_GBSERVER_TOKEN

# Tuning
webhook_frequency: int      # GB_DASHBOARD_WEBHOOK_FREQUENCY (default: 15)
```

### 4. DB schema additions (`db_schema.py`)

```python
class GbdWebhookSubscription(Base):
    __tablename__ = "gbd_webhook_subscriptions"

    id: UUID (primary key, default uuid4)
    space_name: str
    gbserver_subscription_id: UUID   # returned by gbserver API
    webhook_url: str                 # our receiver URL
    active: bool (default True)
    created_at: datetime
    updated_at: datetime

class GbdWebhookDelivery(Base):
    __tablename__ = "gbd_webhook_deliveries"

    id: int (autoincrement primary key)
    delivery_id: UUID               # from X-GB-Delivery header
    build_id: UUID
    space_name: str
    event_count: int
    received_at: datetime
    payload: JSONB                  # full payload for debugging
```

### 5. Registration in `main.py`

```python
from gb_dashboard.api import webhooks
dashboard_app.include_router(webhooks.router)  # /api/webhooks prefix

# In lifespan():
if config.webhook_enabled:
    from gb_dashboard.services.webhook_subscriber import ensure_subscriptions
    await ensure_subscriptions(config, session_factory)
```

Add `/api/webhooks` to `AuthMiddleware.PUBLIC_PATHS` (HMAC replaces session auth).

## Webhook Payload (from gbserver)

```json
{
  "delivery_id": "uuid",
  "build_id": "uuid",
  "build_name": "...",
  "space_name": "...",
  "user": "...",
  "build_start_time": "ISO-8601",
  "batch_start": "ISO-8601",
  "batch_end": "ISO-8601",
  "events": [
    {
      "event_id": "uuid",
      "event_type": "status_event",
      "timestamp": "ISO-8601",
      "target_name": "helloworld",
      "step_name": "space://steps/train",
      "status": "running",
      "message": {"text": "..."}
    }
  ]
}
```

Headers: `X-GB-Signature-256`, `X-GB-Delivery`, `X-GB-Batch-Size`

## HMAC Verification

```python
import hmac, hashlib

def verify_signature(body: bytes, secret: str, signature_header: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature_header, expected)
```

## What Does NOT Change

| Component | Impact |
|-----------|--------|
| Sync Daemon | Unchanged — still provides K8s resource data (pods, GPUs, AppWrappers) |
| AI Daemon | Minor — can optionally react to terminal events faster |
| Frontend (HTMX) | None — already reads from DB |
| Cloud Logs | None — remains pull-based |
| Auth | None — webhook uses HMAC, not session auth |

## Error Handling

- Invalid HMAC → 401, no processing
- Unknown build_id → create placeholder row in gbd_builds (sync daemon will enrich later)
- DB write failure → 500, gbserver retries with exponential backoff (up to 5 retries)
- Duplicate delivery_id → idempotent (check before insert)

## Testing

- Unit test: HMAC verification, event parsing, DB upsert
- Integration test: submit build with webhook_url on standalone gbserver, verify dashboard DB updates
- E2e: reuse pattern from `scripts/test-webhook-e2e.py`
