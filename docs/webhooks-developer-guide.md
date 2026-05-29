# Webhook Push Notifications — Developer Guide

This document explains the internal architecture of gbserver's webhook push
notification system for developers working on the codebase.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BuildRunner Thread                         │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  Build Task  │───▶│  Event Queue │───▶│   Worker Task     │  │
│  │ (run_and_wait)│    │ (asyncio.Q)  │    │ (__worker_task)   │  │
│  └──────────────┘    └──────────────┘    └────────┬──────────┘  │
│                                                    │             │
│                                           ┌────────▼──────────┐  │
│                                           │ __dispatch_to_    │  │
│                                           │   webhooks()      │  │
│                                           │                   │  │
│                                           │ Lazy-creates      │  │
│                                           │ WebhookEventWriter│  │
│                                           └────────┬──────────┘  │
│                                                    │             │
│  ┌─────────────────────────────────────────────────▼──────────┐  │
│  │                  WebhookEventWriter                          │  │
│  │                                                             │  │
│  │  1. Query active subscriptions (status="active")            │  │
│  │  2. Match event against include/exclude filters             │  │
│  │  3. Serialize event → StoredWebhookEvent                    │  │
│  │  4. INSERT into gb_webhook_events table                     │  │
│  │                                                             │  │
│  │  (Fire-and-forget — no HTTP, no batching, no flushing)      │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

                              │
                              │ (Phase 2 — not yet implemented)
                              ▼
                ┌───────────────────────────┐
                │   Delivery Worker         │
                │   Reads pending events    │
                │   Batches by subscription │
                │   HMAC signs + HTTP POST  │
                └───────────────────────────┘
```

## Key Components

### 1. Subscription Storage (`src/gbserver/storage/webhook_subscription_storage.py`)

Webhook subscriptions are stored in the `gb_webhook_subscriptions` table (same
DB as builds). Supports both PostgreSQL and SQLite backends.
SQL implementations in `src/gbserver/storage/sql/webhook_subscription_storage.py`.

Key model: `StoredWebhookSubscription` in `webhooks/models.py`
- `uuid` — subscription identifier
- `status` — lifecycle state: `"pending"` → `"active"` → `"suspended"` / `"disabled"`
- `space_name` — space scope
- `build_filter` — optional per-build filter (`None` = space-wide, UUID = per-build)
- `webhook_url` — subscriber's HTTP endpoint
- `secret` — HMAC shared secret (minimum 8 characters)
- `event_types` — include filter (`["*"]` for all)
- `excluded_types` — exclude filter (takes priority)
- `frequency` — batch interval in seconds (min 15) — used by Phase 2 delivery worker
- `log_pattern` — optional regex for log scanning — used by Phase 2

### 2. WebhookEventWriter (`src/gbserver/webhooks/event_writer.py`)

Replaces the old `WebhookDispatcher` + `BatchBuffer` combination. One instance
per active build, created lazily when the first event arrives.

Responsibilities:
- Query active subscriptions matching the build (by `build_filter` and space-wide)
- Deduplicate subscriptions by UUID across all lookup paths
- For each matching subscription, serialize the event into a `StoredWebhookEvent`
- Persist the event to the `gb_webhook_events` table via the event storage layer
- Periodically re-read subscriptions (every 50 events) to pick up late arrivals
- No HTTP calls, no batching, no flushing — purely a DB write

### 3. Event Storage (`src/gbserver/storage/webhook_event_storage.py`)

Persistence layer for webhook events. Writes `StoredWebhookEvent` records to the
`gb_webhook_events` table. SQL implementations in
`src/gbserver/storage/sql/webhook_event_storage.py`. These records sit in
`"pending"` state until a delivery worker (Phase 2) picks them up.

### 4. Event Models (`src/gbserver/webhooks/event_models.py`)

Pydantic model for `StoredWebhookEvent`:
- `uuid` — event identifier
- `subscription_id` — FK to the subscription
- `build_id` — the build that produced this event
- `event_type` — the build event type
- `payload` — serialized event data (JSON dict)
- `delivered` — boolean, False until delivery worker marks it True
- `created_time` — timestamp

### 5. URL Validator (`src/gbserver/webhooks/url_validator.py`)

SSRF protection layer. Validates webhook URLs before subscription creation:
- Blocks private/reserved IP ranges (10.x, 172.16-31.x, 192.168.x, 127.x, etc.)
- Blocks link-local and loopback addresses
- Enforces HTTPS unless `GBSERVER_WEBHOOKS_ALLOW_HTTP=True`
- Resolves DNS to check the actual IP (prevents DNS rebinding)

### 6. URL Verification (`src/gbserver/webhooks/verification.py`)

Ownership verification challenge. When a subscription is created via the REST API:
1. Subscription starts in `"pending"` status
2. A challenge token is POSTed to the webhook URL with `X-GB-Event: verification` header
3. The endpoint must respond with `{"challenge": "<token>"}` in the body
4. On success, status transitions to `"active"`
5. On failure, subscription remains `"pending"` (can be retried)

Note: Auto-subscriptions via `--webhook-url` on build submission skip verification
and are created directly with `status="active"` (user explicitly provided the URL).

### 7. Integration Point (`src/gbserver/buildwatcher/buildrunner.py`)

The webhook system hooks into the BuildRunner's event processing loop via a
single method:

```python
# Every event goes through dispatch
self.__dispatch_to_webhooks(event)
```

The method:
1. Lazily creates a `WebhookEventWriter` on first event
2. EventWriter queries active subscriptions (status="active")
3. For each matching subscription, writes a `StoredWebhookEvent` to DB
4. No flushing, no delivery, no HTTP calls — just a DB INSERT

No periodic flush or final flush is needed — events are persisted immediately.

### 8. API Endpoints (`src/gbserver/api/webhooks.py`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/webhooks/spaces/{space}/subscriptions` | Create subscription (space-wide or per-build via `build_filter`) |
| GET | `/api/v1/webhooks/spaces/{space}/subscriptions` | List subscriptions (optional `?build_filter=` query) |
| DELETE | `/api/v1/webhooks/{webhook_id}` | Deactivate subscription |

The router uses `include_router()` (not `mount()`) so `AuthMiddleware` applies.
Auth is via `request.state.data["user"].login`.

Subscription creation includes:
- URL validation (SSRF protection)
- Rate limiting (max subscriptions per space)
- Secret length enforcement (minimum 8 characters)

### 9. Subscription Status Lifecycle

```
           ┌─────────────────────────────────────────┐
           │                                         │
  CREATE   │   VERIFY     ┌──────────┐              │
    │      │     │        │          ▼              │
    ▼      │     ▼        │    ┌───────────┐       │
┌─────────┐│ ┌────────┐  │    │ suspended │       │
│ pending │┼▶│ active │──┼───▶│           │───────┘
└─────────┘│ └────────┘  │    └───────────┘  (re-verify)
           │     │        │
           │     ▼        │
           │ ┌──────────┐ │
           │ │ disabled │ │
           │ └──────────┘ │
           └─────────────────────────────────────────┘
```

- **pending** — created, awaiting URL verification challenge
- **active** — verified, events are being persisted for this subscription
- **suspended** — temporarily disabled (e.g., repeated delivery failures in Phase 2)
- **disabled** — permanently deactivated by user or admin

### 10. Auto-Subscribe on Build Submission (`src/gbserver/api/builds.py`)

When a build is submitted with `webhook_url` parameter, a subscription is
automatically created via `_create_webhook_subscription()`:
- Created with `status="active"` (no verification needed — user provided the URL)
- Uses `build_filter=build_id` for per-build scoping
- Created BEFORE the build is enqueued to prevent race with first event
- Requires minimum 8-character secret
- URL validation is applied (SSRF check)
- Returns `webhook_warning` in response if subscription creation is skipped

## Event Flow

1. Build framework emits `BuildEvent` → `event_q.put_nowait(event)`
2. Worker task receives event → `__process_event(event)`
3. Inside `__process_event`: `__dispatch_to_webhooks(event)` is called
4. EventWriter checks `event.type.is_internal_event()` — skips TERMINATE, NEWARTIFACT_IN_ENVIRONMENT
5. EventWriter queries active subscriptions for this build's space
6. For each matching subscription, checks include/exclude filters
7. Serializes the event into a `StoredWebhookEvent`
8. INSERTs the event into the `gb_webhook_events` table
9. (Phase 2) Delivery worker reads pending events, batches by subscription, delivers via HTTP

## Phase 2 — Planned Features

- **Delivery worker** — separate process/CLI command (`gbserver webhook-worker`) that polls pending events and delivers via HTTP with HMAC signing and retry
- **`username_filter`** — optional field on subscriptions to receive events only for builds submitted by a specific user (avoids needing per-build subscriptions when multiple users share a space)
- **Delivery audit log** — `gb_webhook_deliveries` table tracking attempt history, status codes, response times
- **Auto-suspend** — after N consecutive delivery failures, auto-suspend the subscription
- **Log pattern scanning** — regex matching on build log lines to generate LOG_EVENTs

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GBSERVER_WEBHOOKS_ENABLED` | `True` | Master switch for the webhook system |
| `GBSERVER_WEBHOOKS_ALLOW_HTTP` | `False` | Allow non-HTTPS webhook URLs (dev only) |
| `GBSERVER_WEBHOOKS_MAX_PER_SPACE` | `20` | Max subscriptions per space |
| `GBSERVER_WEBHOOKS_MAX_PER_USER` | `50` | Max subscriptions per user |

The EventWriter is only created if webhooks are enabled. Zero overhead for
builds when the feature is disabled.

## Standalone Mode

In standalone mode (`gbserver standalone`):
- Uses SQLite for subscription and event storage
- uvicorn configured with `loop="asyncio"` (not uvloop) to avoid macOS issues
- The e2e test script (`scripts/test-webhook-e2e.py`) exercises the full flow
  including event persistence verification

## Testing

```bash
# Run unit tests for webhook components
pytest test/unit/webhooks/ -v

# Run the e2e test (starts local gbserver, verifies event persistence)
python scripts/test-webhook-e2e.py
```

## File Map

```
src/gbserver/api/
└── webhooks.py                           # APIRouter for subscription CRUD

src/gbserver/webhooks/                    # Webhook-specific logic
├── __init__.py
├── event_models.py                       # StoredWebhookEvent Pydantic model
├── event_writer.py                       # WebhookEventWriter — persists events to DB
├── models.py                             # StoredWebhookSubscription model
├── url_validator.py                      # SSRF protection — blocks private IPs
└── verification.py                       # URL ownership verification challenge

src/gbserver/storage/                     # Storage layer
├── webhook_subscription_storage.py       # IWebhookStorage + BaseWebhookStorage
├── webhook_event_storage.py              # IWebhookEventStorage + BaseWebhookEventStorage
└── sql/
    ├── webhook_subscription_storage.py   # SQLWebhookStorage + factory
    └── webhook_event_storage.py          # SQLWebhookEventStorage + factory
```
