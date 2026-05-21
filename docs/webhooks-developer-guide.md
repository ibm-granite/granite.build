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
│                                           │ Lazy-initializes  │  │
│                                           │ WebhookDispatcher │  │
│                                           └────────┬──────────┘  │
│                                                    │             │
│  ┌─────────────────────────────────────────────────▼──────────┐  │
│  │                  WebhookDispatcher                          │  │
│  │                                                            │  │
│  │  ┌────────────────┐     ┌──────────────┐                  │  │
│  │  │ BatchBuffer    │     │ Delivery     │                  │  │
│  │  │ (per-sub       │────▶│ (aiohttp     │────▶ Subscriber  │  │
│  │  │  accumulator)  │     │  POST+HMAC)  │     Endpoint     │  │
│  │  └────────────────┘     └──────────────┘                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Subscription Storage (`src/gbserver/webhooks/sql_storage.py`)

Webhook subscriptions are stored in the `gb_webhook_subscriptions` table (same
DB as builds). Supports both PostgreSQL and SQLite backends.

Key model: `StoredWebhookSubscription` in `webhooks/models.py`
- `uuid` — subscription identifier
- `build_id` — specific build (null for space-wide)
- `space_name` — space scope
- `webhook_url` — subscriber's HTTP endpoint
- `secret` — HMAC shared secret
- `event_types` — include filter (`["*"]` for all)
- `excluded_types` — exclude filter (takes priority)
- `frequency` — batch interval in seconds (min 15)
- `log_pattern` — optional regex for log scanning

### 2. Dispatcher (`src/gbserver/webhooks/dispatcher.py`)

One `WebhookDispatcher` instance per active build. Created lazily when the
first event arrives. Holds a `BatchBuffer` and subscription registry.

- `start(subscriptions)` — register subscriptions for buffering
- `accept_event(event)` — buffer event for matching subscriptions
- `flush_all_ready()` — deliver batches where frequency elapsed
- `flush_final()` — force-flush all on build completion

### 3. Batch Buffer (`src/gbserver/webhooks/batch_buffer.py`)

Thread-safe in-memory accumulator. Tracks per-subscription:
- Pending events list
- Last flush timestamp
- Configured frequency

`get_ready_subscriptions()` returns IDs where enough time has elapsed
AND buffer is non-empty.

### 4. Delivery (`src/gbserver/webhooks/delivery.py`)

Handles the HTTP POST with:
- JSON serialization of payload
- HMAC-SHA256 signing (`X-GB-Signature-256`)
- Retry with exponential backoff (5 attempts: 1s, 2s, 4s, 8s, 16s)
- 10-second timeout per attempt

Uses `aiohttp.ClientSession` for async HTTP.

### 5. Integration Point (`src/gbserver/buildwatcher/buildrunner.py`)

The webhook system hooks into the BuildRunner's event processing loop:

```python
# Line 746 — every event goes through dispatch
self.__dispatch_to_webhooks(event)

# Line 598 — after each event, check for ready batches
await self.__flush_webhooks()

# Line 600 — on build completion, force-flush everything
if build_finished:
    await self.__flush_webhooks_final()

# Line 610 — on timeout (every 5s), flush ready batches
except TimeoutError:
    await self.__flush_webhooks()
```

### 6. API Endpoints (`src/gbserver/webhooks/api.py`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/webhooks/{build_id}/subscriptions` | Create per-build subscription |
| GET | `/api/v1/webhooks/{build_id}/subscriptions` | List per-build subscriptions |
| POST | `/api/v1/webhooks/spaces/{space}/subscriptions` | Create space-wide subscription |
| GET | `/api/v1/webhooks/spaces/{space}/subscriptions` | List space-wide subscriptions |
| DELETE | `/api/v1/webhooks/{webhook_id}` | Deactivate subscription |

### 7. Auto-Subscribe on Build Submission (`src/gbserver/api/builds.py`)

When a build is submitted with `webhook_url` parameter, a per-build subscription
is automatically created via `_create_webhook_subscription()`.

## Event Flow

1. Build framework emits `BuildEvent` → `event_q.put_nowait(event)`
2. Worker task receives event → `__process_event(event)`
3. Inside `__process_event`: `__dispatch_to_webhooks(event)` is called
4. Dispatcher checks `event.type.is_internal_event()` — skips TERMINATE, NEWARTIFACT_IN_ENVIRONMENT
5. For each registered subscription, checks include/exclude filters
6. Buffers the serialized event data
7. After processing, `__flush_webhooks()` checks `get_ready_subscriptions()`
8. If ready: `flush_subscription()` → serialize payload → sign → POST

## Configuration

Controlled by `GBSERVER_WEBHOOKS_ENABLED` (default: `True`).

The dispatcher is only created if subscriptions exist for the build. Zero
overhead for builds without webhook subscribers.

## Standalone Mode

In standalone mode (`gbserver standalone`):
- Uses SQLite for subscription storage
- uvicorn configured with `loop="asyncio"` (not uvloop) to avoid macOS issues
- The e2e test script (`scripts/test-webhook-e2e.py`) exercises the full flow

## Testing

```bash
# Run the e2e test (starts local gbserver + webhook receiver)
source .venv/bin/activate
python scripts/test-webhook-e2e.py

# Run unit tests for webhook components
pytest -s test/gbserver_test/webhooks/
```

## File Map

```
src/gbserver/webhooks/
├── __init__.py
├── api.py              # FastAPI routes for subscription CRUD
├── batch_buffer.py     # In-memory per-subscription event accumulator
├── delivery.py         # HMAC signing + HTTP POST with retry
├── dispatcher.py       # Orchestrates buffering and flushing per build
├── log_scanner.py      # Regex-based log line scanning
├── models.py           # StoredWebhookSubscription Pydantic model
├── sql_storage.py      # PostgreSQL + SQLite storage backends
└── storage.py          # IWebhookStorage interface
```
