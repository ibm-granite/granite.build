# Design: Push Notifications for Build Events (Webhooks)

**Issue:** [#8 — Feature request: Push notifications of build events](https://github.com/ibm-granite/granite.build/issues/8)
**Branch:** `feat/8-push-notifications-build-events`
**Date:** 2026-05-20
**Revised:** 2026-05-20 (incorporated batching, log patterns, expanded subscription model)

## Problem

Services using granite.build programmatically create many builds and must poll for status changes. A push mechanism (webhooks) allows subscriptions to events for specific builds, eliminating polling loops.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Event granularity | All event types (full firehose) | Batching controls volume; subscribers filter via `event_types` / `excluded_types` |
| Delivery model | **Batched** — accumulate events, flush every N seconds | Reduces webhook call volume; single retry per batch; one payload = full picture |
| Batch frequency | Default 30s, configurable per-subscription, minimum 15s | Balances responsiveness vs. load |
| Subscription scope | Per-build first, space-wide later | Directly solves polling; space-wide is natural extension |
| Retry | 5 attempts, exponential backoff per batch delivery | Handles transient failures without full queue infrastructure |
| Execution model | Background asyncio.Task with periodic flush | Decouples webhook delivery from event processing |
| Authentication | HMAC-SHA256 signature | Industry standard (GitHub/Stripe), secret never in transit |
| Log monitoring | Regex pattern scanned per batch window | Subscriber-defined `log_pattern` matches against build log output in time-blocks |

## Architecture

```
Build Step -> dispatch_event() -> asyncio.Queue -> BuildRunner.__process_event()
                                                        |
                                                   persist event
                                                        |
                                                   accumulate in WebhookBatchBuffer
                                                        |
                                              (every N seconds per subscription)
                                                        |
                                                   flush batch
                                                        |
                                          scan logs for log_pattern matches
                                                        |
                                              build batched payload (list)
                                                        |
                                         sign payload (HMAC-SHA256) + POST
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
| `event_types` | JSON array | Filter: `["STATUS_EVENT"]` or `["*"]` for all |
| `excluded_types` | JSON array | Blocklist: events to exclude even if `*` is used |
| `frequency` | int | Batch window in seconds (default: 30, min: 15) |
| `log_pattern` | str | Nullable — regex to scan build logs; matches emit LOG_EVENT |
| `created_by` | str | Username who created it |
| `active` | bool | Soft-disable without deletion |
| `metadata` | JSONB | Arbitrary key/value pairs for future extensibility |
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
  "event_types": ["*"],
  "excluded_types": ["METRICS_EVENT"],
  "frequency": 30,
  "log_pattern": "(?i)(error|exception|traceback)"
}
```

### Create Subscription Response

```json
{
  "id": "uuid",
  "build_id": "uuid",
  "space_name": "my-space",
  "webhook_url": "https://example.com/hooks/gb",
  "event_types": ["*"],
  "excluded_types": ["METRICS_EVENT"],
  "frequency": 30,
  "log_pattern": "(?i)(error|exception|traceback)",
  "active": true,
  "created_time": "2026-05-20T12:00:00Z"
}
```

## Webhook Delivery Format

### Payload (batched — list of events)

Each delivery is a JSON array of events accumulated since the last delivery:

```json
{
  "delivery_id": "uuid",
  "build_id": "uuid",
  "build_name": "my-build",
  "space_name": "my-space",
  "user": "jane.doe",
  "build_start_time": "2026-05-20T11:00:00Z",
  "batch_start": "2026-05-20T12:00:00Z",
  "batch_end": "2026-05-20T12:00:30Z",
  "events": [
    {
      "event_id": "uuid",
      "event_type": "STATUS_EVENT",
      "timestamp": "2026-05-20T12:00:05Z",
      "target_name": "finetune",
      "step_name": "hfpull",
      "status": "RUNNING",
      "message": {
        "text": "Step started",
        "phase": "initialization"
      }
    },
    {
      "event_id": "uuid",
      "event_type": "STATUS_EVENT",
      "timestamp": "2026-05-20T12:00:28Z",
      "target_name": "finetune",
      "step_name": "hfpull",
      "status": "SUCCESS",
      "message": {
        "text": "Step completed",
        "duration_seconds": 23
      }
    },
    {
      "event_id": "uuid",
      "event_type": "LOG_EVENT",
      "timestamp": "2026-05-20T12:00:15Z",
      "target_name": "finetune",
      "step_name": "hfpull",
      "message": {
        "matched_pattern": "(?i)(error|exception|traceback)",
        "line": "WARNING: Retrying after ConnectionError: timeout",
        "line_number": 142
      }
    }
  ]
}
```

### Headers

```
Content-Type: application/json
X-GB-Delivery: <unique-delivery-uuid>
X-GB-Signature-256: sha256=<hmac-hex-digest>
X-GB-Batch-Size: <number-of-events-in-batch>
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

## Event Type Filtering Logic

```python
def should_include(event_type: str, subscription) -> bool:
    # Check exclusion list first
    if event_type in subscription.excluded_types:
        return False
    # Wildcard includes everything not excluded
    if "*" in subscription.event_types:
        return True
    # Otherwise must be explicitly listed
    return event_type in subscription.event_types
```

## Batching Mechanism

The `WebhookBatchBuffer` accumulates events per subscription:

1. As events flow through `BuildRunner.__process_event()`, they are appended to in-memory buffers keyed by subscription ID.
2. A periodic flush task runs every second, checking if any subscription's buffer has reached its `frequency` threshold since last flush.
3. On flush: collect buffered events, scan logs for `log_pattern` matches since last flush, build batched payload, spawn delivery task.
4. If the buffer is empty at flush time (no events in window), no webhook is sent.

## Log Pattern Scanning

When a subscription has a `log_pattern`:

1. At each batch flush, scan MESSAGE_EVENT payloads that arrived since the last flush (in-process, time-block based).
2. Apply the regex against each message text.
3. For each match, synthesize a `LOG_EVENT` entry in the batch with the matched line, line number, and pattern.
4. Log scanning operates on MESSAGE_EVENTs already flowing through the BuildRunner event queue — no external log backend access required.

**Future enhancement:** For full log coverage (including verbose pod stdout not relayed via MESSAGE_EVENTs), add a `log_source` subscription option (`"event_stream"` vs `"cloud_logs"`). The cloud logs mode would use the IBM Cloud Logs API search pattern from [gb_dashboard/cloud_logs.py](https://github.ibm.com/granite-dot-build/gb_dashboard/blob/678c73e/src/gb_dashboard/services/cloud_logs.py#L227).

## Retry Policy

- Max attempts: 5
- Backoff: exponential — 1s, 2s, 4s, 8s, 16s
- Timeout per attempt: 10s
- Trigger: non-2xx response or connection error
- Failures are logged but do not affect build execution
- Retry is per-batch (the entire batch is re-sent, not individual events)

## Auto-Cleanup

Per-build subscriptions are automatically deactivated when the build reaches a terminal state (SUCCESS, FAILED, CANCELLED) — after delivering the final batch containing the terminal status event.

## New Module: `src/gbserver/webhooks/`

| File | Responsibility |
|------|---------------|
| `__init__.py` | Package init |
| `models.py` | Pydantic models for subscription and delivery payload |
| `storage.py` | Subscription CRUD interface and base implementation |
| `sql_storage.py` | SQL backend for subscriptions |
| `batch_buffer.py` | In-memory event accumulator with per-subscription flush timers |
| `log_scanner.py` | Regex scanning of build log output in time-blocks |
| `dispatcher.py` | Orchestrates batching, log scanning, and delivery |
| `delivery.py` | HMAC signing, HTTP POST, retry with exponential backoff |
| `api.py` | FastAPI router for webhook subscription endpoints |

## Integration Points

1. **`BuildRunner.__process_event()`** — After event persistence, append event to `WebhookBatchBuffer`.
2. **`BuildRunner` lifecycle** — Start flush task when build begins, stop on build completion.
3. **`api/root_api.py`** — Mount webhook API router.
4. **`types/constants.py`** — Add `GBSERVER_WEBHOOKS_ENABLED` (default: True), `GBSERVER_WEBHOOKS_DEFAULT_FREQUENCY` (default: 30), `GBSERVER_WEBHOOKS_MIN_FREQUENCY` (default: 15).

## Existing Code to Leverage

- `src/gbserver/resilience/alert_handlers.py` — `WebhookAlertHandler` and `RetryableAlertHandler` patterns
- `src/gbserver/storage/sql/` — SQLAlchemy model/storage patterns
- `src/gbserver/api/` — FastAPI route patterns and auth middleware
- `src/gbserver/buildwatcher/buildrunner.py:583-653` — Worker task pattern for periodic async loops

## Phase Summary

| Phase | Scope |
|-------|-------|
| 1 | Per-build subscriptions, all event types with filtering, batched delivery, HMAC auth, log_pattern, retry |
| 2 | Space-wide subscriptions (nullable build_id), delivery history/replay endpoint, Slack-specific formatting |
