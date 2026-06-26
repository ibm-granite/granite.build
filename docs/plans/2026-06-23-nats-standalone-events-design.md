# Design: NATS Event Notifications for Standalone Mode

**Issue:** #129
**Date:** 2026-06-23
**Status:** Approved

## Problem

The build event notification feature (PR #85) requires RabbitMQ, which doesn't exist in standalone mode. Standalone already boots an embedded NATS server with JetStream — we should use it for event pub/sub.

## Architecture

In standalone mode (`GB_ENVIRONMENT=STANDALONE`), build event notifications use the embedded NATS server instead of RabbitMQ. The switch is environment-driven — no new env vars needed.

### Publish Path

`BuildEventPublisher.from_env()` checks if `RABBITMQ_HOST` is set. If not, and we're in standalone with NATS available, it creates a `NATSMessaging` instance instead of `RabbitMQBase`. The publisher accepts any `MessagingBase` backend.

Subject format: `gbserver.build.<build_id>.<event_type>` — maps directly to RabbitMQ's `build.<build_id>.<event_type>` routing keys using NATS subject conventions.

### Subscribe Path

`provision_subscription()` checks the environment. In standalone/NATS mode, it skips `RabbitMQAdmin` entirely and returns NATS connection info (url + subject). No credentials needed — single-tenant, localhost only.

### Response Model

`SubscribeResponse` gains optional NATS fields. RabbitMQ-specific fields become optional:

```python
class SubscribeResponse(BaseModel):
    delivery_type: str        # "rabbitmq" or "nats"
    host: str                 # broker host
    port: int                 # broker port
    username: str | None      # None for NATS
    password: str | None      # None for NATS
    exchange: str | None      # None for NATS
    routing_key: str | None   # None for NATS
    queue: str | None         # None for NATS
    url: str | None           # NATS URL (e.g. "nats://localhost:4222")
    subject: str | None       # NATS subject filter (e.g. "gbserver.build.<id>.>")
    expires_at: int           # epoch seconds (far-future for NATS standalone)
```

### Credential Cleanup

The cleanup loop (`start_cleanup_loop`) becomes a no-op in standalone/NATS mode. Gate on backend type — no temp users to clean up with NATS.

### Standalone Defaults

Add `GBSERVER_EVENT_PUBLISHING_ENABLED: "true"` to `STANDALONE_ENV_DEFAULTS` so event publishing is on by default.

## Backend Selection Logic

```
BuildEventPublisher.from_env():
    if RABBITMQ_HOST is set:
        → RabbitMQBase (existing behavior)
    elif GB_ENVIRONMENT == STANDALONE and nats-py available:
        → NATSMessaging with GBSERVER_NATS_URL
    else:
        → not configured (no-op)

provision_subscription(build_id):
    if GB_ENVIRONMENT == STANDALONE:
        → return NATS connection info (url, subject, no credentials)
    else:
        → RabbitMQAdmin.create_scoped_user() (existing behavior)
```

## Testing

- **Unit tests**: Mock NATSMessaging, verify publish path selects NATS in standalone, verify subscribe endpoint returns correct NATS response, verify RabbitMQAdmin is skipped.
- **Integration test**: Starts embedded NATS, publishes an event, subscriber receives it. Gated on `nats-server` being on PATH (skipped otherwise).

## Files to Modify

1. `src/gbserver/messaging/build_event_publisher.py` — backend switch in `from_env()`
2. `src/gbserver/messaging/subscription_service.py` — NATS branch in `provision_subscription()`
3. `src/gbserver/api/event_subscribe.py` — make response fields optional
4. `src/gbserver/api/root_api.py` — gate cleanup loop on non-NATS mode
5. `src/gbserver/types/constants.py` — add to `STANDALONE_ENV_DEFAULTS`
6. `src/gbserver/messaging/nats_messaging.py` — fix `is_available()` to check `HAS_NATS`

## Files to Create

1. `test/unit/messaging/test_nats_event_publisher.py` — unit tests for NATS publish path
2. `test/unit/api/test_event_subscribe_nats.py` — unit tests for NATS subscribe response
3. `test/integration/messaging/test_nats_events_e2e.py` — integration test with real NATS
