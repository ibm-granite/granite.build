# Build Event Notifications

gbserver publishes build status events to a RabbitMQ topic exchange in real time.
Instead of polling the API for status changes, you subscribe to a build's event
stream and receive updates as they happen via an AMQP connection.

Key features:
- **Real-time** — events delivered as they occur (sub-second latency)
- **Scoped access** — each subscription gets credentials that can only read events for the requested build
- **No infrastructure to run** — gbserver provisions everything; you just connect
- **Non-blocking** — event publishing never affects build execution

## Subscribing to Events

### Step 1: Request credentials

Call the subscribe endpoint with your build ID:

```bash
curl -X POST https://gbserver/api/v1/builds/{build_id}/events/subscribe \
  -H "Authorization: Bearer $TOKEN"
```

Response:

```json
{
  "delivery_type": "rabbitmq",
  "host": "rabbitmq.internal",
  "port": 5671,
  "username": "tmp-build-abc12345-xK9f",
  "password": "...",
  "exchange": "build-events",
  "routing_key": "build.abc12345-full-uuid.#",
  "queue": "events.abc12345-full-uuid.xK9f",
  "expires_at": 1748886600
}
```

### Step 2: Connect to RabbitMQ

Use the returned credentials to connect and consume events. The credentials
are short-lived (60 seconds by default) — connect promptly after receiving them.
Once connected, the AMQP connection persists regardless of credential expiry.

### Access Control

| Caller | Can subscribe to |
|--------|-----------------|
| Build owner | Their own builds |
| Space member | Any build in their space |
| Admin | Any build |

Unauthenticated requests receive `401`. Unauthorized requests receive `403`.

### Reconnection

If your connection drops, call the subscribe endpoint again to get fresh
credentials and reconnect. Events published during the disconnection gap are
not recoverable (the temporary queue is deleted on disconnect).

## Event Payload

Each event is a JSON object published to the exchange:

```json
{
  "build_id": "abc12345-full-uuid",
  "event_type": "status_event",
  "timestamp": 1748857872,
  "target_name": "training",
  "step_name": "space://steps/sft",
  "source": "build-framework",
  "status": "running",
  "message": "Step started"
}
```

| Field | Always present | Description |
|-------|---------------|-------------|
| `build_id` | Yes | UUID of the build |
| `event_type` | Yes | Always `status_event` for published events |
| `timestamp` | Yes | Unix epoch seconds |
| `target_name` | Yes | Target that produced the event (empty for build-level) |
| `step_name` | Yes | Step URI (empty for target-level) |
| `source` | Yes | Event source identifier |
| `status` | Yes | Status value (pending, running, success, failed, cancelled) |
| `message` | Yes | Human-readable message |

## Event Types

**Only `status_event` is published to RabbitMQ.** Other event types (message,
artifact, metrics, workload status) are stored in the `gb_events` table but are
NOT published to the exchange.

| Event Type | Published to RabbitMQ | Stored in gb_events | Description |
|------------|:---------------------:|:-------------------:|-------------|
| `status_event` | Yes | Yes | Build/target/step status change |
| `message_event` | No | Yes | Log message from build execution |
| `artifact_event` | No | Yes | Artifact created |
| `artifact_pushed_event` | No | Yes | Artifact pushed to registry |
| `metrics_event` | No | Yes | Training metrics published |
| `workload_status_event` | No | Yes | K8s workload state change |

### Status Values

| Status | Meaning |
|--------|---------|
| `pending` | Build accepted, waiting for resources |
| `running` | Build actively executing |
| `success` | Build completed successfully |
| `failed` | Build failed |
| `cancelled` | Build was cancelled |

### Scope Identification

| `target_name` | `step_name` | Scope |
|---------------|-------------|-------|
| empty | empty | Build-level event |
| set | empty | Target-level event |
| set | set | Step-level event |

## Routing Key Patterns

Routing key format: `build.<build_id>.<event_type>`

| Consumer | Binding | Receives |
|----------|---------|----------|
| Single build subscriber | `build.abc123.#` | All events for one build |
| Status-only subscriber | `build.abc123.status_event` | Only status changes |
| Dashboard (all builds) | `build.#` | Everything |

## Client SDK

The `subscribe(build_id, callback)` pattern handles authentication, connection,
and reconnection transparently:

```python
from gbclient import subscribe

async def on_event(event):
    if event.get("status") == "failed":
        print(f"BUILD FAILED: {event['build_id']}")
    elif event.get("status") == "success":
        print(f"Build complete: {event['build_id']}")

# Handles: authenticate -> get credentials -> connect -> consume -> reconnect
await subscribe(build_id="abc123", callback=on_event)
```

## Best Practices

**Connect promptly.** Credentials expire in 60 seconds. Call the subscribe
endpoint and connect to RabbitMQ within that window.

**Use temporary queues.** Declare your queue as `exclusive=True, auto_delete=True`
so it's cleaned up automatically when you disconnect.

**Handle reconnection.** If your connection drops, call the subscribe endpoint
again for fresh credentials. Don't cache old credentials.

**Filter by routing key.** If you only need status events, bind with
`build.<id>.status_event` instead of `build.<id>.#` to reduce noise.

**Don't poll.** Events are pushed in real-time. There's no need to poll the
builds API for status — subscribe once and let events flow.

**For long-running monitoring** (dashboards), use a durable named queue instead
of an exclusive queue. This survives brief disconnections without losing events.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GBSERVER_EVENT_PUBLISHING_ENABLED` | `false` | Enable RabbitMQ event publishing (also requires `RABBITMQ_HOST`) |
| `GBSERVER_BUILD_EVENTS_EXCHANGE` | `build-events` | Topic exchange name |
| `GBSERVER_RABBITMQ_MGMT_URL` | `http://localhost:15672` | Management API URL |
| `GBSERVER_RABBITMQ_MGMT_USER` | `guest` | Management API user |
| `GBSERVER_RABBITMQ_MGMT_PASSWORD` | `guest` | Management API password |
| `GBSERVER_EVENT_SUBSCRIBE_TTL` | `60` | Credential TTL in seconds |
| `RABBITMQ_HOST` | `localhost` | RabbitMQ broker host |
| `RABBITMQ_PORT` | `5672` | RabbitMQ broker port |
| `RABBITMQ_USERNAME` | `guest` | RabbitMQ publish credentials |
| `RABBITMQ_PASSWORD` | `guest` | RabbitMQ publish credentials |

## Running RabbitMQ Locally

```bash
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management
```

Management UI: http://localhost:15672 (guest/guest)

Set the environment variables:

```bash
export RABBITMQ_HOST="localhost"
export RABBITMQ_PORT="5672"
export RABBITMQ_USERNAME="guest"
export RABBITMQ_PASSWORD="guest"
export GBSERVER_EVENT_PUBLISHING_ENABLED="true"
export GBSERVER_RABBITMQ_MGMT_URL="http://localhost:15672"
export GBSERVER_RABBITMQ_MGMT_USER="guest"
export GBSERVER_RABBITMQ_MGMT_PASSWORD="guest"
```

Verify connectivity:

```bash
curl -s http://localhost:15672/api/overview -u guest:guest | jq .cluster_name
```

To stop and remove the container:

```bash
docker stop rabbitmq && docker rm rabbitmq
```

---

## Internal Architecture

Everything below is for developers working on the gbserver codebase.

### Logger Framework Integration

`BuildEventPublishLogger` is integrated into the build logger stack via `get_message_logger()`:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           BuildRunner                                     │
│                                                                          │
│  get_message_logger(stored_build, event_source)                          │
│       │                                                                  │
│       ▼                                                                  │
│  BuildMultiMessageLogger                                                 │
│       │                                                                  │
│       ├── BuildEventMessageLogger ──────▶ gb_events table (always)       │
│       │                                                                  │
│       ├── BuildPRLogger ────────────────▶ GitHub PR comment (if PR)      │
│       │                                                                  │
│       └── BuildEventPublishLogger ──────▶ RabbitMQ (if enabled)          │
│               │                                                          │
│               │  filters: STATUS_EVENT only                              │
│               │  fire-and-forget, non-blocking                           │
│               ▼                                                          │
│         BuildEventPublisher.publish_event()                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

Loggers are wired via a registry pattern — each logger type registers itself
with a predicate (activation condition) and a factory. `get_message_logger()`
iterates the registry and collects active loggers. The `BuildEventPublishLogger`
is activated when `GBSERVER_EVENT_PUBLISHING_ENABLED=true` AND `RABBITMQ_HOST`
is set. It filters internally to `STATUS_EVENT` only, so even though all events
flow through the logger framework, only status changes are published to RabbitMQ.

### BuildEventPublisher

`src/gbserver/messaging/build_event_publisher.py`

Publishes `BuildEvent` objects to the RabbitMQ topic exchange.

- Exchange: `build-events` (configurable via `GBSERVER_BUILD_EVENTS_EXCHANGE`)
- Routing key format: `build.<build_id>.<event_type>`
- Skips internal events (TERMINATE, NEWARTIFACT_IN_ENVIRONMENT, NEW_MULTIARTIFACT)
- Thread-safe: uses `asyncio.Lock` to serialize concurrent publishes
- Serializes events to JSON: `{build_id, event_type, timestamp, target_name, step_name, source, status, message}`

### Subscribe Endpoint

`src/gbserver/api/event_subscribe.py`

```
POST /api/v1/builds/{build_id}/events/subscribe
Authorization: Bearer <token>
```

- Authenticates caller via existing auth middleware
- Verifies build exists
- Delegates to `messaging/subscription_service.py` which provisions scoped credentials via `RabbitMQAdmin`
- Returns transport-agnostic connection info (`delivery_type`, `host`, `port`, ...)

### Credential Lifecycle

RabbitMQ checks credentials at connection time only. Once connected,
the AMQP connection persists regardless of credential expiry.

```
t=0s    Client calls POST /events/subscribe -> gets credentials (TTL: 60s)
t=2s    Client connects to RabbitMQ (credentials valid)
t=60s   Credentials expire — no NEW connections possible
t=???   Events still flowing on existing connection
t=end   Client disconnects -> queue auto-deletes
t+60s   Cleanup task deletes the expired temp user
```

Background cleanup (`src/gbserver/messaging/credential_cleanup.py`):
- Polls RabbitMQ Management API every 60 seconds
- Deletes temp users with expired TTL
- Started automatically on server startup when event publishing is enabled

### Design Principles

1. **gbserver owns nothing about subscriptions** — RabbitMQ manages queue bindings,
   consumer lifecycle, and message routing. No subscription tables in our DB.

2. **Scoped credentials** — each temp user can only read events for the specific
   build they subscribed to. Compromised credentials can't read other builds.

3. **Non-blocking** — event publishing is fire-and-forget. RabbitMQ failures never
   affect build execution.

4. **No webhook delivery code** — external HTTP consumers (Slack, PagerDuty) are
   handled by off-the-shelf tools (Svix, n8n, etc.) that consume from RabbitMQ.
   We don't own delivery logic.

5. **Logger framework integration** — publishing uses the same `AbstractBuildLogger`
   interface as PR logging and event table logging. No separate dispatch paths.

### File Map

```
src/gbserver/buildrunner/
├── buildlogger.py                  # Logger framework: get_message_logger() factory,
│                                   #   AbstractBuildLogger, BuildMultiMessageLogger,
│                                   #   BuildEventMessageLogger, BuildPRLogger
└── build_event_publish_logger.py   # BuildEventPublishLogger (STATUS_EVENT -> RabbitMQ)

src/gbserver/messaging/
├── build_event_publisher.py        # Publishes events to RabbitMQ exchange
├── subscription_service.py         # Credential provisioning for subscribers
├── rabbitmq_admin.py               # RabbitMQ Management API client
├── credential_cleanup.py           # Background cleanup of expired temp users
├── messaging_base.py               # Abstract messaging interface
└── rabbitmq_base.py                # aio-pika RabbitMQ implementation

src/gbserver/api/
└── event_subscribe.py              # POST /builds/{id}/events/subscribe (thin endpoint)
```

### Testing

```bash
# Unit tests for event publishing and admin client
pytest test/unit/messaging/ -v

# Unit tests for the publish logger
pytest test/unit/buildrunner/test_build_event_publish_logger.py -v

# Integration test (requires running RabbitMQ)
RABBITMQ_HOST=localhost pytest test/integration/messaging/test_event_subscribe_e2e.py -v
```
