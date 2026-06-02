# Build Event Notifications — Developer Guide

This document explains the internal architecture of gbserver's build event
notification system for developers working on the codebase.

## Architecture Overview

gbserver publishes build events to a RabbitMQ topic exchange. Consumers
subscribe directly to RabbitMQ using short-lived, scoped credentials
provisioned by a REST endpoint. In standalone mode (no RabbitMQ), events
are delivered directly via platform-native adapters (macOS notifications,
email).

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BuildRunner Thread                             │
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Build Task  │───▶│  Event Queue │───▶│     Worker Task       │  │
│  │ (run_and_wait)│    │ (asyncio.Q)  │    │   (__worker_task)     │  │
│  └──────────────┘    └──────────────┘    └──────────┬────────────┘  │
│                                                      │               │
│                                         ┌────────────┼────────────┐  │
│                                         │            │            │  │
│                                         ▼            ▼            │  │
│                              ┌────────────────┐ ┌──────────────┐  │  │
│                              │ Event Bus      │ │ Standalone   │  │  │
│                              │ (RabbitMQ)     │ │ Notifications│  │  │
│                              │                │ │ (macOS/email)│  │  │
│                              │ Publishes to   │ │              │  │  │
│                              │ topic exchange │ │ Direct       │  │  │
│                              │ build.<id>.    │ │ delivery via │  │  │
│                              │ <event_type>   │ │ adapters     │  │  │
│                              └───────┬────────┘ └──────────────┘  │  │
│                                      │                            │  │
└──────────────────────────────────────┼────────────────────────────┘  │
                                       │                               │
                                       ▼                               │
                          ┌─────────────────────────┐                  │
                          │   RabbitMQ Topic Exchange │                  │
                          │   "build-events"          │                  │
                          └──────────┬────────────────┘                  │
                                     │                                   │
                    ┌────────────────┼────────────────┐                 │
                    │                │                │                  │
                    ▼                ▼                ▼                  │
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │
           │  User CLI    │ │  Dashboard   │ │  Off-shelf   │          │
           │  (temp queue)│ │  (durable q) │ │  tool        │          │
           └──────────────┘ └──────────────┘ └──────────────┘          │
```

## Key Components

### 1. BuildEventPublisher (`src/gbserver/messaging/build_event_publisher.py`)

Publishes `BuildEvent` objects to the RabbitMQ topic exchange.

- Exchange: `build-events` (configurable via `GBSERVER_BUILD_EVENTS_EXCHANGE`)
- Routing key format: `build.<build_id>.<event_type>`
- Skips internal events (TERMINATE, NEWARTIFACT_IN_ENVIRONMENT, NEW_MULTIARTIFACT)
- Thread-safe: uses `asyncio.Lock` to serialize concurrent publishes
- Serializes events to JSON: `{build_id, event_type, timestamp, target_name, step_name, source, status, message}`

### 2. RabbitMQ Admin (`src/gbserver/messaging/rabbitmq_admin.py`)

Client for the RabbitMQ Management HTTP API. Provisions and cleans up
temporary users for event consumers.

Methods:
- `create_scoped_user(build_id, exchange, ttl_seconds)` — creates a temp user with read-only permissions scoped to `build.<build_id>.*`
- `cleanup_expired_users()` — deletes expired temp users (runs on background loop)
- `delete_user(username)` — deletes a specific user

Username format: `tmp-build-{build_id[:8]}-{random_6}`

### 3. Subscribe Endpoint (`src/gbserver/api/event_subscribe.py`)

```
POST /api/v1/builds/{build_id}/events/subscribe
Authorization: Bearer <token>

Response:
{
  "rabbitmq_host": "...",
  "rabbitmq_port": 5671,
  "username": "tmp-build-abc12345-xK9f",
  "password": "<short-lived>",
  "exchange": "build-events",
  "routing_key": "build.abc12345-full-uuid.#",
  "queue": "events.abc12345-full-uuid.xK9f",
  "expires_at": "2026-06-02T18:30:00+00:00"
}
```

- Authenticates caller via existing auth middleware
- Verifies build exists
- Provisions scoped RabbitMQ credentials via `RabbitMQAdmin`
- Returns connection info for the consumer

### 4. Credential Cleanup (`src/gbserver/messaging/credential_cleanup.py`)

Background task that runs in the REST server process:
- Polls RabbitMQ Management API every 60 seconds
- Deletes temp users with expired TTL
- Started automatically on server startup when event publishing is enabled

### 5. Standalone Notifications (`src/gbserver/notifications/`)

For standalone mode (no RabbitMQ), events are delivered directly to the
user via platform-native adapters.

| Adapter | File | Target |
|---------|------|--------|
| macOS | `macos_adapter.py` | Notification Center via `osascript` |
| Email | `email_adapter.py` | SMTP to any email address |

Configuration: `~/.gbserver/notifications.yaml`

```yaml
notifications:
  - type: macos
    events: [status_event]

  - type: email
    to: "user@example.com"
    smtp_host: "smtp.example.com"
    smtp_port: 587
    smtp_user_env: "SMTP_USER"
    smtp_password_env: "SMTP_PASSWORD"
    events: [status_event]
```

The `StandaloneDispatcher` loads config, creates adapters, and routes
events to matching adapters based on the `events` filter list.

### 6. Integration Point (`src/gbserver/buildwatcher/buildrunner.py`)

Two dispatch methods fire for every event (via `asyncio.ensure_future`):

```python
# Publishes to RabbitMQ (when GBSERVER_EVENT_PUBLISHING_ENABLED=true)
asyncio.ensure_future(self.__dispatch_to_event_bus(event))

# Delivers locally (when GB_ENVIRONMENT=STANDALONE and no RabbitMQ)
asyncio.ensure_future(self.__dispatch_standalone_notification(event))
```

Both are fire-and-forget, non-blocking, non-fatal on error.

## Event Flow

### Clustered Mode (with RabbitMQ)

1. Build framework emits `BuildEvent` → `event_q.put_nowait(event)`
2. Worker task receives event → `__process_event(event)`
3. `__dispatch_to_event_bus(event)` is scheduled via `asyncio.ensure_future`
4. `BuildEventPublisher` checks `event.type.is_internal_event()` — skips internal events
5. Event is serialized to JSON dict
6. Published to `build-events` exchange with routing key `build.<build_id>.<event_type>`
7. RabbitMQ routes to all bound queues matching the routing key pattern
8. Consumers receive the event in real-time

### Standalone Mode (no RabbitMQ)

1. Same event flow through steps 1-2
2. `__dispatch_standalone_notification(event)` is scheduled
3. `StandaloneDispatcher` loads config (lazy init on first call)
4. Filters by event type against each adapter's `events` list
5. Matching adapters call `deliver(event)` — shows macOS notification or sends email

## Credential Lifecycle

RabbitMQ checks credentials at connection time only. Once connected,
the AMQP connection persists regardless of credential expiry.

```
t=0s    Client calls POST /events/subscribe → gets credentials (TTL: 60s)
t=2s    Client connects to RabbitMQ (credentials valid) ✓
t=60s   Credentials expire — no NEW connections possible
t=???   Events still flowing on existing connection ✓
t=end   Client disconnects → queue auto-deletes
t+60s   Cleanup task deletes the expired temp user
```

## Routing Key Patterns

| Consumer | Binding | Receives |
|----------|---------|----------|
| Single build subscriber | `build.abc123.#` | All events for one build |
| Status-only subscriber | `build.abc123.status_event` | Only status changes |
| Dashboard (all builds) | `build.#` | Everything |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GBSERVER_EVENT_PUBLISHING_ENABLED` | `false` | Enable RabbitMQ event publishing |
| `GBSERVER_BUILD_EVENTS_EXCHANGE` | `build-events` | Topic exchange name |
| `GBSERVER_RABBITMQ_MGMT_URL` | `http://localhost:15672` | Management API URL |
| `GBSERVER_RABBITMQ_MGMT_USER` | `guest` | Management API user |
| `GBSERVER_RABBITMQ_MGMT_PASSWORD` | `guest` | Management API password |
| `GBSERVER_EVENT_SUBSCRIBE_TTL` | `60` | Credential TTL in seconds |
| `RABBITMQ_HOST` | `localhost` | RabbitMQ broker host |
| `RABBITMQ_PORT` | `5672` | RabbitMQ broker port |
| `RABBITMQ_USERNAME` | `guest` | RabbitMQ publish credentials |
| `RABBITMQ_PASSWORD` | `guest` | RabbitMQ publish credentials |

## Design Principles

1. **gbserver owns nothing about subscriptions** — RabbitMQ manages queue bindings,
   consumer lifecycle, and message routing. No subscription tables in our DB.

2. **Scoped credentials** — each temp user can only read events for the specific
   build they subscribed to. Compromised credentials can't read other builds.

3. **Non-blocking** — event publishing is fire-and-forget. RabbitMQ failures never
   affect build execution.

4. **No webhook delivery code** — external HTTP consumers (Slack, PagerDuty) are
   handled by off-the-shelf tools (Svix, n8n, etc.) that consume from RabbitMQ.
   We don't own delivery logic.

5. **Standalone is self-contained** — no infrastructure dependencies for local use.
   Config is a YAML file, delivery is direct.

## Getting Started (after cloning the repo)

### Standalone Mode (no RabbitMQ needed)

For local development and testing, standalone notifications work without any
infrastructure. Just create a config file:

```bash
mkdir -p ~/.gbserver
cat > ~/.gbserver/notifications.yaml << 'EOF'
notifications:
  - type: macos
    events: [status_event]
EOF
```

Run a build with `GB_ENVIRONMENT=STANDALONE` and you'll get macOS notifications
on status changes.

### Adding Email Notifications

To receive email notifications, you need SMTP credentials. Add them as
environment variables (never in the YAML file):

```bash
# Example: Gmail with App Password
export SMTP_USER="you@gmail.com"
export SMTP_PASSWORD="xxxx xxxx xxxx xxxx"  # App Password from Google Account settings
```

Then update your config:

```yaml
# ~/.gbserver/notifications.yaml
notifications:
  - type: macos
    events: [status_event]

  - type: email
    to: "you@gmail.com"
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    smtp_user_env: "SMTP_USER"
    smtp_password_env: "SMTP_PASSWORD"
    use_tls: true
    events: [status_event]
```

#### SMTP Settings for Common Providers

| Provider | smtp_host | smtp_port | use_tls | Notes |
|----------|-----------|-----------|---------|-------|
| Gmail | `smtp.gmail.com` | 587 | true | Requires App Password (2FA must be enabled) |
| Outlook/M365 | `smtp.office365.com` | 587 | true | Requires App Password |
| IBM internal | `smtp.ibm.com` | 25 | false | May not require auth on corporate network |
| SendGrid | `smtp.sendgrid.net` | 587 | true | Username is literal `"apikey"`, password is API key |

#### Getting an App Password (Gmail)

1. Enable 2-Factor Authentication: https://myaccount.google.com/security
2. Create App Password: https://myaccount.google.com/apppasswords
3. Select "Mail" → copy the 16-character password
4. Use this as `SMTP_PASSWORD` (not your Google login password)

#### Getting an App Password (Outlook/M365)

1. Enable 2FA: https://account.microsoft.com/security
2. Create App Password in Security settings
3. Use this as `SMTP_PASSWORD`

### Clustered Mode (with RabbitMQ)

For full event publishing to RabbitMQ, you need:

1. A running RabbitMQ instance with the management plugin enabled
2. Set environment variables:

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

#### Running RabbitMQ Locally (Docker)

```bash
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management
```

Management UI: http://localhost:15672 (guest/guest)

#### Verifying the Setup

```bash
# Check RabbitMQ is reachable
curl -s http://localhost:15672/api/overview -u guest:guest | jq .cluster_name

# Run the e2e test
python scripts/test-webhook-e2e.py
```

## Testing

```bash
# Unit tests for event publishing, admin client, adapters
pytest test/unit/messaging/ test/unit/notifications/ test/unit/buildwatcher/ -v

# Integration test (requires running RabbitMQ)
RABBITMQ_HOST=localhost pytest test/integration/messaging/test_event_subscribe_e2e.py -v

# E2E script (standalone mode + optional RabbitMQ)
python scripts/test-webhook-e2e.py [--skip-rabbitmq]
```

## File Map

```
src/gbserver/messaging/
├── build_event_publisher.py     # Publishes events to RabbitMQ exchange
├── rabbitmq_admin.py            # RabbitMQ Management API client
├── credential_cleanup.py        # Background cleanup of expired temp users
├── messaging_base.py            # Abstract messaging interface
└── rabbitmq_base.py             # aio-pika RabbitMQ implementation

src/gbserver/api/
└── event_subscribe.py           # POST /builds/{id}/events/subscribe

src/gbserver/notifications/      # Standalone mode notifications
├── adapter.py                   # NotificationAdapter base class
├── email_adapter.py             # SMTP email delivery
├── macos_adapter.py             # macOS Notification Center
├── config.py                    # YAML config loader
└── dispatcher.py                # Routes events to matching adapters
```
