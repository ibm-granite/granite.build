# NATS Standalone Events Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable build event notifications in standalone mode using the embedded NATS server.

**Architecture:** Environment-driven backend switch — when `GB_ENVIRONMENT=STANDALONE` and `RABBITMQ_HOST` is unset, use `NATSMessaging` for publishing and return NATS connection info from the subscribe endpoint. No credential provisioning needed (single-tenant, localhost).

**Tech Stack:** nats-py, NATSMessaging (existing), FastAPI, pytest

---

### Task 1: Fix NATSMessaging.is_available()

**Files:**
- Modify: `src/gbserver/messaging/nats_messaging.py`
- Test: `test/unit/messaging/test_nats_event_publisher.py` (created in Task 3)

**Step 1: Add the is_available classmethod**

In `src/gbserver/messaging/nats_messaging.py`, add after line 69 (end of `__init__`):

```python
@classmethod
def is_available(cls) -> bool:
    """Return True if nats-py is installed."""
    return HAS_NATS
```

**Step 2: Verify discover_backends picks it up**

Run: `source .venv/bin/activate && python3 -c "from gbserver.messaging import discover_backends; print(discover_backends())"`

Expected: Output includes `natsmessaging` key (if nats-py is installed).

**Step 3: Commit**

```bash
git add src/gbserver/messaging/nats_messaging.py
git commit -m "fix: add is_available() to NATSMessaging for backend discovery"
```

---

### Task 2: Add GBSERVER_EVENT_PUBLISHING_ENABLED to STANDALONE_ENV_DEFAULTS

**Files:**
- Modify: `src/gbserver/types/constants.py:372-377`

**Step 1: Add the default**

In `STANDALONE_ENV_DEFAULTS` dict (line 372), add the entry:

```python
STANDALONE_ENV_DEFAULTS = {
    ENV_VAR_METADATA_STORAGE: "sqlite",
    ENV_VAR_DEFAULT_BUILDRUNNER_TYPE: "thread",
    ENV_VAR_PREFIX + "_PROCEED_WITHOUT_SECRETS": "true",
    ENV_VAR_AUTH_MODE: "apikey",
    ENV_VAR_GBSERVER_EVENT_PUBLISHING_ENABLED: "true",
}
```

**Step 2: Verify**

Run: `source .venv/bin/activate && GB_ENVIRONMENT=STANDALONE python3 -c "from gbserver.types.constants import GBSERVER_EVENT_PUBLISHING_ENABLED; print(GBSERVER_EVENT_PUBLISHING_ENABLED)"`

Expected: `True`

**Step 3: Commit**

```bash
git add src/gbserver/types/constants.py
git commit -m "feat: enable event publishing by default in standalone mode"
```

---

### Task 3: Refactor BuildEventPublisher to support NATS backend

**Files:**
- Modify: `src/gbserver/messaging/build_event_publisher.py`
- Create: `test/unit/messaging/test_nats_event_publisher.py`

**Step 1: Write the failing test**

Create `test/unit/messaging/test_nats_event_publisher.py`:

```python
"""Unit tests for NATS-based event publishing in standalone mode."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.messaging.build_event_publisher import BuildEventPublisher


class TestBuildEventPublisherBackendSelection:
    """Tests for from_env() backend selection logic."""

    @patch.dict(os.environ, {"GB_ENVIRONMENT": "STANDALONE"}, clear=False)
    @patch.dict(os.environ, {}, clear=False)
    @patch("gbserver.messaging.build_event_publisher.HAS_NATS", True)
    @patch("gbserver.messaging.build_event_publisher.NATSMessaging")
    def test_from_env_uses_nats_in_standalone(self, mock_nats_cls):
        """In standalone without RABBITMQ_HOST, from_env() creates NATSMessaging."""
        # Ensure RABBITMQ_HOST is not set
        os.environ.pop("RABBITMQ_HOST", None)
        mock_nats_instance = MagicMock()
        mock_nats_cls.return_value = mock_nats_instance

        publisher = BuildEventPublisher.from_env()

        mock_nats_cls.assert_called_once()
        assert publisher._backend is mock_nats_instance

    @patch.dict(os.environ, {"RABBITMQ_HOST": "rmq.example.com"}, clear=False)
    def test_from_env_uses_rabbitmq_when_host_set(self):
        """When RABBITMQ_HOST is set, from_env() creates RabbitMQBase."""
        publisher = BuildEventPublisher.from_env()

        from gbserver.messaging.rabbitmq_base import RabbitMQBase
        assert isinstance(publisher._backend, RabbitMQBase)

    @patch.dict(os.environ, {"GB_ENVIRONMENT": "STANDALONE"}, clear=False)
    @patch("gbserver.messaging.build_event_publisher.HAS_NATS", True)
    @patch("gbserver.messaging.build_event_publisher.NATSMessaging")
    def test_is_configured_true_in_standalone_with_nats(self, mock_nats_cls):
        """is_configured() returns True in standalone when nats-py is available."""
        os.environ.pop("RABBITMQ_HOST", None)
        assert BuildEventPublisher.is_configured() is True

    @patch.dict(os.environ, {"GB_ENVIRONMENT": "STANDALONE"}, clear=False)
    @patch("gbserver.messaging.build_event_publisher.HAS_NATS", False)
    def test_is_configured_false_without_nats_or_rabbitmq(self):
        """is_configured() returns False when neither backend is available."""
        os.environ.pop("RABBITMQ_HOST", None)
        assert BuildEventPublisher.is_configured() is False
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest test/unit/messaging/test_nats_event_publisher.py -v`

Expected: FAIL (no `_backend` attribute, no `NATSMessaging` import, no `HAS_NATS` in module)

**Step 3: Implement the refactor**

Modify `src/gbserver/messaging/build_event_publisher.py`:

```python
"""
BuildEventPublisher — publishes BuildEvents to a messaging backend.

Supports RabbitMQ (dev/prod) and NATS (standalone).
Routing key format: build.<build_id>.<event_type>
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

from gbserver.messaging.messaging_base import Address, MessagingBase
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
)
from gbserver.types.constants import (
    GBSERVER_BUILD_EVENTS_EXCHANGE,
    GBSERVER_NATS_URL,
)
from gbserver.utils.logger import get_logger
from gbserver.utils.optional_imports import HAS_NATS

logger = get_logger(__name__)


class BuildEventPublisher:
    """
    Publishes BuildEvents to a messaging backend (RabbitMQ or NATS).

    Routing key format: build.<build_id>.<event_type>
    Internal events (TERMINATE, NEWARTIFACT_IN_ENVIRONMENT, NEW_MULTIARTIFACT_IN_ENVIRONMENT)
    are silently skipped.
    """

    def __init__(self, backend: MessagingBase) -> None:
        self._backend = backend
        self._publish_lock = asyncio.Lock()

    @classmethod
    def from_env(
        cls,
        messaging_secret: Optional[Any] = None,
    ) -> "BuildEventPublisher":
        """
        Factory that creates the publisher from environment variables.

        Backend selection:
        - If RABBITMQ_HOST is set → RabbitMQ
        - If GB_ENVIRONMENT=STANDALONE and nats-py is available → NATS
        - Otherwise → raises RuntimeError
        """
        if os.getenv("RABBITMQ_HOST"):
            from gbserver.messaging.rabbitmq_base import RabbitMQBase

            backend = RabbitMQBase.from_env_and_args(
                exchange_name=GBSERVER_BUILD_EVENTS_EXCHANGE,
                queue_name="build",
                routing_key=None,
                messaging_secret=messaging_secret,
            )
        elif _is_standalone() and HAS_NATS:
            from gbserver.messaging.nats_messaging import NATSMessaging

            backend = NATSMessaging(
                addr=Address(
                    exchange=None,
                    queue="build",
                    routing_key=None,
                ),
                nats_url=GBSERVER_NATS_URL,
            )
        else:
            raise RuntimeError(
                "No messaging backend configured. "
                "Set RABBITMQ_HOST or run in standalone mode with nats-py installed."
            )
        return cls(backend=backend)

    @classmethod
    def is_configured(cls) -> bool:
        """Return True if a messaging backend is available."""
        if os.getenv("RABBITMQ_HOST"):
            return True
        if _is_standalone() and HAS_NATS:
            return True
        return False

    async def setup(self) -> None:
        """Initialize the messaging backend connection."""
        await self._backend.setup()

    async def close(self) -> None:
        """Close the messaging backend connection."""
        await self._backend.close()

    async def publish_event(self, event: BuildEvent) -> None:
        """
        Publish a BuildEvent to the messaging backend.

        Internal events are silently skipped.
        If the backend is unavailable, a warning is logged and the error is swallowed
        so that build processing is not interrupted.
        """
        # Skip internal events
        if event.type.is_internal_event():
            logger.debug(
                "Skipping internal event type=%s build_id=%s",
                event.type.value,
                event.run_metadata.build_id,
            )
            return

        build_id = event.run_metadata.build_id or "unknown"
        event_type = event.type.value

        payload = self._serialize_event(event)

        async with self._publish_lock:
            original_addr = self._backend.addr
            publish_addr = Address(
                exchange=GBSERVER_BUILD_EVENTS_EXCHANGE if original_addr.exchange else None,
                queue=f"build.{build_id}",
                routing_key=None,
            )
            try:
                self._backend.addr = publish_addr  # type: ignore[misc]
                await self._backend.publish(payload=payload, suffix=event_type)
                logger.info(
                    "Published event type=%s build_id=%s subject/rk=%s",
                    event_type,
                    build_id,
                    publish_addr.rk(event_type),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to publish event type=%s build_id=%s: %s",
                    event_type,
                    build_id,
                    exc,
                )
            finally:
                self._backend.addr = original_addr  # type: ignore[misc]

    @staticmethod
    def _serialize_event(event: BuildEvent) -> Dict[str, Any]:
        """Serialize a BuildEvent to a JSON-compatible dict."""
        payload: Dict[str, Any] = {
            "build_id": event.run_metadata.build_id or "unknown",
            "event_type": event.type.value,
            "timestamp": int(event.timestamp.timestamp()),
            "target_name": event.run_metadata.target_name or "",
            "step_name": event.run_metadata.targetstep_uri or "",
            "source": event.source,
        }
        if isinstance(event.payload, BuildEventStatusPayload):
            payload["status"] = event.payload.status.value
            payload["message"] = event.payload.msg
        return payload


def _is_standalone() -> bool:
    """Check if running in standalone mode."""
    return os.getenv("GB_ENVIRONMENT", "").upper() == "STANDALONE"
```

**Step 4: Run tests**

Run: `source .venv/bin/activate && pytest test/unit/messaging/test_nats_event_publisher.py test/unit/buildrunner/test_build_event_publish_logger.py -v`

Expected: All pass

**Step 5: Commit**

```bash
git add src/gbserver/messaging/build_event_publisher.py test/unit/messaging/test_nats_event_publisher.py
git commit -m "feat: support NATS backend in BuildEventPublisher for standalone mode"
```

---

### Task 4: Update SubscribeResponse and provision_subscription for NATS

**Files:**
- Modify: `src/gbserver/api/event_subscribe.py`
- Modify: `src/gbserver/messaging/subscription_service.py`
- Create: `test/unit/api/test_event_subscribe_nats.py`

**Step 1: Write the failing test**

Create `test/unit/api/test_event_subscribe_nats.py`:

```python
"""Unit tests for the NATS subscribe endpoint in standalone mode."""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from gbserver.api.event_subscribe import event_subscribe_router
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.auth import User


def _make_fake_user(login: str = "testuser") -> User:
    return User(
        login=login, id=1, url="", html_url="",
        name=login, email=f"{login}@test.com", auth_provider="apikey",
    )


class _FakeAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.data = {"user": _make_fake_user()}
        return await call_next(request)


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(event_subscribe_router, prefix="/api/v1")
    app.add_middleware(_FakeAuthMiddleware)
    return app


def _make_stored_build(build_id: str = "test-build-123") -> StoredBuild:
    return StoredBuild(
        uuid=build_id, name="test-build", space_name="test-space",
        source_uri="", username="testuser", build_archive="", status="submitted",
    )


class TestNATSSubscribeEndpoint:
    """Tests for NATS subscribe response in standalone mode."""

    @patch.dict(os.environ, {"GB_ENVIRONMENT": "STANDALONE"}, clear=False)
    @patch("gbserver.messaging.subscription_service.HAS_NATS", True)
    @patch("gbserver.api.event_subscribe.get_admin_storage")
    def test_returns_nats_delivery_type(self, mock_get_storage):
        """In standalone, returns delivery_type='nats' with url and subject."""
        build_id = "abc-123-def"
        mock_storage = MagicMock()
        mock_storage.build_storage.get_by_uuid.return_value = _make_stored_build(build_id)
        mock_get_storage.return_value = mock_storage

        # Remove RABBITMQ_HOST to ensure NATS path
        os.environ.pop("RABBITMQ_HOST", None)

        app = _make_app()
        client = TestClient(app)
        response = client.post(
            f"/api/v1/builds/{build_id}/events/subscribe",
            headers={"Authorization": "Bearer valid-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["delivery_type"] == "nats"
        assert data["url"] == "nats://localhost:4222"
        assert data["subject"] == f"gbserver.build.{build_id}.>"
        assert data["username"] is None
        assert data["password"] is None
        assert data["exchange"] is None
        assert data["routing_key"] is None
        assert data["queue"] is None
```

**Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest test/unit/api/test_event_subscribe_nats.py -v`

Expected: FAIL

**Step 3: Update SubscribeResponse model**

In `src/gbserver/api/event_subscribe.py`, update the model:

```python
class SubscribeResponse(BaseModel):
    delivery_type: str
    host: str
    port: int
    username: str | None = None
    password: str | None = None
    exchange: str | None = None
    routing_key: str | None = None
    queue: str | None = None
    url: str | None = None
    subject: str | None = None
    expires_at: int
```

**Step 4: Update provision_subscription**

In `src/gbserver/messaging/subscription_service.py`:

```python
"""Service for provisioning event subscription credentials."""

from __future__ import annotations

import os
import time
from typing import Any, Dict

from gbserver.types.constants import (
    GBSERVER_BUILD_EVENTS_EXCHANGE,
    GBSERVER_EVENT_SUBSCRIBE_TTL,
    GBSERVER_NATS_URL,
    GBSERVER_RABBITMQ_MGMT_PASSWORD,
    GBSERVER_RABBITMQ_MGMT_URL,
    GBSERVER_RABBITMQ_MGMT_USER,
)
from gbserver.utils.logger import get_logger
from gbserver.utils.optional_imports import HAS_NATS

logger = get_logger(__name__)


def _is_standalone() -> bool:
    return os.getenv("GB_ENVIRONMENT", "").upper() == "STANDALONE"


async def provision_subscription(build_id: str) -> Dict[str, Any]:
    """Provision credentials/connection info for consuming build events.

    In standalone/NATS mode: returns NATS url + subject (no credentials).
    In RabbitMQ mode: provisions scoped temporary user via Management API.
    """
    if _is_standalone() and HAS_NATS and not os.getenv("RABBITMQ_HOST"):
        return _provision_nats(build_id)

    return await _provision_rabbitmq(build_id)


def _provision_nats(build_id: str) -> Dict[str, Any]:
    """Return NATS connection info for standalone mode."""
    from urllib.parse import urlparse

    parsed = urlparse(GBSERVER_NATS_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 4222

    # No expiry in standalone — set far-future
    expires_at = int(time.time()) + 86400 * 365

    return {
        "delivery_type": "nats",
        "host": host,
        "port": port,
        "username": None,
        "password": None,
        "exchange": None,
        "routing_key": None,
        "queue": None,
        "url": GBSERVER_NATS_URL,
        "subject": f"gbserver.build.{build_id}.>",
        "expires_at": expires_at,
    }


async def _provision_rabbitmq(build_id: str) -> Dict[str, Any]:
    """Provision scoped RabbitMQ credentials via Management API."""
    from gbserver.messaging.rabbitmq_admin import RabbitMQAdmin

    admin = RabbitMQAdmin(
        management_url=GBSERVER_RABBITMQ_MGMT_URL,
        admin_user=GBSERVER_RABBITMQ_MGMT_USER,
        admin_password=GBSERVER_RABBITMQ_MGMT_PASSWORD,
    )

    credentials = await admin.create_scoped_user(
        build_id=build_id,
        exchange=GBSERVER_BUILD_EVENTS_EXCHANGE,
        ttl_seconds=GBSERVER_EVENT_SUBSCRIBE_TTL,
    )

    host = os.getenv("RABBITMQ_HOST", "localhost")
    port = int(os.getenv("RABBITMQ_PORT", "5672"))
    username = credentials["username"]
    username_suffix = username.rsplit("-", 1)[-1] if "-" in username else username

    return {
        "delivery_type": "rabbitmq",
        "host": host,
        "port": port,
        "username": credentials["username"],
        "password": credentials["password"],
        "exchange": GBSERVER_BUILD_EVENTS_EXCHANGE,
        "routing_key": f"build.{build_id}.#",
        "queue": f"events.{build_id}.{username_suffix}",
        "url": None,
        "subject": None,
        "expires_at": credentials["expires_at"],
    }
```

**Step 5: Run tests**

Run: `source .venv/bin/activate && pytest test/unit/api/test_event_subscribe_nats.py test/unit/api/test_event_subscribe.py -v`

Expected: All pass

**Step 6: Commit**

```bash
git add src/gbserver/api/event_subscribe.py src/gbserver/messaging/subscription_service.py test/unit/api/test_event_subscribe_nats.py
git commit -m "feat: return NATS connection info from subscribe endpoint in standalone"
```

---

### Task 5: Gate credential cleanup loop on non-NATS mode

**Files:**
- Modify: `src/gbserver/api/root_api.py:79-86`

**Step 1: Update the startup task**

```python
@root_api.on_event("startup")
async def _start_background_tasks():
    """Launch background tasks that run for the lifetime of the server."""
    if GBSERVER_EVENT_PUBLISHING_ENABLED:
        # Credential cleanup only needed for RabbitMQ (temp users expire)
        # In standalone/NATS mode, no credentials are provisioned
        import os
        if os.getenv("RABBITMQ_HOST"):
            from gbserver.messaging.credential_cleanup import start_cleanup_loop

            logger.info("Event publishing enabled — starting credential cleanup task")
            asyncio.create_task(start_cleanup_loop())
        else:
            logger.info("Event publishing enabled (NATS mode) — no credential cleanup needed")
```

**Step 2: Verify existing tests still pass**

Run: `source .venv/bin/activate && pytest test/unit/messaging/test_credential_cleanup.py -v`

Expected: All pass (existing tests mock the env)

**Step 3: Commit**

```bash
git add src/gbserver/api/root_api.py
git commit -m "fix: skip credential cleanup loop in NATS/standalone mode"
```

---

### Task 6: Integration test with embedded NATS

**Files:**
- Create: `test/integration/messaging/test_nats_events_e2e.py`

**Step 1: Write the integration test**

```python
"""End-to-end test: publish and subscribe to build events via NATS.

Requires nats-server on PATH. Skipped otherwise.
"""

import asyncio
import json
import os
import shutil
import subprocess
import time

import pytest

HAS_NATS_SERVER = shutil.which("nats-server") is not None
HAS_NATS_PY = False
try:
    import nats
    HAS_NATS_PY = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not (HAS_NATS_SERVER and HAS_NATS_PY),
    reason="nats-server not on PATH or nats-py not installed",
)


@pytest.fixture
def nats_server(tmp_path):
    """Start and stop a temporary nats-server with JetStream."""
    port = 14222  # non-default to avoid conflicts
    proc = subprocess.Popen(
        ["nats-server", "-js", "-p", str(port), "-sd", str(tmp_path / "nats-data")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to be ready
    import socket
    for _ in range(50):
        try:
            s = socket.create_connection(("localhost", port), timeout=0.2)
            s.close()
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.1)
    yield f"nats://localhost:{port}"
    proc.terminate()
    proc.wait(timeout=5)


@pytest.mark.asyncio
async def test_publish_and_consume_via_nats(nats_server):
    """Publish an event and verify a subscriber receives it."""
    from gbserver.messaging.messaging_base import Address
    from gbserver.messaging.nats_messaging import NATSMessaging

    build_id = "test-build-e2e-001"

    # Publisher
    pub = NATSMessaging(
        addr=Address(exchange=None, queue="build", routing_key=None),
        nats_url=nats_server,
    )
    await pub.setup()

    # Subscriber
    sub = NATSMessaging(
        addr=Address(exchange=None, queue=f"build.{build_id}", routing_key=None),
        nats_url=nats_server,
    )
    await sub.setup()

    received = []

    async def handler(data: bytes, routing_key: str):
        received.append(json.loads(data))

    # Start consuming in background
    consume_task = asyncio.create_task(sub.consume_stream(handler))

    # Give consumer time to subscribe
    await asyncio.sleep(0.3)

    # Publish an event
    event_payload = {
        "build_id": build_id,
        "event_type": "status_event",
        "timestamp": int(time.time()),
        "target_name": "training",
        "step_name": "space://steps/sft",
        "source": "test",
        "status": "running",
        "message": "Step started",
    }

    # Publish to the subject the subscriber is listening on
    pub._backend_addr = pub.addr
    pub.addr = Address(exchange=None, queue=f"build.{build_id}", routing_key=None)
    await pub.publish(payload=event_payload, suffix="status_event")
    pub.addr = pub._backend_addr

    # Wait for delivery
    await asyncio.sleep(0.5)

    # Cleanup
    consume_task.cancel()
    try:
        await consume_task
    except asyncio.CancelledError:
        pass
    await sub.close()
    await pub.close()

    assert len(received) == 1
    assert received[0]["build_id"] == build_id
    assert received[0]["status"] == "running"
```

**Step 2: Run the integration test**

Run: `source .venv/bin/activate && pytest test/integration/messaging/test_nats_events_e2e.py -v -s`

Expected: PASS if nats-server on PATH, SKIPPED otherwise

**Step 3: Commit**

```bash
git add test/integration/messaging/test_nats_events_e2e.py
git commit -m "test: add NATS event pub/sub integration test"
```

---

### Task 7: Run full test suite and format

**Step 1: Run formatter**

Run: `source .venv/bin/activate && make format`

**Step 2: Run unit tests**

Run: `source .venv/bin/activate && pytest test/unit/messaging/ test/unit/api/test_event_subscribe.py test/unit/api/test_event_subscribe_nats.py test/unit/buildrunner/test_build_event_publish_logger.py -v`

Expected: All pass

**Step 3: Fix any issues and commit**

```bash
git add -A
git commit -m "style: format code with black and isort"
```

---

### Task 8: Push and create PR

**Step 1: Push**

```bash
git push origin feat/129-nats-standalone-events -u
```

**Step 2: Create PR**

```bash
gh pr create --repo ibm-granite/granite.build --base main \
  --title "feat: NATS event notifications for standalone mode" \
  --body-file docs/plans/2026-06-23-nats-standalone-events-design.md
```
