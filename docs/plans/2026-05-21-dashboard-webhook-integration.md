# Dashboard Webhook Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a webhook receiver to the GB Dashboard so it receives real-time build events from gbserver, updating build status without waiting for the 5s K8s sync cycle.

**Architecture:** New FastAPI router in the dashboard's web app receives HMAC-signed POST payloads from gbserver, verifies signatures, upserts build status in PostgreSQL. On startup, the app registers space-wide subscriptions with gbserver.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy (async/asyncpg), httpx, PostgreSQL (JSONB)

**Target repo:** `github.ibm.com/granite-dot-build/gb_dashboard` (clone into worktree)

**Design doc:** `docs/plans/2026-05-21-dashboard-webhook-integration-design.md` (in gbserver repo)

---

## Task 1: Add config fields for webhook settings

**Files:**
- Modify: `src/gb_dashboard/config.py` (add fields to `DashboardConfig`)

**Step 1: Add webhook config fields**

In `DashboardConfig`, add after the existing `sync_poll_interval` field:

```python
# Webhook receiver settings
webhook_enabled: bool = field(default_factory=lambda: _get_bool_env("GB_DASHBOARD_WEBHOOK_ENABLED", False))
webhook_secret: str = field(default_factory=lambda: os.getenv("GB_DASHBOARD_WEBHOOK_SECRET", ""))
webhook_frequency: int = field(default_factory=lambda: int(os.getenv("GB_DASHBOARD_WEBHOOK_FREQUENCY", "15")))

# gbserver API (for webhook subscription registration)
gbserver_api_url: str = field(default_factory=lambda: os.getenv("GB_DASHBOARD_GBSERVER_URL", ""))
gbserver_token: str = field(default_factory=lambda: os.getenv("GB_DASHBOARD_GBSERVER_TOKEN", ""))
```

**Step 2: Commit**

```bash
git add src/gb_dashboard/config.py
git commit -m "feat(webhooks): add config fields for webhook receiver settings"
```

---

## Task 2: Add DB schema for webhook subscriptions and deliveries

**Files:**
- Modify: `src/gb_dashboard/services/db_schema.py`

**Step 1: Add GbdWebhookSubscription model**

Add after the `GbdSyncState` class:

```python
class GbdWebhookSubscription(Base):
    """Tracks webhook subscriptions registered with gbserver."""

    __tablename__ = "gbd_webhook_subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    space_name: Mapped[str] = mapped_column(String(255), nullable=False)
    gbserver_subscription_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    webhook_url: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("idx_gbd_webhook_subs_space", "space_name"),
        Index("idx_gbd_webhook_subs_active", "active"),
    )


class GbdWebhookDelivery(Base):
    """Audit log of received webhook deliveries."""

    __tablename__ = "gbd_webhook_deliveries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    delivery_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    build_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    space_name: Mapped[str] = mapped_column(String(255), nullable=False)
    event_count: Mapped[int] = mapped_column(Integer, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    payload: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)

    __table_args__ = (
        UniqueConstraint("delivery_id", name="uq_gbd_webhook_delivery_id"),
        Index("idx_gbd_webhook_deliveries_build", "build_id"),
        Index("idx_gbd_webhook_deliveries_received", "received_at"),
    )
```

**Step 2: Commit**

```bash
git add src/gb_dashboard/services/db_schema.py
git commit -m "feat(webhooks): add DB models for webhook subscriptions and deliveries"
```

---

## Task 3: Create the webhook receiver endpoint

**Files:**
- Create: `src/gb_dashboard/api/webhooks.py`

**Step 1: Write the receiver**

```python
#!/usr/bin/env python3

"""Webhook receiver endpoint for gbserver build events."""

import hashlib
import hmac
import logging
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, Request, Response
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from gb_dashboard.config import get_config
from gb_dashboard.services.db_schema import GbdBuild, GbdWebhookDelivery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])


def verify_signature(body: bytes, secret: str, signature_header: str) -> bool:
    """Verify HMAC-SHA256 signature from gbserver."""
    if not signature_header or not secret:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature_header, expected)


@router.post("/build-events")
async def receive_build_events(request: Request):
    """Receive batched build events from gbserver.

    Verifies HMAC signature, upserts build status, stores delivery audit log.
    """
    config = get_config()
    if not config.webhook_enabled:
        return Response(status_code=404)

    # Read raw body for signature verification
    body = await request.body()
    signature = request.headers.get("X-GB-Signature-256", "")

    if not verify_signature(body, config.webhook_secret, signature):
        logger.warning("Webhook signature verification failed")
        return Response(status_code=401)

    import json
    payload = json.loads(body)

    delivery_id = payload.get("delivery_id")
    build_id = payload.get("build_id")
    space_name = payload.get("space_name", "")
    events = payload.get("events", [])

    logger.info(
        "Webhook received: delivery=%s build=%s events=%d",
        delivery_id, build_id, len(events),
    )

    # Process events and update DB
    try:
        from gb_dashboard.services.webhook_processor import process_webhook_delivery
        await process_webhook_delivery(payload)
    except Exception as e:
        logger.error("Failed to process webhook delivery %s: %s", delivery_id, e)
        return Response(status_code=500)

    return {"ok": True, "events_processed": len(events)}
```

**Step 2: Commit**

```bash
git add src/gb_dashboard/api/webhooks.py
git commit -m "feat(webhooks): add webhook receiver endpoint with HMAC verification"
```

---

## Task 4: Create the webhook event processor

**Files:**
- Create: `src/gb_dashboard/services/webhook_processor.py`

**Step 1: Write the processor**

```python
#!/usr/bin/env python3

"""Process incoming webhook deliveries and update dashboard database."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from gb_dashboard.config import get_config
from gb_dashboard.services.db_schema import GbdBuild, GbdWebhookDelivery

logger = logging.getLogger(__name__)

# Terminal statuses that indicate a build is done
TERMINAL_STATUSES = {"success", "failed", "cancelled"}

# Map gbserver status strings to dashboard status strings
STATUS_MAP = {
    "submitted": "pending",
    "pending": "pending",
    "running": "running",
    "success": "success",
    "failed": "failed",
    "cancelled": "cancelled",
}


async def process_webhook_delivery(payload: Dict[str, Any]) -> None:
    """Process a webhook delivery payload.

    1. Check for duplicate delivery (idempotency)
    2. Store delivery in audit table
    3. For build-level status events, upsert gbd_builds
    """
    config = get_config()
    from gb_dashboard.services.db_session import get_session_factory

    session_factory = get_session_factory()
    if session_factory is None:
        logger.warning("No DB session factory available, skipping webhook processing")
        return

    delivery_id = payload["delivery_id"]
    build_id = payload["build_id"]
    space_name = payload.get("space_name", "")
    build_name = payload.get("build_name", "")
    username = payload.get("user", "")
    events = payload.get("events", [])

    async with session_factory() as session:
        # Idempotency check
        existing = await session.execute(
            select(GbdWebhookDelivery).where(
                GbdWebhookDelivery.delivery_id == delivery_id
            )
        )
        if existing.scalar_one_or_none():
            logger.debug("Duplicate delivery %s, skipping", delivery_id)
            return

        # Store audit record
        delivery_record = GbdWebhookDelivery(
            delivery_id=delivery_id,
            build_id=build_id,
            space_name=space_name,
            event_count=len(events),
            payload=payload,
        )
        session.add(delivery_record)

        # Process build-level status events
        for event in events:
            event_type = event.get("event_type")
            status = event.get("status")

            if event_type != "status_event" or not status:
                continue

            # Only process build-level status (not target/step level)
            # Build-level events have empty target_name
            target_name = event.get("target_name", "")
            if target_name:
                continue  # Skip target/step-level events for now

            mapped_status = STATUS_MAP.get(status, status)
            now = datetime.now(timezone.utc)

            # Upsert build status
            stmt = insert(GbdBuild).values(
                id=build_id,
                name=build_name,
                space_name=space_name,
                username=username,
                status=mapped_status,
                cluster_name="",  # Will be enriched by sync daemon
                created_at=now,
                updated_at=now,
                finished_at=now if mapped_status in TERMINAL_STATUSES else None,
            ).on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "status": mapped_status,
                    "updated_at": now,
                    "finished_at": now if mapped_status in TERMINAL_STATUSES else None,
                },
            )
            await session.execute(stmt)
            logger.info(
                "Build %s status updated to %s via webhook",
                build_id, mapped_status,
            )

        await session.commit()
```

**Step 2: Commit**

```bash
git add src/gb_dashboard/services/webhook_processor.py
git commit -m "feat(webhooks): add event processor with build status upsert"
```

---

## Task 5: Create the DB session helper

**Files:**
- Create: `src/gb_dashboard/services/db_session.py`

**Step 1: Write session factory singleton**

```python
#!/usr/bin/env python3

"""Database session factory singleton for use outside of request context."""

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gb_dashboard.config import get_config

logger = logging.getLogger(__name__)

_session_factory: Optional[async_sessionmaker] = None


def init_session_factory() -> async_sessionmaker:
    """Initialize the global session factory (call once at startup)."""
    global _session_factory
    config = get_config()
    engine = create_async_engine(config.dashboard_db_url, pool_size=5, max_overflow=10)
    _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


def get_session_factory() -> Optional[async_sessionmaker]:
    """Get the global session factory."""
    return _session_factory
```

**Step 2: Commit**

```bash
git add src/gb_dashboard/services/db_session.py
git commit -m "feat(webhooks): add DB session factory singleton"
```

---

## Task 6: Create the subscription registration service

**Files:**
- Create: `src/gb_dashboard/services/webhook_subscriber.py`

**Step 1: Write the subscriber**

```python
#!/usr/bin/env python3

"""Register webhook subscriptions with gbserver on startup."""

import logging
from datetime import datetime, timezone
from typing import List

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from gb_dashboard.config import DashboardConfig
from gb_dashboard.services.db_schema import GbdWebhookSubscription

logger = logging.getLogger(__name__)


async def ensure_subscriptions(
    config: DashboardConfig,
    session_factory: async_sessionmaker,
    spaces: List[str],
    webhook_url: str,
) -> None:
    """Ensure space-wide webhook subscriptions exist for all monitored spaces.

    Idempotent: skips spaces that already have active subscriptions.
    """
    if not config.gbserver_api_url:
        logger.warning("gbserver_api_url not configured, skipping webhook subscription")
        return

    async with session_factory() as session:
        for space_name in spaces:
            # Check if subscription already exists
            result = await session.execute(
                select(GbdWebhookSubscription).where(
                    GbdWebhookSubscription.space_name == space_name,
                    GbdWebhookSubscription.active == True,
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                logger.info(
                    "Space %s already has active subscription %s",
                    space_name, existing.gbserver_subscription_id,
                )
                continue

            # Register with gbserver
            try:
                sub_id = await _register_with_gbserver(
                    config, space_name, webhook_url
                )
                # Store in DB
                sub = GbdWebhookSubscription(
                    space_name=space_name,
                    gbserver_subscription_id=sub_id,
                    webhook_url=webhook_url,
                    active=True,
                )
                session.add(sub)
                await session.commit()
                logger.info(
                    "Registered webhook subscription for space %s: %s",
                    space_name, sub_id,
                )
            except Exception as e:
                logger.error(
                    "Failed to register webhook for space %s: %s", space_name, e
                )


async def _register_with_gbserver(
    config: DashboardConfig, space_name: str, webhook_url: str
) -> str:
    """Call gbserver API to create a space-wide subscription."""
    url = f"{config.gbserver_api_url}/api/v1/webhooks/spaces/{space_name}/subscriptions"
    headers = {"Authorization": f"Bearer {config.gbserver_token}"}
    payload = {
        "webhook_url": webhook_url,
        "secret": config.webhook_secret,
        "event_types": ["*"],
        "frequency": config.webhook_frequency,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["id"]
```

**Step 2: Commit**

```bash
git add src/gb_dashboard/services/webhook_subscriber.py
git commit -m "feat(webhooks): add subscription registration service"
```

---

## Task 7: Wire everything into main.py

**Files:**
- Modify: `src/gb_dashboard/main.py`

**Step 1: Import and register the webhook router**

Add to the imports section (around line 25):
```python
from gb_dashboard.api import webhooks
```

Add with the other `include_router` calls (around line 401):
```python
dashboard_app.include_router(webhooks.router)  # HMAC-authenticated, no session needed
```

**Step 2: Add /api/webhooks to PUBLIC_PATHS**

In `AuthMiddleware.PUBLIC_PATHS` (around line 299), add:
```python
PUBLIC_PATHS = {"/login", "/health", "/ready", "/static", "/api/metrics", "/api/builds/progress", "/version", "/mcp", "/api/webhooks"}
```

**Step 3: Add startup registration in lifespan**

In the `lifespan()` function, after the chat trajectory init block (around line 270), add:

```python
    # Register webhook subscriptions with gbserver
    if config.webhook_enabled and config.is_dashboard_db_mode:
        try:
            from gb_dashboard.services.db_session import init_session_factory
            from gb_dashboard.services.webhook_subscriber import ensure_subscriptions

            wh_session_factory = init_session_factory()
            # Derive webhook URL from dashboard's own URL
            webhook_url = f"http://{config.host}:{config.port}{config.base_path}/api/webhooks/build-events"
            # Get spaces from configured clusters or gbserver
            spaces = [c.name for c in config.clusters] if config.clusters else []
            await ensure_subscriptions(config, wh_session_factory, spaces, webhook_url)
            logger.info("Webhook subscriptions initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize webhook subscriptions: {e}")
```

**Step 4: Commit**

```bash
git add src/gb_dashboard/main.py
git commit -m "feat(webhooks): wire receiver and subscriber into main app"
```

---

## Task 8: Add unit tests

**Files:**
- Create: `tests/test_webhook_receiver.py`

**Step 1: Write tests**

```python
#!/usr/bin/env python3

"""Tests for webhook receiver HMAC verification and event processing."""

import hashlib
import hmac
import json
import pytest


def sign_payload(payload: dict, secret: str) -> str:
    """Create HMAC signature for a payload."""
    body = json.dumps(payload).encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


class TestHMACVerification:
    """Test HMAC signature verification."""

    def test_valid_signature(self):
        from gb_dashboard.api.webhooks import verify_signature

        payload = {"test": True}
        body = json.dumps(payload).encode("utf-8")
        secret = "test-secret"
        signature = sign_payload(payload, secret)

        assert verify_signature(body, secret, signature) is True

    def test_invalid_signature(self):
        from gb_dashboard.api.webhooks import verify_signature

        body = b'{"test": true}'
        assert verify_signature(body, "secret", "sha256=wrong") is False

    def test_empty_signature(self):
        from gb_dashboard.api.webhooks import verify_signature

        body = b'{"test": true}'
        assert verify_signature(body, "secret", "") is False

    def test_empty_secret(self):
        from gb_dashboard.api.webhooks import verify_signature

        body = b'{"test": true}'
        assert verify_signature(body, "", "sha256=something") is False


class TestStatusMapping:
    """Test gbserver status to dashboard status mapping."""

    def test_maps_known_statuses(self):
        from gb_dashboard.services.webhook_processor import STATUS_MAP

        assert STATUS_MAP["submitted"] == "pending"
        assert STATUS_MAP["pending"] == "pending"
        assert STATUS_MAP["running"] == "running"
        assert STATUS_MAP["success"] == "success"
        assert STATUS_MAP["failed"] == "failed"
        assert STATUS_MAP["cancelled"] == "cancelled"
```

**Step 2: Run tests**

```bash
pytest tests/test_webhook_receiver.py -v
```

**Step 3: Commit**

```bash
git add tests/test_webhook_receiver.py
git commit -m "test(webhooks): add unit tests for HMAC verification and status mapping"
```

---

## Task 9: Update Helm chart with new env vars

**Files:**
- Modify: `k8s/chart/values.yaml`

**Step 1: Add webhook defaults to values.yaml**

```yaml
# Webhook receiver (push notifications from gbserver)
webhook:
  enabled: false
  secret: ""
  frequency: 15
  gbserverUrl: ""
  gbserverToken: ""
```

**Step 2: Wire into deployment template**

In the deployment template's env section, add:
```yaml
{{- if .Values.webhook.enabled }}
- name: GB_DASHBOARD_WEBHOOK_ENABLED
  value: "true"
- name: GB_DASHBOARD_WEBHOOK_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-webhook
      key: secret
- name: GB_DASHBOARD_WEBHOOK_FREQUENCY
  value: {{ .Values.webhook.frequency | quote }}
- name: GB_DASHBOARD_GBSERVER_URL
  value: {{ .Values.webhook.gbserverUrl | quote }}
- name: GB_DASHBOARD_GBSERVER_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ .Release.Name }}-webhook
      key: gbserver-token
{{- end }}
```

**Step 3: Commit**

```bash
git add k8s/chart/
git commit -m "feat(webhooks): add Helm chart values for webhook configuration"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Config fields | `config.py` |
| 2 | DB models | `db_schema.py` |
| 3 | Receiver endpoint | `api/webhooks.py` |
| 4 | Event processor | `services/webhook_processor.py` |
| 5 | DB session helper | `services/db_session.py` |
| 6 | Subscription registration | `services/webhook_subscriber.py` |
| 7 | Wire into main.py | `main.py` |
| 8 | Unit tests | `tests/test_webhook_receiver.py` |
| 9 | Helm chart | `k8s/chart/` |
