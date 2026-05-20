# Webhook Push Notifications Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add webhook push notifications so clients can subscribe to build events instead of polling.

**Architecture:** New `src/gbserver/webhooks/` module with Pydantic models, SQLAlchemy-backed storage, FastAPI endpoints, and an async dispatcher that hooks into `BuildRunner.__process_event()`. Follows existing patterns from `storage/`, `api/`, and `resilience/alert_handlers.py`.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy, aiohttp, tenacity, HMAC-SHA256

---

## Task 1: Webhook Subscription Storage Model

**Files:**
- Create: `src/gbserver/webhooks/__init__.py`
- Create: `src/gbserver/webhooks/models.py`
- Test: `test/unit/webhooks/__init__.py`
- Test: `test/unit/webhooks/test_webhook_models.py`

**Step 1: Write the failing test**

```python
# test/unit/webhooks/__init__.py
# (empty)

# test/unit/webhooks/test_webhook_models.py
"""Tests for webhook subscription model."""

import datetime

from gbserver.webhooks.models import StoredWebhookSubscription


class TestStoredWebhookSubscription:
    def test_create_subscription(self):
        sub = StoredWebhookSubscription(
            space_name="my-space",
            build_id="build-uuid-123",
            webhook_url="https://example.com/hook",
            secret="my-secret",
            event_types=["STATUS_EVENT"],
            created_by="testuser",
        )
        assert sub.space_name == "my-space"
        assert sub.build_id == "build-uuid-123"
        assert sub.webhook_url == "https://example.com/hook"
        assert sub.secret == "my-secret"
        assert sub.event_types == ["STATUS_EVENT"]
        assert sub.created_by == "testuser"
        assert sub.active is True
        assert sub.uuid is not None
        assert sub.created_time is not None

    def test_subscription_defaults(self):
        sub = StoredWebhookSubscription(
            space_name="my-space",
            webhook_url="https://example.com/hook",
            secret="s",
            event_types=["STATUS_EVENT"],
            created_by="user",
        )
        assert sub.build_id is None
        assert sub.active is True

    def test_subscription_matches_event_type(self):
        sub = StoredWebhookSubscription(
            space_name="my-space",
            build_id="b1",
            webhook_url="https://example.com/hook",
            secret="s",
            event_types=["STATUS_EVENT", "ARTIFACT_EVENT"],
            created_by="user",
        )
        assert sub.matches_event_type("STATUS_EVENT") is True
        assert sub.matches_event_type("ARTIFACT_EVENT") is True
        assert sub.matches_event_type("METRICS_EVENT") is False
```

**Step 2: Run test to verify it fails**

Run: `pytest test/unit/webhooks/test_webhook_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gbserver.webhooks'`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/__init__.py
"""Webhook push notification module for build events."""

# src/gbserver/webhooks/models.py
"""Pydantic models for webhook subscriptions."""

import datetime
from typing import List, Optional, Self

from pydantic import Field

from gbserver.storage.storage import BaseStoredItem
from gbserver.utils.utils import get_utc_time


class StoredWebhookSubscription(BaseStoredItem):
    """
    A webhook subscription that receives push notifications for build events.

    Attributes:
        space_name: The space this subscription is scoped to
        build_id: Specific build to subscribe to (None = space-wide, phase 2)
        webhook_url: URL to POST event payloads to
        secret: Shared secret for HMAC-SHA256 signature verification
        event_types: List of BuildEventType names to filter on
        created_by: Username who created this subscription
        active: Whether this subscription is currently delivering events
        created_time: When the subscription was created
        updated_time: Last time the subscription was modified
    """

    space_name: str
    build_id: Optional[str] = None
    webhook_url: str
    secret: str
    event_types: List[str]
    created_by: str
    active: bool = True
    created_time: datetime.datetime = Field(default_factory=get_utc_time)
    updated_time: datetime.datetime = Field(default_factory=get_utc_time)

    def matches_event_type(self: Self, event_type: str) -> bool:
        """Check if this subscription should receive the given event type.

        Args:
            event_type: The BuildEventType name to check

        Returns:
            True if the event type is in this subscription's filter list
        """
        return event_type in self.event_types
```

**Step 4: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_webhook_models.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/gbserver/webhooks/__init__.py src/gbserver/webhooks/models.py \
        test/unit/webhooks/__init__.py test/unit/webhooks/test_webhook_models.py
git commit -m "feat(webhooks): add webhook subscription storage model (#8)"
```

---

## Task 2: Webhook Subscription Storage Layer

**Files:**
- Create: `src/gbserver/webhooks/storage.py`
- Create: `src/gbserver/webhooks/sql_storage.py`
- Test: `test/unit/webhooks/test_webhook_storage.py`

**Step 1: Write the failing test**

```python
# test/unit/webhooks/test_webhook_storage.py
"""Tests for webhook subscription storage."""

import pytest

from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.webhooks.models import StoredWebhookSubscription
from gbserver.webhooks.storage import BaseWebhookStorage, IWebhookStorage


class TestWebhookStorage:
    """Test webhook storage using SQLite backend."""

    def setup_method(self):
        """Create a fresh in-memory storage for each test."""
        from gbserver.webhooks.sql_storage import SQLWebhookStorage

        self.storage: IWebhookStorage = SQLWebhookStorage(
            table_name="test_webhook_subscriptions"
        )

    def _make_subscription(self, **kwargs) -> StoredWebhookSubscription:
        defaults = {
            "space_name": "test-space",
            "build_id": "build-123",
            "webhook_url": "https://example.com/hook",
            "secret": "test-secret",
            "event_types": ["STATUS_EVENT"],
            "created_by": "testuser",
        }
        defaults.update(kwargs)
        return StoredWebhookSubscription(**defaults)

    def test_add_and_get(self):
        sub = self._make_subscription()
        self.storage.add(sub)
        result = self.storage.get(sub.uuid)
        assert result is not None
        assert result.webhook_url == "https://example.com/hook"
        assert result.space_name == "test-space"

    def test_get_subscriptions_for_build(self):
        sub1 = self._make_subscription(build_id="build-1")
        sub2 = self._make_subscription(build_id="build-2")
        sub3 = self._make_subscription(build_id="build-1", active=False)
        self.storage.add(sub1)
        self.storage.add(sub2)
        self.storage.add(sub3)

        results = self.storage.get_active_for_build("build-1")
        assert len(results) == 1
        assert results[0].build_id == "build-1"
        assert results[0].active is True

    def test_deactivate_subscription(self):
        sub = self._make_subscription()
        self.storage.add(sub)
        self.storage.deactivate(sub.uuid)
        result = self.storage.get(sub.uuid)
        assert result is not None
        assert result.active is False

    def test_get_by_space(self):
        sub1 = self._make_subscription(space_name="space-a")
        sub2 = self._make_subscription(space_name="space-b")
        self.storage.add(sub1)
        self.storage.add(sub2)

        results = self.storage.get_by_space("space-a")
        assert len(results) == 1
        assert results[0].space_name == "space-a"
```

**Step 2: Run test to verify it fails**

Run: `pytest test/unit/webhooks/test_webhook_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gbserver.webhooks.storage'`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/storage.py
"""Interface and base implementation for webhook subscription storage.

Follows the same pattern as gbserver.storage.node_failure_storage.
"""

from typing import List

from gbserver.storage.storage import BaseItemStorage, IItemStorage
from gbserver.webhooks.models import StoredWebhookSubscription

GB_WEBHOOK_SUBSCRIPTIONS_TABLE_NAME = "gb_webhook_subscriptions"


class IWebhookStorage(IItemStorage[StoredWebhookSubscription]):
    """Interface for webhook subscription storage."""

    def get_active_for_build(self, build_id: str) -> List[StoredWebhookSubscription]:
        """Get all active subscriptions for a specific build.

        Args:
            build_id: The build UUID to find subscriptions for

        Returns:
            List of active subscriptions matching the build
        """
        raise NotImplementedError

    def get_by_space(self, space_name: str) -> List[StoredWebhookSubscription]:
        """Get all subscriptions for a space.

        Args:
            space_name: The space name to filter by

        Returns:
            List of subscriptions in the space
        """
        raise NotImplementedError

    def deactivate(self, subscription_id: str) -> None:
        """Deactivate a subscription (soft delete).

        Args:
            subscription_id: UUID of the subscription to deactivate
        """
        raise NotImplementedError

    def deactivate_for_build(self, build_id: str) -> int:
        """Deactivate all subscriptions for a build. Used on build completion.

        Args:
            build_id: The build UUID

        Returns:
            Number of subscriptions deactivated
        """
        raise NotImplementedError


class BaseWebhookStorage(BaseItemStorage[StoredWebhookSubscription], IWebhookStorage):
    """Base storage implementation for webhook subscriptions."""

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredWebhookSubscription
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_WEBHOOK_SUBSCRIPTIONS_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredWebhookSubscription) -> dict:
        """Extract indexed columns for storage."""
        from gbserver.storage.storage import CREATED_TIME_FIELD_NAME

        return {
            "space_name": item.space_name,
            "build_id": item.build_id or "",
            "active": item.active,
            "created_by": item.created_by,
            CREATED_TIME_FIELD_NAME: item.created_time,
        }

    @classmethod
    def _get_sample_item(cls) -> StoredWebhookSubscription:
        """Sample item for schema initialization."""
        return StoredWebhookSubscription(
            space_name="sample-space",
            webhook_url="https://example.com",
            secret="sample",
            event_types=["STATUS_EVENT"],
            created_by="system",
        )

    def get_active_for_build(self, build_id: str) -> List[StoredWebhookSubscription]:
        """Get active subscriptions matching a build."""
        results = []
        for page in self.get_paged(
            {"build_id": build_id, "active": True}, page_size=100
        ):
            results.extend(page)
        return results

    def get_by_space(self, space_name: str) -> List[StoredWebhookSubscription]:
        """Get all subscriptions for a space."""
        results = []
        for page in self.get_paged({"space_name": space_name}, page_size=100):
            results.extend(page)
        return results

    def deactivate(self, subscription_id: str) -> None:
        """Deactivate a subscription."""
        self.update_fields(subscription_id, {"active": False})

    def deactivate_for_build(self, build_id: str) -> int:
        """Deactivate all active subscriptions for a build."""
        subs = self.get_active_for_build(build_id)
        for sub in subs:
            self.deactivate(sub.uuid)
        return len(subs)


# src/gbserver/webhooks/sql_storage.py
"""SQL (PostgreSQL) storage backend for webhook subscriptions."""

from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.webhooks.models import StoredWebhookSubscription
from gbserver.webhooks.storage import BaseWebhookStorage, IWebhookStorage


class SQLWebhookStorage(
    BaseSQLItemStorage[StoredWebhookSubscription], BaseWebhookStorage, IWebhookStorage
):
    """SQLAlchemy-backed webhook subscription storage."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
```

**Step 4: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_webhook_storage.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/gbserver/webhooks/storage.py src/gbserver/webhooks/sql_storage.py \
        test/unit/webhooks/test_webhook_storage.py
git commit -m "feat(webhooks): add webhook subscription storage layer (#8)"
```

---

## Task 3: Webhook Delivery (HMAC signing + HTTP POST + retry)

**Files:**
- Create: `src/gbserver/webhooks/delivery.py`
- Test: `test/unit/webhooks/test_webhook_delivery.py`

**Step 1: Write the failing test**

```python
# test/unit/webhooks/test_webhook_delivery.py
"""Tests for webhook delivery with HMAC signing and retry."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, patch

import pytest

from gbserver.webhooks.delivery import WebhookDelivery, sign_payload


class TestSignPayload:
    def test_sign_payload_produces_valid_hmac(self):
        payload = b'{"event_type": "STATUS_EVENT"}'
        secret = "my-secret"
        signature = sign_payload(payload, secret)

        expected = "sha256=" + hmac.new(
            secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        assert signature == expected

    def test_sign_payload_different_secrets_differ(self):
        payload = b'{"status": "SUCCESS"}'
        sig1 = sign_payload(payload, "secret-a")
        sig2 = sign_payload(payload, "secret-b")
        assert sig1 != sig2


class TestWebhookDelivery:
    @pytest.mark.asyncio
    async def test_deliver_success(self):
        delivery = WebhookDelivery(
            webhook_url="https://example.com/hook",
            secret="test-secret",
            max_retries=3,
        )
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            payload = {"event_type": "STATUS_EVENT", "build_id": "b1"}
            result = await delivery.deliver(payload)
            assert result is True
            mock_session.post.assert_called_once()

            # Verify HMAC header was sent
            call_kwargs = mock_session.post.call_args.kwargs
            assert "X-GB-Signature-256" in call_kwargs["headers"]

    @pytest.mark.asyncio
    async def test_deliver_failure_retries(self):
        delivery = WebhookDelivery(
            webhook_url="https://example.com/hook",
            secret="test-secret",
            max_retries=2,
            initial_backoff=0.01,  # fast for tests
        )
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            payload = {"event_type": "STATUS_EVENT"}
            result = await delivery.deliver(payload)
            assert result is False
            # 1 initial + 2 retries = 3 calls
            assert mock_session.post.call_count == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest test/unit/webhooks/test_webhook_delivery.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gbserver.webhooks.delivery'`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/delivery.py
"""Webhook delivery with HMAC-SHA256 signing and retry with exponential backoff.

Delivers event payloads to subscriber webhook URLs. Each delivery is signed
with the subscriber's secret so they can verify authenticity.
"""

import asyncio
import hashlib
import hmac as hmac_mod
import json
import uuid
from typing import Any, Dict, Self

import aiohttp

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Defaults
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0  # seconds
DEFAULT_TIMEOUT = 10  # seconds


def sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature for a webhook payload.

    Args:
        payload_bytes: The raw JSON payload bytes
        secret: The shared secret for this subscription

    Returns:
        Signature string in format "sha256=<hex-digest>"
    """
    digest = hmac_mod.new(
        secret.encode("utf-8"), payload_bytes, hashlib.sha256
    ).hexdigest()
    return f"sha256={digest}"


class WebhookDelivery:
    """Delivers a webhook payload to a URL with HMAC signing and retry.

    Parameters:
        webhook_url: The endpoint to POST to
        secret: Shared secret for HMAC-SHA256 signing
        max_retries: Maximum number of retry attempts after initial failure
        initial_backoff: Initial backoff in seconds (doubles each retry)
        timeout: HTTP request timeout in seconds
    """

    def __init__(
        self: Self,
        webhook_url: str,
        secret: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self.webhook_url = webhook_url
        self.secret = secret
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.timeout = timeout

    async def deliver(self: Self, payload: Dict[str, Any]) -> bool:
        """Deliver a payload to the webhook URL with retry on failure.

        Args:
            payload: The event payload dict to serialize and POST

        Returns:
            True if delivery succeeded (2xx response), False if all retries exhausted
        """
        payload_bytes = json.dumps(payload, default=str).encode("utf-8")
        signature = sign_payload(payload_bytes, self.secret)
        delivery_id = str(uuid.uuid4())

        headers = {
            "Content-Type": "application/json",
            "X-GB-Event": payload.get("event_type", "UNKNOWN"),
            "X-GB-Delivery": delivery_id,
            "X-GB-Signature-256": signature,
        }

        total_attempts = 1 + self.max_retries
        for attempt in range(total_attempts):
            try:
                success = await self._attempt_delivery(payload_bytes, headers)
                if success:
                    logger.info(
                        "[WebhookDelivery] Delivered %s to %s (attempt %d/%d)",
                        delivery_id,
                        self.webhook_url,
                        attempt + 1,
                        total_attempts,
                    )
                    return True

                # Non-2xx response
                if attempt < total_attempts - 1:
                    backoff = self.initial_backoff * (2**attempt)
                    logger.warning(
                        "[WebhookDelivery] Non-2xx from %s, retrying in %.1fs "
                        "(attempt %d/%d)",
                        self.webhook_url,
                        backoff,
                        attempt + 1,
                        total_attempts,
                    )
                    await asyncio.sleep(backoff)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < total_attempts - 1:
                    backoff = self.initial_backoff * (2**attempt)
                    logger.warning(
                        "[WebhookDelivery] Error delivering to %s: %s. "
                        "Retrying in %.1fs (attempt %d/%d)",
                        self.webhook_url,
                        e,
                        backoff,
                        attempt + 1,
                        total_attempts,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "[WebhookDelivery] All retries exhausted for %s: %s",
                        self.webhook_url,
                        e,
                    )

        logger.error(
            "[WebhookDelivery] Failed to deliver %s to %s after %d attempts",
            delivery_id,
            self.webhook_url,
            total_attempts,
        )
        return False

    async def _attempt_delivery(self: Self, payload_bytes: bytes, headers: Dict[str, str]) -> bool:
        """Make a single delivery attempt.

        Args:
            payload_bytes: Serialized JSON payload
            headers: HTTP headers including signature

        Returns:
            True if response was 2xx
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.webhook_url,
                data=payload_bytes,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                if response.status < 300:
                    return True
                body = await response.text()
                logger.warning(
                    "[WebhookDelivery] HTTP %d from %s: %s",
                    response.status,
                    self.webhook_url,
                    body[:200],
                )
                return False
```

**Step 4: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_webhook_delivery.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/gbserver/webhooks/delivery.py test/unit/webhooks/test_webhook_delivery.py
git commit -m "feat(webhooks): add HMAC-signed delivery with exponential backoff (#8)"
```

---

## Task 4: Webhook Dispatcher (matches events to subscriptions)

**Files:**
- Create: `src/gbserver/webhooks/dispatcher.py`
- Test: `test/unit/webhooks/test_webhook_dispatcher.py`

**Step 1: Write the failing test**

```python
# test/unit/webhooks/test_webhook_dispatcher.py
"""Tests for webhook event dispatcher."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.types.buildevent import BuildEvent, BuildEventStatusPayload, BuildEventType
from gbserver.types.status import Status
from gbserver.webhooks.dispatcher import WebhookDispatcher
from gbserver.webhooks.models import StoredWebhookSubscription


class TestWebhookDispatcher:
    def setup_method(self):
        self.mock_storage = MagicMock()
        self.dispatcher = WebhookDispatcher(webhook_storage=self.mock_storage)

    def _make_status_event(self, build_id="build-1", status=Status.SUCCESS):
        """Create a STATUS_EVENT BuildEvent for testing."""
        from gbserver.types.buildevent import EntityRunMetadata

        return BuildEvent(
            type=BuildEventType.STATUS_EVENT,
            run_metadata=EntityRunMetadata(
                build_id=build_id,
                username="testuser",
                target_name="target-1",
                targetrun_id="tr-1",
                targetstep_uri="step://test",
                targetsteprun_id="tsr-1",
            ),
            payload=BuildEventStatusPayload(status=status, msg="Done"),
        )

    def _make_subscription(self, **kwargs):
        defaults = {
            "space_name": "test-space",
            "build_id": "build-1",
            "webhook_url": "https://example.com/hook",
            "secret": "secret",
            "event_types": ["STATUS_EVENT"],
            "created_by": "user",
        }
        defaults.update(kwargs)
        return StoredWebhookSubscription(**defaults)

    @pytest.mark.asyncio
    async def test_dispatch_finds_matching_subscriptions(self):
        sub = self._make_subscription()
        self.mock_storage.get_active_for_build.return_value = [sub]
        event = self._make_status_event()

        with patch(
            "gbserver.webhooks.dispatcher.WebhookDelivery"
        ) as mock_delivery_cls:
            mock_delivery = AsyncMock()
            mock_delivery.deliver = AsyncMock(return_value=True)
            mock_delivery_cls.return_value = mock_delivery

            await self.dispatcher.dispatch(event, space_name="test-space", build_name="test-build")
            self.mock_storage.get_active_for_build.assert_called_once_with("build-1")
            mock_delivery.deliver.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_skips_non_matching_event_types(self):
        sub = self._make_subscription(event_types=["ARTIFACT_EVENT"])
        self.mock_storage.get_active_for_build.return_value = [sub]
        event = self._make_status_event()

        with patch(
            "gbserver.webhooks.dispatcher.WebhookDelivery"
        ) as mock_delivery_cls:
            mock_delivery = AsyncMock()
            mock_delivery_cls.return_value = mock_delivery

            await self.dispatcher.dispatch(event, space_name="test-space", build_name="test-build")
            mock_delivery.deliver.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_skips_internal_events(self):
        from gbserver.types.buildevent import EntityRunMetadata

        event = BuildEvent(
            type=BuildEventType.TERMINATE_EVENT,
            run_metadata=EntityRunMetadata(
                build_id="build-1",
                username="u",
                target_name="t",
                targetrun_id="tr",
                targetstep_uri="s",
                targetsteprun_id="tsr",
            ),
            payload=MagicMock(),
        )

        with patch(
            "gbserver.webhooks.dispatcher.WebhookDelivery"
        ) as mock_delivery_cls:
            mock_delivery = AsyncMock()
            mock_delivery_cls.return_value = mock_delivery

            await self.dispatcher.dispatch(event, space_name="space", build_name="build")
            # Should not even query storage for internal events
            self.mock_storage.get_active_for_build.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pytest test/unit/webhooks/test_webhook_dispatcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gbserver.webhooks.dispatcher'`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/dispatcher.py
"""Webhook event dispatcher.

Matches build events to active subscriptions and spawns background
delivery tasks. This is the integration point called from BuildRunner.
"""

import asyncio
from typing import Any, Dict, Optional, Self

from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
    BuildEventType,
)
from gbserver.utils.logger import get_logger
from gbserver.webhooks.delivery import WebhookDelivery
from gbserver.webhooks.storage import IWebhookStorage

logger = get_logger(__name__)


class WebhookDispatcher:
    """Dispatches build events to matching webhook subscriptions.

    Looks up active subscriptions for the event's build_id, filters by
    event type, and spawns async delivery tasks.

    Args:
        webhook_storage: Storage backend for webhook subscriptions
    """

    def __init__(self: Self, webhook_storage: IWebhookStorage) -> None:
        self.webhook_storage = webhook_storage

    async def dispatch(
        self: Self,
        event: BuildEvent,
        space_name: str,
        build_name: str,
    ) -> None:
        """Dispatch an event to all matching webhook subscriptions.

        Skips internal events. For each matching subscription, spawns a
        background asyncio.Task to deliver the payload.

        Args:
            event: The build event to dispatch
            space_name: Name of the space the build belongs to
            build_name: Name of the build
        """
        # Skip internal events that shouldn't be exposed externally
        if event.type.is_internal_event():
            return

        build_id = event.run_metadata.build_id
        if not build_id:
            return

        event_type_name = event.type.value

        # Find matching subscriptions
        subscriptions = self.webhook_storage.get_active_for_build(build_id)
        if not subscriptions:
            return

        # Build the payload once, share across deliveries
        payload = self._build_payload(event, space_name, build_name)

        for sub in subscriptions:
            if not sub.matches_event_type(event_type_name):
                continue

            # Spawn background delivery task
            delivery = WebhookDelivery(
                webhook_url=sub.webhook_url,
                secret=sub.secret,
            )
            asyncio.create_task(
                self._deliver_with_error_handling(delivery, payload, sub.uuid)
            )

    def _build_payload(
        self: Self, event: BuildEvent, space_name: str, build_name: str
    ) -> Dict[str, Any]:
        """Build the webhook payload dict from a BuildEvent.

        Args:
            event: The source build event
            space_name: Space name for context
            build_name: Build name for context

        Returns:
            Dictionary payload ready for JSON serialization
        """
        meta = event.run_metadata
        payload: Dict[str, Any] = {
            "event_type": event.type.value,
            "timestamp": event.timestamp,
            "build_id": meta.build_id,
            "build_name": build_name,
            "space_name": space_name,
            "target_name": meta.target_name,
            "step_name": meta.targetstep_uri,
        }

        # Add status-specific fields
        if isinstance(event.payload, BuildEventStatusPayload):
            payload["status"] = event.payload.status.value
            payload["message"] = event.payload.msg

        return payload

    async def _deliver_with_error_handling(
        self: Self,
        delivery: WebhookDelivery,
        payload: Dict[str, Any],
        subscription_id: str,
    ) -> None:
        """Wrapper that catches and logs delivery errors without propagating.

        Args:
            delivery: The delivery instance to use
            payload: The payload to deliver
            subscription_id: UUID of the subscription (for logging)
        """
        try:
            await delivery.deliver(payload)
        except Exception as e:
            logger.error(
                "[WebhookDispatcher] Unexpected error delivering to subscription %s: %s",
                subscription_id,
                e,
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_webhook_dispatcher.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/gbserver/webhooks/dispatcher.py test/unit/webhooks/test_webhook_dispatcher.py
git commit -m "feat(webhooks): add event dispatcher matching events to subscriptions (#8)"
```

---

## Task 5: REST API for Webhook Subscriptions

**Files:**
- Create: `src/gbserver/webhooks/api.py`
- Test: `test/unit/webhooks/test_webhook_api.py`

**Step 1: Write the failing test**

```python
# test/unit/webhooks/test_webhook_api.py
"""Tests for webhook subscription REST API."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gbserver.webhooks.api import webhooks_api
from gbserver.webhooks.models import StoredWebhookSubscription


class TestWebhookAPI:
    def setup_method(self):
        self.client = TestClient(webhooks_api)

    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_create_subscription(self, mock_admin_storage, mock_get_storage):
        mock_storage = MagicMock()
        mock_get_storage.return_value = mock_storage

        # Mock build storage to return a valid build
        mock_build_storage = MagicMock()
        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        mock_build_storage.get.return_value = mock_build
        mock_admin = MagicMock()
        mock_admin.build_storage = mock_build_storage
        mock_admin_storage.return_value = mock_admin

        response = self.client.post(
            "/build-123/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "my-secret",
                "event_types": ["STATUS_EVENT"],
            },
            headers={"X-Forwarded-User": "testuser"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["webhook_url"] == "https://example.com/hook"
        assert data["build_id"] == "build-123"
        assert "secret" not in data  # Secret should not be returned

    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_list_subscriptions(self, mock_admin_storage, mock_get_storage):
        mock_storage = MagicMock()
        sub = StoredWebhookSubscription(
            space_name="test-space",
            build_id="build-123",
            webhook_url="https://example.com/hook",
            secret="hidden",
            event_types=["STATUS_EVENT"],
            created_by="testuser",
        )
        mock_storage.get_active_for_build.return_value = [sub]
        mock_get_storage.return_value = mock_storage

        mock_build_storage = MagicMock()
        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        mock_build_storage.get.return_value = mock_build
        mock_admin = MagicMock()
        mock_admin.build_storage = mock_build_storage
        mock_admin_storage.return_value = mock_admin

        response = self.client.get(
            "/build-123/subscriptions",
            headers={"X-Forwarded-User": "testuser"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["subscriptions"]) == 1
        assert "secret" not in data["subscriptions"][0]

    @patch("gbserver.webhooks.api.get_webhook_storage")
    def test_delete_subscription(self, mock_get_storage):
        mock_storage = MagicMock()
        sub = StoredWebhookSubscription(
            space_name="test-space",
            build_id="build-123",
            webhook_url="https://example.com/hook",
            secret="s",
            event_types=["STATUS_EVENT"],
            created_by="testuser",
        )
        mock_storage.get.return_value = sub
        mock_get_storage.return_value = mock_storage

        response = self.client.delete(
            f"/{sub.uuid}",
            headers={"X-Forwarded-User": "testuser"},
        )
        assert response.status_code == 204
        mock_storage.deactivate.assert_called_once_with(sub.uuid)
```

**Step 2: Run test to verify it fails**

Run: `pytest test/unit/webhooks/test_webhook_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gbserver.webhooks.api'`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/api.py
"""FastAPI routes for webhook subscription management.

Endpoints:
    POST   /{build_id}/subscriptions   — Create a per-build webhook subscription
    GET    /{build_id}/subscriptions   — List active subscriptions for a build
    DELETE /{webhook_id}               — Deactivate a subscription
"""

from typing import List, Optional, Self

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.utils.logger import get_logger
from gbserver.webhooks.models import StoredWebhookSubscription
from gbserver.webhooks.storage import IWebhookStorage

logger = get_logger(__name__)

webhooks_api = FastAPI()

# Module-level storage reference (set during app startup)
_webhook_storage: Optional[IWebhookStorage] = None


def set_webhook_storage(storage: IWebhookStorage) -> None:
    """Set the webhook storage instance. Called during app initialization."""
    global _webhook_storage
    _webhook_storage = storage


def get_webhook_storage() -> IWebhookStorage:
    """Get the webhook storage instance.

    Raises:
        RuntimeError: If storage has not been initialized
    """
    if _webhook_storage is None:
        from gbserver.webhooks.sql_storage import SQLWebhookStorage

        set_webhook_storage(SQLWebhookStorage())
    return _webhook_storage  # type: ignore[return-value]


# ── Request/Response models ──────────────────────────────────────────


class CreateWebhookRequest(BaseModel):
    """Request body for creating a webhook subscription.

    Attributes:
        webhook_url: URL to receive POST notifications
        secret: Shared secret for HMAC-SHA256 signing
        event_types: List of event type names to subscribe to
    """

    webhook_url: str
    secret: str
    event_types: List[str]


class WebhookResponse(BaseModel):
    """Response body for a webhook subscription (secret excluded).

    Attributes:
        id: Subscription UUID
        build_id: Build this subscription is for (None if space-wide)
        space_name: Space this subscription belongs to
        webhook_url: Delivery URL
        event_types: Subscribed event types
        active: Whether subscription is active
        created_by: Who created it
        created_time: When it was created
    """

    id: str
    build_id: Optional[str]
    space_name: str
    webhook_url: str
    event_types: List[str]
    active: bool
    created_by: str
    created_time: str


class ListWebhooksResponse(BaseModel):
    """Response body for listing webhook subscriptions."""

    subscriptions: List[WebhookResponse]


def _to_response(sub: StoredWebhookSubscription) -> WebhookResponse:
    """Convert a stored subscription to an API response (excluding secret)."""
    return WebhookResponse(
        id=sub.uuid,
        build_id=sub.build_id,
        space_name=sub.space_name,
        webhook_url=sub.webhook_url,
        event_types=sub.event_types,
        active=sub.active,
        created_by=sub.created_by,
        created_time=sub.created_time.isoformat(),
    )


# ── Routes ───────────────────────────────────────────────────────────


@webhooks_api.post(
    "/{build_id}/subscriptions",
    status_code=status.HTTP_201_CREATED,
    response_model=WebhookResponse,
)
def create_subscription(build_id: str, body: CreateWebhookRequest, request: Request):
    """Create a webhook subscription for a specific build.

    Validates the build exists and creates a subscription scoped to that
    build's space. The secret is stored but never returned in responses.
    """
    username = request.headers.get("X-Forwarded-User", "")
    if not username:
        raise HTTPException(status_code=401, detail="Missing user identity")

    # Verify the build exists
    admin_storage = get_admin_storage()
    build = admin_storage.build_storage.get(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail=f"Build {build_id} not found")

    storage = get_webhook_storage()
    subscription = StoredWebhookSubscription(
        space_name=build.space_name,
        build_id=build_id,
        webhook_url=body.webhook_url,
        secret=body.secret,
        event_types=body.event_types,
        created_by=username,
    )
    storage.add(subscription)

    logger.info(
        "[WebhookAPI] Created subscription %s for build %s by %s",
        subscription.uuid,
        build_id,
        username,
    )
    return _to_response(subscription)


@webhooks_api.get(
    "/{build_id}/subscriptions",
    response_model=ListWebhooksResponse,
)
def list_subscriptions(build_id: str, request: Request):
    """List active webhook subscriptions for a build."""
    username = request.headers.get("X-Forwarded-User", "")
    if not username:
        raise HTTPException(status_code=401, detail="Missing user identity")

    # Verify build exists
    admin_storage = get_admin_storage()
    build = admin_storage.build_storage.get(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail=f"Build {build_id} not found")

    storage = get_webhook_storage()
    subs = storage.get_active_for_build(build_id)
    return ListWebhooksResponse(subscriptions=[_to_response(s) for s in subs])


@webhooks_api.delete(
    "/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_subscription(webhook_id: str, request: Request):
    """Deactivate (soft-delete) a webhook subscription.

    Only the subscription creator can delete it.
    """
    username = request.headers.get("X-Forwarded-User", "")
    if not username:
        raise HTTPException(status_code=401, detail="Missing user identity")

    storage = get_webhook_storage()
    sub = storage.get(webhook_id)
    if sub is None:
        raise HTTPException(
            status_code=404, detail=f"Subscription {webhook_id} not found"
        )

    if sub.created_by != username:
        raise HTTPException(
            status_code=403, detail="Only the subscription creator can delete it"
        )

    storage.deactivate(webhook_id)
    logger.info("[WebhookAPI] Deactivated subscription %s by %s", webhook_id, username)
```

**Step 4: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_webhook_api.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/gbserver/webhooks/api.py test/unit/webhooks/test_webhook_api.py
git commit -m "feat(webhooks): add REST API for subscription management (#8)"
```

---

## Task 6: Integration — Wire Dispatcher into BuildRunner

**Files:**
- Modify: `src/gbserver/buildwatcher/buildrunner.py:740` (after event persistence)
- Modify: `src/gbserver/api/root_api.py:65` (mount webhook API)
- Modify: `src/gbserver/types/constants.py` (add GBSERVER_WEBHOOKS_ENABLED)
- Test: `test/unit/webhooks/test_webhook_integration.py`

**Step 1: Write the failing test**

```python
# test/unit/webhooks/test_webhook_integration.py
"""Tests for webhook integration with BuildRunner event processing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.webhooks.dispatcher import WebhookDispatcher


class TestBuildRunnerWebhookIntegration:
    """Verify that the dispatcher is called from the event processing path."""

    @pytest.mark.asyncio
    async def test_dispatcher_called_on_status_event(self):
        """When GBSERVER_WEBHOOKS_ENABLED=true and a STATUS_EVENT is processed,
        the webhook dispatcher should be called."""
        from gbserver.types.buildevent import (
            BuildEvent,
            BuildEventStatusPayload,
            BuildEventType,
            EntityRunMetadata,
        )
        from gbserver.types.status import Status

        event = BuildEvent(
            type=BuildEventType.STATUS_EVENT,
            run_metadata=EntityRunMetadata(
                build_id="build-1",
                username="user",
                target_name="t",
                targetrun_id="tr",
                targetstep_uri="s",
                targetsteprun_id="tsr",
            ),
            payload=BuildEventStatusPayload(status=Status.SUCCESS, msg="ok"),
        )

        mock_storage = MagicMock()
        mock_storage.get_active_for_build.return_value = []
        dispatcher = WebhookDispatcher(webhook_storage=mock_storage)

        await dispatcher.dispatch(event, space_name="space", build_name="build")
        mock_storage.get_active_for_build.assert_called_once_with("build-1")
```

**Step 2: Run test to verify it passes (this tests the dispatcher in isolation)**

Run: `pytest test/unit/webhooks/test_webhook_integration.py -v`
Expected: PASS

**Step 3: Modify BuildRunner to call dispatcher**

In `src/gbserver/buildwatcher/buildrunner.py`, after line 740 (`self.event_storage.add(StoredEvent(build_event=event))`), add the webhook dispatch call:

```python
# After: self.event_storage.add(StoredEvent(build_event=event))
# Add:
        self.__dispatch_webhook(event)
```

Add new method to the BuildRunner class:

```python
    def __dispatch_webhook(self: Self, event: BuildEvent) -> None:
        """Dispatch event to webhook subscribers if webhooks are enabled."""
        from gbserver.types.constants import GBSERVER_WEBHOOKS_ENABLED

        if not GBSERVER_WEBHOOKS_ENABLED:
            return

        try:
            from gbserver.webhooks.dispatcher import WebhookDispatcher
            from gbserver.webhooks.sql_storage import SQLWebhookStorage

            if not hasattr(self, "_webhook_dispatcher"):
                storage = SQLWebhookStorage()
                self._webhook_dispatcher = WebhookDispatcher(webhook_storage=storage)

            asyncio.ensure_future(
                self._webhook_dispatcher.dispatch(
                    event,
                    space_name=self.stored_build.space_name,
                    build_name=self.stored_build.name,
                )
            )
        except Exception as e:
            logger.warning("[BuildRunner] Webhook dispatch failed (non-fatal): %s", e)
```

**Step 4: Add constant to `src/gbserver/types/constants.py`**

Add near other feature flags:

```python
ENV_VAR_GBSERVER_WEBHOOKS_ENABLED = f"{ENV_VAR_PREFIX}WEBHOOKS_ENABLED"
GBSERVER_WEBHOOKS_ENABLED: bool = getenv_boolean(ENV_VAR_GBSERVER_WEBHOOKS_ENABLED, True)
```

**Step 5: Mount webhook API in `src/gbserver/api/root_api.py`**

Add import and mount:

```python
from gbserver.webhooks.api import webhooks_api

# After existing mounts:
root_api.mount(f"{API_BASE_PATH}/webhooks", webhooks_api)
```

**Step 6: Run all webhook tests**

Run: `pytest test/unit/webhooks/ -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/gbserver/buildwatcher/buildrunner.py \
        src/gbserver/api/root_api.py \
        src/gbserver/types/constants.py \
        test/unit/webhooks/test_webhook_integration.py
git commit -m "feat(webhooks): wire dispatcher into BuildRunner event loop (#8)"
```

---

## Task 7: Auto-cleanup on Build Completion

**Files:**
- Modify: `src/gbserver/buildwatcher/build_utils.py` (in `finalize_build_status()`)
- Test: `test/unit/webhooks/test_webhook_cleanup.py`

**Step 1: Write the failing test**

```python
# test/unit/webhooks/test_webhook_cleanup.py
"""Tests for automatic subscription deactivation on build completion."""

from unittest.mock import MagicMock, patch

import pytest

from gbserver.webhooks.models import StoredWebhookSubscription


class TestWebhookAutoCleanup:
    def test_deactivate_for_build_on_completion(self):
        """When a build reaches terminal state, its subscriptions are deactivated."""
        from gbserver.webhooks.storage import BaseWebhookStorage

        mock_storage = MagicMock(spec=BaseWebhookStorage)
        sub = StoredWebhookSubscription(
            space_name="s",
            build_id="build-done",
            webhook_url="https://example.com",
            secret="s",
            event_types=["STATUS_EVENT"],
            created_by="u",
        )
        mock_storage.get_active_for_build.return_value = [sub]

        mock_storage.deactivate_for_build("build-done")
        mock_storage.deactivate_for_build.assert_called_once_with("build-done")
```

**Step 2: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_webhook_cleanup.py -v`
Expected: PASS

**Step 3: Modify `finalize_build_status()` in `src/gbserver/buildwatcher/build_utils.py`**

At the end of `finalize_build_status()`, after the build status is updated to a terminal state, add:

```python
    # Deactivate webhook subscriptions for completed builds
    try:
        from gbserver.types.constants import GBSERVER_WEBHOOKS_ENABLED

        if GBSERVER_WEBHOOKS_ENABLED:
            from gbserver.webhooks.sql_storage import SQLWebhookStorage

            webhook_storage = SQLWebhookStorage()
            count = webhook_storage.deactivate_for_build(stored_build.uuid)
            if count > 0:
                logger.info(
                    "Deactivated %d webhook subscription(s) for completed build %s",
                    count,
                    stored_build.uuid,
                )
    except Exception as e:
        logger.warning("Failed to deactivate webhook subscriptions: %s", e)
```

**Step 4: Run all tests**

Run: `pytest test/unit/webhooks/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/gbserver/buildwatcher/build_utils.py \
        test/unit/webhooks/test_webhook_cleanup.py
git commit -m "feat(webhooks): auto-deactivate subscriptions on build completion (#8)"
```

---

## Task 8: Format, Lint, and Final Validation

**Files:**
- All new files in `src/gbserver/webhooks/` and `test/unit/webhooks/`

**Step 1: Format**

Run: `make xformat`

**Step 2: Lint**

Run: `make xcheck`

Fix any issues reported by pylint/mypy.

**Step 3: Run full standalone test suite**

Run: `make test-standalone`

**Step 4: Commit any formatting/lint fixes**

```bash
git add -u
git commit -m "chore: apply formatting and fix lint issues (#8)"
```

---

## Summary of Deliverables

| Task | Component | Files |
|------|-----------|-------|
| 1 | Storage model | `webhooks/models.py` |
| 2 | Storage layer | `webhooks/storage.py`, `webhooks/sql_storage.py` |
| 3 | Delivery | `webhooks/delivery.py` |
| 4 | Dispatcher | `webhooks/dispatcher.py` |
| 5 | REST API | `webhooks/api.py` |
| 6 | Integration | `buildrunner.py`, `root_api.py`, `constants.py` |
| 7 | Auto-cleanup | `build_utils.py` |
| 8 | Polish | Formatting + lint + full test run |
