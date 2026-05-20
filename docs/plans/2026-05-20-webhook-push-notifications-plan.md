# Webhook Push Notifications Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add batched webhook push notifications so clients can subscribe to build events instead of polling.

**Architecture:** New `src/gbserver/webhooks/` module with batched event delivery. Events accumulate in an in-memory buffer per subscription and flush every N seconds (default 30s, min 15s). Each flush produces a single signed HTTP POST containing all events since last delivery plus any log pattern matches. Follows existing patterns from `storage/`, `api/`, and `resilience/alert_handlers.py`.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy, aiohttp, HMAC-SHA256, asyncio periodic tasks

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
    def test_create_subscription_with_defaults(self):
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
        assert sub.excluded_types == []
        assert sub.frequency == 30
        assert sub.log_pattern is None
        assert sub.metadata == {}
        assert sub.created_by == "testuser"
        assert sub.active is True

    def test_subscription_with_all_fields(self):
        sub = StoredWebhookSubscription(
            space_name="my-space",
            build_id="b1",
            webhook_url="https://example.com/hook",
            secret="s",
            event_types=["*"],
            excluded_types=["METRICS_EVENT"],
            frequency=60,
            log_pattern="(?i)error",
            metadata={"slack_channel": "#builds"},
            created_by="user",
        )
        assert sub.event_types == ["*"]
        assert sub.excluded_types == ["METRICS_EVENT"]
        assert sub.frequency == 60
        assert sub.log_pattern == "(?i)error"
        assert sub.metadata == {"slack_channel": "#builds"}

    def test_should_include_event_wildcard(self):
        sub = StoredWebhookSubscription(
            space_name="s", webhook_url="u", secret="s",
            event_types=["*"], excluded_types=["METRICS_EVENT"],
            created_by="u",
        )
        assert sub.should_include_event("STATUS_EVENT") is True
        assert sub.should_include_event("ARTIFACT_EVENT") is True
        assert sub.should_include_event("METRICS_EVENT") is False

    def test_should_include_event_explicit_list(self):
        sub = StoredWebhookSubscription(
            space_name="s", webhook_url="u", secret="s",
            event_types=["STATUS_EVENT", "ARTIFACT_EVENT"],
            excluded_types=[],
            created_by="u",
        )
        assert sub.should_include_event("STATUS_EVENT") is True
        assert sub.should_include_event("ARTIFACT_EVENT") is True
        assert sub.should_include_event("METRICS_EVENT") is False

    def test_frequency_minimum_enforced(self):
        sub = StoredWebhookSubscription(
            space_name="s", webhook_url="u", secret="s",
            event_types=["*"], created_by="u",
            frequency=5,  # below minimum
        )
        assert sub.effective_frequency() >= 15
```

**Step 2: Run test to verify it fails**

Run: `pytest test/unit/webhooks/test_webhook_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gbserver.webhooks'`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/__init__.py
"""Webhook push notification module for build events."""

# src/gbserver/webhooks/models.py
"""Pydantic models for webhook subscriptions and delivery payloads."""

import datetime
from typing import Any, Dict, List, Optional, Self

from pydantic import Field

from gbserver.storage.storage import BaseStoredItem
from gbserver.utils.utils import get_utc_time

WEBHOOK_DEFAULT_FREQUENCY = 30
WEBHOOK_MIN_FREQUENCY = 15


class StoredWebhookSubscription(BaseStoredItem):
    """
    A webhook subscription that receives batched push notifications for build events.

    Attributes:
        space_name: The space this subscription is scoped to
        build_id: Specific build to subscribe to (None = space-wide, phase 2)
        webhook_url: URL to POST batched event payloads to
        secret: Shared secret for HMAC-SHA256 signature verification
        event_types: List of BuildEventType names to include, or ["*"] for all
        excluded_types: List of BuildEventType names to exclude (takes precedence)
        frequency: Batch flush interval in seconds (default 30, min 15)
        log_pattern: Optional regex to scan build logs; matches emit LOG_EVENT
        created_by: Username who created this subscription
        active: Whether this subscription is currently delivering events
        metadata: Arbitrary JSONB key/value pairs for future extensibility
        created_time: When the subscription was created
        updated_time: Last time the subscription was modified
    """

    space_name: str
    build_id: Optional[str] = None
    webhook_url: str
    secret: str
    event_types: List[str] = Field(default_factory=lambda: ["*"])
    excluded_types: List[str] = Field(default_factory=list)
    frequency: int = WEBHOOK_DEFAULT_FREQUENCY
    log_pattern: Optional[str] = None
    created_by: str
    active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_time: datetime.datetime = Field(default_factory=get_utc_time)
    updated_time: datetime.datetime = Field(default_factory=get_utc_time)

    def should_include_event(self: Self, event_type: str) -> bool:
        """Determine if an event type passes this subscription's filters.

        Exclusion list is checked first. Then wildcard or explicit inclusion.

        Args:
            event_type: The BuildEventType name to check

        Returns:
            True if the event should be included in the batch
        """
        if event_type in self.excluded_types:
            return False
        if "*" in self.event_types:
            return True
        return event_type in self.event_types

    def effective_frequency(self: Self) -> int:
        """Return the batch frequency, clamped to the minimum.

        Returns:
            Frequency in seconds, at least WEBHOOK_MIN_FREQUENCY
        """
        return max(self.frequency, WEBHOOK_MIN_FREQUENCY)
```

**Step 4: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_webhook_models.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/gbserver/webhooks/__init__.py src/gbserver/webhooks/models.py \
        test/unit/webhooks/__init__.py test/unit/webhooks/test_webhook_models.py
git commit -m "feat(webhooks): add webhook subscription model with batching and log_pattern (#8)"
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

from gbserver.webhooks.models import StoredWebhookSubscription
from gbserver.webhooks.storage import IWebhookStorage


class TestWebhookStorage:
    """Test webhook storage using SQL backend."""

    def setup_method(self):
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
            "event_types": ["*"],
            "excluded_types": ["METRICS_EVENT"],
            "frequency": 30,
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
        assert result.frequency == 30
        assert result.excluded_types == ["METRICS_EVENT"]

    def test_get_active_for_build(self):
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

    def test_deactivate(self):
        sub = self._make_subscription()
        self.storage.add(sub)
        self.storage.deactivate(sub.uuid)
        result = self.storage.get(sub.uuid)
        assert result is not None
        assert result.active is False

    def test_deactivate_for_build(self):
        sub1 = self._make_subscription(build_id="build-done")
        sub2 = self._make_subscription(build_id="build-done")
        self.storage.add(sub1)
        self.storage.add(sub2)

        count = self.storage.deactivate_for_build("build-done")
        assert count == 2

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
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/storage.py
"""Interface and base implementation for webhook subscription storage."""

from typing import List

from gbserver.storage.storage import BaseItemStorage, CREATED_TIME_FIELD_NAME, IItemStorage
from gbserver.webhooks.models import StoredWebhookSubscription

GB_WEBHOOK_SUBSCRIPTIONS_TABLE_NAME = "gb_webhook_subscriptions"


class IWebhookStorage(IItemStorage[StoredWebhookSubscription]):
    """Interface for webhook subscription storage."""

    def get_active_for_build(self, build_id: str) -> List[StoredWebhookSubscription]:
        """Get all active subscriptions for a specific build."""
        raise NotImplementedError

    def get_by_space(self, space_name: str) -> List[StoredWebhookSubscription]:
        """Get all subscriptions for a space."""
        raise NotImplementedError

    def deactivate(self, subscription_id: str) -> None:
        """Deactivate a subscription (soft delete)."""
        raise NotImplementedError

    def deactivate_for_build(self, build_id: str) -> int:
        """Deactivate all subscriptions for a completed build. Returns count."""
        raise NotImplementedError


class BaseWebhookStorage(BaseItemStorage[StoredWebhookSubscription], IWebhookStorage):
    """Base storage implementation for webhook subscriptions."""

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredWebhookSubscription
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_WEBHOOK_SUBSCRIPTIONS_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredWebhookSubscription) -> dict:
        """Extract indexed columns for efficient querying."""
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
            event_types=["*"],
            created_by="system",
        )

    def get_active_for_build(self, build_id: str) -> List[StoredWebhookSubscription]:
        results = []
        for page in self.get_paged({"build_id": build_id, "active": True}, page_size=100):
            results.extend(page)
        return results

    def get_by_space(self, space_name: str) -> List[StoredWebhookSubscription]:
        results = []
        for page in self.get_paged({"space_name": space_name}, page_size=100):
            results.extend(page)
        return results

    def deactivate(self, subscription_id: str) -> None:
        self.update_fields(subscription_id, {"active": False})

    def deactivate_for_build(self, build_id: str) -> int:
        subs = self.get_active_for_build(build_id)
        for sub in subs:
            self.deactivate(sub.uuid)
        return len(subs)
```

```python
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
Expected: PASS (5 tests)

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

    def test_different_secrets_produce_different_signatures(self):
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

            payload = {
                "delivery_id": "d1",
                "events": [{"event_type": "STATUS_EVENT"}],
            }
            result = await delivery.deliver(payload)
            assert result is True

            # Verify HMAC and batch-size headers
            call_kwargs = mock_session.post.call_args.kwargs
            assert "X-GB-Signature-256" in call_kwargs["headers"]
            assert "X-GB-Batch-Size" in call_kwargs["headers"]

    @pytest.mark.asyncio
    async def test_deliver_retries_on_failure(self):
        delivery = WebhookDelivery(
            webhook_url="https://example.com/hook",
            secret="test-secret",
            max_retries=2,
            initial_backoff=0.01,
        )
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            payload = {"events": []}
            result = await delivery.deliver(payload)
            assert result is False
            assert mock_session.post.call_count == 3  # 1 initial + 2 retries
```

**Step 2: Run test to verify it fails**

Run: `pytest test/unit/webhooks/test_webhook_delivery.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/delivery.py
"""Webhook delivery with HMAC-SHA256 signing and retry with exponential backoff.

Delivers batched event payloads to subscriber webhook URLs. Each delivery is
signed with the subscriber's secret for authenticity verification.
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

DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_TIMEOUT = 10


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
    """Delivers a batched webhook payload with HMAC signing and retry.

    Args:
        webhook_url: The endpoint to POST to
        secret: Shared secret for HMAC-SHA256 signing
        max_retries: Maximum retry attempts after initial failure
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
        """Deliver a batched payload with retry on failure.

        Args:
            payload: The batched event payload dict to POST

        Returns:
            True if delivery succeeded (2xx), False if all retries exhausted
        """
        payload_bytes = json.dumps(payload, default=str).encode("utf-8")
        signature = sign_payload(payload_bytes, self.secret)
        delivery_id = payload.get("delivery_id", str(uuid.uuid4()))
        batch_size = str(len(payload.get("events", [])))

        headers = {
            "Content-Type": "application/json",
            "X-GB-Delivery": delivery_id,
            "X-GB-Signature-256": signature,
            "X-GB-Batch-Size": batch_size,
        }

        total_attempts = 1 + self.max_retries
        for attempt in range(total_attempts):
            try:
                success = await self._attempt(payload_bytes, headers)
                if success:
                    logger.info(
                        "[WebhookDelivery] Delivered batch %s (%s events) to %s",
                        delivery_id, batch_size, self.webhook_url,
                    )
                    return True

                if attempt < total_attempts - 1:
                    backoff = self.initial_backoff * (2 ** attempt)
                    logger.warning(
                        "[WebhookDelivery] Non-2xx from %s, retry in %.1fs (%d/%d)",
                        self.webhook_url, backoff, attempt + 1, total_attempts,
                    )
                    await asyncio.sleep(backoff)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < total_attempts - 1:
                    backoff = self.initial_backoff * (2 ** attempt)
                    logger.warning(
                        "[WebhookDelivery] Error %s: %s, retry in %.1fs (%d/%d)",
                        self.webhook_url, e, backoff, attempt + 1, total_attempts,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "[WebhookDelivery] All retries exhausted for %s: %s",
                        self.webhook_url, e,
                    )

        logger.error(
            "[WebhookDelivery] Failed batch %s to %s after %d attempts",
            delivery_id, self.webhook_url, total_attempts,
        )
        return False

    async def _attempt(self: Self, payload_bytes: bytes, headers: Dict[str, str]) -> bool:
        """Single delivery attempt. Returns True on 2xx."""
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
                    response.status, self.webhook_url, body[:200],
                )
                return False
```

**Step 4: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_webhook_delivery.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/gbserver/webhooks/delivery.py test/unit/webhooks/test_webhook_delivery.py
git commit -m "feat(webhooks): add HMAC-signed batch delivery with retry (#8)"
```

---

## Task 4: Log Scanner

**Files:**
- Create: `src/gbserver/webhooks/log_scanner.py`
- Test: `test/unit/webhooks/test_log_scanner.py`

**Step 1: Write the failing test**

```python
# test/unit/webhooks/test_log_scanner.py
"""Tests for build log regex scanning."""

from gbserver.webhooks.log_scanner import scan_log_lines


class TestLogScanner:
    def test_scan_finds_matching_lines(self):
        lines = [
            "2026-05-20 INFO Starting step",
            "2026-05-20 ERROR Connection refused",
            "2026-05-20 INFO Retrying...",
            "2026-05-20 ERROR Timeout exceeded",
        ]
        matches = scan_log_lines(lines, pattern=r"(?i)error")
        assert len(matches) == 2
        assert matches[0]["line"] == "2026-05-20 ERROR Connection refused"
        assert matches[0]["line_number"] == 2
        assert matches[1]["line"] == "2026-05-20 ERROR Timeout exceeded"
        assert matches[1]["line_number"] == 4

    def test_scan_returns_empty_for_no_matches(self):
        lines = ["INFO all good", "INFO still good"]
        matches = scan_log_lines(lines, pattern=r"FATAL")
        assert matches == []

    def test_scan_with_invalid_regex_returns_empty(self):
        lines = ["some text"]
        matches = scan_log_lines(lines, pattern=r"[invalid")
        assert matches == []

    def test_scan_includes_pattern_in_result(self):
        lines = ["Traceback (most recent call last):"]
        matches = scan_log_lines(lines, pattern=r"Traceback")
        assert matches[0]["matched_pattern"] == r"Traceback"
```

**Step 2: Run test to verify it fails**

Run: `pytest test/unit/webhooks/test_log_scanner.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/log_scanner.py
"""Regex-based build log scanner for webhook LOG_EVENT generation.

Scans build log lines against subscriber-defined patterns and produces
structured match results to include in webhook batches.
"""

import re
from typing import Any, Dict, List

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def scan_log_lines(
    lines: List[str],
    pattern: str,
    start_line_number: int = 1,
) -> List[Dict[str, Any]]:
    """Scan log lines for regex matches and return structured results.

    Args:
        lines: List of log line strings to scan
        pattern: Regex pattern to match against each line
        start_line_number: Line number offset for the first line in the batch

    Returns:
        List of match dicts with keys: line, line_number, matched_pattern
    """
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        logger.warning("[LogScanner] Invalid regex pattern '%s': %s", pattern, e)
        return []

    matches = []
    for i, line in enumerate(lines):
        if compiled.search(line):
            matches.append({
                "line": line,
                "line_number": start_line_number + i,
                "matched_pattern": pattern,
            })

    return matches
```

**Step 4: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_log_scanner.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/gbserver/webhooks/log_scanner.py test/unit/webhooks/test_log_scanner.py
git commit -m "feat(webhooks): add log scanner for regex-based LOG_EVENT generation (#8)"
```

---

## Task 5: Batch Buffer

**Files:**
- Create: `src/gbserver/webhooks/batch_buffer.py`
- Test: `test/unit/webhooks/test_batch_buffer.py`

**Step 1: Write the failing test**

```python
# test/unit/webhooks/test_batch_buffer.py
"""Tests for the webhook event batch buffer."""

import time
from unittest.mock import MagicMock

from gbserver.webhooks.batch_buffer import WebhookBatchBuffer
from gbserver.webhooks.models import StoredWebhookSubscription


class TestWebhookBatchBuffer:
    def _make_subscription(self, sub_id="sub-1", frequency=30, **kwargs):
        defaults = {
            "space_name": "s", "build_id": "b1",
            "webhook_url": "https://example.com", "secret": "s",
            "event_types": ["*"], "created_by": "u",
            "frequency": frequency,
        }
        defaults.update(kwargs)
        sub = StoredWebhookSubscription(**defaults)
        # Override UUID for predictable testing
        sub.uuid = sub_id
        return sub

    def test_add_event_to_buffer(self):
        buf = WebhookBatchBuffer()
        sub = self._make_subscription()
        buf.register_subscription(sub)

        event_data = {"event_type": "STATUS_EVENT", "status": "RUNNING"}
        buf.add_event(sub.uuid, event_data)

        assert buf.pending_count(sub.uuid) == 1

    def test_flush_returns_accumulated_events(self):
        buf = WebhookBatchBuffer()
        sub = self._make_subscription()
        buf.register_subscription(sub)

        buf.add_event(sub.uuid, {"event_type": "STATUS_EVENT", "status": "RUNNING"})
        buf.add_event(sub.uuid, {"event_type": "STATUS_EVENT", "status": "SUCCESS"})

        events = buf.flush(sub.uuid)
        assert len(events) == 2
        assert buf.pending_count(sub.uuid) == 0

    def test_flush_empty_buffer_returns_empty_list(self):
        buf = WebhookBatchBuffer()
        sub = self._make_subscription()
        buf.register_subscription(sub)

        events = buf.flush(sub.uuid)
        assert events == []

    def test_is_ready_to_flush(self):
        buf = WebhookBatchBuffer()
        sub = self._make_subscription(frequency=1)  # 1 second for testing
        buf.register_subscription(sub)
        buf.add_event(sub.uuid, {"event_type": "STATUS_EVENT"})

        # Not ready immediately
        assert buf.is_ready_to_flush(sub.uuid) is False

        # Simulate time passage by backdating last_flush
        buf._last_flush[sub.uuid] = time.time() - 2
        assert buf.is_ready_to_flush(sub.uuid) is True

    def test_unregister_subscription(self):
        buf = WebhookBatchBuffer()
        sub = self._make_subscription()
        buf.register_subscription(sub)
        buf.add_event(sub.uuid, {"event_type": "STATUS_EVENT"})

        buf.unregister_subscription(sub.uuid)
        assert buf.pending_count(sub.uuid) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest test/unit/webhooks/test_batch_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/batch_buffer.py
"""In-memory event accumulator with per-subscription flush timers.

Buffers incoming events per subscription and tracks when each subscription
is due for its next batch delivery based on its configured frequency.
"""

import time
import threading
from typing import Any, Dict, List, Self

from gbserver.utils.logger import get_logger
from gbserver.webhooks.models import StoredWebhookSubscription

logger = get_logger(__name__)


class WebhookBatchBuffer:
    """Accumulates events per subscription with time-based flush readiness.

    Thread-safe buffer that holds events until a subscription's frequency
    interval has elapsed, at which point the buffer can be flushed.
    """

    def __init__(self: Self) -> None:
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._frequencies: Dict[str, int] = {}
        self._last_flush: Dict[str, float] = {}
        self._lock = threading.Lock()

    def register_subscription(self: Self, subscription: StoredWebhookSubscription) -> None:
        """Register a subscription for event buffering.

        Args:
            subscription: The subscription to start buffering for
        """
        sub_id = subscription.uuid
        with self._lock:
            if sub_id not in self._buffers:
                self._buffers[sub_id] = []
                self._frequencies[sub_id] = subscription.effective_frequency()
                self._last_flush[sub_id] = time.time()

    def unregister_subscription(self: Self, subscription_id: str) -> None:
        """Remove a subscription from the buffer, discarding pending events.

        Args:
            subscription_id: UUID of the subscription to remove
        """
        with self._lock:
            self._buffers.pop(subscription_id, None)
            self._frequencies.pop(subscription_id, None)
            self._last_flush.pop(subscription_id, None)

    def add_event(self: Self, subscription_id: str, event_data: Dict[str, Any]) -> None:
        """Add an event to a subscription's buffer.

        Args:
            subscription_id: UUID of the subscription
            event_data: Serialized event dict to accumulate
        """
        with self._lock:
            if subscription_id in self._buffers:
                self._buffers[subscription_id].append(event_data)

    def pending_count(self: Self, subscription_id: str) -> int:
        """Return number of pending events for a subscription.

        Args:
            subscription_id: UUID of the subscription

        Returns:
            Number of buffered events, 0 if subscription not registered
        """
        with self._lock:
            return len(self._buffers.get(subscription_id, []))

    def is_ready_to_flush(self: Self, subscription_id: str) -> bool:
        """Check if a subscription's batch interval has elapsed.

        Args:
            subscription_id: UUID of the subscription

        Returns:
            True if frequency seconds have passed since last flush and buffer is non-empty
        """
        with self._lock:
            if subscription_id not in self._buffers:
                return False
            if not self._buffers[subscription_id]:
                return False
            elapsed = time.time() - self._last_flush.get(subscription_id, 0)
            return elapsed >= self._frequencies.get(subscription_id, 30)

    def flush(self: Self, subscription_id: str) -> List[Dict[str, Any]]:
        """Flush and return all buffered events for a subscription.

        Resets the flush timer. Returns empty list if no events buffered.

        Args:
            subscription_id: UUID of the subscription

        Returns:
            List of accumulated event dicts
        """
        with self._lock:
            events = self._buffers.get(subscription_id, [])
            self._buffers[subscription_id] = []
            self._last_flush[subscription_id] = time.time()
            return events

    def get_ready_subscriptions(self: Self) -> List[str]:
        """Return subscription IDs that are ready for flushing.

        Returns:
            List of subscription UUIDs whose batch interval has elapsed
        """
        ready = []
        with self._lock:
            now = time.time()
            for sub_id, buf in self._buffers.items():
                if not buf:
                    continue
                elapsed = now - self._last_flush.get(sub_id, 0)
                if elapsed >= self._frequencies.get(sub_id, 30):
                    ready.append(sub_id)
        return ready
```

**Step 4: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_batch_buffer.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/gbserver/webhooks/batch_buffer.py test/unit/webhooks/test_batch_buffer.py
git commit -m "feat(webhooks): add batch buffer for per-subscription event accumulation (#8)"
```

---

## Task 6: Webhook Dispatcher (orchestrates batching, scanning, delivery)

**Files:**
- Create: `src/gbserver/webhooks/dispatcher.py`
- Test: `test/unit/webhooks/test_webhook_dispatcher.py`

**Step 1: Write the failing test**

```python
# test/unit/webhooks/test_webhook_dispatcher.py
"""Tests for webhook batch dispatcher."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.types.buildevent import (
    BuildEvent, BuildEventStatusPayload, BuildEventType, EntityRunMetadata,
)
from gbserver.types.status import Status
from gbserver.webhooks.dispatcher import WebhookDispatcher
from gbserver.webhooks.models import StoredWebhookSubscription


class TestWebhookDispatcher:
    def setup_method(self):
        self.mock_storage = MagicMock()
        self.dispatcher = WebhookDispatcher(
            webhook_storage=self.mock_storage,
            build_id="build-1",
            space_name="test-space",
            build_name="test-build",
            username="testuser",
            build_start_time="2026-05-20T11:00:00Z",
        )

    def _make_subscription(self, **kwargs):
        defaults = {
            "space_name": "test-space",
            "build_id": "build-1",
            "webhook_url": "https://example.com/hook",
            "secret": "secret",
            "event_types": ["*"],
            "created_by": "user",
            "frequency": 30,
        }
        defaults.update(kwargs)
        return StoredWebhookSubscription(**defaults)

    def _make_status_event(self, status=Status.SUCCESS):
        return BuildEvent(
            type=BuildEventType.STATUS_EVENT,
            run_metadata=EntityRunMetadata(
                build_id="build-1", username="user",
                target_name="target-1", targetrun_id="tr-1",
                targetstep_uri="step://test", targetsteprun_id="tsr-1",
            ),
            payload=BuildEventStatusPayload(status=status, msg="Done"),
        )

    def test_accept_event_buffers_matching_events(self):
        sub = self._make_subscription()
        self.mock_storage.get_active_for_build.return_value = [sub]
        self.dispatcher.start([sub])

        event = self._make_status_event()
        self.dispatcher.accept_event(event)

        assert self.dispatcher.buffer.pending_count(sub.uuid) == 1

    def test_accept_event_skips_excluded_types(self):
        sub = self._make_subscription(
            event_types=["*"], excluded_types=["STATUS_EVENT"]
        )
        self.dispatcher.start([sub])

        event = self._make_status_event()
        self.dispatcher.accept_event(event)

        assert self.dispatcher.buffer.pending_count(sub.uuid) == 0

    def test_accept_event_skips_internal_events(self):
        sub = self._make_subscription()
        self.dispatcher.start([sub])

        event = BuildEvent(
            type=BuildEventType.TERMINATE_EVENT,
            run_metadata=EntityRunMetadata(
                build_id="build-1", username="u",
                target_name="t", targetrun_id="tr",
                targetstep_uri="s", targetsteprun_id="tsr",
            ),
            payload=MagicMock(),
        )
        self.dispatcher.accept_event(event)
        assert self.dispatcher.buffer.pending_count(sub.uuid) == 0

    @pytest.mark.asyncio
    async def test_flush_subscription_delivers_batch(self):
        sub = self._make_subscription()
        self.dispatcher.start([sub])

        event = self._make_status_event()
        self.dispatcher.accept_event(event)

        with patch("gbserver.webhooks.dispatcher.WebhookDelivery") as mock_cls:
            mock_delivery = AsyncMock()
            mock_delivery.deliver = AsyncMock(return_value=True)
            mock_cls.return_value = mock_delivery

            await self.dispatcher.flush_subscription(sub)

            mock_delivery.deliver.assert_called_once()
            payload = mock_delivery.deliver.call_args[0][0]
            assert payload["build_id"] == "build-1"
            assert payload["user"] == "testuser"
            assert len(payload["events"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest test/unit/webhooks/test_webhook_dispatcher.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/dispatcher.py
"""Webhook batch dispatcher.

Orchestrates the batching lifecycle: accepts events, buffers them per
subscription, and flushes batches on a periodic schedule. Also coordinates
log pattern scanning at flush time.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Self

from gbserver.types.buildevent import BuildEvent, BuildEventStatusPayload, BuildEventType
from gbserver.utils.logger import get_logger
from gbserver.webhooks.batch_buffer import WebhookBatchBuffer
from gbserver.webhooks.delivery import WebhookDelivery
from gbserver.webhooks.log_scanner import scan_log_lines
from gbserver.webhooks.models import StoredWebhookSubscription
from gbserver.webhooks.storage import IWebhookStorage

logger = get_logger(__name__)


class WebhookDispatcher:
    """Orchestrates batched webhook delivery for a single build.

    One dispatcher instance is created per active build. It holds the
    batch buffer, accepts events from the BuildRunner, and runs a
    periodic flush loop.

    Args:
        webhook_storage: Storage backend for subscriptions
        build_id: UUID of the build being monitored
        space_name: Space the build belongs to
        build_name: Name of the build
        username: User who created the build
        build_start_time: ISO timestamp of when the build started
    """

    def __init__(
        self: Self,
        webhook_storage: IWebhookStorage,
        build_id: str,
        space_name: str,
        build_name: str,
        username: str,
        build_start_time: str,
    ) -> None:
        self.webhook_storage = webhook_storage
        self.build_id = build_id
        self.space_name = space_name
        self.build_name = build_name
        self.username = username
        self.build_start_time = build_start_time
        self.buffer = WebhookBatchBuffer()
        self._subscriptions: Dict[str, StoredWebhookSubscription] = {}
        self._flush_task: Optional[asyncio.Task] = None
        self._log_lines_cursor: int = 0  # Track log position for scanning

    def start(self: Self, subscriptions: List[StoredWebhookSubscription]) -> None:
        """Register subscriptions and prepare for event buffering.

        Args:
            subscriptions: Active subscriptions for this build
        """
        for sub in subscriptions:
            self._subscriptions[sub.uuid] = sub
            self.buffer.register_subscription(sub)

    def accept_event(self: Self, event: BuildEvent) -> None:
        """Accept a build event and buffer it for matching subscriptions.

        Skips internal events. Checks each subscription's filter to determine
        if the event should be buffered.

        Args:
            event: The build event from the event processing loop
        """
        if event.type.is_internal_event():
            return

        event_type_name = event.type.value
        event_data = self._serialize_event(event)

        for sub_id, sub in self._subscriptions.items():
            if sub.should_include_event(event_type_name):
                self.buffer.add_event(sub_id, event_data)

    def _serialize_event(self: Self, event: BuildEvent) -> Dict[str, Any]:
        """Serialize a BuildEvent into a dict for the batch payload.

        Args:
            event: The raw build event

        Returns:
            Serialized event dict
        """
        meta = event.run_metadata
        data: Dict[str, Any] = {
            "event_id": str(uuid.uuid4()),
            "event_type": event.type.value,
            "timestamp": event.timestamp,
            "target_name": meta.target_name,
            "step_name": meta.targetstep_uri,
        }

        if isinstance(event.payload, BuildEventStatusPayload):
            data["status"] = event.payload.status.value
            data["message"] = {"text": event.payload.msg}
        else:
            # Generic payload — serialize data field as message
            payload_data = getattr(event.payload, "data", None)
            data["message"] = payload_data if payload_data else {}

        return data

    async def flush_subscription(
        self: Self,
        subscription: StoredWebhookSubscription,
        log_lines: Optional[List[str]] = None,
    ) -> None:
        """Flush the batch buffer for a subscription and deliver.

        Args:
            subscription: The subscription to flush
            log_lines: Optional new log lines to scan for log_pattern matches
        """
        events = self.buffer.flush(subscription.uuid)

        # Scan logs for pattern matches if configured
        if subscription.log_pattern and log_lines:
            matches = scan_log_lines(
                log_lines,
                subscription.log_pattern,
                start_line_number=self._log_lines_cursor + 1,
            )
            for match in matches:
                events.append({
                    "event_id": str(uuid.uuid4()),
                    "event_type": "LOG_EVENT",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "target_name": None,
                    "step_name": None,
                    "message": match,
                })

        if not events:
            return

        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "delivery_id": str(uuid.uuid4()),
            "build_id": self.build_id,
            "build_name": self.build_name,
            "space_name": self.space_name,
            "user": self.username,
            "build_start_time": self.build_start_time,
            "batch_start": now,  # Simplified; could track actual window
            "batch_end": now,
            "events": events,
        }

        delivery = WebhookDelivery(
            webhook_url=subscription.webhook_url,
            secret=subscription.secret,
        )
        await delivery.deliver(payload)

    async def flush_all_ready(self: Self, log_lines: Optional[List[str]] = None) -> None:
        """Flush all subscriptions whose batch interval has elapsed.

        Args:
            log_lines: New log lines since last flush (for log_pattern scanning)
        """
        ready_ids = self.buffer.get_ready_subscriptions()
        for sub_id in ready_ids:
            sub = self._subscriptions.get(sub_id)
            if sub:
                try:
                    await self.flush_subscription(sub, log_lines)
                except Exception as e:
                    logger.error(
                        "[WebhookDispatcher] Error flushing subscription %s: %s",
                        sub_id, e,
                    )

    async def flush_final(self: Self, log_lines: Optional[List[str]] = None) -> None:
        """Force-flush all subscriptions (called on build completion).

        Args:
            log_lines: Final log lines to scan
        """
        for sub_id, sub in self._subscriptions.items():
            try:
                await self.flush_subscription(sub, log_lines)
            except Exception as e:
                logger.error(
                    "[WebhookDispatcher] Error in final flush for %s: %s",
                    sub_id, e,
                )

    def stop(self: Self) -> None:
        """Stop the dispatcher and clean up buffers."""
        for sub_id in list(self._subscriptions.keys()):
            self.buffer.unregister_subscription(sub_id)
        self._subscriptions.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_webhook_dispatcher.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/gbserver/webhooks/dispatcher.py test/unit/webhooks/test_webhook_dispatcher.py
git commit -m "feat(webhooks): add batch dispatcher with log scanning (#8)"
```

---

## Task 7: REST API for Webhook Subscriptions

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

        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        mock_admin = MagicMock()
        mock_admin.build_storage.get.return_value = mock_build
        mock_admin_storage.return_value = mock_admin

        response = self.client.post(
            "/build-123/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "my-secret",
                "event_types": ["*"],
                "excluded_types": ["METRICS_EVENT"],
                "frequency": 45,
                "log_pattern": "(?i)error",
            },
            headers={"X-Forwarded-User": "testuser"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["webhook_url"] == "https://example.com/hook"
        assert data["build_id"] == "build-123"
        assert data["event_types"] == ["*"]
        assert data["excluded_types"] == ["METRICS_EVENT"]
        assert data["frequency"] == 45
        assert data["log_pattern"] == "(?i)error"
        assert "secret" not in data

    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_create_subscription_enforces_min_frequency(self, mock_admin, mock_get_storage):
        mock_storage = MagicMock()
        mock_get_storage.return_value = mock_storage

        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        admin = MagicMock()
        admin.build_storage.get.return_value = mock_build
        mock_admin.return_value = admin

        response = self.client.post(
            "/build-123/subscriptions",
            json={
                "webhook_url": "https://example.com/hook",
                "secret": "s",
                "event_types": ["*"],
                "frequency": 5,  # Below minimum
            },
            headers={"X-Forwarded-User": "testuser"},
        )
        assert response.status_code == 400
        assert "minimum" in response.json()["detail"].lower()

    @patch("gbserver.webhooks.api.get_webhook_storage")
    @patch("gbserver.webhooks.api.get_admin_storage")
    def test_list_subscriptions(self, mock_admin_storage, mock_get_storage):
        mock_storage = MagicMock()
        sub = StoredWebhookSubscription(
            space_name="test-space", build_id="build-123",
            webhook_url="https://example.com/hook", secret="hidden",
            event_types=["*"], created_by="testuser",
        )
        mock_storage.get_active_for_build.return_value = [sub]
        mock_get_storage.return_value = mock_storage

        mock_build = MagicMock()
        mock_build.space_name = "test-space"
        admin = MagicMock()
        admin.build_storage.get.return_value = mock_build
        mock_admin_storage.return_value = admin

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
            space_name="test-space", build_id="build-123",
            webhook_url="https://example.com/hook", secret="s",
            event_types=["*"], created_by="testuser",
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
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/gbserver/webhooks/api.py
"""FastAPI routes for webhook subscription management.

Endpoints:
    POST   /{build_id}/subscriptions — Create per-build webhook subscription
    GET    /{build_id}/subscriptions — List active subscriptions for a build
    DELETE /{webhook_id}             — Deactivate a subscription
"""

from typing import Any, Dict, List, Optional, Self

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.utils.logger import get_logger
from gbserver.webhooks.models import WEBHOOK_MIN_FREQUENCY, StoredWebhookSubscription
from gbserver.webhooks.storage import IWebhookStorage

logger = get_logger(__name__)

webhooks_api = FastAPI()

_webhook_storage: Optional[IWebhookStorage] = None


def set_webhook_storage(storage: IWebhookStorage) -> None:
    """Set the webhook storage instance (called during app init)."""
    global _webhook_storage
    _webhook_storage = storage


def get_webhook_storage() -> IWebhookStorage:
    """Get the webhook storage, lazily initializing if needed."""
    global _webhook_storage
    if _webhook_storage is None:
        from gbserver.webhooks.sql_storage import SQLWebhookStorage
        _webhook_storage = SQLWebhookStorage()
    return _webhook_storage


# ── Request/Response Models ──────────────────────────────────────────


class CreateWebhookRequest(BaseModel):
    """Request body for creating a webhook subscription."""
    webhook_url: str
    secret: str
    event_types: List[str] = ["*"]
    excluded_types: List[str] = []
    frequency: int = 30
    log_pattern: Optional[str] = None
    metadata: Dict[str, Any] = {}


class WebhookResponse(BaseModel):
    """Response body for a subscription (secret excluded)."""
    id: str
    build_id: Optional[str]
    space_name: str
    webhook_url: str
    event_types: List[str]
    excluded_types: List[str]
    frequency: int
    log_pattern: Optional[str]
    active: bool
    created_by: str
    created_time: str


class ListWebhooksResponse(BaseModel):
    """Response body for listing subscriptions."""
    subscriptions: List[WebhookResponse]


def _to_response(sub: StoredWebhookSubscription) -> WebhookResponse:
    """Convert stored subscription to API response (excluding secret)."""
    return WebhookResponse(
        id=sub.uuid,
        build_id=sub.build_id,
        space_name=sub.space_name,
        webhook_url=sub.webhook_url,
        event_types=sub.event_types,
        excluded_types=sub.excluded_types,
        frequency=sub.frequency,
        log_pattern=sub.log_pattern,
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
    """Create a webhook subscription for a build."""
    username = request.headers.get("X-Forwarded-User", "")
    if not username:
        raise HTTPException(status_code=401, detail="Missing user identity")

    # Validate minimum frequency
    if body.frequency < WEBHOOK_MIN_FREQUENCY:
        raise HTTPException(
            status_code=400,
            detail=f"Frequency must be at minimum {WEBHOOK_MIN_FREQUENCY} seconds",
        )

    # Verify build exists
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
        excluded_types=body.excluded_types,
        frequency=body.frequency,
        log_pattern=body.log_pattern,
        metadata=body.metadata,
        created_by=username,
    )
    storage.add(subscription)

    logger.info(
        "[WebhookAPI] Created subscription %s for build %s (freq=%ds) by %s",
        subscription.uuid, build_id, body.frequency, username,
    )
    return _to_response(subscription)


@webhooks_api.get("/{build_id}/subscriptions", response_model=ListWebhooksResponse)
def list_subscriptions(build_id: str, request: Request):
    """List active webhook subscriptions for a build."""
    username = request.headers.get("X-Forwarded-User", "")
    if not username:
        raise HTTPException(status_code=401, detail="Missing user identity")

    admin_storage = get_admin_storage()
    build = admin_storage.build_storage.get(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail=f"Build {build_id} not found")

    storage = get_webhook_storage()
    subs = storage.get_active_for_build(build_id)
    return ListWebhooksResponse(subscriptions=[_to_response(s) for s in subs])


@webhooks_api.delete("/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_subscription(webhook_id: str, request: Request):
    """Deactivate a webhook subscription. Only the creator can delete."""
    username = request.headers.get("X-Forwarded-User", "")
    if not username:
        raise HTTPException(status_code=401, detail="Missing user identity")

    storage = get_webhook_storage()
    sub = storage.get(webhook_id)
    if sub is None:
        raise HTTPException(status_code=404, detail=f"Subscription {webhook_id} not found")

    if sub.created_by != username:
        raise HTTPException(status_code=403, detail="Only the creator can delete this subscription")

    storage.deactivate(webhook_id)
    logger.info("[WebhookAPI] Deactivated subscription %s by %s", webhook_id, username)
```

**Step 4: Run test to verify it passes**

Run: `pytest test/unit/webhooks/test_webhook_api.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/gbserver/webhooks/api.py test/unit/webhooks/test_webhook_api.py
git commit -m "feat(webhooks): add REST API with frequency validation and log_pattern (#8)"
```

---

## Task 8: Integration — Wire into BuildRunner and Root API

**Files:**
- Modify: `src/gbserver/buildwatcher/buildrunner.py` (~line 740)
- Modify: `src/gbserver/buildwatcher/build_utils.py` (finalize_build_status)
- Modify: `src/gbserver/api/root_api.py` (mount webhook API)
- Modify: `src/gbserver/types/constants.py` (add webhook env vars)
- Test: `test/unit/webhooks/test_webhook_integration.py`

**Step 1: Add constants**

In `src/gbserver/types/constants.py`, add:

```python
ENV_VAR_GBSERVER_WEBHOOKS_ENABLED = f"{ENV_VAR_PREFIX}WEBHOOKS_ENABLED"
GBSERVER_WEBHOOKS_ENABLED: bool = getenv_boolean(ENV_VAR_GBSERVER_WEBHOOKS_ENABLED, True)

ENV_VAR_GBSERVER_WEBHOOKS_DEFAULT_FREQUENCY = f"{ENV_VAR_PREFIX}WEBHOOKS_DEFAULT_FREQUENCY"
GBSERVER_WEBHOOKS_DEFAULT_FREQUENCY: int = int(
    os.getenv(ENV_VAR_GBSERVER_WEBHOOKS_DEFAULT_FREQUENCY, "30")
)

ENV_VAR_GBSERVER_WEBHOOKS_MIN_FREQUENCY = f"{ENV_VAR_PREFIX}WEBHOOKS_MIN_FREQUENCY"
GBSERVER_WEBHOOKS_MIN_FREQUENCY: int = int(
    os.getenv(ENV_VAR_GBSERVER_WEBHOOKS_MIN_FREQUENCY, "15")
)
```

**Step 2: Mount API in `src/gbserver/api/root_api.py`**

Add after existing mounts:

```python
from gbserver.webhooks.api import webhooks_api

root_api.mount(f"{API_BASE_PATH}/webhooks", webhooks_api)
```

**Step 3: Integrate dispatcher into BuildRunner**

In `src/gbserver/buildwatcher/buildrunner.py`, in the `__process_event` method, after line 740 (`self.event_storage.add(StoredEvent(build_event=event))`):

```python
        # Dispatch to webhook subscribers
        if self._webhook_dispatcher is not None:
            self._webhook_dispatcher.accept_event(event)
```

In the BuildRunner `__init__` or setup, initialize the dispatcher:

```python
    def _init_webhook_dispatcher(self: Self) -> None:
        """Initialize webhook dispatcher if enabled and subscriptions exist."""
        from gbserver.types.constants import GBSERVER_WEBHOOKS_ENABLED
        if not GBSERVER_WEBHOOKS_ENABLED:
            self._webhook_dispatcher = None
            return

        try:
            from gbserver.webhooks.dispatcher import WebhookDispatcher
            from gbserver.webhooks.sql_storage import SQLWebhookStorage

            storage = SQLWebhookStorage()
            subs = storage.get_active_for_build(self.stored_build.uuid)
            if not subs:
                self._webhook_dispatcher = None
                return

            self._webhook_dispatcher = WebhookDispatcher(
                webhook_storage=storage,
                build_id=self.stored_build.uuid,
                space_name=self.stored_build.space_name,
                build_name=self.stored_build.name,
                username=self.stored_build.username,
                build_start_time=self.stored_build.created_time.isoformat(),
            )
            self._webhook_dispatcher.start(subs)
        except Exception as e:
            logger.warning("[BuildRunner] Failed to init webhook dispatcher: %s", e)
            self._webhook_dispatcher = None
```

In the worker task periodic loop (already runs every ~1s), add a flush check:

```python
        # Flush ready webhook batches
        if self._webhook_dispatcher is not None:
            await self._webhook_dispatcher.flush_all_ready()
```

**Step 4: Auto-deactivate on build completion in `build_utils.py`**

At end of `finalize_build_status()`:

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
                    count, stored_build.uuid,
                )
    except Exception as e:
        logger.warning("Failed to deactivate webhook subscriptions: %s", e)
```

**Step 5: Write integration test**

```python
# test/unit/webhooks/test_webhook_integration.py
"""Tests verifying webhook dispatcher integrates with build event flow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.types.buildevent import (
    BuildEvent, BuildEventStatusPayload, BuildEventType, EntityRunMetadata,
)
from gbserver.types.status import Status
from gbserver.webhooks.dispatcher import WebhookDispatcher
from gbserver.webhooks.models import StoredWebhookSubscription


class TestBuildRunnerWebhookIntegration:
    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Simulate: event accepted -> buffered -> flushed -> delivered."""
        mock_storage = MagicMock()
        sub = StoredWebhookSubscription(
            space_name="s", build_id="b1",
            webhook_url="https://example.com/hook", secret="sec",
            event_types=["*"], created_by="u", frequency=1,
        )
        mock_storage.get_active_for_build.return_value = [sub]

        dispatcher = WebhookDispatcher(
            webhook_storage=mock_storage,
            build_id="b1", space_name="s", build_name="build",
            username="u", build_start_time="2026-05-20T11:00:00Z",
        )
        dispatcher.start([sub])

        # Accept event
        event = BuildEvent(
            type=BuildEventType.STATUS_EVENT,
            run_metadata=EntityRunMetadata(
                build_id="b1", username="u", target_name="t",
                targetrun_id="tr", targetstep_uri="s", targetsteprun_id="tsr",
            ),
            payload=BuildEventStatusPayload(status=Status.RUNNING, msg="go"),
        )
        dispatcher.accept_event(event)

        # Force flush
        with patch("gbserver.webhooks.dispatcher.WebhookDelivery") as mock_cls:
            mock_delivery = AsyncMock()
            mock_delivery.deliver = AsyncMock(return_value=True)
            mock_cls.return_value = mock_delivery

            await dispatcher.flush_final()

            mock_delivery.deliver.assert_called_once()
            payload = mock_delivery.deliver.call_args[0][0]
            assert payload["build_id"] == "b1"
            assert len(payload["events"]) == 1
            assert payload["events"][0]["event_type"] == "status_event"
```

**Step 6: Run all webhook tests**

Run: `pytest test/unit/webhooks/ -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/gbserver/buildwatcher/buildrunner.py src/gbserver/buildwatcher/build_utils.py \
        src/gbserver/api/root_api.py src/gbserver/types/constants.py \
        test/unit/webhooks/test_webhook_integration.py
git commit -m "feat(webhooks): integrate dispatcher into BuildRunner and mount API (#8)"
```

---

## Task 9: Format, Lint, Final Validation

**Step 1: Format**
Run: `make xformat`

**Step 2: Lint**
Run: `make xcheck`
Fix any issues.

**Step 3: Run standalone tests**
Run: `make test-standalone`

**Step 4: Commit fixes**
```bash
git add -u
git commit -m "chore: formatting and lint fixes (#8)"
```

---

## Summary

| Task | Component | Key Files |
|------|-----------|-----------|
| 1 | Storage model | `webhooks/models.py` |
| 2 | Storage layer | `webhooks/storage.py`, `webhooks/sql_storage.py` |
| 3 | Delivery | `webhooks/delivery.py` |
| 4 | Log scanner | `webhooks/log_scanner.py` |
| 5 | Batch buffer | `webhooks/batch_buffer.py` |
| 6 | Dispatcher | `webhooks/dispatcher.py` |
| 7 | REST API | `webhooks/api.py` |
| 8 | Integration | `buildrunner.py`, `build_utils.py`, `root_api.py`, `constants.py` |
| 9 | Polish | Formatting, lint, test validation |
