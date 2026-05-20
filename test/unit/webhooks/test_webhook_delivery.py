"""Tests for webhook delivery module with HMAC signing and retry."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.webhooks.delivery import (
    DEFAULT_INITIAL_BACKOFF,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    WebhookDelivery,
    sign_payload,
)


class TestSignPayload:
    """Tests for the sign_payload helper function."""

    def test_sign_payload_produces_valid_hmac(self):
        """Verify signature matches manual hmac.new() computation."""
        payload = b'{"event": "build.started"}'
        secret = "my-secret-key"

        result = sign_payload(payload, secret)

        expected_digest = hmac.new(
            secret.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()
        assert result == f"sha256={expected_digest}"

    def test_different_secrets_produce_different_signatures(self):
        """Same payload with different secrets should produce different signatures."""
        payload = b'{"event": "build.completed"}'
        secret_a = "secret-alpha"
        secret_b = "secret-beta"

        sig_a = sign_payload(payload, secret_a)
        sig_b = sign_payload(payload, secret_b)

        assert sig_a != sig_b


class TestWebhookDelivery:
    """Tests for the WebhookDelivery class."""

    @pytest.mark.asyncio
    async def test_deliver_success(self):
        """Mock aiohttp.ClientSession to return 200, verify success."""
        delivery = WebhookDelivery(
            webhook_url="https://example.com/hook",
            secret="test-secret",
            max_retries=3,
            initial_backoff=0.01,
            timeout=5,
        )
        payload = {"event": "build.started", "build_id": "abc123"}

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "gbserver.webhooks.delivery.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            result = await delivery.deliver(payload)

        assert result is True
        mock_session.post.assert_called_once()

        # Verify headers include signature and batch size
        call_kwargs = mock_session.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert "X-GB-Signature-256" in headers
        assert "X-GB-Delivery" in headers
        assert headers["X-GB-Batch-Size"] == str(1)
        assert headers["Content-Type"] == "application/json"

        # Verify the signature is correct
        payload_bytes = json.dumps(payload).encode("utf-8")
        expected_sig = sign_payload(payload_bytes, "test-secret")
        assert headers["X-GB-Signature-256"] == expected_sig

    @pytest.mark.asyncio
    async def test_deliver_retries_on_failure(self):
        """Mock to always return 500, verify retries and final failure."""
        max_retries = 2
        delivery = WebhookDelivery(
            webhook_url="https://example.com/hook",
            secret="test-secret",
            max_retries=max_retries,
            initial_backoff=0.01,
            timeout=5,
        )
        payload = {"event": "build.failed", "build_id": "def456"}

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "gbserver.webhooks.delivery.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await delivery.deliver(payload)

        assert result is False
        # 1 initial attempt + 2 retries = 3 total calls
        assert mock_session.post.call_count == 1 + max_retries
        # Sleep called once per retry (2 times)
        assert mock_sleep.call_count == max_retries

    @pytest.mark.asyncio
    async def test_deliver_retries_on_exception(self):
        """Verify delivery retries when aiohttp raises an exception."""
        max_retries = 2
        delivery = WebhookDelivery(
            webhook_url="https://example.com/hook",
            secret="test-secret",
            max_retries=max_retries,
            initial_backoff=0.01,
            timeout=5,
        )
        payload = {"event": "build.started", "build_id": "ghi789"}

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=Exception("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "gbserver.webhooks.delivery.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await delivery.deliver(payload)

        assert result is False
        assert mock_session.post.call_count == 1 + max_retries

    @pytest.mark.asyncio
    async def test_deliver_succeeds_after_retry(self):
        """Verify delivery succeeds when server recovers after initial failure."""
        delivery = WebhookDelivery(
            webhook_url="https://example.com/hook",
            secret="test-secret",
            max_retries=3,
            initial_backoff=0.01,
            timeout=5,
        )
        payload = {"event": "build.completed", "build_id": "jkl012"}

        mock_response_fail = AsyncMock()
        mock_response_fail.status = 503
        mock_response_fail.__aenter__ = AsyncMock(return_value=mock_response_fail)
        mock_response_fail.__aexit__ = AsyncMock(return_value=False)

        mock_response_ok = AsyncMock()
        mock_response_ok.status = 200
        mock_response_ok.__aenter__ = AsyncMock(return_value=mock_response_ok)
        mock_response_ok.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            side_effect=[mock_response_fail, mock_response_fail, mock_response_ok]
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "gbserver.webhooks.delivery.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await delivery.deliver(payload)

        assert result is True
        assert mock_session.post.call_count == 3

    @pytest.mark.asyncio
    async def test_deliver_batch_size_header(self):
        """Verify X-GB-Batch-Size header reflects events list length."""
        delivery = WebhookDelivery(
            webhook_url="https://example.com/hook",
            secret="test-secret",
            max_retries=0,
            initial_backoff=0.01,
            timeout=5,
        )
        payload = {
            "events": [
                {"type": "build.started", "build_id": "a"},
                {"type": "build.completed", "build_id": "b"},
                {"type": "build.failed", "build_id": "c"},
            ]
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "gbserver.webhooks.delivery.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            result = await delivery.deliver(payload)

        assert result is True
        call_kwargs = mock_session.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["X-GB-Batch-Size"] == str(3)


class TestConstants:
    """Tests for module-level constants."""

    def test_default_max_retries(self):
        assert DEFAULT_MAX_RETRIES == 5

    def test_default_initial_backoff(self):
        assert DEFAULT_INITIAL_BACKOFF == 1.0

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 10
