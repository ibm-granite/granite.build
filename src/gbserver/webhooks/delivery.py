"""Webhook delivery with HMAC-SHA256 signing and exponential backoff retry.

This module provides the WebhookDelivery class which handles sending webhook
payloads to subscriber endpoints, signing them with HMAC-SHA256 for authenticity
verification, and retrying on failure with exponential backoff.
"""

import asyncio
import hashlib
import hmac
import json
import uuid
from typing import Any, Dict

import aiohttp

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_TIMEOUT = 10


def sign_payload(payload_bytes: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature for a webhook payload.

    Args:
        payload_bytes: The raw payload bytes to sign.
        secret: The shared secret used for HMAC computation.

    Returns:
        A string in the format "sha256=<hex_digest>".
    """
    digest = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


class WebhookDelivery:  # pylint: disable=too-few-public-methods
    """Delivers a batched webhook payload with HMAC signing and retry.

    Args:
        webhook_url: The target URL to POST the webhook payload to.
        secret: The shared secret for HMAC-SHA256 signing.
        max_retries: Maximum number of retry attempts after initial failure.
        initial_backoff: Initial backoff duration in seconds (doubles each retry).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        webhook_url: str,
        secret: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.webhook_url = webhook_url
        self.secret = secret
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.timeout = timeout

    async def deliver(self, payload: Dict[str, Any]) -> bool:
        """Deliver payload with retry. Returns True on success (2xx).

        Serializes the payload to JSON, signs it with HMAC-SHA256, and POSTs
        it to the webhook URL. Retries with exponential backoff on failure.

        Args:
            payload: The dictionary payload to deliver as JSON.

        Returns:
            True if delivery succeeded (2xx response), False if all attempts exhausted.
        """
        payload_bytes = json.dumps(payload).encode("utf-8")
        signature = sign_payload(payload_bytes, self.secret)
        delivery_id = str(uuid.uuid4())

        # Determine batch size from events list if present
        events = payload.get("events")
        batch_size = len(events) if isinstance(events, list) else 1

        headers = {
            "Content-Type": "application/json",
            "X-GB-Delivery": delivery_id,
            "X-GB-Signature-256": signature,
            "X-GB-Batch-Size": str(batch_size),
        }

        total_attempts = 1 + self.max_retries
        for attempt in range(total_attempts):
            success = await self._attempt(payload_bytes, headers)
            if success:
                logger.info(
                    "Webhook delivered successfully to %s (delivery_id=%s, attempt=%d)",
                    self.webhook_url,
                    delivery_id,
                    attempt + 1,
                )
                return True

            if attempt < total_attempts - 1:
                backoff = self.initial_backoff * (2**attempt)
                logger.warning(
                    "Webhook delivery failed to %s (delivery_id=%s, attempt=%d/%d), "
                    "retrying in %.2fs",
                    self.webhook_url,
                    delivery_id,
                    attempt + 1,
                    total_attempts,
                    backoff,
                )
                await asyncio.sleep(backoff)

        logger.error(
            "Webhook delivery exhausted all %d attempts to %s (delivery_id=%s)",
            total_attempts,
            self.webhook_url,
            delivery_id,
        )
        return False

    async def _attempt(self, payload_bytes: bytes, headers: Dict[str, str]) -> bool:
        """Single POST attempt. Returns True on 2xx.

        Args:
            payload_bytes: The serialized JSON payload bytes.
            headers: The HTTP headers to include in the request.

        Returns:
            True if the response status code is 2xx, False otherwise.
        """
        try:
            client_timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(
                    self.webhook_url, data=payload_bytes, headers=headers
                ) as response:
                    return 200 <= response.status < 300
        except Exception as exc:
            logger.debug(
                "Webhook POST to %s raised exception: %s", self.webhook_url, exc
            )
            return False
