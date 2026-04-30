"""NATS-based messaging backend for standalone and multi-node deployments.

Uses nats-py to connect to a nats-server instance. Provides sub-millisecond
pub/sub messaging that works from single-machine to multi-node with the same
client code.

Supports two modes:
- Lightweight (core pub/sub) — when JetStream is not available on the server
- JetStream (persistent) — auto-detected; durable streams, ack/nack, replay

Dependencies: nats-py (pip install nats-py)
"""

import asyncio
import json
import logging
from typing import Awaitable, Callable

from gbserver.utils.optional_imports import HAS_NATS

if HAS_NATS:
    import nats
    from nats.aio.client import Client as NATSClient
    from nats.js.api import (
        AckPolicy,
        ConsumerConfig,
        RetentionPolicy,
        StorageType,
        StreamConfig,
    )

from gbserver.messaging.messaging_base import JSON, Address, MessagingBase

logger = logging.getLogger(__name__)


class NATSMessaging(MessagingBase):
    """NATS-based messaging using nats-py.

    Connects to a nats-server instance for pub/sub messaging.
    Auto-detects JetStream availability on setup — if JetStream is enabled,
    messages are persisted to durable streams with ack/nack semantics.
    Falls back to lightweight pub/sub when JetStream is not available.
    """

    def __init__(
        self,
        addr: Address,
        nats_url: str = "nats://localhost:4222",
        stream_max_age: int = 604800,
        max_deliver: int = 5,
        ack_wait: int = 30,
    ):
        if not HAS_NATS:
            raise ImportError(
                "The 'nats-py' library is required for NATS messaging. "
                "Install it with: pip install nats-py"
            )
        super().__init__(addr)
        self._nats_url = nats_url
        self._stream_max_age = stream_max_age
        self._max_deliver = max_deliver
        self._ack_wait = ack_wait
        self._nc: "NATSClient | None" = None
        self._js = None
        self._jetstream_available = False
        self._sub = None
        self._closed = False
        self._stop_event: asyncio.Event | None = None

    async def setup(self) -> None:
        """Connect to the NATS server and auto-detect JetStream."""
        self._nc = await nats.connect(self._nats_url)
        logger.info(
            "NATSMessaging connected: queue='%s', server='%s'",
            self.addr.queue,
            self._nats_url,
        )

        # Auto-detect JetStream
        try:
            self._js = self._nc.jetstream()  # type: ignore[assignment]
            await self._js.account_info()  # type: ignore[attr-defined]
            self._jetstream_available = True
            logger.info("JetStream available, using persistent mode")
            await self._ensure_stream()
        except Exception:
            self._js = None
            self._jetstream_available = False
            logger.info("JetStream unavailable, using lightweight pub/sub")
            logger.debug("JetStream probe error:", exc_info=True)

    async def _ensure_stream(self) -> None:
        """Create or update the JetStream stream for this queue."""
        stream_name = f"GBSERVER_{self.addr.queue}".upper().replace(".", "_")
        config = StreamConfig(
            name=stream_name,
            subjects=[f"gbserver.{self.addr.queue}.>"],
            retention=RetentionPolicy.LIMITS,
            max_age=self._stream_max_age * 1_000_000_000,  # seconds to nanoseconds
            storage=StorageType.FILE,
            num_replicas=1,
        )
        await self._js.add_stream(config=config)  # type: ignore[attr-defined]
        logger.info(
            "JetStream stream '%s' ready (max_age=%ds)",
            stream_name,
            self._stream_max_age,
        )

    async def publish(self, payload: JSON, suffix: str) -> None:
        """Publish a JSON message. Uses JetStream if available, else core NATS."""
        if self._nc is None or self._nc.is_closed:
            raise RuntimeError("NATSMessaging not connected. Call setup() first.")
        body = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload
        subject = f"gbserver.{self.addr.queue}"
        if suffix:
            subject = f"{subject}.{suffix}"

        if self._jetstream_available:
            ack = await self._js.publish(subject, body)  # type: ignore[attr-defined]
            logger.info(
                "JetStream publish: subject='%s', stream='%s', seq=%d",
                subject,
                ack.stream,
                ack.seq,
            )
        else:
            await self._nc.publish(subject, body)

    async def consume_stream(self, handler: Callable[[bytes, str], Awaitable[None]]) -> None:
        """Subscribe and consume messages. Uses JetStream durable consumer if available."""
        if self._nc is None or self._nc.is_closed:
            raise RuntimeError("NATSMessaging not connected. Call setup() first.")

        subject = f"gbserver.{self.addr.queue}.>"

        if self._jetstream_available:
            consumer_name = f"gbserver_{self.addr.queue}_consumer".replace(".", "_")
            self._sub = await self._js.subscribe(  # type: ignore[attr-defined]
                subject,
                durable=consumer_name,
                manual_ack=True,
                config=ConsumerConfig(
                    ack_policy=AckPolicy.EXPLICIT,
                    max_deliver=self._max_deliver,
                    ack_wait=self._ack_wait * 1_000_000_000,  # seconds to nanoseconds
                ),
            )
            logger.info(
                "JetStream consuming: subject='%s', consumer='%s'",
                subject,
                consumer_name,
            )
            try:
                async for msg in self._sub.messages:  # type: ignore[attr-defined]
                    if self._closed:
                        break
                    routing_key = msg.subject.removeprefix(f"gbserver.{self.addr.queue}.")
                    try:
                        await handler(msg.data, routing_key)
                        await msg.ack()
                    except Exception:
                        logger.warning(
                            "Handler failed for subject '%s', nak-ing message",
                            msg.subject,
                            exc_info=True,
                        )
                        await msg.nak()
            except asyncio.CancelledError:
                pass
        else:
            # Lightweight mode — plain NATS subscribe, no ack/nack
            self._sub = await self._nc.subscribe(subject)  # type: ignore[assignment]
            logger.info("NATSMessaging consuming from subject '%s'", subject)
            try:
                async for msg in self._sub.messages:  # type: ignore[attr-defined]
                    if self._closed:
                        break
                    routing_key = msg.subject.removeprefix(f"gbserver.{self.addr.queue}.")
                    await handler(msg.data, routing_key)
            except asyncio.CancelledError:
                pass

    async def run(self) -> None:
        """Block until stop_event is set (JetStream mode) or return immediately (lightweight)."""
        if self._jetstream_available:
            self._stop_event = asyncio.Event()
            await self._stop_event.wait()

    async def close(self) -> None:
        """Close the NATS connection."""
        self._closed = True
        if self._stop_event is not None:
            self._stop_event.set()
        if self._sub is not None:
            try:
                await self._sub.unsubscribe()
            except Exception:
                logger.debug("Error unsubscribing during close", exc_info=True)
            self._sub = None
        if self._nc is not None and not self._nc.is_closed:
            await self._nc.close()
            self._nc = None
        logger.info("NATSMessaging closed for queue '%s'", self.addr.queue)
