#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The base class for the RabbitMQ messenger.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import ssl
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Self

from gbserver.utils.optional_imports import HAS_RABBITMQ

if HAS_RABBITMQ:
    import aio_pika
    from aio_pika.exceptions import (
        AMQPConnectionError,
    )
    from aio_pika.exceptions import ChannelInvalidStateError as AioPikaChannelInvalidStateError
    from aio_pika.exceptions import ConnectionClosed as AioPikaConnectionClosed
    from aiormq.exceptions import ChannelInvalidStateError as AiormqChannelInvalidStateError
    from aiormq.exceptions import ConnectionClosed as AiormqConnectionClosed
else:
    # Provide stand-in exception classes so retry decorators and except
    # clauses can reference them at class-definition time without crashing.
    class AMQPConnectionError(Exception):  # type: ignore[no-redef]
        """A M Q P Connection Error implementation."""

        pass

    class AioPikaChannelInvalidStateError(Exception):  # type: ignore[no-redef]
        """Aio Pika Channel Invalid State Error implementation."""

        pass

    class AioPikaConnectionClosed(Exception):  # type: ignore[no-redef]
        """Aio Pika Connection Closed implementation."""

        pass

    class AiormqChannelInvalidStateError(Exception):  # type: ignore[no-redef]
        """Aiormq Channel Invalid State Error implementation."""

        pass

    class AiormqConnectionClosed(Exception):  # type: ignore[no-redef]
        """Aiormq Connection Closed implementation."""

        pass


from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gbserver.messaging.messaging_base import JSON, Address, MessagingBase
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RabbitSettings:
    """
    Helper class used for connection configuration in RabbitMQ
    """

    uri: str = os.getenv("RABBITMQ_URI", "amqps")
    host: str = os.getenv("RABBITMQ_HOST", "localhost")
    port: int = int(os.getenv("RABBITMQ_PORT", 5672))
    user: str = os.getenv("RABBITMQ_USERNAME", "guest")
    password: str = os.getenv("RABBITMQ_PASSWORD", "guest")
    vhost: str = "/"
    # if True, open new channel, passive declare exchange, publish, close channel
    # if False, reuse one robust channel/exchange with keepalive/refresh logic
    per_publish_channel: bool = False

    @property
    def url(self) -> str:
        """Url."""
        return f"{self.uri}://{self.user}:{self.password}@{self.host}:{self.port}{self.vhost}"

    # This property returns the SSL context for the following two RabbitMQ servers:
    #   (1) an IBM Cloud service that uses SSL, but has a self-signed certificate
    #   (2) a Docker container running locally that has no SSL
    @property
    def ssl_context(self) -> Optional[ssl.SSLContext]:
        """Ssl context."""
        ctx: ssl.SSLContext | None = None
        if self.uri == "amqps":
            # --- build an "insecure" SSLContext -------------
            ctx = ssl.create_default_context()
            ctx.check_hostname = False  # skip CN / SAN hostname match
            ctx.verify_mode = ssl.CERT_NONE  # accept any certificate
        return ctx


# -------- RabbitMQBase -------------------------
class RabbitMQBase(MessagingBase):
    """
    aio_pika-based implementation for RabbitMQBase
    """

    def __init__(
        self: Self,
        addr: Address,
        stop_event: asyncio.Event | None = None,
        settings: RabbitSettings = RabbitSettings(),
    ) -> None:
        """
        Parameters
        ----------
        addr: Address - canonical id for a logical channel with the following fields:
            * exchange: str - RabbitMQ Topic Exchange name.
            * queue: str - Stream/queue name (this is the build-id).
            * routing_key: str | None - used in Address.key() to build the routing prefix for all events from a particular launch-id
              format: target-id.step-id.launch-id
        stop_event: asyncio.Event() - set this event to stop the RabbitMQ consumer instance
        settings: RabbitSettings - helper class that automatically instantiates, from environment variables, the connection parameters
        """
        if not HAS_RABBITMQ:
            raise ImportError(
                "aio-pika required for RabbitMQ messaging. "
                "Install with: pip install gbserver[rabbitmq]"
            )
        logger.info("addr: %s", addr)
        self.launch_id = "unset"
        if addr.routing_key:
            self.launch_id = addr.routing_key.split(".")[-1]
        logger.info("self.launch_id: %s", self.launch_id)
        super().__init__(addr)
        self.settings = settings
        self.stop_event: asyncio.Event = stop_event or asyncio.Event()

        # aio-pika handles
        self._conn: Optional[aio_pika.RobustConnection] = None
        self._chan: Optional[aio_pika.abc.AbstractChannel] = None
        self._exchange: Optional[aio_pika.abc.AbstractExchange] = None
        self._setup_done: asyncio.Event = asyncio.Event()

    @classmethod
    def from_env_and_args(
        cls,
        exchange_name: str,
        queue_name: str,
        routing_key: Optional[str] = None,
        messaging_secret: Optional[Any] = None,
        stop_evt: Optional[asyncio.Event] = None,
    ):
        """From env and args."""
        addr = Address(exchange=exchange_name, queue=queue_name, routing_key=routing_key)
        if messaging_secret:
            rabbit_settings = RabbitSettings(
                host=messaging_secret.get("host"),
                port=int(messaging_secret.get("port")),
                user=messaging_secret.get("username"),
                password=messaging_secret.get("password"),
            )
            return cls(addr=addr, stop_event=stop_evt, settings=rabbit_settings)
        else:
            return cls(addr=addr, stop_event=stop_evt)

    # ------------------------ connection setup -----------------------------
    @retry(
        retry=retry_if_exception_type(
            (
                AMQPConnectionError,
                AioPikaConnectionClosed,
                asyncio.TimeoutError,
                StopAsyncIteration,
                ssl.SSLError,
            )
        ),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _connect(self) -> aio_pika.RobustConnection:
        logger.info(
            f"[RabbitMQ launch_id {self.launch_id}] Connecting to {self.settings.uri}://{self.settings.host}:{self.settings.port}"
        )
        return await aio_pika.connect_robust(
            url=self.settings.url,
            timeout=15,  # gives broker time to finish TLS handshake
            ssl_context=self.settings.ssl_context,
        )

    # FIX: ensure the underlying connection exists and is open
    async def _ensure_connection(self):
        if self._conn is None or getattr(self._conn, "is_closed", True):
            self._conn = await self._connect()

    # ------------------------- async setup -------------------------------- #
    async def setup(self):
        """Initialize the RabbitMQ connection and channel."""
        if self._setup_done.is_set():  # already initialized
            return
        # only the TCP/TLS + auth handshake is retried here
        self._conn = await self._connect()  # tenacity-wrapped
        # create channel + exchange via single helper
        await self._setup_channel_exchange()
        self._setup_done.set()
        logger.info(f"[RabbitMQ launch_id {self.launch_id}] setup finished")

    # ------------ private helper: create channel + exchange ------------
    async def _setup_channel_exchange(self):
        await self._ensure_connection()
        self._chan = await self._conn.channel(publisher_confirms=True)
        if not self._chan.is_initialized:
            await self._chan.initialize()
        await self._chan.set_qos(prefetch_count=1)

        # Exchange (durable topic); use default exchange if addr.exchange is None
        if self.addr.exchange:
            self._exchange = await self._chan.declare_exchange(
                self.addr.exchange,
                type=aio_pika.ExchangeType.TOPIC,
                durable=True,  # survive broker restart
                auto_delete=False,  # survive periods with zero bindings
                passive=False,  # create the exchange on first run
            )
        else:
            self._exchange = self._chan.default_exchange

    # ------------------------- channel guard -------------------------- #
    async def _ensure_channel(self):
        """
        Re-create channel & exchange after reconnect or if transport vanished.
        """
        await self._ensure_connection()

        # Hard check for closed or missing transport (robust channel may not flag .is_closed during reconnect race)
        needs_new = (
            self._chan is None
            or getattr(self._chan, "is_closed", True)
            or getattr(self._chan, "_channel", None) is None  # transport missing
        )

        if needs_new:
            await self._setup_channel_exchange()
            logger.info(
                "[RabbitMQ launch_id %s] channel and %s exchange ready",
                self.launch_id,
                self.addr.exchange if self.addr.exchange else "default",
            )
            return

        # Also refresh the exchange handle if necessary (passive declare = health check)
        if self.addr.exchange:
            try:
                exch = await self._chan.declare_exchange(
                    self.addr.exchange,
                    type=aio_pika.ExchangeType.TOPIC,
                    durable=True,
                    auto_delete=False,
                    passive=True,
                )
                self._exchange = exch  # refresh handle in case it was invalidated
            except Exception as e:  # rebuild on *any* hiccup here
                logger.warning(
                    "[RabbitMQ launch_id %s] passive declare failed (%s); rebuilding channel/exchange",
                    self.launch_id,
                    type(e).__name__,
                )
                await self._setup_channel_exchange()
                logger.info(
                    "[RabbitMQ launch_id %s] exchange refreshed after passive declare",
                    self.launch_id,
                )

    # ------ retry wrapper for publish / consume, invokes channel guard --------- #
    def _with_retries(self, coro_factory: Callable[..., Awaitable[Any]]):
        """
        Retry wrapper for publish / consume calls. Retry on both aio-pika and aiormq
        channel/connection errors plus common transient transport issues.
        """

        @retry(
            retry=retry_if_exception_type(
                (
                    AioPikaChannelInvalidStateError,
                    AiormqChannelInvalidStateError,
                    AioPikaConnectionClosed,
                    AiormqConnectionClosed,
                    AMQPConnectionError,
                    asyncio.TimeoutError,
                    ConnectionResetError,
                    BrokenPipeError,
                    RuntimeError,  # in the tests, aiormq sometimes raises RuntimeError("... closed")
                    asyncio.CancelledError,  # cancellation from aiormq tasks during close
                )
            ),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
            stop=stop_after_attempt(10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def wrapper(*a, **kw):
            await self._ensure_channel()
            return await coro_factory(*a, **kw)

        return wrapper

    async def declare_event_queue(self):
        """Declare event queue."""
        await self._setup_done.wait()
        await self._ensure_channel()
        queue = await self._chan.declare_queue(
            name=self.addr.queue,
            durable=True,
            arguments={
                "x-queue-type": "stream",
            },
        )
        # Bind queue -> exchange
        bind_key = f"{self.addr.queue}.#"
        await queue.bind(self._exchange, routing_key=bind_key)

    # ------------------------- MessagingBase API ------------------------- #
    # ------------------------- publish helper ---------------------------- #
    async def publish(
        self,
        payload: JSON,
        suffix: str = "event",
        persistent: bool = True,
    ) -> None:
        """Publish a message to the RabbitMQ exchange."""
        await asyncio.wait_for(self._setup_done.wait(), timeout=30)  # wait for setup to complete

        body = json.dumps(payload).encode()
        rk = self.addr.rk(suffix)
        message = aio_pika.Message(
            body,
            delivery_mode=(
                aio_pika.DeliveryMode.PERSISTENT
                if persistent
                else aio_pika.DeliveryMode.NOT_PERSISTENT
            ),
        )
        logger.info(
            f"[RabbitMQ launch_id {self.launch_id}] Publishing message with routing key {rk}"
        )

        if self.settings.per_publish_channel:
            # fresh channel for this publish only
            await self._ensure_connection()
            ch = await self._conn.channel(publisher_confirms=True)  # type: ignore[union-attr]
            try:
                exch = (
                    (
                        await ch.declare_exchange(
                            self.addr.exchange,
                            type=aio_pika.ExchangeType.TOPIC,
                            durable=True,
                            auto_delete=False,
                            passive=True,
                        )
                    )
                    if self.addr.exchange
                    else ch.default_exchange
                )
                await self._with_retries(lambda *a, **kw: exch.publish(*a, **kw))(
                    message, routing_key=rk
                )
            finally:
                with contextlib.suppress(Exception):
                    await ch.close()
        else:
            # Use a lambda to resolve the *current* exchange each attempt, so retries don't call a
            # stale bound method from a dead channel.
            publish_call = self._with_retries(lambda *a, **kw: self._exchange.publish(*a, **kw))  # type: ignore[union-attr]
            await publish_call(message, routing_key=rk)

        logger.info(
            f"[RabbitMQ launch_id {self.launch_id}] Published message {json.dumps(payload, indent=2)}\nwith routing key {rk}"
        )

    # ------------------------- consume helper ---------------------------- #
    async def consume_stream(  # type: ignore[override]
        self,
        handler: Callable[[bytes, str, int], Awaitable[None]],
        *,
        consumer_name: Optional[str] = None,
        stream_offset: str | int = "first",
        no_ack: bool = False,
    ):
        """
        Declare queue, bind queue to exchange, and start consuming events
        For resume semantics:
          - pass a stable `consumer_name` (e.g., sidecar/monitor identity)
          - use `stream_offset="next"` to continue after the last committed record
        """
        await asyncio.wait_for(self._setup_done.wait(), timeout=30)  # wait for setup to complete
        queue_holder = {"queue": None}
        tag_holder = {"tag": None}

        async def _start_consume():
            # Declare STREAM queue
            queue = await self._chan.declare_queue(
                name=self.addr.queue,
                durable=True,
                arguments={
                    "x-queue-type": "stream",
                },
            )
            logger.info(
                "[RabbitMQ launch_id %s] %s stream queue declared",
                self.launch_id,
                self.addr.queue,
            )
            queue_holder["queue"] = queue

            # Bind queue -> exchange
            bind_key = f"{self.addr.queue}.#"
            await queue.bind(self._exchange, routing_key=bind_key)
            logger.info(
                "[RabbitMQ launch_id %s] %s stream queue bound to %s",
                self.launch_id,
                self.addr.queue,
                bind_key,
            )

            args = {"x-stream-offset": stream_offset}
            if consumer_name:
                args["name"] = consumer_name  # enables broker-tracked offsets

            async def _inner(msg: aio_pika.IncomingMessage):
                async with msg.process():
                    await handler(msg.body, msg.routing_key, msg.delivery_tag)

            tag = await queue.consume(callback=_inner, arguments=args, no_ack=no_ack)
            tag_holder["tag"] = tag

        await self._with_retries(_start_consume)()
        logger.info(
            "[RabbitMQ launch_id %s] consuming stream %s (offset=%s, name=%s)",
            self.launch_id,
            self.addr.queue,
            str(stream_offset),
            consumer_name or "-",
        )

        async def cancel():
            # cancel only this consumer; safe to call multiple times
            logger.info(
                "[RabbitMQ launch_id %s] starting cancel() method",
                self.launch_id,
            )
            q: aio_pika.abc.AbstractQueue = queue_holder.get("queue")
            logger.info(
                "[RabbitMQ launch_id %s] in cancel() method, q = %s, type(q) = %s",
                self.launch_id,
                q,
                type(q).__name__,
            )
            tag = tag_holder.get("tag")
            logger.info(
                "[RabbitMQ launch_id %s] in cancel() method, tag = %s",
                self.launch_id,
                tag,
            )
            if q and tag:
                try:
                    logger.info(
                        "[RabbitMQ launch_id %s] in cancel() method, trying to cancel q with tag %s",
                        self.launch_id,
                        tag,
                    )
                    assert isinstance(
                        q, aio_pika.abc.AbstractQueue
                    ), f"Invalid queue type {type(q).__name__}, expected AbstractQueue"
                    logger.info("queue_closed = %s", q.channel.is_closed)
                    if not q.channel.is_closed:
                        await q.cancel(consumer_tag=tag, nowait=True, timeout=5)
                        logger.info(
                            "[RabbitMQ launch_id %s] consumer %s cancelled from stream %s",
                            self.launch_id,
                            tag,
                            q.name,
                        )
                    else:
                        logger.warning(
                            "[RabbitMQ launch_id %s] will not cancel consumer %s from stream %s because channel already closed",
                            self.launch_id,
                            tag,
                            q.name,
                        )
                except Exception as e:
                    logger.warning(
                        "[RabbitMQ launch_id %s] failed to cancel consumer %s from stream %s (%s)",
                        self.launch_id,
                        tag,
                        q.name,
                        e,
                    )
            else:
                logger.warning(
                    "[RabbitMQ launch_id %s] in cancel() method, invalid q %s or tag %s",
                    self.launch_id,
                    q,
                    tag,
                )

        return cancel

    # ------------------------- run / close ------------------------------- #
    async def run(self):
        """Run the messaging loop until stop is signaled."""
        await self._setup_done.wait()
        await self.stop_event.wait()

    async def close(self):
        """Close the RabbitMQ connection."""
        if self._conn and not self._conn.is_closed:
            await self._conn.close()
            logger.info("[RabbitMQ launch_id %s] connection closed", self.launch_id)
        logger.info("[RabbitMQ launch_id %s] close() completed", self.launch_id)
