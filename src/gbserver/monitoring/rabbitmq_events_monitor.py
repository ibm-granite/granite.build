"""Rabbitmq events monitor module."""

import argparse
import asyncio
import json
import signal
from typing import Any, Awaitable, Callable, Dict, Optional, Self

from gbserver.messaging.messaging_base import Address
from gbserver.monitoring.monitor_base import MonitorBase

try:
    from gbserver.messaging.rabbitmq_base import RabbitMQBase
except ImportError:
    RabbitMQBase = None  # type: ignore[assignment,misc]
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    EntityRunMetadata,
    EventPayload,
)
from gbserver.types.constants import GBSERVER_MONITORING_GRACE_PERIOD
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class EventDeduplicator:
    """In-memory deduplicator that keeps all seen event ids for the life of the process"""

    def __init__(self):
        self._seen = set()
        self._lock = asyncio.Lock()

    async def add_if_new(self, event_id: Optional[str]) -> bool:
        """
        Returns True if event_id was not seen before (and records it).
        Returns False if it's a duplicate.
        If event_id is None/empty, returns True (treat as processable).
        """
        if not event_id:
            return True
        async with self._lock:
            if event_id in self._seen:
                return False
            self._seen.add(event_id)
            return True


class RabbitMQEventMonitor(MonitorBase):
    """Rabbit M Q Event Monitor implementation."""

    def __init__(
        self: Self,
        messaging_config: Dict[str, Any],
        messaging_secret: Any,
        launch_id: str,
        event_configs: Optional[str] = None,
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_queue: Optional[asyncio.Queue] = None,
        stop_event: Optional[asyncio.Event] = None,
    ) -> None:
        logger.info(
            "constructing RabbitMQEventMonitor launch_id %s entityrun_metadata: %s",
            launch_id,
            entityrun_metadata,
        )
        super().__init__(
            launch_id=launch_id,
            entityrun_metadata=entityrun_metadata,
            event_queue=event_queue,
            stop_event=stop_event,
        )

        exchange_name = messaging_config.get("exchange", "build-exchange")
        if entityrun_metadata is not None:
            stream_name = entityrun_metadata.build_id
            routing_key = ".".join(
                [
                    entityrun_metadata.targetrun_id,
                    entityrun_metadata.targetsteprun_id,
                    launch_id,
                ]
            )
        else:
            stream_name = launch_id
            routing_key = None

        self.msg = RabbitMQBase.from_env_and_args(
            exchange_name=exchange_name,
            queue_name=stream_name,
            routing_key=routing_key,
            messaging_secret=messaging_secret,
            stop_evt=stop_event,
        )

        self.routing_prefix = self.msg.addr.rk()
        self.event_configs = event_configs
        self.last_processed_msg: int = 0
        # private flag so we only attach consumer once
        self._consumer_attached = False
        self._dedup: EventDeduplicator = EventDeduplicator()
        self._cancel_consumer: Optional[Callable[[], Awaitable[None]]] = None

    async def _consume_events(self):
        """Logs every non-config, routing prefix matching message arriving on the stream queue."""
        logger.info(
            "launch_id %s _consume_events called with routing_prefix: %s",
            self.launch_id,
            self.routing_prefix,
        )

        async def _handler(body: bytes, rk: str, delivery_tag: int) -> None:
            # use the message delivery_tag to update last_processed_msg variable or
            # skip the message if it was already handled and is a replay
            # payload = json.loads(body.decode())
            # build_event_type = BuildEventType(payload.get("type", "").lower())
            # logger.info("%s before skips! launch_id %s last %s tag %s rk %s type %s payload %s", ">"*20, self.launch_id, self.last_processed_msg, delivery_tag, rk, build_event_type, payload)
            if self.last_processed_msg >= delivery_tag:
                logger.warning(
                    "[Monitor]: Skipping message with delivery tag %d because last processed message is %d",
                    delivery_tag,
                    self.last_processed_msg,
                )
                return
            self.last_processed_msg = delivery_tag
            # filter out the events that do not match launch monitor's
            # build_id,target_id.step_id.launch_id routing prefix
            if not rk.startswith(self.routing_prefix):
                logger.debug(
                    "launch_id %s Skipping message %d with routing key %s",
                    self.launch_id,
                    delivery_tag,
                    rk,
                )
                logger.debug(
                    "launch_id %s     delivery_tag %d   routing_prefix %s",
                    self.launch_id,
                    delivery_tag,
                    self.routing_prefix,
                )
                return
            # filter out config messages
            if rk.endswith(".config"):
                logger.debug(
                    "launch_id %s Skipping configuration message with routing key %s",
                    self.launch_id,
                    rk,
                )
                return

            payload = json.loads(body.decode())
            build_event_type = BuildEventType(payload.get("type", "").lower())
            logger.info("[Monitor] Event %s via %s: %s", delivery_tag, rk, payload)

            build_event_id = payload.get("event_id")
            is_new = await self._dedup.add_if_new(build_event_id)
            if not is_new:
                logger.warning("[Monitor] duplicate ignored event_id=%s", build_event_id)
                return

            # build_event_type = BuildEventType(payload.get("type", "").lower())
            # logger.info("%s after skips! launch_id %s last %s tag %s rk %s type %s payload %s", ">"*20, self.launch_id, self.last_processed_msg, delivery_tag, rk, build_event_type, payload)
            event = BuildEvent(
                run_metadata=self.entityrun_metadata,
                type=build_event_type,
                payload=EventPayload.payload_parser(
                    event_type=build_event_type,
                    data=payload.get("data"),
                ),
            )
            logger.info("[Monitor] Built event %s", event)
            if self.event_queue is not None:
                await self.event_queue.put(event)
            # return event

        if not self._consumer_attached:
            self._cancel_consumer = await self.msg.consume_stream(_handler)
            self._consumer_attached = True

    async def monitor(self):
        # setup the messenger
        logger.info("%s: starting the monitor()", self.launch_id)
        await self.msg.setup()
        logger.info("%s: setup completed", self.launch_id)
        # ensure the queue is present and bound **before** publishing
        await self.msg.declare_event_queue()
        # start the consumer using RabbitMQBase.consume_stream(), keep its cancel handle
        logger.info("%s: before consume_events", self.launch_id)
        await self._consume_events()
        logger.info("%s: after consume_events", self.launch_id)
        # block until the stop_event is set by somebody
        try:
            await self.stop_event.wait()
            logger.info(
                "launch_id %s stop event has been set! exitting monitoring",
                self.launch_id,
            )
        finally:
            # graceful shutdown: cancel the consumer
            try:
                logger.info(
                    "launch_id %s before sleeping for %d seconds",
                    self.launch_id,
                    GBSERVER_MONITORING_GRACE_PERIOD,
                )
                await asyncio.sleep(GBSERVER_MONITORING_GRACE_PERIOD)
                logger.info(
                    "launch_id %s after sleeping for %d seconds",
                    self.launch_id,
                    GBSERVER_MONITORING_GRACE_PERIOD,
                )
                await self._cancel_consumer()
                logger.info("launch_id %s cancelled consumer", self.launch_id)
            except Exception as e:
                # in case it is already stopped
                logger.info("launch_id %s failed to cancel the consumer: %s", self.launch_id, e)
        await self.msg.close()
        logger.info("launch_id %s messenger closed", self.launch_id)


async def main():
    """Main."""
    import os

    import yaml

    from gbserver.messaging.rabbitmq_base import RabbitMQBase
    from gbserver.monitoring.dummy_monitor import DummyMonitor

    default_events_config_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../test-data/gbserver_test/monitoring/event_configs_digit.yaml",
        )
    )

    parser = argparse.ArgumentParser("Monitor with pluggable messaging and process monitoring")
    parser.add_argument("--exchange", default="build_events")
    parser.add_argument("--stream", required=True)
    parser.add_argument("--routing-key", help="routing key")
    parser.add_argument(
        "--events-config-file",
        default=default_events_config_file,
        help="yaml file with events config",
    )
    parser.add_argument("--termination-delay", type=int, default=30)
    args = parser.parse_args()

    # Inject RabbitMQ implementation
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop_event.set)  # Ctrl-C means stop
    addr: Address = Address(exchange=args.exchange, queue=args.stream, routing_key=args.routing_key)
    messenger = RabbitMQBase(addr=addr, stop_event=stop_event)
    await messenger.setup()

    with open(args.events_config_file) as stream:
        try:
            event_configs = yaml.safe_load(stream)
        except yaml.YAMLError as ex:
            logger.error(ex)

    # Create RabbitMQ-based launch monitor
    event_queue: asyncio.Queue = asyncio.Queue()
    msg_config = {
        "exchange": args.exchange,
    }

    launch_id = "dummy-launch_id"
    if args.routing_key is not None:
        entityrun_metadata = EntityRunMetadata(build_id=args.stream)
        tokens = args.routing_key.split(".")
        if len(tokens) == 3:
            entityrun_metadata.targetrun_id = tokens[0]
            entityrun_metadata.targetsteprun_id = tokens[1]
            launch_id = tokens[2]
        else:
            entityrun_metadata = None
            launch_id = args.stream
    else:
        entityrun_metadata = None
        launch_id = args.stream

    rmq_monitor = RabbitMQEventMonitor(
        messaging_config=msg_config,
        messaging_secret=None,
        launch_id=launch_id,
        event_configs=json.dumps(event_configs),
        entityrun_metadata=None,
        event_queue=event_queue,
        stop_event=stop_event,
    )
    dummy_monitor: DummyMonitor = DummyMonitor(
        delay_sec=args.termination_delay,
        launch_id=launch_id,
        entityrun_metadata=None,
        event_queue=event_queue,
        stop_event=stop_event,
    )
    await asyncio.gather(
        dummy_monitor.monitor(),
        rmq_monitor.monitor(),
    )


if __name__ == "__main__":
    asyncio.run(main())
