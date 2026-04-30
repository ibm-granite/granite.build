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
Unit tests for RetryHandler.
"""

import asyncio
import json
from typing import Optional, Self
from unittest.mock import AsyncMock, call, patch

import pytest

from gbserver.resilience import (
    RetryHandler,
    RetryStrategy,
    UnhealthyInsufficientPodsRetryStrategy,
)
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    EntityRunMetadata,
    EventPayload,
)


class MockEnvironment:
    """Mock environment for testing RetryHandler."""

    def __init__(self: Self) -> None:
        self.retry_called = False
        self.retry_launch_id: Optional[str] = None
        self.retry_nodes_to_avoid: Optional[list] = None
        self.retry_call_count = 0

    async def retry_workload(
        self: Self,
        launch_id: str,
        nodes_to_avoid: Optional[list] = None,
        **kwargs,
    ) -> None:
        """Mock retry_workload method."""
        self.retry_called = True
        self.retry_launch_id = launch_id
        self.retry_nodes_to_avoid = nodes_to_avoid
        self.retry_call_count += 1


class AlwaysRetryStrategy(RetryStrategy):
    """Test strategy that always recommends retry."""

    def should_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        return True

    def extract_nodes_to_avoid(self: Self, event: BuildEvent) -> set:
        return {"test-node"}


class NeverRetryStrategy(RetryStrategy):
    """Test strategy that never recommends retry."""

    def should_retry(
        self: Self,
        event: BuildEvent,
    ) -> bool:
        return False


def create_test_event(msg: str = "test message") -> BuildEvent:
    """Create a simple test BuildEvent."""
    payload = EventPayload.payload_parser(
        event_type=BuildEventType.MESSAGE_EVENT,
        data={"msg": msg},
    )
    return BuildEvent(
        run_metadata=EntityRunMetadata(build_id="test-build-id"),
        type=BuildEventType.MESSAGE_EVENT,
        payload=payload,
    )


def create_unhealthy_event(node_name: str = "worker-node-1") -> BuildEvent:
    """Create an unhealthy event for integration testing."""
    events = [
        {
            "object_type": "AppWrapper",
            "object_name": "test-appwrapper",
            "reason": "Unhealthy",
            "message": "InsufficientPodsReady: 0/1 pods are ready",
        },
        {
            "object_type": "Pod",
            "object_name": "test-pod-1",
            "reason": "FailedMount",
            "message": "Unable to attach or mount volumes",
        },
    ]

    pod_placement = {"test-pod-1": node_name}
    data = {"events": events, "state": "Unhealthy", "pod_placement": pod_placement}
    msg = f"```json\n{json.dumps(data, indent=2)}\n```"

    payload = EventPayload.payload_parser(
        event_type=BuildEventType.MESSAGE_EVENT,
        data={"msg": msg},
    )
    return BuildEvent(
        run_metadata=EntityRunMetadata(build_id="test-build-id"),
        type=BuildEventType.MESSAGE_EVENT,
        payload=payload,
    )


class TestRetryHandler:
    """Tests for RetryHandler orchestration logic."""

    @pytest.mark.asyncio
    async def test_retry_handler_triggers_retry(self: Self) -> None:
        """Test that RetryHandler triggers retry when strategy recommends it."""
        downstream_queue = asyncio.Queue()
        env = MockEnvironment()
        strategy = AlwaysRetryStrategy()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
            max_retries=3,
            strategies=[strategy],
        )

        event = create_test_event()
        retry_triggered = await handler._evaluate_and_retry(event)

        assert retry_triggered is True
        assert env.retry_called is True
        assert env.retry_launch_id == "test-launch-123"
        assert "test-node" in env.retry_nodes_to_avoid

    @pytest.mark.asyncio
    async def test_retry_handler_forwards_non_retryable_event(self: Self) -> None:
        """Test that RetryHandler does not trigger retry when strategy doesn't recommend it."""
        downstream_queue = asyncio.Queue()
        env = MockEnvironment()
        strategy = NeverRetryStrategy()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
            max_retries=3,
            strategies=[strategy],
        )

        event = create_test_event()
        retry_triggered = await handler._evaluate_and_retry(event)

        assert retry_triggered is False
        assert env.retry_called is False

    @pytest.mark.asyncio
    async def test_retry_handler_respects_max_retries(self: Self) -> None:
        """Test that RetryHandler stops retrying after max_retries is reached."""
        downstream_queue = asyncio.Queue()
        env = MockEnvironment()
        strategy = AlwaysRetryStrategy()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
            max_retries=2,
            strategies=[strategy],
        )

        event = create_test_event()

        # First retry should succeed
        retry_1 = await handler._evaluate_and_retry(event)
        assert retry_1 is True
        assert handler.retry_count == 1

        # Second retry should succeed
        retry_2 = await handler._evaluate_and_retry(event)
        assert retry_2 is True
        assert handler.retry_count == 2

        # Third retry should fail (max_retries=2)
        retry_3 = await handler._evaluate_and_retry(event)
        assert retry_3 is False
        assert handler.retry_count == 2  # Should not increment

    @pytest.mark.asyncio
    async def test_retry_handler_accumulates_nodes_to_avoid(self: Self) -> None:
        """Test that RetryHandler accumulates nodes across multiple retries."""
        downstream_queue = asyncio.Queue()
        env = MockEnvironment()

        # Strategy that returns different nodes
        class NodeIncrementingStrategy(RetryStrategy):
            def __init__(self):
                self.call_count = 0

            def should_retry(self, event):
                return True

            def extract_nodes_to_avoid(self, event):
                self.call_count += 1
                return {f"node-{self.call_count}"}

        strategy = NodeIncrementingStrategy()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
            max_retries=3,
            strategies=[strategy],
        )

        # First retry
        event1 = create_test_event("event1")
        await handler._evaluate_and_retry(event1)
        assert "node-1" in handler.nodes_to_avoid

        # Second retry
        event2 = create_test_event("event2")
        await handler._evaluate_and_retry(event2)
        assert "node-1" in handler.nodes_to_avoid
        assert "node-2" in handler.nodes_to_avoid

        # Third retry
        event3 = create_test_event("event3")
        await handler._evaluate_and_retry(event3)
        assert len(handler.nodes_to_avoid) == 3
        assert env.retry_call_count == 3

    @pytest.mark.asyncio
    async def test_retry_handler_uses_default_strategy(self: Self) -> None:
        """Test that RetryHandler uses default strategies when none provided."""
        from gbserver.resilience.strategies import NCCLErrorRetryStrategy

        downstream_queue = asyncio.Queue()
        env = MockEnvironment()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
            max_retries=3,
            # No strategies parameter
        )

        # Default includes UnhealthyInsufficientPodsRetryStrategy and NCCLErrorRetryStrategy
        assert len(handler.strategies) == 2
        assert any(
            isinstance(s, UnhealthyInsufficientPodsRetryStrategy) for s in handler.strategies
        )
        assert any(isinstance(s, NCCLErrorRetryStrategy) for s in handler.strategies)

    @pytest.mark.asyncio
    async def test_retry_handler_empty_strategies_list_uses_default(self: Self) -> None:
        """Test that RetryHandler uses default strategies when given empty list."""
        from gbserver.resilience.strategies import NCCLErrorRetryStrategy

        downstream_queue = asyncio.Queue()
        env = MockEnvironment()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
            max_retries=3,
            strategies=[],  # Empty list should fall back to default
        )

        # Default includes UnhealthyInsufficientPodsRetryStrategy and NCCLErrorRetryStrategy
        assert len(handler.strategies) == 2
        assert any(
            isinstance(s, UnhealthyInsufficientPodsRetryStrategy) for s in handler.strategies
        )
        assert any(isinstance(s, NCCLErrorRetryStrategy) for s in handler.strategies)

    @pytest.mark.asyncio
    async def test_wrapper_queue_pattern(self: Self) -> None:
        """Test the wrapper queue pattern for event interception."""
        downstream_queue = asyncio.Queue()
        env = MockEnvironment()
        strategy = UnhealthyInsufficientPodsRetryStrategy(object_types=["AppWrapper"])

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
            max_retries=3,
            strategies=[strategy],
        )

        # Start event processor
        processor_task = asyncio.create_task(handler.process_events())

        # Get wrapper queue
        wrapper_queue = handler.get_wrapper_queue()

        # Put a retryable event (unhealthy)
        retry_event = create_unhealthy_event(node_name="worker-node-1")
        await wrapper_queue.put(retry_event)

        # Put a normal event (non-retryable)
        normal_event = create_test_event("normal status")
        await wrapper_queue.put(normal_event)

        # Wait for all events to be processed and forwarded to downstream queue.
        # _execute_retry emits a MESSAGE_EVENT before forwarding the retryable event,
        # so we expect 3 events total: message + retryable + normal.
        for _ in range(100):  # Max 1 second wait (100 * 0.01s)
            if downstream_queue.qsize() >= 3:
                break
            await asyncio.sleep(0.01)

        # Stop processor
        handler.stop()
        await processor_task

        # Verify retry was triggered
        assert env.retry_called is True
        assert env.retry_launch_id == "test-launch-123"
        assert "worker-node-1" in env.retry_nodes_to_avoid

        # Verify all three events were forwarded to downstream queue:
        # 1. MESSAGE_EVENT from _execute_retry, 2. retryable event, 3. normal event
        assert downstream_queue.qsize() == 3

        # Get the retry message event (first) — emitted by _execute_retry
        retry_message_event = await downstream_queue.get()
        assert retry_message_event.type == BuildEventType.MESSAGE_EVENT
        assert retry_message_event.payload is not None
        assert "Retrying workload" in retry_message_event.payload.msg
        assert "1/3" in retry_message_event.payload.msg

        # Get the retryable event (second) — forwarded with retry metadata
        retry_forwarded_event = await downstream_queue.get()
        assert retry_forwarded_event.payload is not None
        assert hasattr(retry_forwarded_event.payload, "data")
        assert retry_forwarded_event.payload.data is not None
        # Verify retry metadata was added
        assert retry_forwarded_event.payload.data["retry_triggered"] is True
        assert retry_forwarded_event.payload.data["retry_count"] == 1
        assert retry_forwarded_event.payload.data["max_retries"] == 3
        assert "worker-node-1" in retry_forwarded_event.payload.data["nodes_to_avoid"]

        # Get the normal event (third)
        normal_forwarded_event = await downstream_queue.get()
        # Verify normal event was forwarded without retry metadata
        if normal_forwarded_event.payload and hasattr(normal_forwarded_event.payload, "data"):
            assert (
                normal_forwarded_event.payload.data is None
                or "retry_triggered" not in normal_forwarded_event.payload.data
            )

    @pytest.mark.asyncio
    async def test_multiple_strategies_first_match_wins(self: Self) -> None:
        """Test that when multiple strategies match, the first one is used."""
        downstream_queue = asyncio.Queue()
        env = MockEnvironment()

        class Strategy1(RetryStrategy):
            def should_retry(self, event):
                return True

            def extract_nodes_to_avoid(self, event):
                return {"strategy1-node"}

        class Strategy2(RetryStrategy):
            def should_retry(self, event):
                return True

            def extract_nodes_to_avoid(self, event):
                return {"strategy2-node"}

        strategy1 = Strategy1()
        strategy2 = Strategy2()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
            max_retries=3,
            strategies=[strategy1, strategy2],
        )

        event = create_test_event()
        await handler._evaluate_and_retry(event)

        # Should use first strategy
        assert "strategy1-node" in handler.nodes_to_avoid
        # Second strategy should not be evaluated since first one matched
        assert "strategy2-node" not in handler.nodes_to_avoid

    @pytest.mark.asyncio
    async def test_get_wrapper_queue(self: Self) -> None:
        """Test that get_wrapper_queue returns the correct queue."""
        downstream_queue = asyncio.Queue()
        env = MockEnvironment()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
        )

        wrapper_queue = handler.get_wrapper_queue()

        assert wrapper_queue is handler.wrapper_queue
        assert isinstance(wrapper_queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_stop_event_processing(self: Self) -> None:
        """Test that stop() properly terminates event processing."""
        downstream_queue = asyncio.Queue()
        env = MockEnvironment()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
        )

        # Start processor
        processor_task = asyncio.create_task(handler.process_events())

        # Stop it (tests that stop works immediately)
        handler.stop()
        await processor_task

        # Verify it stopped
        assert handler.stop_processing is True

    @pytest.mark.asyncio
    async def test_retry_handler_respects_backoff_delay(self: Self) -> None:
        """Test that RetryHandler calls asyncio.sleep with the strategy's backoff delay."""
        downstream_queue = asyncio.Queue()
        env = MockEnvironment()

        class BackoffStrategy(RetryStrategy):
            """Strategy that returns a specific backoff delay."""

            def should_retry(self, event):
                return True

            def extract_nodes_to_avoid(self, event):
                return set()

            def get_retry_delay(self, retry_count):
                return 30.0 * (2**retry_count)

        strategy = BackoffStrategy()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
            max_retries=3,
            strategies=[strategy],
        )

        event = create_test_event()

        with patch(
            "gbserver.resilience.retry_handler.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await handler._evaluate_and_retry(event)

            # Should have called sleep with 30.0 (30 * 2^0, retry_count=0 at time of delay)
            mock_sleep.assert_called_once_with(30.0)

    @pytest.mark.asyncio
    async def test_retry_handler_no_backoff_for_zero_delay(self: Self) -> None:
        """Test that RetryHandler skips sleep when strategy returns 0 delay."""
        downstream_queue = asyncio.Queue()
        env = MockEnvironment()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
            max_retries=3,
            strategies=[AlwaysRetryStrategy()],
        )

        event = create_test_event()

        with patch(
            "gbserver.resilience.retry_handler.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            await handler._evaluate_and_retry(event)

            # AlwaysRetryStrategy inherits default get_retry_delay() returning 0.0
            # so the retry handler should NOT call sleep with that delay.
            # Note: patching module.asyncio.sleep replaces it on the global
            # asyncio object, so unrelated async code may still call it with
            # other values (e.g. 0.1 from polling loops). We only assert that
            # no call was made with the retry delay of 0.0.
            retry_delay_calls = [c for c in mock_sleep.call_args_list if c == call(0.0)]
            assert (
                len(retry_delay_calls) == 0
            ), f"sleep should not be called with delay 0.0, but was: {retry_delay_calls}"

    @pytest.mark.asyncio
    async def test_terminal_failure_raises_with_zero_max_retries(self: Self) -> None:
        """Test that a Failed AppWrapper raises WorkloadFailedException even with max_retries=0.

        This verifies the fix for issue #1810: when retry is disabled, the
        RetryHandler is still created with max_retries=0 so that terminal
        failures are detected and raised instead of being silently ignored.
        """
        from gbserver.types.errors import WorkloadFailedException

        downstream_queue = asyncio.Queue()
        env = MockEnvironment()

        handler = RetryHandler(
            launch_id="test-launch-123",
            downstream_queue=downstream_queue,
            environment=env,
            max_retries=0,
            strategies=[NeverRetryStrategy()],
        )

        # Create a terminal failure event (AppWrapper Failed state)
        failed_payload = {
            "appwrapper": "test-aw",
            "state": "Failed",
            "previous_state": "Running",
            "events": [],
            "failed_pods": {},
        }
        failed_event = create_test_event(
            f"\n```json\n{json.dumps(failed_payload, indent=4)}\n```\n"
        )

        # Start processor
        processor_task = asyncio.create_task(handler.process_events())

        # Put the terminal failure event
        await handler.get_wrapper_queue().put(failed_event)

        # The processor should raise WorkloadFailedException
        with pytest.raises(WorkloadFailedException):
            await processor_task

        # Event should still be forwarded downstream before the exception
        assert downstream_queue.qsize() == 1
        assert env.retry_called is False
