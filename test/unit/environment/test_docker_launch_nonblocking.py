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

"""Regression test: Docker.launch_docker must not block the asyncio event loop.

The blocking docker-SDK calls in ``launch_docker`` (image pull and
``containers.run``) used to run directly in the coroutine, freezing the event
loop.  An enclosing ``asyncio.wait_for`` could then never enforce its timeout,
so a slow/stalled pull or daemon hung the build indefinitely (observed as the
standalone docker e2e tests never completing).

This test makes ``containers.run`` block on a threading.Event and asserts that a
short ``asyncio.wait_for`` around ``launch_docker`` still times out — which is
only possible if the blocking call is offloaded off the event-loop thread.
"""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from gbserver.environment.docker import Docker


def _make_docker_env(client: MagicMock) -> Docker:
    """Construct a Docker environment with just enough state for launch_docker.

    Bypasses the heavyweight Environment base ``__init__`` and stubs the methods
    launch_docker touches so the test exercises only its event-loop behaviour.

    Args:
        client: Mock docker client whose ``containers.run`` is driven by the test.

    Returns:
        A Docker instance ready for ``launch_docker``.
    """
    env = object.__new__(Docker)
    env._launched_containers = {}
    env._launched_workspaces = {}
    env._extra_volumes = {}
    env._get_docker = MagicMock(return_value=(MagicMock(), client))  # type: ignore[method-assign]
    env._resolve_image = MagicMock(return_value="img:latest")  # type: ignore[method-assign]
    env._pull_image = MagicMock()  # type: ignore[method-assign]
    env._get_defaults = MagicMock(return_value={})  # type: ignore[method-assign]
    env._release_monitors = MagicMock()  # type: ignore[method-assign]
    return env


@pytest.mark.asyncio
async def test_launch_docker_does_not_block_event_loop():
    """A blocking containers.run must not freeze the loop: wait_for must time out."""
    release = threading.Event()
    started = threading.Event()

    def blocking_run(*args, **kwargs):
        # Simulate a slow/stalled daemon or inline image pull.
        started.set()
        release.wait(timeout=5)
        container = MagicMock()
        container.id = "container-123"
        return container

    client = MagicMock()
    client.containers.run.side_effect = blocking_run
    env = _make_docker_env(client)

    try:
        # If launch_docker offloads the blocking call, the loop stays alive and
        # wait_for fires at 0.3s.  If it blocks the loop (the bug), the timer can
        # never run and this would instead hang until blocking_run returns.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                env.launch_docker(
                    launch_id="launchid123",
                    config={},
                    launcher_config={"command": "echo hi"},
                    step={"name": "s"},
                    run_metadata={"target_name": "t"},
                ),
                timeout=0.3,
            )
        # The blocking call must actually have been reached (in a worker thread).
        assert started.is_set(), "containers.run was never invoked"
    finally:
        # Release the worker thread so it does not linger after the test.
        release.set()
