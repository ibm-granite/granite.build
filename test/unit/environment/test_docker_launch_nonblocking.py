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

This test makes ``containers.run`` block on a threading.Event and asserts that
``launch_docker`` is still pending after a short wait — which is only possible if
the blocking call was offloaded — and that it really ran on another thread.

Two things it deliberately does *not* do, both of which made it flaky before:

* It waits with ``asyncio.wait``, not ``asyncio.wait_for``. ``wait_for``
  **cancels** on timeout, and cancelling a ``run_in_executor`` future that has
  not started yet drops it from the executor queue, so ``containers.run`` then
  never runs at all and any later "did it start?" assertion fails for a reason
  that has nothing to do with the event loop.
* It does not treat the short wait as a deadline for the worker thread to start.
  ``launch_docker`` submits to the default executor (``run_in_executor(None,
  ...)``), which is shared process-wide and only ``min(32, cpu_count + 4)``
  threads wide — 8 on a 4-core CI runner against 18 on a typical dev machine. A
  loaded suite can keep the submission queued well past any short timeout, so
  sampling the start flag once made a green run depend on how busy that queue was.
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from gbserver.environment.docker import Docker

# How long the loop is given to come back to us. Only needs to be long enough to
# show the loop is not frozen; it is not a budget for the executor queue.
LOOP_ALIVE_SECONDS = 0.3

# How long the executor may keep the submission queued before we call it broken.
# Generous on purpose; pytest-timeout (1200s) is the backstop.
OFFLOAD_WAIT_SECONDS = 60

# How long the fake docker call pretends to hang. Only has to outlast the wait
# above, and kept short because a regressed offload parks the loop inside it —
# so this is also how long a failing run takes.
HANG_SECONDS = 5


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
    """A blocking containers.run must not freeze the loop, and must run off it."""
    release = threading.Event()
    started = threading.Event()
    loop_thread = threading.get_ident()
    call_threads: list[int] = []

    def blocking_run(*args, **kwargs):
        # Simulate a slow/stalled daemon or inline image pull.
        call_threads.append(threading.get_ident())
        started.set()
        release.wait(timeout=HANG_SECONDS)
        container = MagicMock()
        container.id = "container-123"
        return container

    client = MagicMock()
    client.containers.run.side_effect = blocking_run
    env = _make_docker_env(client)

    task = asyncio.ensure_future(
        env.launch_docker(
            launch_id="launchid123",
            config={},
            launcher_config={"command": "echo hi"},
            step={"name": "s"},
            run_metadata={"target_name": "t"},
        )
    )
    try:
        # asyncio.wait returns on timeout without cancelling, so the executor
        # submission survives and can still run. If launch_docker blocked the
        # loop, control would not come back here until blocking_run returned and
        # the task would be done.
        done, _ = await asyncio.wait({task}, timeout=LOOP_ALIVE_SECONDS)
        assert task not in done, (
            "launch_docker completed within "
            f"{LOOP_ALIVE_SECONDS}s while containers.run was still blocked — it "
            "ran the blocking call on the event loop"
        )

        # Now wait for the offload itself, which may sit queued behind a busy
        # default executor for far longer than the wait above.
        deadline = time.monotonic() + OFFLOAD_WAIT_SECONDS
        while not started.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert (
            started.is_set()
        ), f"containers.run was never invoked within {OFFLOAD_WAIT_SECONDS}s"

        # And it ran somewhere other than the event loop. This states the property
        # directly rather than inferring it from timing.
        assert call_threads[0] != loop_thread, (
            "containers.run ran on the event-loop thread — launch_docker did not "
            "offload it"
        )
    finally:
        # Release the worker thread so it does not linger after the test.
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
