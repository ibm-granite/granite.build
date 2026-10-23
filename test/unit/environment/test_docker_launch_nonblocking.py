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
the blocking call was offloaded — and that, once released, it completes and
records the container.

Two things it deliberately does *not* do, both of which made it flaky before:

* It waits with ``asyncio.wait``, not ``asyncio.wait_for``. ``wait_for``
  **cancels** on timeout, and cancelling a ``run_in_executor`` future that has
  not started yet drops it from the executor queue, so ``containers.run`` then
  never runs at all and any later "did it start?" assertion fails for a reason
  that has nothing to do with the event loop.
* It does not treat the short wait as a deadline for the worker thread to start.
  The default executor is created lazily per event loop, so ``launch_docker``'s
  own submissions are the only work on it — but on a loaded CI runner, thread
  startup and CPU contention can still delay the call well past any short
  timeout, so sampling the start flag once made a green run depend on how busy
  the machine was.

Any exception ``launch_docker`` raises is re-raised as-is rather than being
reported as one of the timing failures above.
"""

import asyncio
import contextlib
import threading
from unittest.mock import MagicMock

import pytest

from gbserver.environment.docker import Docker

LAUNCH_ID = "launchid123"
CONTAINER_ID = "container-123"

# How long the loop is given to come back to us. Only needs to be long enough to
# show the loop is not frozen; it is not a budget for the worker thread.
LOOP_ALIVE_SECONDS = 0.3

# How long the worker thread may take to start the call, and later to finish it
# once released, before we call it broken. Generous on purpose; pytest-timeout
# (1200s) is the backstop.
OFFLOAD_WAIT_SECONDS = 60

# How long the fake docker call pretends to hang. Only has to outlast the
# LOOP_ALIVE_SECONDS wait, and kept short because a regressed offload parks the
# loop inside it — so this is also how long a failing run takes.
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


def _start_launch(env: Docker) -> asyncio.Task:
    """Schedule ``env.launch_docker`` with minimal arguments.

    Args:
        env: Docker environment from ``_make_docker_env``.

    Returns:
        The running launch task.
    """
    return asyncio.create_task(
        env.launch_docker(
            launch_id=LAUNCH_ID,
            config={},
            launcher_config={"command": "echo hi"},
            step={"name": "s"},
            run_metadata={"target_name": "t"},
        )
    )


def _raise_if_failed(task: asyncio.Task) -> None:
    """Re-raise the task's own exception, if it has already finished with one.

    Called before each timing assertion so a real error in launch_docker
    surfaces with its traceback instead of being misreported as a timing failure.

    Args:
        task: The launch task.
    """
    if task.done():
        task.result()


async def _await_started(
    started: asyncio.Future, task: asyncio.Task, timeout: float
) -> None:
    """Wait until containers.run is invoked, or until the launch task finishes.

    Args:
        started: Future that the fake ``containers.run`` resolves on entry.
        task: The launch task; if it finishes first, its exception is re-raised.
        timeout: Upper bound on the wait, in seconds.
    """
    await asyncio.wait(
        {started, task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
    )
    _raise_if_failed(task)
    assert started.done(), f"containers.run was never invoked within {timeout}s"


async def _cancel_if_pending(task: asyncio.Task) -> None:
    """Cancel the task if a failed assertion left it running.

    Args:
        task: The launch task.
    """
    if not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_launch_docker_does_not_block_event_loop():
    """A blocking containers.run must not freeze the loop, and must run off it."""
    loop = asyncio.get_running_loop()
    started = loop.create_future()
    release = threading.Event()

    def blocking_run(*args, **kwargs):
        # Simulate a slow/stalled daemon or inline image pull.
        loop.call_soon_threadsafe(started.set_result, None)
        release.wait(timeout=HANG_SECONDS)
        container = MagicMock()
        container.id = CONTAINER_ID
        return container

    client = MagicMock()
    client.containers.run.side_effect = blocking_run
    env = _make_docker_env(client)

    task = _start_launch(env)
    try:
        # asyncio.wait returns on timeout without cancelling, so the executor
        # submission survives and can still run. If launch_docker blocked the
        # loop, control would not come back here until blocking_run returned and
        # the task would be done.
        done, _ = await asyncio.wait({task}, timeout=LOOP_ALIVE_SECONDS)
        _raise_if_failed(task)
        assert task not in done, (
            "launch_docker completed within "
            f"{LOOP_ALIVE_SECONDS}s while containers.run was still blocked — it "
            "ran the blocking call on the event loop"
        )

        # Now wait for the offload itself, which can take far longer than the
        # wait above on a loaded machine.
        await _await_started(started, task, OFFLOAD_WAIT_SECONDS)

        # Let the call return and the launch finish; wait_for is right here
        # because the task is expected to complete, and it propagates anything
        # the post-run bookkeeping raises.
        release.set()
        await asyncio.wait_for(task, timeout=OFFLOAD_WAIT_SECONDS)
        assert env._launched_containers == {LAUNCH_ID: CONTAINER_ID}
    finally:
        # Release the worker thread so it does not linger after the test.
        release.set()
        await _cancel_if_pending(task)
