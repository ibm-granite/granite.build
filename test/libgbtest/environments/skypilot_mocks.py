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

"""Shared launch-mock scaffolding for the SkyPilot unit tests.

The SkyPilot unit tests (``test_skypilot_slurm.py``,
``test_skypilot_sbatch_options.py``, …) all drive ``launch_skypilot`` under a
mocked ``sky`` module and then assert on the arguments passed to
``sky.Resources``. These helpers factor out that scaffolding so it lives in one
place rather than drifting across test files.
"""

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from gbserver.environment.skypilot import Skypilot
from gbserver.types.environmentconfig import EnvironmentConfig


def _mock_sky(launch_request_id: str = "req-sky") -> MagicMock:
    """Build a ``MagicMock`` standing in for the ``sky`` module during a launch.

    :param launch_request_id: value returned by the mocked ``sky.launch`` (an
        opaque request id; tests generally don't assert on it).
    :returns: a MagicMock with ``Resources``, ``Task``, ``launch`` and
        ``stream_and_get`` pre-wired for a successful launch.
    """
    mock = MagicMock()
    mock.Resources = MagicMock(return_value=MagicMock())
    mock.Task = MagicMock(return_value=MagicMock())
    mock.launch = MagicMock(return_value=launch_request_id)
    mock.stream_and_get = MagicMock(return_value=(1, MagicMock()))
    return mock


def _make_env(config: dict) -> Skypilot:
    """Build a Skypilot environment from a raw env-config dict.

    :param config: the ``EnvironmentConfig.config`` payload (``default_cloud``,
        ``cluster``, ``zone``, ``sbatch_options``, etc.).
    :returns: a Skypilot instance wired to a fresh event queue.
    """
    return Skypilot(
        event_q=asyncio.Queue(),
        environment_config=EnvironmentConfig(
            name="test-sky", type="Skypilot", config=config
        ),
    )


async def _launch_and_get_resources(
    env: Skypilot, launch_id: str, **launch_kwargs
) -> Dict[str, Any]:
    """Launch under a mocked ``sky`` and return the ``sky.Resources`` call kwargs.

    :param env: the Skypilot environment under test.
    :param launch_id: unique id for this launch (arms the ready event).
    :param launch_kwargs: forwarded to ``launch_skypilot`` (``launcher_config``,
        ``config``).
    :returns: the keyword-argument dict passed to the mocked ``sky.Resources``
        constructor (e.g. ``["_cluster_config_overrides"]``, ``["infra"]``).
    """
    mock_sky = _mock_sky()
    with (
        patch("gbserver.environment.skypilot.sky", mock_sky),
        patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
    ):
        env._get_launch_ready_event(launch_id)
        await env.launch_skypilot(launch_id=launch_id, **launch_kwargs)
    return mock_sky.Resources.call_args[1]
