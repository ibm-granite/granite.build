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

"""Tests for reporting the SLURM job id behind a SkyPilot launch.

Covers the pure handle reader in ``_skypilot_slurm`` and the launch-time
MESSAGE_EVENT. No cluster required.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from libgbtest.environments.skypilot_mocks import _make_env, _mock_sky

from gbserver.environment._skypilot_slurm import slurm_job_id_from_handle
from gbserver.types.buildevent import BuildEventType


def _handle(tags=None, head_instance_id="job4242-node1"):
    """A handle shaped like SkyPilot's: cached_cluster_info with a head instance."""
    head = SimpleNamespace(tags=tags if tags is not None else {})
    info = SimpleNamespace(
        head_instance_id=head_instance_id, get_head_instance=lambda: head
    )
    return SimpleNamespace(cached_cluster_info=info)


def test_reads_job_id_tag():
    assert slurm_job_id_from_handle(_handle(tags={"job_id": "1234"})) == "1234"


def test_falls_back_to_head_instance_id():
    assert slurm_job_id_from_handle(_handle()) == "4242"


@pytest.mark.parametrize("bad", [True, " ", "None", "12a", None])
def test_malformed_tag_falls_back_to_head_instance_id(bad):
    assert slurm_job_id_from_handle(_handle(tags={"job_id": bad})) == "4242"


@pytest.mark.parametrize(
    "handle",
    [
        None,
        SimpleNamespace(cached_cluster_info=None),
        _handle(head_instance_id="pod-abc"),  # not a SLURM instance id
        _handle(head_instance_id=None),
        MagicMock(),  # attribute access yields mocks, never a real id
    ],
)
def test_returns_none_when_unavailable(handle):
    assert slurm_job_id_from_handle(handle) is None


def test_never_raises():
    info = SimpleNamespace(get_head_instance=MagicMock(side_effect=ValueError("x")))
    assert slurm_job_id_from_handle(SimpleNamespace(cached_cluster_info=info)) is None


async def _launch(cloud: str, handle, run_metadata=None) -> list:
    """Launch under a mocked sky returning ``handle``; return the queued events."""
    env = _make_env({"default_cloud": cloud, "idle_minutes_to_autostop": 0})
    mock_sky = _mock_sky()
    mock_sky.stream_and_get.return_value = (1, handle)
    with (
        patch("gbserver.environment.skypilot.sky", mock_sky),
        patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
    ):
        env._get_launch_ready_event("l1")
        await env.launch_skypilot(
            launch_id="l1",
            launcher_config={"run": "hostname", "resources": {}},
            config={},
            run_metadata=(
                {"build_id": "b1", "targetsteprun_id": "s1"}
                if run_metadata is None
                else run_metadata
            ),
        )
    events = []
    while not env.event_q.empty():
        events.append(env.event_q.get_nowait())
    return events


def _messages(events) -> list:
    return [e.payload.msg for e in events if e.type is BuildEventType.MESSAGE_EVENT]


@pytest.mark.asyncio
async def test_slurm_launch_posts_job_id():
    events = await _launch("slurm", _handle(tags={"job_id": "777"}))
    assert "⚡ SLURM job submitted with JobID=777" in _messages(events)
    msg_event = next(e for e in events if e.type is BuildEventType.MESSAGE_EVENT)
    assert msg_event.run_metadata.targetsteprun_id == "s1"


@pytest.mark.asyncio
async def test_slurm_launch_without_job_id_posts_nothing():
    events = await _launch("slurm", MagicMock())
    assert not any("SLURM job" in m for m in _messages(events))


@pytest.mark.asyncio
async def test_build_less_slurm_launch_posts_nothing():
    events = await _launch("slurm", _handle(tags={"job_id": "777"}), run_metadata={})
    assert not any("SLURM job" in m for m in _messages(events))


@pytest.mark.asyncio
async def test_non_slurm_launch_posts_nothing():
    events = await _launch("kubernetes", _handle(tags={"job_id": "777"}))
    assert not any("SLURM job" in m for m in _messages(events))
