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

"""What each SkyPilot cloud is asked to do with an idle cluster at ``sky.launch``.

SkyPilot treats ``idle_minutes_to_autostop`` WITHOUT ``down=True`` as a request
for autostop, and WITH it as autodown — and the clouds disagree on which exist:

* VM clouds (aws, ...) support autostop: the idle value passes through, no down.
* Kubernetes supports autodown only (a pod cannot be stopped). Without
  ``down=True`` every launch fails provisioning with "Auto-stop is not supported
  on Kubernetes", so the idle value must arrive WITH ``down=True`` — for both the
  ``kubernetes`` name and its ``k8s`` alias.
* slurm/lsf support neither: the idle value is forced to None.

Asserted at the ``sky.launch`` call itself, since that is what SkyPilot sees.
"""

from unittest.mock import patch

import pytest
from libgbtest.environments.skypilot_mocks import _make_env, _mock_sky

# A fast poll, as the build fixtures use, so the autodown floor (a multiple of the
# step's poll interval) stays below the windows these tests set.
_FAST_POLL = {"poll_interval_seconds": 5}


async def _launch_kwargs(
    env_config: dict, launcher_config: dict, step_config: dict | None = None
) -> dict:
    """Run one mocked launch and return the kwargs ``sky.launch`` received."""
    env = _make_env(env_config)
    mock_sky = _mock_sky()
    with (
        patch("gbserver.environment.skypilot.sky", mock_sky),
        patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
    ):
        env._get_launch_ready_event("idle-1")
        await env.launch_skypilot(
            launch_id="idle-1",
            launcher_config={"run": "hostname", **launcher_config},
            config=_FAST_POLL if step_config is None else step_config,
        )
    mock_sky.launch.assert_called_once()
    return mock_sky.launch.call_args.kwargs


class TestIdleActionPerCloud:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cloud", ["kubernetes", "k8s"])
    async def test_kubernetes_idle_window_is_autodown(self, cloud):
        kwargs = await _launch_kwargs(
            {"default_cloud": cloud, "idle_minutes_to_autostop": 5}, {}
        )
        assert kwargs["idle_minutes_to_autostop"] == 5
        assert kwargs["down"] is True

    @pytest.mark.asyncio
    async def test_kubernetes_zero_is_raised_to_the_floor_and_still_autodown(self):
        # `main`'s env used 0, i.e. delete the moment the job ends: the race the
        # floor exists for. Even with a 5s poll it becomes 1 minute.
        kwargs = await _launch_kwargs(
            {"default_cloud": "kubernetes", "idle_minutes_to_autostop": 0}, {}
        )
        assert kwargs["idle_minutes_to_autostop"] == 1
        assert kwargs["down"] is True

    @pytest.mark.asyncio
    async def test_kubernetes_null_requests_no_idle_action(self):
        kwargs = await _launch_kwargs(
            {"default_cloud": "kubernetes", "idle_minutes_to_autostop": None}, {}
        )
        assert kwargs["idle_minutes_to_autostop"] is None
        assert kwargs["down"] is False

    @pytest.mark.asyncio
    async def test_kubernetes_default_idle_window_is_autodown(self):
        # Omitting the key falls back to the env default (10) — which, before
        # this fix, was itself an autostop request and failed every launch. With
        # the default 300s poll the floor is also 10, so it is kept.
        kwargs = await _launch_kwargs({"default_cloud": "kubernetes"}, {}, {})
        assert kwargs["idle_minutes_to_autostop"] == 10
        assert kwargs["down"] is True

    @pytest.mark.asyncio
    async def test_step_override_to_kubernetes_is_autodown(self):
        # Decided by the cloud actually provisioned, not the env's default_cloud.
        kwargs = await _launch_kwargs(
            {"default_cloud": "aws", "idle_minutes_to_autostop": 5},
            {"resources": {"cloud": "kubernetes"}},
        )
        assert kwargs["down"] is True

    @pytest.mark.asyncio
    async def test_vm_cloud_keeps_autostop(self):
        kwargs = await _launch_kwargs(
            {"default_cloud": "aws", "idle_minutes_to_autostop": 5}, {}
        )
        assert kwargs["idle_minutes_to_autostop"] == 5
        assert kwargs["down"] is False

    @pytest.mark.asyncio
    async def test_slurm_has_no_idle_action(self):
        kwargs = await _launch_kwargs(
            {"default_cloud": "slurm", "idle_minutes_to_autostop": 5}, {}
        )
        assert kwargs["idle_minutes_to_autostop"] is None
        assert kwargs["down"] is False


class TestAutodownFloor:
    """On autodown-only clouds the window must outlast the step's poll interval.

    Otherwise a job finishing just after a poll is deleted before the next one and
    the poller marks the step FAILED (pod "does not exist").
    """

    @pytest.mark.asyncio
    async def test_short_window_with_default_poll_is_raised(self, caplog):
        # 5 minutes vs the default 300s poll: raised to 2 x 300s = 10 minutes.
        with caplog.at_level("WARNING"):
            kwargs = await _launch_kwargs(
                {"default_cloud": "kubernetes", "idle_minutes_to_autostop": 5}, {}, {}
            )
        assert kwargs["idle_minutes_to_autostop"] == 10
        assert kwargs["down"] is True
        assert "raising it to 10 minutes" in caplog.text

    @pytest.mark.asyncio
    async def test_window_above_the_floor_is_kept(self):
        kwargs = await _launch_kwargs(
            {"default_cloud": "kubernetes", "idle_minutes_to_autostop": 15}, {}, {}
        )
        assert kwargs["idle_minutes_to_autostop"] == 15

    @pytest.mark.asyncio
    async def test_templated_string_poll_interval_is_honoured(self):
        # "120" (as a template renders it) -> floor ceil(240/60) = 4.
        kwargs = await _launch_kwargs(
            {"default_cloud": "kubernetes", "idle_minutes_to_autostop": 2},
            {},
            {"poll_interval_seconds": "120"},
        )
        assert kwargs["idle_minutes_to_autostop"] == 4

    @pytest.mark.asyncio
    async def test_null_window_is_left_alone(self):
        kwargs = await _launch_kwargs(
            {"default_cloud": "kubernetes", "idle_minutes_to_autostop": None}, {}, {}
        )
        assert kwargs["idle_minutes_to_autostop"] is None
        assert kwargs["down"] is False

    @pytest.mark.asyncio
    async def test_vm_clouds_are_not_floored(self):
        # Autostop (not autodown) keeps the disk, so there is no race to guard.
        kwargs = await _launch_kwargs(
            {"default_cloud": "aws", "idle_minutes_to_autostop": 1}, {}, {}
        )
        assert kwargs["idle_minutes_to_autostop"] == 1
        assert kwargs["down"] is False
