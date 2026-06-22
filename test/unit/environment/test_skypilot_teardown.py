import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gbserver.environment.skypilot import Skypilot
from gbserver.types.environmentconfig import EnvironmentConfig


@pytest.fixture
def lsf_env():
    event_q = asyncio.Queue()
    config = EnvironmentConfig(
        name="test-lsf",
        type="Skypilot",
        config={"default_cloud": "lsf"},
    )
    return Skypilot(event_q=event_q, environment_config=config)


def _teardown_config(names):
    # Mirrors the step config block surfaced from bindings in build.yaml.
    return {"config": {"teardown_config": {"cluster_names": names}}}


class TestSkypilotTeardown:
    @pytest.mark.asyncio
    async def test_downs_each_bound_cluster_via_cleanup(self, lsf_env):
        lsf_env._cluster_names["rm-launch-id-1"] = "gb-rm-launch-i"
        lsf_env._cluster_names["code-launch-id"] = "gb-code-launch"

        cleanup = AsyncMock()
        with patch.object(lsf_env, "cleanup_skypilot", cleanup):
            await lsf_env.launch_skypilot_teardown(
                launch_id="teardown-1",
                **_teardown_config(["gb-rm-launch-i", "gb-code-launch"]),
            )

        called_ids = {c.kwargs["launch_id"] for c in cleanup.await_args_list}
        assert called_ids == {"rm-launch-id-1", "code-launch-id"}

    @pytest.mark.asyncio
    async def test_unknown_cluster_falls_back_to_sky_down(self, lsf_env):
        mock_sky = AsyncMock()
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await lsf_env.launch_skypilot_teardown(
                launch_id="teardown-2",
                **_teardown_config(["gb-orphan-xxxx"]),
            )

        mock_sky.down.assert_called_once_with("gb-orphan-xxxx", purge=True)

    @pytest.mark.asyncio
    async def test_one_failure_does_not_skip_the_other(self, lsf_env):
        lsf_env._cluster_names["id-a"] = "gb-a"
        lsf_env._cluster_names["id-b"] = "gb-b"

        async def flaky(launch_id, **kw):
            if launch_id == "id-a":
                raise RuntimeError("down failed")

        cleanup = AsyncMock(side_effect=flaky)
        with patch.object(lsf_env, "cleanup_skypilot", cleanup):
            await lsf_env.launch_skypilot_teardown(
                launch_id="teardown-3",
                **_teardown_config(["gb-a", "gb-b"]),
            )

        called_ids = {c.kwargs["launch_id"] for c in cleanup.await_args_list}
        assert called_ids == {"id-a", "id-b"}

    @pytest.mark.asyncio
    async def test_empty_or_blank_names_are_skipped(self, lsf_env):
        cleanup = AsyncMock()
        with patch.object(lsf_env, "cleanup_skypilot", cleanup):
            await lsf_env.launch_skypilot_teardown(
                launch_id="teardown-4",
                **_teardown_config(["", "   ", None]),
            )
        cleanup.assert_not_awaited()
