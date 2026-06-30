import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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
        mock_sky = MagicMock()
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await lsf_env.launch_skypilot_teardown(
                launch_id="teardown-2",
                **_teardown_config(["gb-orphan-xxxx"]),
            )

        mock_sky.down.assert_called_once_with("gb-orphan-xxxx", purge=True)
        mock_sky.get.assert_called_once()

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

    @pytest.mark.asyncio
    async def test_teardown_records_cluster_names_globally(self, lsf_env):
        # Even with NO tracked launch_ids (teardown runs in its own instance),
        # the cluster names are recorded in the process-global set so the
        # SERVICE monitors (in other instances) can match by cluster name.
        Skypilot._intentionally_torn_down_clusters.clear()
        mock_sky = MagicMock()
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await lsf_env.launch_skypilot_teardown(
                launch_id="teardown-5",
                **_teardown_config(["gb-rm", "gb-code"]),
            )
        assert {"gb-rm", "gb-code"} <= Skypilot._intentionally_torn_down_clusters
        Skypilot._intentionally_torn_down_clusters.clear()


class TestMonitorTreatsTeardownAsSuccess:
    """A monitor whose cluster was intentionally torn down must NOT raise.

    The teardown records cluster names in the CLASS-level set, so a monitor on
    a *different* Skypilot instance still matches by its own cluster name.
    """

    @pytest.fixture(autouse=True)
    def _clear_global(self):
        Skypilot._intentionally_torn_down_clusters.clear()
        yield
        Skypilot._intentionally_torn_down_clusters.clear()

    @pytest.mark.asyncio
    async def test_poll_returns_cleanly_when_cluster_gone_after_teardown(self, lsf_env):
        launch_id = "srv-1"
        lsf_env._cluster_names[launch_id] = "gb-srv-1"
        lsf_env._job_ids[launch_id] = 1
        # A *different* instance's teardown recorded this cluster name.
        Skypilot._intentionally_torn_down_clusters.add("gb-srv-1")

        mock_sky = MagicMock()
        # Mirrors a poll hitting a cluster that sky.down already removed.
        mock_sky.job_status.side_effect = RuntimeError(
            "Cluster 'gb-srv-1' does not exist"
        )
        failed = MagicMock()
        failed.is_terminal.return_value = True
        mock_sky.JobStatus.FAILED = failed

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            # Must return cleanly (no WorkloadFailedException) -> step SUCCESS.
            await lsf_env._poll_skypilot_job(launch_id=launch_id, poll_interval=0)

    @pytest.mark.asyncio
    async def test_poll_still_raises_when_not_intentional(self, lsf_env):
        from gbserver.types.errors import WorkloadFailedException

        launch_id = "srv-2"
        lsf_env._cluster_names[launch_id] = "gb-srv-2"
        lsf_env._job_ids[launch_id] = 1
        # NOT recorded: a genuine cluster loss must still fail the step.

        mock_sky = MagicMock()
        mock_sky.job_status.side_effect = RuntimeError(
            "Cluster 'gb-srv-2' does not exist"
        )
        failed = MagicMock()
        failed.is_terminal.return_value = True
        mock_sky.JobStatus.FAILED = failed

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            pytest.raises(WorkloadFailedException),
        ):
            await lsf_env._poll_skypilot_job(launch_id=launch_id, poll_interval=0)
