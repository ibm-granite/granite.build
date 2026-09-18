import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.environment.skypilot import Skypilot
from gbserver.types.buildevent import EntityRunMetadata
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

    @pytest.mark.asyncio
    async def test_teardown_cluster_name_embeds_target_and_build(self):
        # setup_skypilot stashes target_name/build_id (keyed by setup_id) so
        # teardown_skypilot names its cleanup cluster the same human-identifiable
        # way as the launch cluster: gb-<target>-<build8>-...
        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-skypilot",
            type="Skypilot",
            config={"default_cloud": "k8s", "shared_workdir": "/shared"},
        )
        env = Skypilot(event_q=event_q, environment_config=config)
        setup_id = "3168aa02-1234-5678-9abc-def012345678"
        await env.setup_skypilot(
            setup_id,
            runmetadata=EntityRunMetadata(
                build_id="9f3ac1d2-aaaa-bbbb-cccc-ddddeeeeffff",
                target_name="train",
                targetrun_id="run-1",
            ),
        )
        assert env._setup_run_meta[setup_id] == {
            "target_name": "train",
            "build_id": "9f3ac1d2-aaaa-bbbb-cccc-ddddeeeeffff",
            "build_config_name": "",
        }

        mock_sky = MagicMock()
        mock_sky.Resources = MagicMock(return_value=MagicMock())
        mock_sky.Task = MagicMock(return_value=MagicMock())
        mock_sky.launch = MagicMock(return_value="req-td")
        mock_sky.stream_and_get = MagicMock(return_value=None)
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await env.teardown_skypilot(setup_id)

        cluster_name = mock_sky.launch.call_args.kwargs["cluster_name"]
        assert cluster_name.startswith("gb-9f3ac1d2-aaaa-bbbb-cccc-ddddeeeeffff-train-")


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


def make_skypilot_env(config):
    """Build a Skypilot env whose ``shared_filesystem`` block passes the
    EnvironmentConfig gate (Skypilot/aws). Injects ``default_cloud: aws`` (which
    the gate requires) unless the caller set it."""
    event_q = asyncio.Queue()
    ec = EnvironmentConfig(
        name="test-shared-fs",
        type="Skypilot",
        subtype="aws",
        config={"default_cloud": "aws", **config},
    )
    return Skypilot(event_q=event_q, environment_config=ec)


class TestTeardownWithProvider:
    @pytest.mark.asyncio
    async def test_teardown_runs_cleanup_script_and_warns_on_failure(
        self, monkeypatch, caplog
    ):
        env = make_skypilot_env(
            {
                "shared_filesystem": {
                    "provider": "efs",
                    "mount_point": "/mnt/gb-shared",
                    "efs": {
                        "file_system_id": "fs-1",
                        "region": "us-east-1",
                        "cleanup_zone": "us-east-1a",
                    },
                }
            }
        )

        class _Prov:
            mount_point = "/mnt/gb-shared"

            def cleanup_run_script(self, workdir):
                return f"rm -rf {workdir}"

            def cleanup_zone(self):
                return "us-east-1a"

        monkeypatch.setattr(
            "gbserver.environment.skypilot.build_provider", lambda cfg: _Prov()
        )

        # Force the throwaway launch to fail, assert it is logged (not swallowed).
        def _boom(*a, **k):
            raise RuntimeError("no capacity in us-east-1a")

        monkeypatch.setattr(
            "gbserver.environment.skypilot.sky.launch", _boom, raising=False
        )

        env._setup_workdirs["sid"] = "/mnt/gb-shared/builds/b1/runs/r1"
        env._setup_run_meta["sid"] = {
            "target_name": "t",
            "build_id": "b1",
            "build_config_name": "c",
        }

        with caplog.at_level("WARNING"):
            await env.teardown_skypilot("sid")
        assert "/mnt/gb-shared/builds/b1/runs/r1" in caplog.text  # orphan surfaced

    @pytest.mark.asyncio
    async def test_teardown_runs_cleanup_run_script_and_pins_zone(self, monkeypatch):
        env = make_skypilot_env(
            {
                "shared_filesystem": {
                    "provider": "efs",
                    "mount_point": "/mnt/gb-shared",
                    "efs": {
                        "file_system_id": "fs-1",
                        "region": "us-east-1",
                        "cleanup_zone": "us-east-1a",
                    },
                }
            }
        )

        class _Prov:
            mount_point = "/mnt/gb-shared"

            def cleanup_run_script(self, workdir):
                return f"CLEANUP {workdir}"

            def cleanup_zone(self):
                return "us-east-1a"

        monkeypatch.setattr(
            "gbserver.environment.skypilot.build_provider", lambda cfg: _Prov()
        )

        mock_sky = MagicMock()
        mock_sky.Resources = MagicMock(return_value=MagicMock())
        mock_sky.Task = MagicMock(return_value=MagicMock())
        mock_sky.launch = MagicMock(return_value="req-td")
        mock_sky.stream_and_get = MagicMock(return_value=None)

        env._setup_workdirs["sid"] = "/mnt/gb-shared/builds/b1/runs/r1"
        env._setup_run_meta["sid"] = {
            "target_name": "t",
            "build_id": "b1",
            "build_config_name": "c",
        }

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await env.teardown_skypilot("sid")

        # cleanup_run_script drove the throwaway VM's run script.
        run_script = mock_sky.Task.call_args.kwargs["run"]
        assert run_script == "CLEANUP /mnt/gb-shared/builds/b1/runs/r1"
        # Zone pinned onto the resources for the AZ with a mount target.
        assert mock_sky.Resources.call_args.kwargs.get("zone") == "us-east-1a"

    @pytest.mark.asyncio
    async def test_teardown_no_provider_uses_plain_rm_rf(self):
        # No shared_filesystem/shared_workdir -> no provider -> legacy rm -rf path.
        event_q = asyncio.Queue()
        ec = EnvironmentConfig(
            name="test-plain",
            type="Skypilot",
            config={"default_cloud": "k8s"},
        )
        env = Skypilot(event_q=event_q, environment_config=ec)

        mock_sky = MagicMock()
        mock_sky.Resources = MagicMock(return_value=MagicMock())
        mock_sky.Task = MagicMock(return_value=MagicMock())
        mock_sky.launch = MagicMock(return_value="req-td")
        mock_sky.stream_and_get = MagicMock(return_value=None)

        env._setup_workdirs["sid"] = "/shared/builds/b1/runs/r1"
        env._setup_run_meta["sid"] = {
            "target_name": "t",
            "build_id": "b1",
            "build_config_name": "c",
        }

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await env.teardown_skypilot("sid")

        run_script = mock_sky.Task.call_args.kwargs["run"]
        assert run_script == "rm -rf /shared/builds/b1/runs/r1"


class TestWorkdirLauncherEnvVars:
    def test_gb_local_scratch_exported_when_provider_active(self):
        # GB_LOCAL_SCRATCH is only exported when a shared_filesystem provider is
        # active (the provider prologue creates it); a Skypilot/aws env with an
        # efs shared_filesystem block makes build_provider() return a provider.
        event_q = asyncio.Queue()
        ec = EnvironmentConfig(
            name="test-scratch",
            type="Skypilot",
            subtype="aws",
            config={
                "default_cloud": "aws",
                "shared_filesystem": {
                    "provider": "efs",
                    "mount_point": "/mnt/gb-shared",
                    "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
                },
            },
        )
        env = Skypilot(event_q=event_q, environment_config=ec)
        env_vars = env._skypilot_builtin_env(
            launch_id="L1",
            cluster_name="gb-c",
            build_workdir="/mnt/gb-shared/builds/b/runs/r",
        )
        assert env_vars["GB_LOCAL_SCRATCH"] == "/tmp/gb-scratch"
        assert env_vars["GB_SHARED_WORKDIR"] == "/mnt/gb-shared"

    def test_gb_local_scratch_absent_for_plain_shared_workdir(self):
        # A plain shared_workdir env (no provider) must NOT export
        # GB_LOCAL_SCRATCH: nothing creates it (only the provider prologue does).
        event_q = asyncio.Queue()
        ec = EnvironmentConfig(
            name="test-plain-scratch",
            type="Skypilot",
            config={"default_cloud": "k8s", "shared_workdir": "/shared"},
        )
        env = Skypilot(event_q=event_q, environment_config=ec)
        env_vars = env._skypilot_builtin_env(
            launch_id="L1",
            cluster_name="gb-c",
            build_workdir="/shared/builds/b/runs/r",
        )
        assert "GB_LOCAL_SCRATCH" not in env_vars
        assert env_vars["GB_SHARED_WORKDIR"] == "/shared"

    def test_gb_local_scratch_absent_without_shared_workdir(self):
        event_q = asyncio.Queue()
        ec = EnvironmentConfig(
            name="test-noscratch",
            type="Skypilot",
            config={"default_cloud": "k8s"},
        )
        env = Skypilot(event_q=event_q, environment_config=ec)
        env_vars = env._skypilot_builtin_env(
            launch_id="L1", cluster_name="gb-c", build_workdir=None
        )
        assert "GB_LOCAL_SCRATCH" not in env_vars
