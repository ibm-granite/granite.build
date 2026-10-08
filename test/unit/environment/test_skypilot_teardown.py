import asyncio
import enum
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
            "targetrun_id": "run-1",
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

    @pytest.mark.asyncio
    async def test_cleanup_vm_is_floored_to_a_small_instance(self):
        # The throwaway teardown VM only mounts the shared FS and rm's the per-run
        # workdir, so it must be floored to a small instance. Without a cpus floor
        # SkyPilot falls back to its oversized default (e.g. m6i.2xlarge, 8 vCPU),
        # which is wasteful for an `rm`. See issue #425.
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

        res_kwargs = mock_sky.Resources.call_args.kwargs
        assert res_kwargs.get("cpus") == "1+", (
            "teardown cleanup VM must pin a single-vCPU floor, got: " f"{res_kwargs!r}"
        )
        # On cloud catalogs 1 vCPU alone could match a sub-1-GiB t2.nano too small
        # for Ray, so a 2-GiB memory floor is paired with it (cloud only).
        assert res_kwargs.get("memory") == "2+", (
            "cloud cleanup VM must pin a memory floor so Ray fits, got: "
            f"{res_kwargs!r}"
        )

    @pytest.mark.parametrize("cloud", ["slurm", "lsf"])
    @pytest.mark.asyncio
    async def test_cleanup_vm_cpus_floor_is_a_bare_int_on_hpc_clouds(self, cloud):
        # SkyPilot's LSF/SLURM cloud matches CPUs directly and rejects the "N+"
        # minimum form (sky.Resources(infra="lsf", cpus="2+") raises), so the
        # floor must be a bare int there -- the same gating _resources_from_
        # compute_config applies. Regression guard: an unconditional "2+" would
        # crash teardown on these backends, get swallowed by the except, and leak
        # the per-run workdir. See issue #425.
        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-skypilot",
            type="Skypilot",
            config={"default_cloud": cloud, "shared_workdir": "/shared"},
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

        res_kwargs = mock_sky.Resources.call_args.kwargs
        assert res_kwargs.get("cpus") == 1, (
            f"teardown on {cloud} must pass a bare int cpus (the 'N+' form crashes "
            f"the HPC cloud), got: {res_kwargs!r}"
        )
        # slurm/lsf match CPUs directly and don't track memory as a consumable, so
        # a --memory request fails resource matching: the floor must be skipped.
        assert "memory" not in res_kwargs, (
            f"teardown on {cloud} must NOT pass memory (breaks HPC matching), got: "
            f"{res_kwargs!r}"
        )


async def _run_workdir_teardown(default_cloud, stream_side_effect=None):
    """Run teardown_skypilot against a mocked sky for ``default_cloud``.

    :param default_cloud: the env's ``default_cloud`` (e.g. "slurm", "k8s").
    :param stream_side_effect: optional side effect for ``sky.stream_and_get``.
    :returns: (mock_sky, td_cluster_name).
    """
    config = EnvironmentConfig(
        name="test-skypilot",
        type="Skypilot",
        config={"default_cloud": default_cloud, "shared_workdir": "/shared"},
    )
    env = Skypilot(event_q=asyncio.Queue(), environment_config=config)
    setup_id = "3168aa02-1234-5678-9abc-def012345678"
    await env.setup_skypilot(
        setup_id,
        runmetadata=EntityRunMetadata(
            build_id="9f3ac1d2-aaaa-bbbb-cccc-ddddeeeeffff",
            target_name="train",
            targetrun_id="run-1",
        ),
    )
    mock_sky = MagicMock()
    mock_sky.launch = MagicMock(return_value="req-td")
    mock_sky.stream_and_get = MagicMock(side_effect=stream_side_effect)
    mock_sky.down = MagicMock(return_value="req-down")
    with (
        patch("gbserver.environment.skypilot.sky", mock_sky),
        patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
    ):
        await env.teardown_skypilot(setup_id)
    return mock_sky, mock_sky.launch.call_args.kwargs["cluster_name"]


class TestWorkdirTeardownReleasesHpcCluster:
    """teardown_skypilot's td- cluster relies on autodown, which SLURM/LSF do
    not support, so on those clouds it must be downed explicitly or it keeps
    its allocation (observed on BlueVela: td- jobs still RUNNING 40 min on)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cloud", ["slurm", "lsf"])
    async def test_hpc_td_cluster_is_downed(self, cloud):
        mock_sky, td_name = await _run_workdir_teardown(cloud)
        mock_sky.down.assert_called_once_with(td_name, purge=True)

    @pytest.mark.asyncio
    async def test_hpc_td_cluster_is_downed_even_if_rm_fails(self):
        mock_sky, td_name = await _run_workdir_teardown(
            "slurm", stream_side_effect=RuntimeError("rm failed")
        )
        mock_sky.down.assert_called_once_with(td_name, purge=True)

    @pytest.mark.asyncio
    async def test_autodown_cloud_is_left_to_autodown(self):
        mock_sky, _ = await _run_workdir_teardown("k8s")
        mock_sky.down.assert_not_called()


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


class _FakeJobStatus(enum.Enum):
    """Stand-in for sky.JobStatus, so these tests don't need the skypilot extra.
    Teardown only uses ``is_terminal()`` and ``SUCCEEDED``."""

    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

    def is_terminal(self):
        return self is not _FakeJobStatus.RUNNING


def _provider_teardown_sky(
    monkeypatch,
    launch_result,
    get_return=None,
    get_side_effect=None,
    cleanup_timeout_s=None,
):
    """Build a provider-backed Skypilot env and a mocked ``sky`` whose cleanup
    VM launch returns ``launch_result`` (the (job_id, handle) tuple) and whose
    ``sky.get`` of the job status returns ``get_return`` (or raises)."""

    class _Prov:
        mount_point = "/mnt/gb-shared"

        def cleanup_run_script(self, workdir):
            return f"rm -rf {workdir}"

        def cleanup_zone(self):
            return "us-east-1a"

        def cleanup_timeout_s(self):
            return cleanup_timeout_s

    monkeypatch.setattr(
        "gbserver.environment.skypilot.build_providers", lambda cfg: [_Prov()]
    )
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
            },
            "shared_workdir": "/mnt/gb-shared/gbroot",
        }
    )
    env._setup_workdirs["sid"] = "/mnt/gb-shared/gbroot/builds/b1/runs/r1"
    env._setup_run_meta["sid"] = {
        "target_name": "t",
        "build_id": "b1",
        "build_config_name": "c",
    }
    mock_sky = MagicMock()
    mock_sky.JobStatus = _FakeJobStatus
    mock_sky.Resources = MagicMock(return_value=MagicMock())
    mock_sky.Task = MagicMock(return_value=MagicMock())
    mock_sky.launch = MagicMock(return_value="req-launch")
    mock_sky.stream_and_get = MagicMock(return_value=launch_result)
    mock_sky.job_status = MagicMock(return_value="req-status")
    mock_sky.get = MagicMock(return_value=get_return, side_effect=get_side_effect)
    return env, mock_sky


class TestCleanupJobStatusChecked:
    """The reap VM's run script failing exits the job non-zero, which
    ``sky.stream_and_get`` does NOT raise on (it returns (job_id, handle)).
    Teardown must poll the job's final status so a reap that mounted but failed
    (e.g. a failed rm) is surfaced as an ORPHANED tree, not silently dropped."""

    @pytest.mark.asyncio
    async def test_failed_cleanup_job_surfaces_orphan(self, monkeypatch, caplog):
        env, mock_sky = _provider_teardown_sky(
            monkeypatch,
            launch_result=(7, object()),
            get_return={7: _FakeJobStatus.FAILED},
        )
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            caplog.at_level("WARNING"),
        ):
            await env.teardown_skypilot("sid")
        mock_sky.job_status.assert_called_once()
        assert "/mnt/gb-shared/gbroot/builds/b1/runs/r1" in caplog.text
        assert "ORPHAN" in caplog.text.upper()

    @pytest.mark.asyncio
    async def test_succeeded_cleanup_job_raises_no_orphan(self, monkeypatch, caplog):
        env, mock_sky = _provider_teardown_sky(
            monkeypatch,
            launch_result=(7, object()),
            get_return={7: _FakeJobStatus.SUCCEEDED},
        )
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            caplog.at_level("WARNING"),
        ):
            await env.teardown_skypilot("sid")
        assert "ORPHAN" not in caplog.text.upper()

    @pytest.mark.asyncio
    async def test_unconfirmable_status_is_tolerated(self, monkeypatch, caplog):
        # down=True autodowns the cleanup VM right after the job; a status poll
        # that finds the cluster already gone must NOT be read as a failed reap.
        env, mock_sky = _provider_teardown_sky(
            monkeypatch,
            launch_result=(7, object()),
            get_side_effect=RuntimeError("Cluster 'gb-td-sid' does not exist"),
        )
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            caplog.at_level("WARNING"),
        ):
            await env.teardown_skypilot("sid")
        assert "ORPHAN" not in caplog.text.upper()

    @pytest.mark.asyncio
    async def test_running_then_succeeded_raises_no_orphan(self, monkeypatch, caplog):
        # stream_and_get returns at submission, so the first poll sees the reap
        # still RUNNING. A non-terminal status must NOT be read as failure; wait
        # for the terminal SUCCEEDED. Regression for the real-AWS false positive.
        monkeypatch.setattr(
            "gbserver.environment.skypilot._CLEANUP_STATUS_POLL_INTERVAL_S", 0
        )
        env, mock_sky = _provider_teardown_sky(
            monkeypatch,
            launch_result=(7, object()),
            get_side_effect=[
                {7: _FakeJobStatus.RUNNING},
                {7: _FakeJobStatus.SUCCEEDED},
            ],
        )
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            caplog.at_level("WARNING"),
        ):
            await env.teardown_skypilot("sid")
        assert mock_sky.get.call_count == 2
        assert "ORPHAN" not in caplog.text.upper()

    @pytest.mark.asyncio
    async def test_still_running_at_timeout_warns_unconfirmed(
        self, monkeypatch, caplog
    ):
        # A reap that outlasts the wait budget is not a confirmed failure, but
        # it must be visible to operators (WARNING naming the tree), not hidden
        # at debug.
        monkeypatch.setattr(
            "gbserver.environment.skypilot._CLEANUP_STATUS_TIMEOUT_S", 0
        )
        env, mock_sky = _provider_teardown_sky(
            monkeypatch,
            launch_result=(7, object()),
            get_return={7: _FakeJobStatus.RUNNING},
        )
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            caplog.at_level("WARNING"),
        ):
            await env.teardown_skypilot("sid")
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "/mnt/gb-shared/gbroot/builds/b1/runs/r1" in r.getMessage()
            and "may be orphaned" in r.getMessage()
            for r in warnings
        )
        assert "post-build cleanup failed" not in caplog.text

    @pytest.mark.asyncio
    async def test_provider_cleanup_timeout_overrides_default(
        self, monkeypatch, caplog
    ):
        # The provider's cleanup_timeout_s() budget (e.g. the longer AP-mode
        # budget) replaces the module default.
        monkeypatch.setattr(
            "gbserver.environment.skypilot._CLEANUP_STATUS_TIMEOUT_S", 10_000
        )
        monkeypatch.setattr(
            "gbserver.environment.skypilot._CLEANUP_STATUS_POLL_INTERVAL_S", 0
        )
        env, mock_sky = _provider_teardown_sky(
            monkeypatch,
            launch_result=(7, object()),
            get_side_effect=[{7: _FakeJobStatus.RUNNING}] * 3,
            cleanup_timeout_s=0,
        )
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            caplog.at_level("WARNING"),
        ):
            await env.teardown_skypilot("sid")
        assert mock_sky.get.call_count == 1
        assert "may be orphaned" in caplog.text


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
                },
                "shared_workdir": "/mnt/gb-shared/gbroot",
            }
        )

        class _Prov:
            mount_point = "/mnt/gb-shared"

            def cleanup_run_script(self, workdir):
                return f"rm -rf {workdir}"

            def cleanup_zone(self):
                return "us-east-1a"

        monkeypatch.setattr(
            "gbserver.environment.skypilot.build_providers", lambda cfg: [_Prov()]
        )

        # Force the throwaway launch to fail, assert it is logged (not swallowed).
        # Patch the whole `sky` module (not `sky.launch`) so the test does not
        # require the skypilot extra to be installed -- mirrors the sibling
        # teardown tests, whose sky.launch is a MagicMock on a patched module.
        mock_sky = MagicMock()
        mock_sky.Resources = MagicMock(return_value=MagicMock())
        mock_sky.Task = MagicMock(return_value=MagicMock())
        mock_sky.launch = MagicMock(
            side_effect=RuntimeError("no capacity in us-east-1a")
        )

        env._setup_workdirs["sid"] = "/mnt/gb-shared/gbroot/builds/b1/runs/r1"
        env._setup_run_meta["sid"] = {
            "target_name": "t",
            "build_id": "b1",
            "build_config_name": "c",
        }

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            caplog.at_level("WARNING"),
        ):
            await env.teardown_skypilot("sid")
        # orphan surfaced (per-run dir under shared_workdir, mount at mount_point)
        assert "/mnt/gb-shared/gbroot/builds/b1/runs/r1" in caplog.text

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
                },
                "shared_workdir": "/mnt/gb-shared/gbroot",
            }
        )

        class _Prov:
            mount_point = "/mnt/gb-shared"

            def cleanup_run_script(self, workdir):
                return f"CLEANUP {workdir}"

            def cleanup_zone(self):
                return "us-east-1a"

        monkeypatch.setattr(
            "gbserver.environment.skypilot.build_providers", lambda cfg: [_Prov()]
        )

        mock_sky = MagicMock()
        mock_sky.Resources = MagicMock(return_value=MagicMock())
        mock_sky.Task = MagicMock(return_value=MagicMock())
        mock_sky.launch = MagicMock(return_value="req-td")
        mock_sky.stream_and_get = MagicMock(return_value=None)

        env._setup_workdirs["sid"] = "/mnt/gb-shared/gbroot/builds/b1/runs/r1"
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

        # cleanup_run_script drove the throwaway VM's run script (per-run dir
        # under shared_workdir, not the bare mount_point).
        run_script = mock_sky.Task.call_args.kwargs["run"]
        assert run_script == "CLEANUP /mnt/gb-shared/gbroot/builds/b1/runs/r1"
        # Zone pinned onto the resources for the AZ with a mount target.
        assert mock_sky.Resources.call_args.kwargs.get("zone") == "us-east-1a"

    @pytest.mark.asyncio
    async def test_teardown_server_side_cleanup_when_no_run_script(self, monkeypatch):
        # A provider whose cleanup_run_script returns None (a non-mount / object-store
        # backend) reaps server-side via cleanup() and launches NO throwaway VM.
        env = make_skypilot_env(
            {
                "shared_filesystem": {
                    "provider": "efs",
                    "mount_point": "/mnt/gb-shared",
                    "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
                },
                "shared_workdir": "/mnt/gb-shared/gbroot",
            }
        )

        class _Prov:
            mount_point = "/mnt/gb-shared"
            cleaned = False

            def cleanup_run_script(self, workdir):
                return None  # no VM-side cleanup

            def cleanup_zone(self):
                return None

            async def cleanup(self):
                _Prov.cleaned = True

        monkeypatch.setattr(
            "gbserver.environment.skypilot.build_providers", lambda cfg: [_Prov()]
        )
        mock_sky = MagicMock()
        env._setup_workdirs["sid"] = "/mnt/gb-shared/gbroot/builds/b1/runs/r1"
        env._setup_run_meta["sid"] = {"target_name": "t", "build_id": "b1"}

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await env.teardown_skypilot("sid")

        assert _Prov.cleaned is True
        mock_sky.launch.assert_not_called()

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


class TestTeardownDeprovisionsEphemeral:
    """Task 11 (#391): teardown deprovisions every ephemeral mount, skips the
    throwaway rm-VM when the workdir mount is ephemeral, and WARNs (not raises)
    naming the orphan on a deprovision failure."""

    @pytest.mark.asyncio
    async def test_deprovisions_all_ephemeral_and_skips_rm_for_ephemeral_workdir(self):
        from unittest import mock

        from gbserver.environment.shared_fs.base import ProvisionedResources

        env = make_skypilot_env(
            {
                "shared_workdir": "/mnt/e/work",
                "shared_filesystem": [
                    {
                        "provider": "efs",
                        "mount_point": "/mnt/e",
                        "efs": {"provision": "ephemeral", "region": "us-east-1"},
                    }
                ],
            }
        )
        pr = ProvisionedResources(
            region="us-east-1",
            file_system_id="fs-x",
            dns_name="d",
            mount_target_ids=["mt-1"],
            subnet_ids=["subnet-a"],
            security_group_id="sg-1",
            created_sg=True,
        )
        prov = env._shared_fs_providers()[0]
        env._setup_workdirs["sid-1"] = "/mnt/e/work/builds/b1/runs/r1"
        env._setup_run_meta["sid-1"] = {"build_id": "b1"}
        env._setup_provisioned["sid-1"] = [(prov, pr)]
        with (
            mock.patch.object(type(prov), "deprovision", new=mock.AsyncMock()) as dep,
            patch("gbserver.environment.skypilot.sky") as sky_mod,
        ):
            await env.teardown_skypilot("sid-1")
        dep.assert_awaited_once()
        sky_mod.launch.assert_not_called()  # no rm-VM for an ephemeral workdir mount

    @pytest.mark.asyncio
    async def test_warns_orphan_on_deprovision_failure(self, caplog):
        from unittest import mock

        from gbserver.environment.shared_fs.base import ProvisionedResources

        env = make_skypilot_env(
            {
                "shared_workdir": "/mnt/e/work",
                "shared_filesystem": [
                    {
                        "provider": "efs",
                        "mount_point": "/mnt/e",
                        "efs": {"provision": "ephemeral", "region": "us-east-1"},
                    }
                ],
            }
        )
        pr = ProvisionedResources(
            region="us-east-1",
            file_system_id="fs-orphan",
            dns_name="d",
            mount_target_ids=["mt-1"],
            subnet_ids=["subnet-a"],
            security_group_id="sg-1",
            created_sg=True,
        )
        prov = env._shared_fs_providers()[0]
        env._setup_workdirs["sid-2"] = "/mnt/e/work/builds/b1/runs/r1"
        env._setup_run_meta["sid-2"] = {"build_id": "b1"}
        env._setup_provisioned["sid-2"] = [(prov, pr)]
        with (
            mock.patch.object(
                type(prov),
                "deprovision",
                new=mock.AsyncMock(side_effect=RuntimeError("boom")),
            ),
            patch("gbserver.environment.skypilot.sky"),
        ):
            with caplog.at_level("WARNING"):
                await env.teardown_skypilot("sid-2")
        assert "fs-orphan" in caplog.text and "ORPHAN" in caplog.text

    @pytest.mark.asyncio
    async def test_orphan_warning_names_real_targetrun_id_not_setup_id(self, caplog):
        """The EFS is tagged with the gb-targetrun-id and the WARNING tells
        operators to reclaim by tag, so it must print the real targetrun_id --
        not the internal setup_id, which won't match any tag (issue #391)."""
        from unittest import mock

        from gbserver.environment.shared_fs.base import ProvisionedResources

        env = make_skypilot_env(
            {
                "shared_workdir": "/mnt/e/work",
                "shared_filesystem": [
                    {
                        "provider": "efs",
                        "mount_point": "/mnt/e",
                        "efs": {"provision": "ephemeral", "region": "us-east-1"},
                    }
                ],
            }
        )
        pr = ProvisionedResources(
            region="us-east-1",
            file_system_id="fs-orphan",
            dns_name="d",
            mount_target_ids=["mt-1"],
            subnet_ids=["subnet-a"],
            security_group_id="sg-1",
            created_sg=True,
        )
        prov = env._shared_fs_providers()[0]
        setup_id = "3168aa02-1234-5678-9abc-def012345678"
        # Drive setup so the run metadata is stashed by real code, then fail the
        # deprovision and inspect the orphan WARNING.
        with mock.patch.object(
            type(prov), "provision", new=mock.AsyncMock(return_value=pr)
        ):
            await env.setup_skypilot(
                setup_id,
                runmetadata=EntityRunMetadata(
                    build_id="b1",
                    target_name="train",
                    targetrun_id="run-abc123",
                ),
            )
        with (
            mock.patch.object(
                type(prov),
                "deprovision",
                new=mock.AsyncMock(side_effect=RuntimeError("boom")),
            ),
            patch("gbserver.environment.skypilot.sky"),
            caplog.at_level("WARNING"),
        ):
            await env.teardown_skypilot(setup_id)
        orphan = [r.getMessage() for r in caplog.records if "ORPHAN" in r.getMessage()]
        assert orphan, "expected an ORPHAN deprovision WARNING"
        assert "targetrun=run-abc123" in orphan[0]
        assert setup_id not in orphan[0]  # not the internal setup_id


class TestWorkdirMountMemoization:
    def test_workdir_mount_resolved_once_per_env(self, monkeypatch):
        """resolve_workdir_mount re-runs full pydantic validation plus the
        uniqueness/nesting scan on every call; the env resolves it once (like the
        memoized provider list) so repeated launches/teardowns don't re-validate
        (issue #391)."""
        import gbserver.environment.skypilot as sky_mod

        env = make_skypilot_env(
            {
                "shared_workdir": "/mnt/e/work",
                "shared_filesystem": [
                    {
                        "provider": "efs",
                        "mount_point": "/mnt/e",
                        "efs": {"provision": "ephemeral", "region": "us-east-1"},
                    }
                ],
            }
        )
        calls = {"n": 0}
        real = sky_mod.resolve_workdir_mount

        def counting(cfg):
            calls["n"] += 1
            return real(cfg)

        monkeypatch.setattr(sky_mod, "resolve_workdir_mount", counting)

        m1 = env._workdir_mount()
        m2 = env._workdir_mount()
        assert calls["n"] == 1  # resolved once, then cached
        assert m1 is m2 and m1 is not None
        assert m1.mount_point == "/mnt/e"


class TestWorkdirLauncherEnvVars:
    def test_gb_local_scratch_exported_when_provider_active(self):
        # GB_LOCAL_SCRATCH is only exported when a shared_filesystem provider is
        # active (the provider prologue creates it); a Skypilot/aws env with an
        # efs shared_filesystem block makes build_providers() return a provider.
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
                "shared_workdir": "/mnt/gb-shared/gbroot",
            },
        )
        env = Skypilot(event_q=event_q, environment_config=ec)
        env_vars = env._skypilot_builtin_env(
            launch_id="L1",
            cluster_name="gb-c",
            build_workdir="/mnt/gb-shared/gbroot/builds/b/runs/r",
        )
        assert env_vars["GB_LOCAL_SCRATCH"] == "/tmp/gb-scratch"
        # GB_SHARED_WORKDIR is the explicit subdir under mount_point, not the mount.
        assert env_vars["GB_SHARED_WORKDIR"] == "/mnt/gb-shared/gbroot"

    def test_gb_local_scratch_path_is_configurable(self):
        # The scratch path defaults to /tmp/gb-scratch but is configurable via the
        # environment's `local_scratch` (e.g. to point at an instance-store NVMe
        # mount the image actually provides, rather than the EBS root /tmp).
        event_q = asyncio.Queue()
        ec = EnvironmentConfig(
            name="test-scratch-cfg",
            type="Skypilot",
            subtype="aws",
            config={
                "default_cloud": "aws",
                "shared_filesystem": {
                    "provider": "efs",
                    "mount_point": "/mnt/gb-shared",
                    "local_scratch": "/opt/dlami/nvme/gb-scratch",
                    "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
                },
                "shared_workdir": "/mnt/gb-shared/gbroot",
            },
        )
        env = Skypilot(event_q=event_q, environment_config=ec)
        env_vars = env._skypilot_builtin_env(
            launch_id="L1",
            cluster_name="gb-c",
            build_workdir="/mnt/gb-shared/gbroot/builds/b/runs/r",
        )
        assert env_vars["GB_LOCAL_SCRATCH"] == "/opt/dlami/nvme/gb-scratch"

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
