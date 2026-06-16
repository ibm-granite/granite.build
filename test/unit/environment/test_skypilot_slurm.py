import asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment.skypilot import Skypilot
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    BuildEventWorkloadStatusPayload,
    EntityRunMetadata,
)
from gbserver.types.environmentconfig import EnvironmentConfig
from gbserver.types.errors import WorkloadFailedException
from gbserver.types.status import Status


@pytest.fixture
def slurm_env():
    event_q = asyncio.Queue()
    config = EnvironmentConfig(
        name="test-slurm",
        type="Skypilot",
        config={
            "default_cloud": "slurm",
            "idle_minutes_to_autostop": 0,
        },
    )
    return Skypilot(event_q=event_q, environment_config=config)


def _mock_sky():
    mock = MagicMock()
    mock.Resources = MagicMock(return_value=MagicMock())
    mock.Task = MagicMock(return_value=MagicMock())
    mock.launch = MagicMock(return_value="req-slurm")
    mock.stream_and_get = MagicMock(return_value=(1, MagicMock()))
    return mock


class TestSlurmInfraPath:
    @pytest.mark.asyncio
    async def test_infra_includes_cluster_and_partition(self, slurm_env):
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("slurm-1")
            await slurm_env.launch_skypilot(
                launch_id="slurm-1",
                launcher_config={
                    "run": "hostname",
                    "resources": {
                        "cloud": "slurm",
                        "cluster": "slurm-docker",
                        "zone": "normal",
                        "accelerators": "GPU:1",
                    },
                },
                config={},
            )

        mock_sky.Resources.assert_called_once()
        call_kwargs = mock_sky.Resources.call_args[1]
        assert call_kwargs["infra"] == "slurm/slurm-docker/normal"
        assert call_kwargs["zone"] is None
        assert call_kwargs["accelerators"] == "GPU:1"

    @pytest.mark.asyncio
    async def test_infra_cluster_without_partition(self, slurm_env):
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("slurm-2")
            await slurm_env.launch_skypilot(
                launch_id="slurm-2",
                launcher_config={
                    "run": "hostname",
                    "resources": {
                        "cloud": "slurm",
                        "cluster": "slurm-docker",
                    },
                },
                config={},
            )

        call_kwargs = mock_sky.Resources.call_args[1]
        assert call_kwargs["infra"] == "slurm/slurm-docker"
        assert call_kwargs["zone"] is None

    @pytest.mark.asyncio
    async def test_infra_bare_cloud_without_cluster(self, slurm_env):
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("slurm-3")
            await slurm_env.launch_skypilot(
                launch_id="slurm-3",
                launcher_config={
                    "run": "hostname",
                    "resources": {"cloud": "slurm"},
                },
                config={},
            )

        call_kwargs = mock_sky.Resources.call_args[1]
        assert call_kwargs["infra"] == "slurm"
        assert call_kwargs["zone"] is None

    @pytest.mark.asyncio
    async def test_explicit_infra_string_takes_precedence(self, slurm_env):
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("slurm-4")
            await slurm_env.launch_skypilot(
                launch_id="slurm-4",
                launcher_config={
                    "run": "hostname",
                    "resources": {
                        "infra": "slurm/my-cluster/gpu-partition",
                        "cloud": "slurm",
                        "cluster": "ignored",
                    },
                },
                config={},
            )

        call_kwargs = mock_sky.Resources.call_args[1]
        assert call_kwargs["infra"] == "slurm/my-cluster/gpu-partition"

    @pytest.mark.asyncio
    async def test_defaults_to_env_config_cloud(self, slurm_env):
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("slurm-5")
            await slurm_env.launch_skypilot(
                launch_id="slurm-5",
                launcher_config={
                    "run": "hostname",
                    "resources": {},
                },
                config={},
            )

        call_kwargs = mock_sky.Resources.call_args[1]
        assert call_kwargs["infra"] == "slurm"


class TestSharedWorkdirEnvVar:
    @pytest.mark.asyncio
    async def test_shared_workdir_exposed_as_env_var(self):
        """When env config sets shared_workdir, GB_SHARED_WORKDIR is exported to the task."""
        env = Skypilot(
            event_q=asyncio.Queue(),
            environment_config=EnvironmentConfig(
                name="test-slurm",
                type="Skypilot",
                config={
                    "default_cloud": "slurm",
                    "idle_minutes_to_autostop": 0,
                    "shared_workdir": "/shared",
                },
            ),
        )
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            env._get_launch_ready_event("workdir-1")
            await env.launch_skypilot(
                launch_id="workdir-1",
                launcher_config={"run": "hostname", "resources": {}},
                config={},
            )

        envs = mock_sky.Task.call_args[1]["envs"]
        assert envs["GB_SHARED_WORKDIR"] == "/shared"

    @pytest.mark.asyncio
    async def test_shared_workdir_omitted_when_unset(self, slurm_env):
        """No shared_workdir on env config -> GB_SHARED_WORKDIR is not exported."""
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("workdir-2")
            await slurm_env.launch_skypilot(
                launch_id="workdir-2",
                launcher_config={"run": "hostname", "resources": {}},
                config={},
            )

        envs = mock_sky.Task.call_args[1]["envs"] or {}
        assert "GB_SHARED_WORKDIR" not in envs


class TestBuildWorkdir:
    @pytest.mark.asyncio
    async def test_setup_skypilot_returns_workdir_and_stashes(self):
        """setup_skypilot returns the build_workdir path and stashes it."""
        env = Skypilot(
            event_q=asyncio.Queue(),
            environment_config=EnvironmentConfig(
                name="test-slurm",
                type="Skypilot",
                config={
                    "default_cloud": "slurm",
                    "shared_workdir": "/shared",
                },
            ),
        )
        runmetadata = EntityRunMetadata(build_id="b-123", targetrun_id="tr-456")

        result = await env.setup_skypilot(setup_id="setup-1", runmetadata=runmetadata)

        expected = "/shared/builds/b-123/runs/tr-456"
        assert result == {"skypilot": {"build_workdir": expected}}
        assert env._setup_workdirs["setup-1"] == expected

    @pytest.mark.asyncio
    async def test_setup_skypilot_returns_empty_when_shared_workdir_unset(
        self, slurm_env
    ):
        """No shared_workdir -> setup_skypilot is a no-op returning {}."""
        runmetadata = EntityRunMetadata(build_id="b-1", targetrun_id="tr-1")
        result = await slurm_env.setup_skypilot(
            setup_id="setup-2", runmetadata=runmetadata
        )
        assert result == {}
        assert "setup-2" not in slurm_env._setup_workdirs

    @pytest.mark.asyncio
    async def test_launch_skypilot_exports_build_workdir_and_prepends_cd(
        self, slurm_env
    ):
        """launch_skypilot reads setup_config.skypilot.build_workdir,
        exports GB_BUILD_WORKDIR, and prepends mkdir+cd to the run script."""
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("bw-1")
            await slurm_env.launch_skypilot(
                launch_id="bw-1",
                launcher_config={"run": "hostname", "resources": {}},
                config={},
                setup_config={"skypilot": {"build_workdir": "/shared/builds/b/runs/r"}},
            )

        task_kwargs = mock_sky.Task.call_args[1]
        assert task_kwargs["envs"]["GB_BUILD_WORKDIR"] == "/shared/builds/b/runs/r"
        run_script = task_kwargs["run"]
        assert run_script.startswith(
            'mkdir -p "$GB_BUILD_WORKDIR"\ncd "$GB_BUILD_WORKDIR"\n'
        )
        assert run_script.endswith("hostname")

    @pytest.mark.asyncio
    async def test_launch_skypilot_skips_workdir_wiring_when_unset(self, slurm_env):
        """No build_workdir in setup_config -> run script is unchanged
        and GB_BUILD_WORKDIR is not exported."""
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("bw-2")
            await slurm_env.launch_skypilot(
                launch_id="bw-2",
                launcher_config={"run": "hostname", "resources": {}},
                config={},
            )

        task_kwargs = mock_sky.Task.call_args[1]
        envs = task_kwargs["envs"] or {}
        assert "GB_BUILD_WORKDIR" not in envs
        assert task_kwargs["run"] == "hostname"

    @pytest.mark.asyncio
    async def test_teardown_skypilot_removes_stashed_workdir(self, slurm_env):
        """teardown_skypilot launches a sky task that rm -rf's the stashed
        path and pops it from _setup_workdirs."""
        slurm_env._setup_workdirs["setup-td"] = "/shared/builds/b/runs/r"
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await slurm_env.teardown_skypilot(setup_id="setup-td")

        mock_sky.Task.assert_called_once()
        task_kwargs = mock_sky.Task.call_args[1]
        # shlex.quote leaves shell-safe paths unquoted (no special chars).
        assert task_kwargs["run"] == "rm -rf /shared/builds/b/runs/r"
        mock_sky.launch.assert_called_once()
        assert "setup-td" not in slurm_env._setup_workdirs

    @pytest.mark.asyncio
    async def test_teardown_skypilot_escapes_unsafe_path(self, slurm_env):
        """A workdir containing shell-meta chars (quotes, semicolons) must
        be shlex-quoted so the `rm -rf` command can't be hijacked. The
        sentinel `; rm -rf /;` here would be a shell injection if the path
        were naively interpolated as f'rm -rf "{workdir}"'."""
        unsafe = '/shared/foo"; rm -rf /;"'
        slurm_env._setup_workdirs["setup-unsafe"] = unsafe
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await slurm_env.teardown_skypilot(setup_id="setup-unsafe")

        run = mock_sky.Task.call_args[1]["run"]
        # shlex.quote single-quotes the whole token; the embedded `"` and
        # `;` survive verbatim inside the single quotes — no breakout.
        import shlex

        assert run == f"rm -rf {shlex.quote(unsafe)}"
        # And the rendered command should be a single rm token followed by
        # a single quoted argument — the second `;` should be inside, not
        # outside, the quoted token.
        assert run.startswith("rm -rf '")
        assert run.endswith("'")

    @pytest.mark.asyncio
    async def test_teardown_skypilot_noop_when_no_stashed_workdir(self, slurm_env):
        """teardown_skypilot is a no-op when setup_id was not provisioned."""
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await slurm_env.teardown_skypilot(setup_id="never-set-up")

        mock_sky.Task.assert_not_called()
        mock_sky.launch.assert_not_called()


class TestSkypilotRetry:
    @pytest.mark.asyncio
    async def test_launch_skypilot_stashes_kwargs_for_replay(self, slurm_env):
        """launch_skypilot must populate _launch_kwargs[launch_id] so
        retry_workload can replay the same args."""
        mock_sky = _mock_sky()
        launcher_config = {"run": "hostname", "resources": {"cloud": "slurm"}}

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("retry-1")
            await slurm_env.launch_skypilot(
                launch_id="retry-1",
                launcher_config=launcher_config,
                config={"foo": "bar"},
                run_metadata={"build_id": "b-1"},
                retry_enabled=True,
                retry_transparently=False,
            )

        stashed = slurm_env._launch_kwargs["retry-1"]
        assert stashed["launcher_config"] == launcher_config
        assert stashed["config"] == {"foo": "bar"}
        assert stashed["run_metadata"] == {"build_id": "b-1"}
        assert stashed["retry_enabled"] is True
        assert stashed["retry_transparently"] is False

    def test_get_default_retry_strategies_returns_any_failure(self, slurm_env):
        """Skypilot ships AnyFailureRetryStrategy as the sole default."""
        from gbserver.resilience.strategies.any_failure import AnyFailureRetryStrategy

        strategies = slurm_env._get_default_retry_strategies()
        assert len(strategies) == 1
        assert isinstance(strategies[0], AnyFailureRetryStrategy)

    @pytest.mark.asyncio
    async def test_retry_workload_cleans_relaunches_and_signals(self, slurm_env):
        """retry_workload calls cleanup_skypilot, then launch_skypilot with the
        stashed kwargs, and sets the per-launch retry-complete event."""
        slurm_env._launch_kwargs["retry-2"] = {
            "launcher_config": {"run": "echo", "resources": {}},
            "config": {},
            "run_metadata": None,
            "setup_config": None,
            "retry_enabled": True,
            "retry_transparently": None,
        }
        slurm_env._cluster_names["retry-2"] = "gb-retry-2"
        retry_event = asyncio.Event()
        slurm_env._skypilot_retry_complete_events["retry-2"] = retry_event

        cleanup_calls: list = []
        relaunch_calls: list = []

        async def fake_cleanup(launch_id, **_):
            cleanup_calls.append(launch_id)
            slurm_env._cluster_names.pop(launch_id, None)

        async def fake_launch(launch_id, **kw):
            relaunch_calls.append((launch_id, kw))
            slurm_env._cluster_names[launch_id] = f"gb-{launch_id}-new"

        with (
            patch.object(slurm_env, "cleanup_skypilot", fake_cleanup),
            patch.object(slurm_env, "launch_skypilot", fake_launch),
        ):
            await slurm_env.retry_workload(
                launch_id="retry-2", nodes_to_avoid=["bad-node"]
            )

        assert cleanup_calls == ["retry-2"]
        assert len(relaunch_calls) == 1
        assert relaunch_calls[0][0] == "retry-2"
        # The stashed kwargs are forwarded verbatim (modulo missing keys
        # filtered by launch_skypilot's `kwargs.get` calls).
        assert relaunch_calls[0][1]["launcher_config"] == {
            "run": "echo",
            "resources": {},
        }
        assert retry_event.is_set()

    def test_cluster_name_for_suffixes_only_on_relaunch(self, slurm_env):
        """_cluster_name_for is unchanged on the initial launch (attempt 0) and
        appends an -r<attempt> suffix on relaunches so each attempt is distinct."""
        assert slurm_env._cluster_name_for("abcdef123456789") == "gb-abcdef123456"
        assert slurm_env._cluster_name_for("abcdef123456789", 0) == "gb-abcdef123456"
        assert slurm_env._cluster_name_for("abcdef123456789", 1) == "gb-abcdef123456-r1"
        assert slurm_env._cluster_name_for("abcdef123456789", 2) == "gb-abcdef123456-r2"

    @pytest.mark.asyncio
    async def test_retry_workload_relaunches_with_fresh_cluster_name(self, slurm_env):
        """retry_workload records the attempt so the relaunch provisions a fresh,
        uniquely-named cluster instead of reusing the draining original name."""
        slurm_env._launch_kwargs["retry-9"] = {
            "launcher_config": {"run": "echo", "resources": {}},
            "config": {},
            "run_metadata": None,
            "setup_config": None,
            "retry_enabled": True,
            "retry_transparently": None,
        }
        slurm_env._cluster_names["retry-9"] = "gb-retry-9"
        slurm_env._skypilot_retry_complete_events["retry-9"] = asyncio.Event()

        # Attempt value observed at the instant launch_skypilot is invoked — this
        # is what _launch_skypilot_inner reads to derive the cluster name.
        attempt_at_launch: list = []

        async def fake_cleanup(launch_id, **_):
            slurm_env._cluster_names.pop(launch_id, None)
            slurm_env._relaunch_attempts.pop(launch_id, None)

        async def fake_launch(launch_id, **_):
            attempt_at_launch.append(slurm_env._relaunch_attempts.get(launch_id))

        with (
            patch.object(slurm_env, "cleanup_skypilot", fake_cleanup),
            patch.object(slurm_env, "launch_skypilot", fake_launch),
        ):
            await slurm_env.retry_workload(launch_id="retry-9", retry_count=2)

        assert attempt_at_launch == [2]
        # The name the relaunch would provision under is the suffixed, fresh one.
        assert slurm_env._cluster_name_for("retry-9", 2) == "gb-retry-9-r2"

    @pytest.mark.asyncio
    async def test_retry_workload_propagates_relaunch_failure(self, slurm_env):
        """If launch_skypilot raises during retry, retry_workload re-raises but
        still sets the retry-complete event (in its finally) so the monitor
        doesn't hang; the monitor then fails the step on the missing cluster."""
        slurm_env._launch_kwargs["retry-3"] = {
            "launcher_config": {"run": "echo"},
            "config": {},
            "run_metadata": None,
            "setup_config": None,
            "retry_enabled": True,
            "retry_transparently": None,
        }
        retry_event = asyncio.Event()
        slurm_env._skypilot_retry_complete_events["retry-3"] = retry_event

        async def fake_cleanup(launch_id, **_):
            pass

        async def fake_launch(*_args, **_kw):
            raise RuntimeError("boom")

        with (
            patch.object(slurm_env, "cleanup_skypilot", fake_cleanup),
            patch.object(slurm_env, "launch_skypilot", fake_launch),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await slurm_env.retry_workload(launch_id="retry-3")

        assert retry_event.is_set()


class TestProvisionRetry:
    """Bounded provision-retry on transient resource-acquisition failures
    (the slurm teardown→relaunch race)."""

    @staticmethod
    def _patches(mock_sky, attempts=4):
        """Common patch set: mocked sky, HAS_SKYPILOT, and 0-backoff so the
        tenacity wait is instant. Constants are imported inside
        _provision_with_retry, so patch them at their definition module."""
        return (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            patch(
                "gbserver.types.constants.GBSERVER_SKYPILOT_PROVISION_BACKOFF_MAX", 0
            ),
            patch(
                "gbserver.types.constants.GBSERVER_SKYPILOT_PROVISION_MAX_ATTEMPTS",
                attempts,
            ),
        )

    @pytest.mark.asyncio
    async def test_transient_failure_is_retried_then_succeeds(self, slurm_env):
        """A transient resource-acquisition error on the first provision attempt
        tears down the partial cluster and the relaunch succeeds."""
        mock_sky = _mock_sky()
        mock_sky.stream_and_get.side_effect = [
            Exception("Failed to acquire resources in normal for {Slurm(cpus=1+)}"),
            (1, MagicMock()),
        ]
        s, h, bmax, batt = self._patches(mock_sky)
        with s, h, bmax, batt:
            slurm_env._get_launch_ready_event("prov-1")
            await slurm_env.launch_skypilot(
                launch_id="prov-1",
                launcher_config={"run": "hostname", "resources": {"cloud": "slurm"}},
                config={},
            )

        assert mock_sky.stream_and_get.call_count == 2
        # partial cluster torn down once between the two attempts
        assert mock_sky.down.call_count == 1
        assert slurm_env._cluster_names["prov-1"] == "gb-prov-1"

    @pytest.mark.asyncio
    async def test_non_retriable_failure_propagates_without_retry(self, slurm_env):
        """A non-provision error (e.g. bad image) is re-raised on the first
        attempt — never retried, never masked, no teardown."""
        mock_sky = _mock_sky()
        mock_sky.stream_and_get.side_effect = Exception("Image not found: badimage")
        s, h, bmax, batt = self._patches(mock_sky)
        with s, h, bmax, batt:
            slurm_env._get_launch_ready_event("prov-2")
            with pytest.raises(Exception, match="Image not found"):
                await slurm_env.launch_skypilot(
                    launch_id="prov-2",
                    launcher_config={"run": "hostname", "resources": {}},
                    config={},
                )

        assert mock_sky.stream_and_get.call_count == 1
        assert mock_sky.down.call_count == 0

    @pytest.mark.asyncio
    async def test_exhaustion_reraises_original_error(self, slurm_env):
        """When every attempt hits a transient failure, the original provision
        error surfaces after exactly max_attempts tries."""
        mock_sky = _mock_sky()
        mock_sky.stream_and_get.side_effect = Exception(
            "Failed to provision all possible launchable resources"
        )
        s, h, bmax, batt = self._patches(mock_sky, attempts=2)
        with s, h, bmax, batt:
            slurm_env._get_launch_ready_event("prov-3")
            with pytest.raises(Exception, match="Failed to provision"):
                await slurm_env.launch_skypilot(
                    launch_id="prov-3",
                    launcher_config={"run": "hostname", "resources": {}},
                    config={},
                )

        assert mock_sky.stream_and_get.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_tolerates_cluster_already_gone(self, slurm_env):
        """_teardown swallows ClusterDoesNotExist (already gone) and
        cleanup_skypilot still clears the per-launch bookkeeping."""

        class _ClusterGone(Exception):
            pass

        mock_sky = _mock_sky()
        mock_sky.exceptions.ClusterDoesNotExist = _ClusterGone
        mock_sky.down.side_effect = _ClusterGone("gb-td-1 does not exist")
        slurm_env._cluster_names["td-1"] = "gb-td-1"

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await slurm_env.cleanup_skypilot(launch_id="td-1")  # no raise

        mock_sky.down.assert_called_once()
        assert "td-1" not in slurm_env._cluster_names


class TestMonitorRetryHandoff:
    """monitor_skypilot_monitor must AWAIT a (possibly slow) relaunch rather
    than racing retry_complete_event and abandoning the relaunched job."""

    @staticmethod
    def _kwargs(launch_id):
        return {
            "launcher_config": {"run": "echo", "resources": {}},
            "config": {},
            "run_metadata": None,
            "setup_config": None,
            "retry_enabled": True,
            "retry_transparently": None,
        }

    @pytest.mark.asyncio
    async def test_monitor_awaits_slow_relaunch_and_polls_fresh_cluster(
        self, slurm_env
    ):
        """First poll triggers a retry; the relaunch finishes only after the
        monitor begins waiting. The monitor must poll the FRESH cluster (a 2nd
        poll) instead of returning early. On the old code only 1 poll happened."""
        mock_sky = _mock_sky()
        slurm_env._launch_kwargs["race-1"] = self._kwargs("race-1")
        slurm_env._cluster_names["race-1"] = "gb-old"
        poll_calls = []
        fresh_seen = []
        retry_task = {}

        @asynccontextmanager
        async def fake_with_retry_handler(*_a, **_k):
            # No handler task: _poll raises directly on terminal failure; the
            # retry path is driven by retry_workload setting stop_event.
            yield slurm_env.event_q, None

        async def fake_cleanup(launch_id, **_):
            slurm_env._cluster_names.pop(launch_id, None)

        async def fake_launch(launch_id, **_):
            await asyncio.sleep(0.05)  # slow: completes after monitor starts waiting
            slurm_env._cluster_names[launch_id] = "gb-new"

        async def fake_poll(launch_id, **_):
            poll_calls.append(launch_id)
            if len(poll_calls) == 1:
                # Trigger a retry the way the RetryHandler would, concurrently,
                # then mirror _poll's stop-event return path.
                retry_task["t"] = asyncio.create_task(
                    slurm_env.retry_workload(launch_id=launch_id)
                )
                await slurm_env._get_launch_stopped_event(launch_id).wait()
                return
            fresh_seen.append(
                slurm_env._cluster_names.get(launch_id)
            )  # terminal success

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            patch.object(slurm_env, "_with_retry_handler", fake_with_retry_handler),
            patch.object(slurm_env, "_poll_skypilot_job", fake_poll),
            patch.object(slurm_env, "cleanup_skypilot", fake_cleanup),
            patch.object(slurm_env, "launch_skypilot", fake_launch),
        ):
            await asyncio.wait_for(
                slurm_env.monitor_skypilot_monitor(
                    launch_id="race-1", event_q=slurm_env.event_q
                ),
                timeout=5,
            )
            await retry_task["t"]  # ensure the retry task finished cleanly

        assert poll_calls == ["race-1", "race-1"]  # polled the FRESH cluster
        assert fresh_seen == ["gb-new"]
        assert "race-1" not in slurm_env._skypilot_retry_complete_events
        assert "race-1" not in slurm_env._skypilot_retry_in_progress_events

    @pytest.mark.asyncio
    async def test_monitor_fails_when_relaunch_fails(self, slurm_env):
        """If the relaunch fails (no fresh cluster), the monitor raises
        WorkloadFailedException rather than returning cleanly."""
        mock_sky = _mock_sky()
        slurm_env._launch_kwargs["race-2"] = self._kwargs("race-2")
        slurm_env._cluster_names["race-2"] = "gb-old"
        poll_calls = []
        retry_task = {}

        @asynccontextmanager
        async def fake_with_retry_handler(*_a, **_k):
            # No handler task: _poll raises directly on terminal failure; the
            # retry path is driven by retry_workload setting stop_event.
            yield slurm_env.event_q, None

        async def fake_cleanup(launch_id, **_):
            slurm_env._cluster_names.pop(launch_id, None)

        async def fake_launch(launch_id, **_):
            await asyncio.sleep(0.02)
            raise RuntimeError("relaunch boom")

        async def fake_poll(launch_id, **_):
            poll_calls.append(launch_id)
            retry_task["t"] = asyncio.create_task(
                slurm_env.retry_workload(launch_id=launch_id)
            )
            await slurm_env._get_launch_stopped_event(launch_id).wait()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            patch.object(slurm_env, "_with_retry_handler", fake_with_retry_handler),
            patch.object(slurm_env, "_poll_skypilot_job", fake_poll),
            patch.object(slurm_env, "cleanup_skypilot", fake_cleanup),
            patch.object(slurm_env, "launch_skypilot", fake_launch),
        ):
            with pytest.raises(WorkloadFailedException):
                await asyncio.wait_for(
                    slurm_env.monitor_skypilot_monitor(
                        launch_id="race-2", event_q=slurm_env.event_q
                    ),
                    timeout=5,
                )
            # retrieve the retry task's failure so it isn't an orphan exception
            with pytest.raises(RuntimeError, match="relaunch boom"):
                await retry_task["t"]

        assert poll_calls == ["race-2"]  # never polled a fresh cluster
        assert "race-2" not in slurm_env._skypilot_retry_in_progress_events

    @pytest.mark.asyncio
    async def test_monitor_times_out_if_relaunch_never_signals(self, slurm_env):
        """If retry_complete is never set, the monitor fails (bounded) instead
        of hanging forever."""
        mock_sky = _mock_sky()
        poll_calls = []

        @asynccontextmanager
        async def fake_with_retry_handler(*_a, **_k):
            # No handler task: _poll raises directly on terminal failure; the
            # retry path is driven by retry_workload setting stop_event.
            yield slurm_env.event_q, None

        async def fake_poll(launch_id, **_):
            poll_calls.append(launch_id)
            # Simulate a retry beginning but never completing.
            slurm_env._skypilot_retry_in_progress_events[launch_id].set()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            patch("gbserver.environment.skypilot.RETRY_RELAUNCH_TIMEOUT_SECONDS", 0.05),
            patch.object(slurm_env, "_with_retry_handler", fake_with_retry_handler),
            patch.object(slurm_env, "_poll_skypilot_job", fake_poll),
        ):
            with pytest.raises(WorkloadFailedException):
                await asyncio.wait_for(
                    slurm_env.monitor_skypilot_monitor(
                        launch_id="race-3", event_q=slurm_env.event_q
                    ),
                    timeout=5,
                )

        assert poll_calls == ["race-3"]


class TestMonitorTerminalNoRetry:
    """A genuine terminal failure must be routed through the RetryHandler:
    retried while budget remains, then failed once the handler gives up — never
    hung, never wrongly succeeded. Exercises the poll-vs-handler-task race with a
    realistic handler task (the TestMonitorRetryHandoff fakes have none, so they
    don't cover the handler's terminal-verdict path)."""

    @staticmethod
    def _kwargs():
        return {
            "launcher_config": {"run": "echo", "resources": {}},
            "config": {},
            "run_metadata": None,
            "setup_config": None,
            "retry_enabled": True,
            "retry_transparently": None,
        }

    @staticmethod
    def _fail_event(launch_id):
        return BuildEvent(
            run_metadata=EntityRunMetadata(build_id=launch_id),
            type=BuildEventType.WORKLOAD_STATUS_EVENT,
            payload=BuildEventWorkloadStatusPayload(status=Status.FAILED),
        )

    async def _run_monitor(
        self,
        slurm_env,
        launch_id,
        poll_outcomes,
        decisions,
        poll_calls,
        launch_calls,
        *,
        timeout=5,
    ):
        """Drive monitor_skypilot_monitor against a realistic handler task.

        poll_outcomes: per-poll "fail"|"success" (the SkyPilot job state).
        decisions: per-FAILED-event "retry"|"fail" (the handler's verdict).
        poll_calls/launch_calls: caller-owned lists, populated as side effects so
        they remain inspectable even when the monitor raises.
        """
        slurm_env._launch_kwargs[launch_id] = self._kwargs()
        slurm_env._cluster_names[launch_id] = "gb-initial"
        state = {"idx": 0}

        async def fake_poll(launch_id, event_q=None, defer_terminal_failure=False, **_):
            poll_calls.append(launch_id)
            if poll_outcomes[len(poll_calls) - 1] == "success":
                return
            await event_q.put(self._fail_event(launch_id))
            if defer_terminal_failure:
                await slurm_env._get_launch_stopped_event(launch_id).wait()
                return
            raise WorkloadFailedException(f"no-handler terminal {launch_id}")

        async def fake_cleanup(launch_id, **_):
            slurm_env._cluster_names.pop(launch_id, None)
            slurm_env._relaunch_attempts.pop(launch_id, None)

        async def fake_launch(launch_id, **_):
            launch_calls.append(launch_id)
            slurm_env._cluster_names[launch_id] = f"gb-{launch_id}-r{len(launch_calls)}"

        @asynccontextmanager
        async def handler_cm(*_a, **_k):
            queue: asyncio.Queue = asyncio.Queue()
            stop = {"v": False}

            async def handler():
                while not stop["v"]:
                    try:
                        event = await asyncio.wait_for(queue.get(), 0.05)
                    except asyncio.TimeoutError:
                        continue
                    if event.type != BuildEventType.WORKLOAD_STATUS_EVENT:
                        continue
                    i = state["idx"]
                    state["idx"] += 1
                    decision = decisions[i] if i < len(decisions) else "fail"
                    if decision == "retry":
                        await slurm_env.retry_workload(
                            launch_id=launch_id, retry_count=i + 1
                        )
                    else:
                        raise WorkloadFailedException(f"terminal no-retry {launch_id}")

            task = asyncio.create_task(handler())
            try:
                yield queue, task
            finally:
                # Mirror _with_retry_handler.__aexit__: stop then await the task,
                # surfacing its terminal-verdict raise (or a clean exit).
                stop["v"] = True
                await task

        with (
            patch("gbserver.environment.skypilot.sky", _mock_sky()),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            patch.object(slurm_env, "_with_retry_handler", handler_cm),
            patch.object(slurm_env, "_poll_skypilot_job", fake_poll),
            patch.object(slurm_env, "cleanup_skypilot", fake_cleanup),
            patch.object(slurm_env, "launch_skypilot", fake_launch),
        ):
            await asyncio.wait_for(
                slurm_env.monitor_skypilot_monitor(
                    launch_id=launch_id, event_q=slurm_env.event_q
                ),
                timeout=timeout,
            )

    @pytest.mark.asyncio
    async def test_first_terminal_failure_is_retried(self, slurm_env):
        """A real terminal failure (not simulated) triggers one relaunch, and the
        monitor polls the fresh cluster to a clean success."""
        poll_calls, launch_calls = [], []
        await self._run_monitor(
            slurm_env, "nr-a", ["fail", "success"], ["retry"], poll_calls, launch_calls
        )
        assert poll_calls == ["nr-a", "nr-a"]
        assert launch_calls == ["nr-a"]
        assert "nr-a" not in slurm_env._skypilot_retry_in_progress_events

    @pytest.mark.asyncio
    async def test_consecutive_terminal_failures_retry_until_success(self, slurm_env):
        """A relaunched cluster that ALSO fails terminally is retried again — the
        exact scenario the old code could not recover from."""
        poll_calls, launch_calls = [], []
        await self._run_monitor(
            slurm_env,
            "nr-b",
            ["fail", "fail", "success"],
            ["retry", "retry"],
            poll_calls,
            launch_calls,
        )
        assert poll_calls == ["nr-b", "nr-b", "nr-b"]
        assert launch_calls == ["nr-b", "nr-b"]

    @pytest.mark.asyncio
    async def test_exhausted_budget_fails_step_without_orphan(self, slurm_env):
        """When the handler gives up (no retry), the monitor raises and never
        relaunches — no orphaned cluster, no wrong success."""
        poll_calls, launch_calls = [], []
        with pytest.raises(WorkloadFailedException):
            await self._run_monitor(
                slurm_env, "nr-c", ["fail"], ["fail"], poll_calls, launch_calls
            )
        assert poll_calls == ["nr-c"]
        assert launch_calls == []  # no orphaned relaunch
        assert "nr-c" not in slurm_env._skypilot_retry_in_progress_events

    @pytest.mark.asyncio
    async def test_success_path_unaffected(self, slurm_env):
        """A terminal SUCCESS returns immediately with no retry/relaunch."""
        poll_calls, launch_calls = [], []
        await self._run_monitor(
            slurm_env, "nr-d", ["success"], [], poll_calls, launch_calls
        )
        assert poll_calls == ["nr-d"]
        assert launch_calls == []

    @pytest.mark.asyncio
    async def test_no_retry_resolves_promptly(self, slurm_env):
        """The no-retry verdict must surface within seconds with the real
        RETRY_RELAUNCH_TIMEOUT_SECONDS (1800s) in place — proving it does NOT
        route through the relaunch-completion wait (the old hang)."""
        poll_calls, launch_calls = [], []
        with pytest.raises(WorkloadFailedException):
            await self._run_monitor(
                slurm_env,
                "nr-e",
                ["fail"],
                ["fail"],
                poll_calls,
                launch_calls,
                timeout=2,
            )
