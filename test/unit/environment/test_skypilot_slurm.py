import asyncio
import io
import os
import socket
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment.skypilot import (
    Skypilot,
    _is_interactive_auth_stdin_failure,
)
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventType,
    BuildEventWorkloadStatusPayload,
    EntityRunMetadata,
)
from gbserver.types.environmentconfig import EnvironmentConfig
from gbserver.types.errors import (
    ErrSkypilotInteractiveAuthFailed,
    WorkloadFailedException,
)
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
        assert call_kwargs["zone"] is None  # partition already in the infra

    @pytest.mark.asyncio
    async def test_explicit_infra_passes_separate_zone_through(self, slurm_env):
        # Regression: a 2-segment infra (cloud/cluster, no partition) plus a
        # separate `zone` must keep the zone — it is passed through to
        # sky.Resources as the standalone `zone` arg, not silently dropped.
        mock_sky = _mock_sky()
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("slurm-4b")
            await slurm_env.launch_skypilot(
                launch_id="slurm-4b",
                launcher_config={
                    "run": "hostname",
                    "resources": {
                        "infra": "slurm/my-cluster",
                        "zone": "gpu-partition",
                    },
                },
                config={},
            )

        call_kwargs = mock_sky.Resources.call_args[1]
        assert call_kwargs["infra"] == "slurm/my-cluster"
        assert call_kwargs["zone"] == "gpu-partition"

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


def _make_env(config: dict) -> Skypilot:
    """Build a Skypilot environment from a raw env-config dict.

    :param config: the EnvironmentConfig.config payload (default_cloud,
        cluster, zone, etc.).
    :returns: a Skypilot instance wired to a fresh event queue.
    """
    return Skypilot(
        event_q=asyncio.Queue(),
        environment_config=EnvironmentConfig(
            name="test-slurm", type="Skypilot", config=config
        ),
    )


async def _launch_and_get_resources(env: Skypilot, launch_id: str, **launch_kwargs):
    """Launch under mocked sky and return the sky.Resources call kwargs.

    :param env: the Skypilot environment under test.
    :param launch_id: unique id for this launch (arms the ready event).
    :param launch_kwargs: forwarded to launch_skypilot (launcher_config, config).
    :returns: the kwargs dict passed to the mocked sky.Resources constructor.
    """
    mock_sky = _mock_sky()
    with (
        patch("gbserver.environment.skypilot.sky", mock_sky),
        patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
    ):
        env._get_launch_ready_event(launch_id)
        await env.launch_skypilot(launch_id=launch_id, **launch_kwargs)
    return mock_sky.Resources.call_args[1]


class TestSlurmEnvConfigPartition:
    """The SLURM partition (`zone`) and `cluster` can be set from the
    environment.yaml or step/build `config`, not just the resources override,
    with resources > config > env precedence; the partition is omitted when
    no `zone` resolves."""

    @pytest.mark.asyncio
    async def test_env_config_cluster_and_zone_compose_into_infra(self):
        env = _make_env(
            {"default_cloud": "slurm", "cluster": "bluevela", "zone": "gpu-mid"}
        )
        kw = await _launch_and_get_resources(
            env,
            "envcfg-1",
            launcher_config={"run": "hostname", "resources": {}},
            config={},
        )
        assert kw["infra"] == "slurm/bluevela/gpu-mid"
        assert kw["zone"] is None

    @pytest.mark.asyncio
    async def test_env_config_cluster_without_zone_omits_partition(self):
        env = _make_env({"default_cloud": "slurm", "cluster": "bluevela"})
        kw = await _launch_and_get_resources(
            env,
            "envcfg-2",
            launcher_config={"run": "hostname", "resources": {}},
            config={},
        )
        assert kw["infra"] == "slurm/bluevela"
        assert kw["zone"] is None

    @pytest.mark.asyncio
    async def test_step_config_zone_overrides_env_zone(self):
        env = _make_env(
            {"default_cloud": "slurm", "cluster": "bluevela", "zone": "gpu-mid"}
        )
        kw = await _launch_and_get_resources(
            env,
            "envcfg-3",
            launcher_config={"run": "hostname", "resources": {}},
            config={"zone": "big"},
        )
        assert kw["infra"] == "slurm/bluevela/big"
        assert kw["zone"] is None

    @pytest.mark.asyncio
    async def test_resources_override_beats_config_and_env(self):
        env = _make_env(
            {"default_cloud": "slurm", "cluster": "bluevela", "zone": "gpu-mid"}
        )
        kw = await _launch_and_get_resources(
            env,
            "envcfg-4",
            launcher_config={
                "run": "hostname",
                "resources": {"cluster": "other", "zone": "small"},
            },
            config={"cluster": "cfg", "zone": "cfgzone"},
        )
        assert kw["infra"] == "slurm/other/small"
        assert kw["zone"] is None

    @pytest.mark.asyncio
    async def test_lsf_env_config_composes_into_infra(self):
        """LSF is an HPC cloud, so cluster/queue (`zone`) fall back through env
        config the same way SLURM does — the queue can be set at env level."""
        env = _make_env(
            {"default_cloud": "lsf", "cluster": "bluevela", "zone": "normal"}
        )
        kw = await _launch_and_get_resources(
            env,
            "envcfg-lsf",
            launcher_config={"run": "hostname", "resources": {}},
            config={},
        )
        assert kw["infra"] == "lsf/bluevela/normal"
        assert kw["zone"] is None

    @pytest.mark.asyncio
    async def test_non_hpc_ignores_env_config_zone(self):
        """A non-HPC cloud must NOT pull cluster/zone from env config — only the
        resources override is consulted (behavior unchanged for k8s/aws)."""
        env = _make_env(
            {"default_cloud": "k8s", "cluster": "bluevela", "zone": "normal"}
        )
        kw = await _launch_and_get_resources(
            env,
            "envcfg-5",
            launcher_config={"run": "hostname", "resources": {}},
            config={},
        )
        assert kw["infra"] == "k8s"
        assert kw["zone"] is None

    @pytest.mark.asyncio
    async def test_hpc_zone_without_cluster_raises(self):
        """An HPC zone/partition with no cluster is inexpressible in SkyPilot's
        cloud/region/zone grammar, so it must fail loud rather than silently
        mislabel the partition as the cluster (`slurm/<zone>`)."""
        env = _make_env({"default_cloud": "slurm", "zone": "gpu-mid"})
        with pytest.raises(ValueError, match="requires a cluster"):
            await _launch_and_get_resources(
                env,
                "envcfg-noclust",
                launcher_config={"run": "hostname", "resources": {}},
                config={},
            )


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
        exports GB_BUILD_WORKDIR, and prepends set -eu + mkdir+cd to both the
        setup and run scripts so step authors can use relative paths."""
        mock_sky = _mock_sky()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            slurm_env._get_launch_ready_event("bw-1")
            await slurm_env.launch_skypilot(
                launch_id="bw-1",
                launcher_config={
                    "run": "hostname",
                    "setup": "echo prep",
                    "resources": {},
                },
                config={},
                setup_config={"skypilot": {"build_workdir": "/shared/builds/b/runs/r"}},
            )

        task_kwargs = mock_sky.Task.call_args[1]
        assert task_kwargs["envs"]["GB_BUILD_WORKDIR"] == "/shared/builds/b/runs/r"
        cd_prefix = 'set -eu\nmkdir -p "$GB_BUILD_WORKDIR"\ncd "$GB_BUILD_WORKDIR"\n'
        run_script = task_kwargs["run"]
        assert run_script.startswith(cd_prefix)
        assert run_script.endswith("hostname")
        setup_script = task_kwargs["setup"]
        assert setup_script.startswith(cd_prefix)
        assert setup_script.endswith("echo prep")

    @pytest.mark.asyncio
    async def test_launch_skypilot_skips_cd_when_workdir_unset(self, slurm_env):
        """No build_workdir in setup_config -> GB_BUILD_WORKDIR is not exported
        and no cd is prepended (only the leading set -eu), so the run script runs
        in SkyPilot's default ~/sky_workdir (where relative file_mounts land)."""
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
        # set -eu is always prepended; no mkdir/cd is injected without a workdir.
        assert task_kwargs["run"] == "set -eu\nhostname"

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


def _bind_unix_socket(directory, name):
    """Create a real AF_UNIX socket file ``name`` inside ``directory``.

    Binds with a *relative* path (cwd temporarily set to ``directory``) to stay
    within the platform's ``sun_path`` length limit (~104 bytes on macOS), which
    a long pytest ``tmp_path`` would otherwise blow. Closing the socket does not
    remove the file, so the caller can assert on its (non-)existence afterwards.

    :param directory: pathlib.Path of the (existing) dir to create the socket in.
    :param name: basename for the socket file.
    :returns: the bound ``socket.socket`` (kept open; ref held by caller).
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        s.bind(name)
    finally:
        os.chdir(cwd)
    return s


class TestSshControlSocketClear:
    """The ControlMaster socket clear runs on an HPC launch only when the
    ``GBTEST_SKY_SSH_RESET`` env var is set (manually, during credential-change
    testing), forcing the next SSH to re-authenticate against the fresh config.
    Production leaves SkyPilot's socket management untouched."""

    def test_socket_dir_shape(self):
        # Real SDK path: the stable per-user *root* dir, NOT the hashed
        # control-name subdir (that name is md5(control_name)[:10] on disk, so
        # we address the root and glob one level deeper — see the clear fn).
        from gbserver.environment import skypilot

        control_root = skypilot._ssh_control_socket_dir()
        assert control_root is not None
        assert os.path.basename(control_root).startswith("skypilot_ssh_")
        assert not control_root.endswith("/__default__")

    def test_clears_sockets_every_call(self, tmp_path, monkeypatch):
        from gbserver.environment import skypilot

        # Sockets live one level below the root, under the hashed control name:
        # <root>/<hashed-control-name>/<%C socket>. The clear globs "*/*".
        control_root = tmp_path
        control_name_dir = control_root / "3651d5b8ee"  # md5('__default__')[:10]
        control_name_dir.mkdir()
        monkeypatch.setattr(
            skypilot, "_ssh_control_socket_dir", lambda: str(control_root)
        )

        s1 = _bind_unix_socket(control_name_dir, "abcd1234")  # noqa: F841
        skypilot._clear_skypilot_ssh_control_sockets()
        assert list(control_name_dir.iterdir()) == []

        # No once-guard: a socket that reappears is cleared again next call.
        s2 = _bind_unix_socket(control_name_dir, "efgh5678")  # noqa: F841
        skypilot._clear_skypilot_ssh_control_sockets()
        assert list(control_name_dir.iterdir()) == []

    def test_clears_sockets_across_multiple_control_names(self, tmp_path, monkeypatch):
        # Regression for the literal-"__default__" bug: the on-disk control-name
        # dir is a hash, and there may be more than one; the clear must reach
        # sockets under any subdir, not a name it reconstructs itself.
        from gbserver.environment import skypilot

        monkeypatch.setattr(skypilot, "_ssh_control_socket_dir", lambda: str(tmp_path))
        socks, keep = [], []
        for name in ("3651d5b8ee", "0a1b2c3d4e"):
            d = tmp_path / name
            d.mkdir()
            keep.append(_bind_unix_socket(d, "deadbeef"))
            socks.append(d / "deadbeef")

        skypilot._clear_skypilot_ssh_control_sockets()
        assert all(not s.exists() for s in socks)

    def test_leaves_non_socket_entries_untouched(self, tmp_path, monkeypatch):
        # S_ISSOCK guard: a stray regular file or directory at socket depth is
        # never removed — only actual ControlMaster sockets are.
        from gbserver.environment import skypilot

        monkeypatch.setattr(skypilot, "_ssh_control_socket_dir", lambda: str(tmp_path))
        sub = tmp_path / "3651d5b8ee"
        sub.mkdir()
        sock = _bind_unix_socket(sub, "realsock")  # noqa: F841
        regular = sub / "not-a-socket"
        regular.write_text("keep me")
        nested_dir = sub / "subdir"
        nested_dir.mkdir()

        skypilot._clear_skypilot_ssh_control_sockets()

        assert not (sub / "realsock").exists()  # socket removed
        assert regular.exists()  # regular file kept
        assert nested_dir.is_dir()  # directory kept

    def test_warns_when_dir_unavailable(self, monkeypatch, caplog):
        # The clear runs only in the test scenario, where the socket root is
        # expected to resolve; an unresolved root signals SkyPilot drift, so we
        # warn (rather than silently skip) and must not raise.
        import logging

        from gbserver.environment import skypilot

        monkeypatch.setattr(skypilot, "_ssh_control_socket_dir", lambda: None)
        with caplog.at_level(logging.WARNING):
            skypilot._clear_skypilot_ssh_control_sockets()  # must not raise
        assert any(
            "control-socket root could not be resolved" in r.message
            for r in caplog.records
        )

    def test_logs_when_root_resolves_but_no_sockets(
        self, tmp_path, monkeypatch, caplog
    ):
        # A resolved-but-empty root (nothing cached yet, or SkyPilot relocated its
        # socket layout) must not pass silently: the zero-clear is logged so a
        # dir mismatch under GBTEST_SKY_SSH_RESET is diagnosable, not invisible.
        import logging

        from gbserver.environment import skypilot

        monkeypatch.setattr(skypilot, "_ssh_control_socket_dir", lambda: str(tmp_path))
        with caplog.at_level(logging.INFO):
            skypilot._clear_skypilot_ssh_control_sockets()  # empty root, no raise
        assert any("found no sockets under" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_cleared_on_launch_when_flag_set(self, slurm_env, monkeypatch):
        # GBTEST_SKY_SSH_RESET=true: the HPC launch clears the socket (then
        # materializes the SSH config) so the fresh credentials re-authenticate.
        monkeypatch.setenv("GBTEST_SKY_SSH_RESET", "true")
        order = []
        mock_sky = _mock_sky()
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            patch(
                "gbserver.environment.skypilot._clear_skypilot_ssh_control_sockets",
                side_effect=lambda: order.append("clear"),
            ) as clear,
            patch.object(
                slurm_env,
                "_materialize_ssh_for_launch",
                side_effect=lambda _c: order.append("materialize"),
            ),
        ):
            slurm_env._get_launch_ready_event("clear-1")
            await slurm_env.launch_skypilot(
                launch_id="clear-1",
                launcher_config={"run": "hostname", "resources": {"cloud": "slurm"}},
                config={},
            )
        clear.assert_called_once()
        assert order == ["clear", "materialize"]  # clear precedes materialize

    @pytest.mark.asyncio
    async def test_not_cleared_when_flag_unset(self, slurm_env, monkeypatch):
        # Without GBTEST_SKY_SSH_RESET (the production default) the clear never
        # runs, even for an HPC launch; SkyPilot manages its own sockets.
        monkeypatch.delenv("GBTEST_SKY_SSH_RESET", raising=False)
        mock_sky = _mock_sky()
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            patch(
                "gbserver.environment.skypilot._clear_skypilot_ssh_control_sockets"
            ) as clear,
            patch.object(slurm_env, "_materialize_ssh_for_launch") as mat,
        ):
            slurm_env._get_launch_ready_event("noflag-1")
            await slurm_env.launch_skypilot(
                launch_id="noflag-1",
                launcher_config={"run": "hostname", "resources": {"cloud": "slurm"}},
                config={},
            )
        clear.assert_not_called()
        mat.assert_called_once()  # SSH config is still materialized

    def test_not_cleared_for_non_hpc_cloud(self, slurm_env, monkeypatch):
        # Even with the flag set, a non-HPC cloud (no shared SSH config file) is
        # a no-op: neither the clear nor the SSH materialization runs.
        monkeypatch.setenv("GBTEST_SKY_SSH_RESET", "true")
        with (
            patch(
                "gbserver.environment.skypilot._clear_skypilot_ssh_control_sockets"
            ) as clear,
            patch.object(slurm_env, "_materialize_ssh_for_launch") as mat,
        ):
            slurm_env._prepare_ssh_for_launch("k8s")
        clear.assert_not_called()
        mat.assert_not_called()


def _raise_from_module(filename: str) -> None:
    """Raise ``io.UnsupportedOperation`` from a frame whose code filename is ``filename``.

    Compiles a tiny function under a synthetic filename so the resulting
    traceback carries a frame from that path — reproducing SkyPilot's
    interactive_utils stdin crash without needing the SDK.

    :param filename: The ``co_filename`` to stamp on the raising frame.
    :raises io.UnsupportedOperation: always, from the synthetic frame.
    """
    src = (
        "def _boom():\n"
        "    raise io.UnsupportedOperation("
        "'redirected stdin is pseudofile, has no fileno()')\n"
    )
    namespace = {"io": io}
    exec(compile(src, filename, "exec"), namespace)
    namespace["_boom"]()


class TestInteractiveAuthErrorTranslation:
    """Interactive-auth stdin crash is detected and surfaced as an auth error."""

    _SKY_MODULE = "/venv/site-packages/sky/client/interactive_utils.py"

    def test_detects_interactive_auth_frame(self):
        """A traceback passing through interactive_utils is flagged."""
        with pytest.raises(io.UnsupportedOperation) as excinfo:
            _raise_from_module(self._SKY_MODULE)
        assert _is_interactive_auth_stdin_failure(excinfo.value) is True

    def test_detects_via_context_chain(self):
        """The signal is found when interactive_utils is only in the chain."""
        with pytest.raises(RuntimeError) as excinfo:
            try:
                _raise_from_module(self._SKY_MODULE)
            except io.UnsupportedOperation:
                raise RuntimeError("wrapped by an outer failure")
        assert _is_interactive_auth_stdin_failure(excinfo.value) is True

    def test_ignores_unrelated_error(self):
        """A genuine error with no interactive_utils frame is not relabeled."""
        with pytest.raises(ValueError) as excinfo:
            raise ValueError("some unrelated value error")
        assert _is_interactive_auth_stdin_failure(excinfo.value) is False

    def test_translated_message_surfaces_over_stdin_cause(self):
        """unwrap_errors surfaces the clear auth message, not the stdin crash.

        The translation raises ErrSkypilotInteractiveAuthFailed WITHOUT
        ``from e`` so ``__cause__`` stays None (unwrap_errors follows
        ``__cause__``) while ``__context__`` preserves the original for the
        stack trace.
        """
        from gbserver.utils.unwrap_errors import unwrap_errors

        with pytest.raises(ErrSkypilotInteractiveAuthFailed) as excinfo:
            try:
                _raise_from_module(self._SKY_MODULE)
            except io.UnsupportedOperation:
                raise ErrSkypilotInteractiveAuthFailed(
                    "SSH authentication to the slurm login node failed: bad key"
                )
        err = excinfo.value
        assert err.__cause__ is None
        assert err.__context__ is not None  # original preserved for the trace
        readable = unwrap_errors(err)
        assert "SSH authentication" in readable
        assert "fileno" not in readable

    def test_sky_interactive_auth_module_still_present(self):
        """Guard against SkyPilot relocating the interactive-auth module.

        The other tests in this class raise from a *synthetic* path built to
        match ``_SKY_INTERACTIVE_AUTH_MODULE``, so they exercise the matching
        logic but cannot catch a SkyPilot upgrade that moves the module out
        from under the constant. When that happens ``_is_interactive_auth_stdin_failure``
        silently stops firing and users see the opaque ``io.UnsupportedOperation``
        stdin crash again. This canary pins the constant to the *installed*
        SkyPilot so such a relocation fails loudly here instead. It checks both
        that the module still exists and that it still contains the
        ``stdin``/``fileno`` call that raises in a headless context — catching a
        refactor that keeps the file but moves the crash elsewhere.

        Skips when SkyPilot is not installed (mock-tier venv).
        """
        from pathlib import Path

        sky = pytest.importorskip("sky")

        from gbserver.environment.skypilot import _SKY_INTERACTIVE_AUTH_MODULE

        # The constant is a "sky/<...>" path fragment; resolve it against the
        # installed package root (drop the leading "sky/" so it is not doubled).
        rel = _SKY_INTERACTIVE_AUTH_MODULE.split("sky/", 1)[1]
        module_file = Path(sky.__file__).parent / rel
        assert module_file.is_file(), (
            f"SkyPilot no longer ships {_SKY_INTERACTIVE_AUTH_MODULE} "
            f"(looked for {module_file}); interactive-auth error relabeling in "
            "_is_interactive_auth_stdin_failure has silently stopped firing. "
            "Update _SKY_INTERACTIVE_AUTH_MODULE to the module's new location."
        )
        source = module_file.read_text(encoding="utf-8")
        assert "fileno" in source and "stdin" in source, (
            f"{_SKY_INTERACTIVE_AUTH_MODULE} no longer references the "
            "stdin/fileno call that raises io.UnsupportedOperation in a headless "
            "context; the interactive-auth crash may now originate elsewhere. "
            "Re-verify the detection frame in _is_interactive_auth_stdin_failure."
        )
