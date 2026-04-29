import asyncio
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment.environment import Environment
from gbserver.types.buildevent import BuildEventType, EntityRunMetadata


@pytest.mark.g4os
@pytest.mark.unit
class TestStepSkypilotConfig:
    def test_default_values(self):
        from gbserver.types.environment.skypilot import StepSkypilotConfig

        config = StepSkypilotConfig()
        assert config.resources == {}
        assert config.setup == ""
        assert config.run == ""
        assert config.envs == {}
        assert config.file_mounts == {}
        assert config.idle_minutes_to_autostop == 10
        assert config.image_id is None

    def test_from_dict(self):
        from gbserver.types.environment.skypilot import StepSkypilotConfig

        config = StepSkypilotConfig(
            resources={"cloud": "kubernetes", "accelerators": "A100:1"},
            setup="pip install torch",
            run="python train.py",
            envs={"LR": "0.001"},
            idle_minutes_to_autostop=30,
            image_id="docker:nvcr.io/nvidia/pytorch:24.01-py3",
        )
        assert config.resources["accelerators"] == "A100:1"
        assert config.setup == "pip install torch"
        assert config.run == "python train.py"
        assert config.idle_minutes_to_autostop == 30
        assert config.image_id == "docker:nvcr.io/nvidia/pytorch:24.01-py3"


@pytest.mark.g4os
@pytest.mark.unit
class TestSkypilotDiscovery:
    def test_skypilot_registered(self):
        """Skypilot class is auto-discovered and registered."""
        assert "skypilot" in Environment.environment_types
        assert "Skypilot" in Environment.environment_types

    def test_skypilot_is_environment_subclass(self):
        from gbserver.environment.skypilot import Skypilot

        assert issubclass(Skypilot, Environment)


@pytest.mark.g4os
@pytest.mark.unit
class TestSkypilotInit:
    def test_init_creates_instance(self):
        from gbserver.environment.skypilot import Skypilot

        event_q = asyncio.Queue()
        env = Skypilot(event_q=event_q)
        assert env.type == "Skypilot"
        assert env._cluster_names == {}
        assert env._job_ids == {}

    def test_has_launch_types(self):
        from gbserver.environment.skypilot import Skypilot

        event_q = asyncio.Queue()
        env = Skypilot(event_q=event_q)
        assert "skypilot" in env.launch_types

    def test_has_cleanup_types(self):
        from gbserver.environment.skypilot import Skypilot

        event_q = asyncio.Queue()
        env = Skypilot(event_q=event_q)
        assert "skypilot" in env.cleanup_types

    def test_has_monitor_types(self):
        from gbserver.environment.skypilot import Skypilot

        event_q = asyncio.Queue()
        env = Skypilot(event_q=event_q)
        assert "skypilot_monitor" in env.monitor_types


@pytest.mark.g4os
@pytest.mark.unit
class TestSkypilotClusterNaming:
    def test_cluster_name_format(self):
        from gbserver.environment.skypilot import Skypilot

        name = Skypilot._cluster_name_for("abcdef123456789")
        assert name == "gb-abcdef123456"

    def test_cluster_name_short_id(self):
        from gbserver.environment.skypilot import Skypilot

        name = Skypilot._cluster_name_for("short")
        assert name == "gb-short"


@pytest.mark.g4os
@pytest.mark.unit
class TestLaunchSkypilot:
    @pytest.fixture
    def skypilot_env(self):
        from gbserver.environment.skypilot import Skypilot
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-skypilot",
            type="Skypilot",
            config={
                "default_cloud": "k8s",
                "idle_minutes_to_autostop": 15,
            },
        )
        return Skypilot(event_q=event_q, environment_config=config)

    @pytest.mark.asyncio
    async def test_launch_calls_sky_launch(self, skypilot_env):
        mock_sky = MagicMock()
        mock_sky.Resources = MagicMock(return_value=MagicMock())
        mock_sky.Task = MagicMock(return_value=MagicMock())
        mock_sky.launch = MagicMock(return_value="req-123")
        mock_sky.stream_and_get = MagicMock(return_value=(42, MagicMock()))

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            launch_id = "test-launch-001"
            skypilot_env._get_launch_ready_event(launch_id)

            await skypilot_env.launch_skypilot(
                launch_id=launch_id,
                launcher_config={
                    "run": "python train.py",
                    "setup": "pip install torch",
                    "resources": {"accelerators": "A100:1", "cpus": "4+"},
                    "envs": {"LR": "0.001"},
                },
                config={},
            )

        assert launch_id in skypilot_env._cluster_names
        assert skypilot_env._cluster_names[launch_id] == "gb-test-launch-"
        assert skypilot_env._job_ids[launch_id] == 42
        assert skypilot_env._get_launch_ready_event(launch_id).is_set()
        mock_sky.launch.assert_called_once()
        mock_sky.stream_and_get.assert_called_once_with("req-123")

    @pytest.mark.asyncio
    async def test_launch_sets_readiness_on_error(self, skypilot_env):
        """release_monitors must be called even if launch fails."""
        with patch("gbserver.environment.skypilot.HAS_SKYPILOT", False):
            launch_id = "test-launch-err"
            skypilot_env._get_launch_ready_event(launch_id)

            with pytest.raises(ImportError, match="skypilot"):
                await skypilot_env.launch_skypilot(
                    launch_id=launch_id,
                    launcher_config={"run": "echo hello"},
                    config={},
                )

        assert skypilot_env._get_launch_ready_event(launch_id).is_set()

    @pytest.mark.asyncio
    async def test_launch_uses_env_config_cloud(self, skypilot_env):
        """Cloud defaults to environment.yaml config.default_cloud."""
        mock_sky = MagicMock()
        mock_sky.Resources = MagicMock(return_value=MagicMock())
        mock_sky.Task = MagicMock(return_value=MagicMock())
        mock_sky.launch = MagicMock(return_value="req-456")
        mock_sky.stream_and_get = MagicMock(return_value=(1, MagicMock()))

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            launch_id = "test-launch-cloud"
            skypilot_env._get_launch_ready_event(launch_id)

            await skypilot_env.launch_skypilot(
                launch_id=launch_id,
                launcher_config={"run": "echo hello"},
                config={},
            )

        mock_sky.Resources.assert_called_once()
        call_kwargs = mock_sky.Resources.call_args
        assert call_kwargs.kwargs.get("infra") == "k8s"


@pytest.mark.g4os
@pytest.mark.unit
class TestMonitorSkypilotMonitor:
    @pytest.fixture
    def skypilot_env_with_job(self):
        from gbserver.environment.skypilot import Skypilot
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-skypilot",
            type="Skypilot",
            config={"default_cloud": "k8s"},
        )
        env = Skypilot(event_q=event_q, environment_config=config)
        launch_id = "monitor-test-001"
        env._cluster_names[launch_id] = "gb-monitor-test"
        env._job_ids[launch_id] = 42
        env._release_monitors(launch_id)
        return env, launch_id, event_q

    @pytest.mark.asyncio
    async def test_monitor_detects_terminal(self, skypilot_env_with_job):
        env, launch_id, event_q = skypilot_env_with_job

        mock_status_running = MagicMock()
        mock_status_running.is_terminal.return_value = False
        mock_status_running.__str__ = lambda s: "RUNNING"
        mock_status_running.__eq__ = lambda s, o: False

        mock_status_succeeded = MagicMock()
        mock_status_succeeded.is_terminal.return_value = True
        mock_status_succeeded.__str__ = lambda s: "SUCCEEDED"

        mock_sky = MagicMock()
        call_count = [0]

        def mock_job_status(*args, **kwargs):
            call_count[0] += 1
            return f"req-status-{call_count[0]}"

        def mock_get(req_id):
            if "1" in req_id:
                return {42: mock_status_running}
            return {42: mock_status_succeeded}

        mock_sky.job_status = mock_job_status
        mock_sky.get = mock_get

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await env.monitor_skypilot_monitor(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="build-1"),
                poll_interval=0.01,
            )

        assert call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_monitor_respects_stop_event(self, skypilot_env_with_job):
        env, launch_id, event_q = skypilot_env_with_job

        mock_status = MagicMock()
        mock_status.is_terminal.return_value = False

        mock_sky = MagicMock()
        mock_sky.job_status = MagicMock(return_value="req-status")
        mock_sky.get = MagicMock(return_value={42: mock_status})

        stop_event = env._get_launch_stopped_event(launch_id)

        async def set_stop_after_delay():
            await asyncio.sleep(0.05)
            stop_event.set()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await asyncio.gather(
                env.monitor_skypilot_monitor(
                    launch_id=launch_id,
                    event_q=event_q,
                    entityrun_metadata=EntityRunMetadata(build_id="build-1"),
                    poll_interval=0.01,
                ),
                set_stop_after_delay(),
            )


@pytest.mark.g4os
@pytest.mark.unit
class TestCleanupSkypilot:
    @pytest.fixture
    def skypilot_env_with_cluster(self):
        from gbserver.environment.skypilot import Skypilot

        event_q = asyncio.Queue()
        env = Skypilot(event_q=event_q)
        launch_id = "cleanup-test-001"
        env._cluster_names[launch_id] = "gb-cleanup-test"
        env._job_ids[launch_id] = 99
        return env, launch_id

    @pytest.mark.asyncio
    async def test_cleanup_calls_sky_down(self, skypilot_env_with_cluster):
        env, launch_id = skypilot_env_with_cluster

        mock_sky = MagicMock()
        mock_sky.down = MagicMock(return_value="req-down")
        mock_sky.get = MagicMock(return_value=None)

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await env.cleanup_skypilot(launch_id=launch_id)

        mock_sky.down.assert_called_once_with("gb-cleanup-test", purge=True)
        assert launch_id not in env._cluster_names
        assert launch_id not in env._job_ids

    @pytest.mark.asyncio
    async def test_cleanup_sets_stop_event(self, skypilot_env_with_cluster):
        env, launch_id = skypilot_env_with_cluster
        stop_event = env._get_launch_stopped_event(launch_id)

        mock_sky = MagicMock()
        mock_sky.down = MagicMock(return_value="req-down")
        mock_sky.get = MagicMock(return_value=None)

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await env.cleanup_skypilot(launch_id=launch_id)

        assert stop_event.is_set()

    @pytest.mark.asyncio
    async def test_cleanup_no_cluster_is_noop(self):
        from gbserver.environment.skypilot import Skypilot

        env = Skypilot(event_q=asyncio.Queue())
        await env.cleanup_skypilot(launch_id="nonexistent-launch")


@pytest.mark.g4os
@pytest.mark.unit
class TestSkypilotManagedDiscovery:
    def test_skypilot_managed_registered(self):
        assert "skypilot_managed" in Environment.environment_types
        assert "Skypilot_managed" in Environment.environment_types

    def test_skypilot_managed_is_environment_subclass(self):
        from gbserver.environment.skypilot_managed import Skypilot_managed

        assert issubclass(Skypilot_managed, Environment)


@pytest.mark.g4os
@pytest.mark.unit
class TestSkypilotManagedInit:
    def test_init_creates_instance(self):
        from gbserver.environment.skypilot_managed import Skypilot_managed

        event_q = asyncio.Queue()
        env = Skypilot_managed(event_q=event_q)
        assert env.type == "Skypilot_managed"
        assert env._job_names == {}

    def test_has_launch_types(self):
        from gbserver.environment.skypilot_managed import Skypilot_managed

        event_q = asyncio.Queue()
        env = Skypilot_managed(event_q=event_q)
        assert "skypilot_managed" in env.launch_types

    def test_has_cleanup_types(self):
        from gbserver.environment.skypilot_managed import Skypilot_managed

        event_q = asyncio.Queue()
        env = Skypilot_managed(event_q=event_q)
        assert "skypilot_managed" in env.cleanup_types

    def test_has_monitor_types(self):
        from gbserver.environment.skypilot_managed import Skypilot_managed

        event_q = asyncio.Queue()
        env = Skypilot_managed(event_q=event_q)
        assert "skypilot_managed_monitor" in env.monitor_types


@pytest.mark.g4os
@pytest.mark.unit
class TestLaunchSkypilotManaged:
    @pytest.fixture
    def managed_env(self):
        from gbserver.environment.skypilot_managed import Skypilot_managed
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-managed",
            type="Skypilot_managed",
            config={"default_cloud": "k8s", "idle_minutes_to_autostop": 20},
        )
        return Skypilot_managed(event_q=event_q, environment_config=config)

    @pytest.mark.asyncio
    async def test_launch_calls_sky_jobs_launch(self, managed_env):
        mock_sky = MagicMock()
        mock_sky.Resources = MagicMock(return_value=MagicMock())
        mock_sky.Task = MagicMock(return_value=MagicMock())
        mock_sky.jobs.launch = MagicMock(return_value="req-managed-1")
        mock_sky.stream_and_get = MagicMock(return_value=(101, MagicMock()))

        with (
            patch("gbserver.environment.skypilot_managed.sky", mock_sky),
            patch("gbserver.environment.skypilot_managed.HAS_SKYPILOT", True),
        ):
            launch_id = "managed-launch-001"
            managed_env._get_launch_ready_event(launch_id)

            await managed_env.launch_skypilot_managed(
                launch_id=launch_id,
                launcher_config={
                    "run": "python train.py",
                    "resources": {"accelerators": "H100:4"},
                },
                config={},
            )

        assert launch_id in managed_env._job_names
        # "managed-launch-001"[:12] = "managed-laun"
        assert managed_env._job_names[launch_id] == "gb-managed-laun"
        assert managed_env._get_launch_ready_event(launch_id).is_set()
        mock_sky.jobs.launch.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_sets_readiness_on_error(self, managed_env):
        with patch("gbserver.environment.skypilot_managed.HAS_SKYPILOT", False):
            launch_id = "managed-launch-err"
            managed_env._get_launch_ready_event(launch_id)

            with pytest.raises(ImportError, match="skypilot"):
                await managed_env.launch_skypilot_managed(
                    launch_id=launch_id,
                    launcher_config={"run": "echo hello"},
                    config={},
                )

        assert managed_env._get_launch_ready_event(launch_id).is_set()


@pytest.mark.g4os
@pytest.mark.unit
class TestCleanupSkypilotManaged:
    @pytest.mark.asyncio
    async def test_cleanup_calls_sky_jobs_cancel(self):
        from gbserver.environment.skypilot_managed import Skypilot_managed

        event_q = asyncio.Queue()
        env = Skypilot_managed(event_q=event_q)
        launch_id = "managed-cleanup-001"
        env._job_names[launch_id] = "gb-managed-clea"

        mock_sky = MagicMock()
        mock_sky.jobs.cancel = MagicMock(return_value="req-cancel")
        mock_sky.get = MagicMock(return_value=None)

        with (
            patch("gbserver.environment.skypilot_managed.sky", mock_sky),
            patch("gbserver.environment.skypilot_managed.HAS_SKYPILOT", True),
        ):
            await env.cleanup_skypilot_managed(launch_id=launch_id)

        mock_sky.jobs.cancel.assert_called_once_with(name="gb-managed-clea")
        assert launch_id not in env._job_names

    @pytest.mark.asyncio
    async def test_cleanup_no_job_is_noop(self):
        from gbserver.environment.skypilot_managed import Skypilot_managed

        env = Skypilot_managed(event_q=asyncio.Queue())
        await env.cleanup_skypilot_managed(launch_id="nonexistent-launch")


@pytest.mark.g4os
@pytest.mark.unit
class TestImportGuard:
    def test_skypilot_import_guard(self):
        from gbserver.environment.skypilot import _require_skypilot

        with patch("gbserver.environment.skypilot.HAS_SKYPILOT", False):
            with pytest.raises(ImportError, match="pip install.*gbserver.*skypilot"):
                _require_skypilot()

    def test_skypilot_managed_import_guard(self):
        from gbserver.environment.skypilot_managed import _require_skypilot

        with patch("gbserver.environment.skypilot_managed.HAS_SKYPILOT", False):
            with pytest.raises(ImportError, match="pip install.*gbserver.*skypilot"):
                _require_skypilot()


def _make_terminal_sky_mock():
    """Create a mock sky module where the job immediately reaches terminal (SUCCEEDED) state."""
    mock_sky = MagicMock()

    mock_status_succeeded = MagicMock()
    mock_status_succeeded.is_terminal.return_value = True
    mock_status_succeeded.__str__ = lambda s: "SUCCEEDED"

    mock_sky.job_status = MagicMock(return_value="req-status-terminal")
    mock_sky.get = MagicMock(return_value={42: mock_status_succeeded})

    return mock_sky


@pytest.mark.g4os
@pytest.mark.unit
class TestSkypilotMonitorLogParsing:
    """Tests for log-based artifact detection in the unmanaged SkyPilot monitor."""

    EVENT_CONFIGS = [
        {
            "event_type": "NEWARTIFACT_IN_ENVIRONMENT_EVENT",
            "line_regex": "Generated\\sData:\\s.+",
            "is_json": False,
            "event_fields": [
                {
                    "field_name": "binding_id",
                    "field_value_template": "digit_output",
                },
                {
                    "field_name": "path",
                    "field_regex": "[^\\s]+[.]jsonl",
                    "is_data": True,
                },
                {
                    "field_name": "binding",
                    "field_value_template": '{ "path": "{{ fields.data.path }}" }',
                    "is_json": True,
                },
            ],
        }
    ]

    @pytest.fixture
    def skypilot_env_with_terminal_job(self):
        """Create a Skypilot env with a job already in terminal (SUCCEEDED) state."""
        from gbserver.environment.skypilot import Skypilot
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-skypilot",
            type="Skypilot",
            config={"default_cloud": "k8s"},
        )
        env = Skypilot(event_q=event_q, environment_config=config)
        launch_id = "log-parse-test-001"
        env._cluster_names[launch_id] = "gb-log-parse-te"
        env._job_ids[launch_id] = 42
        env._release_monitors(launch_id)
        return env, launch_id, event_q

    # @pytest.mark.skip(reason="TODO: fix the mock so that it matches the changes in the code")
    @pytest.mark.asyncio
    async def test_log_parsing_emits_artifact_event(
        self, skypilot_env_with_terminal_job, tmp_path
    ):
        """Matching log lines produce NEWARTIFACT_IN_ENVIRONMENT_EVENT on event_q."""
        env, launch_id, event_q = skypilot_env_with_terminal_job

        # Write a log file with a matching line
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "job-42.log"
        log_file.write_text(
            "Starting job...\n"
            "Training epoch 1\n"
            "Generated Data: /tmp/outputs/final_data.jsonl\n"
            "Job complete.\n"
        )

        mock_sky = _make_terminal_sky_mock()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            patch(
                "gbserver.environment.skypilot._download_logs_with_retry",
                return_value=str(tmp_path / "logs"),
            ),
        ):
            await env.monitor_skypilot_monitor(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="build-log-1"),
                poll_interval=0.01,
                event_configs=self.EVENT_CONFIGS,
            )

        # Collect all events from the queue
        events = []
        while not event_q.empty():
            events.append(await event_q.get())

        # There should be at least one NEWARTIFACT_IN_ENVIRONMENT_EVENT
        artifact_events = [
            e
            for e in events
            if e.type == BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT
        ]
        assert len(artifact_events) == 1, (
            f"Expected exactly 1 NEWARTIFACT_IN_ENVIRONMENT_EVENT, "
            f"got {len(artifact_events)}. All events: {events}"
        )

        # Verify the event payload has the expected fields
        artifact_event = artifact_events[0]
        assert artifact_event.payload.binding_id == "digit_output"
        assert artifact_event.payload.binding is not None

    # @pytest.mark.skip(reason="TODO: fix the mock so that it matches the changes in the code")
    @pytest.mark.asyncio
    async def test_no_artifact_events_when_no_matching_lines(
        self, skypilot_env_with_terminal_job, tmp_path
    ):
        """Non-matching log lines produce no artifact events."""
        env, launch_id, event_q = skypilot_env_with_terminal_job

        # Write a log file with NO matching lines
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "job-42.log"
        log_file.write_text(
            "Starting job...\n"
            "Training epoch 1\n"
            "Training epoch 2\n"
            "Job complete.\n"
        )

        mock_sky = _make_terminal_sky_mock()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            patch(
                "gbserver.environment.skypilot._download_logs_with_retry",
                return_value=str(tmp_path / "logs"),
            ),
        ):
            await env.monitor_skypilot_monitor(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="build-log-2"),
                poll_interval=0.01,
                event_configs=self.EVENT_CONFIGS,
            )

        # Collect all events from the queue
        events = []
        while not event_q.empty():
            events.append(await event_q.get())

        # There should be NO NEWARTIFACT_IN_ENVIRONMENT_EVENT events
        artifact_events = [
            e
            for e in events
            if e.type == BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT
        ]
        assert len(artifact_events) == 0, (
            f"Expected 0 NEWARTIFACT_IN_ENVIRONMENT_EVENT, "
            f"got {len(artifact_events)}. Events: {artifact_events}"
        )

    @pytest.mark.asyncio
    async def test_no_event_configs_skips_log_parsing(
        self, skypilot_env_with_terminal_job
    ):
        """When event_configs is not provided, no log download occurs."""
        env, launch_id, event_q = skypilot_env_with_terminal_job

        mock_sky = _make_terminal_sky_mock()
        mock_sky.download_logs = MagicMock(return_value="req-download-logs")

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            await env.monitor_skypilot_monitor(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="build-log-3"),
                poll_interval=0.01,
                # No event_configs passed
            )

        # download_logs should NOT have been called
        mock_sky.download_logs.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_download_failure_does_not_crash_monitor(
        self, skypilot_env_with_terminal_job
    ):
        """If log download fails after all retries, monitor returns normally."""
        env, launch_id, event_q = skypilot_env_with_terminal_job

        mock_sky = _make_terminal_sky_mock()

        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
            patch(
                "gbserver.environment.skypilot._download_logs_with_retry",
                create=True,
                side_effect=RuntimeError("Log download failed after all retries"),
            ),
        ):
            # Should NOT raise — the monitor must handle the error gracefully
            await env.monitor_skypilot_monitor(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="build-log-4"),
                poll_interval=0.01,
                event_configs=self.EVENT_CONFIGS,
            )


def _make_terminal_managed_sky_mock(
    job_name="gb-managed-logp", cluster_name="sky-managed-cluster-1"
):
    """Create a mock sky module where a managed job immediately reaches terminal (SUCCEEDED) state."""
    mock_sky = MagicMock()

    mock_status_succeeded = MagicMock()
    mock_status_succeeded.is_terminal.return_value = True
    mock_status_succeeded.__str__ = lambda s: "ManagedJobStatus.SUCCEEDED"

    mock_sky.jobs.queue = MagicMock(return_value="req-managed-queue")
    mock_sky.get = MagicMock(
        return_value=[
            {
                "name": job_name,
                "status": mock_status_succeeded,
                "cluster_name": cluster_name,
            }
        ]
    )

    return mock_sky


# @pytest.mark.skip(reason="skipped because not using managed for now, TODO: unskip after using managed")
@pytest.mark.g4os
@pytest.mark.unit
class TestSkypilotManagedMonitorLogParsing:
    """Tests for log-based artifact detection in the managed SkyPilot monitor."""

    EVENT_CONFIGS = [
        {
            "event_type": "NEWARTIFACT_IN_ENVIRONMENT_EVENT",
            "line_regex": "Generated\\sData:\\s.+",
            "is_json": False,
            "event_fields": [
                {
                    "field_name": "binding_id",
                    "field_value_template": "digit_output",
                },
                {
                    "field_name": "path",
                    "field_regex": "[^\\s]+[.]jsonl",
                    "is_data": True,
                },
                {
                    "field_name": "binding",
                    "field_value_template": '{ "path": "{{ fields.data.path }}" }',
                    "is_json": True,
                },
            ],
        }
    ]

    @pytest.fixture
    def managed_env_with_terminal_job(self):
        """Create a Skypilot_managed env with a job ready for monitoring."""
        from gbserver.environment.skypilot_managed import Skypilot_managed
        from gbserver.types.environmentconfig import EnvironmentConfig

        event_q = asyncio.Queue()
        config = EnvironmentConfig(
            name="test-managed",
            type="Skypilot_managed",
            config={"default_cloud": "k8s"},
        )
        env = Skypilot_managed(event_q=event_q, environment_config=config)
        launch_id = "managed-logp-001"
        env._job_names[launch_id] = "gb-managed-logp"
        env._release_monitors(launch_id)
        return env, launch_id, event_q

    @pytest.mark.asyncio
    async def test_log_parsing_emits_artifact_event(
        self, managed_env_with_terminal_job, tmp_path
    ):
        """Matching log lines produce NEWARTIFACT_IN_ENVIRONMENT_EVENT on event_q."""
        env, launch_id, event_q = managed_env_with_terminal_job

        # Write a log file with a matching line
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "job-managed.log"
        log_file.write_text(
            "Starting job...\n"
            "Training epoch 1\n"
            "Generated Data: /tmp/outputs/final_data.jsonl\n"
            "Job complete.\n"
        )

        mock_sky = _make_terminal_managed_sky_mock()

        with (
            patch("gbserver.environment.skypilot_managed.sky", mock_sky),
            patch("gbserver.environment.skypilot_managed.HAS_SKYPILOT", True),
            patch(
                "gbserver.environment.skypilot_managed._download_logs_with_retry",
                return_value=str(tmp_path / "logs"),
            ),
        ):
            await env.monitor_skypilot_managed_monitor(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="build-managed-log-1"),
                poll_interval=0.01,
                event_configs=self.EVENT_CONFIGS,
            )

        # Collect all events from the queue
        events = []
        while not event_q.empty():
            events.append(await event_q.get())

        # There should be at least one NEWARTIFACT_IN_ENVIRONMENT_EVENT
        artifact_events = [
            e
            for e in events
            if e.type == BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT
        ]
        assert len(artifact_events) == 1, (
            f"Expected exactly 1 NEWARTIFACT_IN_ENVIRONMENT_EVENT, "
            f"got {len(artifact_events)}. All events: {events}"
        )

        # Verify the event payload has the expected fields
        artifact_event = artifact_events[0]
        assert artifact_event.payload.binding_id == "digit_output"
        assert artifact_event.payload.binding is not None

    @pytest.mark.asyncio
    async def test_no_artifact_events_when_no_matching_lines(
        self, managed_env_with_terminal_job, tmp_path
    ):
        """Non-matching log lines produce no artifact events."""
        env, launch_id, event_q = managed_env_with_terminal_job

        # Write a log file with NO matching lines
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        log_file = log_dir / "job-managed.log"
        log_file.write_text(
            "Starting job...\n"
            "Training epoch 1\n"
            "Training epoch 2\n"
            "Job complete.\n"
        )

        mock_sky = _make_terminal_managed_sky_mock()

        with (
            patch("gbserver.environment.skypilot_managed.sky", mock_sky),
            patch("gbserver.environment.skypilot_managed.HAS_SKYPILOT", True),
            patch(
                "gbserver.environment.skypilot_managed._download_logs_with_retry",
                return_value=str(tmp_path / "logs"),
            ),
        ):
            await env.monitor_skypilot_managed_monitor(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="build-managed-log-2"),
                poll_interval=0.01,
                event_configs=self.EVENT_CONFIGS,
            )

        # Collect all events from the queue
        events = []
        while not event_q.empty():
            events.append(await event_q.get())

        # There should be NO NEWARTIFACT_IN_ENVIRONMENT_EVENT events
        artifact_events = [
            e
            for e in events
            if e.type == BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT
        ]
        assert len(artifact_events) == 0, (
            f"Expected 0 NEWARTIFACT_IN_ENVIRONMENT_EVENT, "
            f"got {len(artifact_events)}. Events: {artifact_events}"
        )

    @pytest.mark.asyncio
    async def test_no_event_configs_skips_log_parsing(
        self, managed_env_with_terminal_job
    ):
        """When event_configs is not provided, no log download occurs."""
        env, launch_id, event_q = managed_env_with_terminal_job

        mock_sky = _make_terminal_managed_sky_mock()
        mock_sky.download_logs = MagicMock(return_value="req-download-logs")

        with (
            patch("gbserver.environment.skypilot_managed.sky", mock_sky),
            patch("gbserver.environment.skypilot_managed.HAS_SKYPILOT", True),
        ):
            await env.monitor_skypilot_managed_monitor(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="build-managed-log-3"),
                poll_interval=0.01,
                # No event_configs passed
            )

        # download_logs should NOT have been called
        mock_sky.download_logs.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_download_failure_does_not_crash_monitor(
        self, managed_env_with_terminal_job
    ):
        """If log download fails after all retries, monitor returns normally."""
        env, launch_id, event_q = managed_env_with_terminal_job

        mock_sky = _make_terminal_managed_sky_mock()

        with (
            patch("gbserver.environment.skypilot_managed.sky", mock_sky),
            patch("gbserver.environment.skypilot_managed.HAS_SKYPILOT", True),
            patch(
                "gbserver.environment.skypilot_managed._download_logs_with_retry",
                create=True,
                side_effect=RuntimeError("Log download failed after all retries"),
            ),
        ):
            # Should NOT raise — the monitor must handle the error gracefully
            await env.monitor_skypilot_managed_monitor(
                launch_id=launch_id,
                event_q=event_q,
                entityrun_metadata=EntityRunMetadata(build_id="build-managed-log-4"),
                poll_interval=0.01,
                event_configs=self.EVENT_CONFIGS,
            )
