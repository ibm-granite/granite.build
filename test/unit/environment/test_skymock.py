from gbserver.testing.skymock.mock_sky import MockJobStatus, MockSky
from gbserver.testing.skymock.scenario import Scenario, ScenarioStep


class TestScenarioStep:
    def test_step_has_required_fields(self):
        step = ScenarioStep(status="RUNNING", is_terminal=False)
        assert step.status == "RUNNING"
        assert step.is_terminal is False
        assert step.error is None
        assert step.logs is None

    def test_step_with_error(self):
        step = ScenarioStep(status="FAILED", is_terminal=True, error="ResourceExhausted")
        assert step.error == "ResourceExhausted"

    def test_step_with_logs(self):
        step = ScenarioStep(
            status="SUCCEEDED", is_terminal=True, logs={"1": "/tmp/logs/job-1"}
        )
        assert step.logs == {"1": "/tmp/logs/job-1"}


class TestScenarioFactory:
    def test_happy_path_produces_correct_steps(self):
        scenario = Scenario.happy_path(cloud="aws")
        assert scenario.cloud == "aws"
        statuses = [s.status for s in scenario.steps]
        assert statuses == ["PENDING", "RUNNING", "SUCCEEDED"]
        assert scenario.steps[-1].is_terminal is True
        assert all(not s.is_terminal for s in scenario.steps[:-1])

    def test_failure_produces_terminal_failed(self):
        scenario = Scenario.failure(cloud="gcp", error="QuotaExceeded")
        statuses = [s.status for s in scenario.steps]
        assert statuses == ["PENDING", "RUNNING", "FAILED"]
        assert scenario.steps[-1].is_terminal is True
        assert scenario.steps[-1].error == "QuotaExceeded"
        assert scenario.cloud == "gcp"

    def test_preemption_then_recovery_sequence(self):
        scenario = Scenario.preemption_then_recovery(cloud="aws")
        statuses = [s.status for s in scenario.steps]
        assert statuses == [
            "PENDING",
            "RUNNING",
            "PREEMPTED",
            "PENDING",
            "RUNNING",
            "SUCCEEDED",
        ]
        assert scenario.steps[-1].is_terminal is True
        terminal_flags = [s.is_terminal for s in scenario.steps]
        assert terminal_flags.count(True) == 1

    def test_cross_cloud_failover_returns_two_scenarios(self):
        primary, fallback = Scenario.cross_cloud_failover(primary="aws", fallback="gcp")
        assert primary.cloud == "aws"
        assert fallback.cloud == "gcp"
        assert primary.steps[-1].status == "FAILED"
        assert primary.steps[-1].is_terminal is True
        assert fallback.steps[-1].status == "SUCCEEDED"
        assert fallback.steps[-1].is_terminal is True

    def test_cloud_parameter_affects_error_messages(self):
        aws = Scenario.failure(cloud="aws")
        gcp = Scenario.failure(cloud="gcp")
        assert "AWS" in aws.steps[-1].error
        assert "GCP" in gcp.steps[-1].error


class TestMockJobStatus:
    def test_terminal_status_is_terminal(self):
        status = MockJobStatus("SUCCEEDED", is_terminal=True)
        assert status.is_terminal() is True

    def test_non_terminal_status_is_not_terminal(self):
        status = MockJobStatus("RUNNING", is_terminal=False)
        assert status.is_terminal() is False

    def test_str_representation_matches_sky_format(self):
        status = MockJobStatus("SUCCEEDED", is_terminal=True)
        assert str(status) == "JobStatus.SUCCEEDED"

    def test_equality_by_name(self):
        s1 = MockJobStatus("RUNNING", is_terminal=False)
        s2 = MockJobStatus("RUNNING", is_terminal=False)
        assert s1 == s2


class TestMockSkyLaunch:
    def test_launch_returns_request_id(self):
        mock = MockSky(default_scenario=Scenario.happy_path())
        task = mock.Task(name="test-cluster", run="echo hi")
        req_id = mock.launch(task, cluster_name="gb-test123")
        assert isinstance(req_id, str)
        assert len(req_id) > 0

    def test_stream_and_get_returns_job_id_and_handle(self):
        mock = MockSky(default_scenario=Scenario.happy_path())
        task = mock.Task(name="test-cluster", run="echo hi")
        req_id = mock.launch(task, cluster_name="gb-test123")
        job_id, handle = mock.stream_and_get(req_id)
        assert isinstance(job_id, int)
        assert handle is not None

    def test_resources_returns_mock_object(self):
        mock = MockSky()
        res = mock.Resources(infra="aws", accelerators="A100:1")
        assert res is not None

    def test_task_accepts_resources(self):
        mock = MockSky()
        res = mock.Resources(infra="aws")
        task = mock.Task(name="test", run="echo", resources=res)
        assert task is not None

    def test_storage_and_storage_mode(self):
        mock = MockSky()
        storage = mock.Storage(source="s3://bucket", mode=mock.StorageMode.MOUNT)
        assert storage is not None

    def test_storage_mode_getitem(self):
        mock = MockSky()
        assert mock.StorageMode["MOUNT"] == "MOUNT"
        assert mock.StorageMode["COPY"] == "COPY"


class TestMockSkyPolling:
    def test_job_status_then_get_advances_scenario(self):
        scenario = Scenario.happy_path()
        mock = MockSky(default_scenario=scenario)
        task = mock.Task(name="t", run="echo")
        mock.launch(task, cluster_name="gb-poll1")

        # Poll 1: PENDING
        req = mock.job_status("gb-poll1", job_ids=[1])
        result = mock.get(req)
        assert result[1] == MockJobStatus("PENDING", is_terminal=False)

        # Poll 2: RUNNING
        req = mock.job_status("gb-poll1", job_ids=[1])
        result = mock.get(req)
        assert result[1] == MockJobStatus("RUNNING", is_terminal=False)

        # Poll 3: SUCCEEDED
        req = mock.job_status("gb-poll1", job_ids=[1])
        result = mock.get(req)
        assert result[1] == MockJobStatus("SUCCEEDED", is_terminal=True)

    def test_terminal_status_has_is_terminal_true(self):
        mock = MockSky(default_scenario=Scenario.happy_path())
        mock.launch(mock.Task(run="x"), cluster_name="gb-term")
        for _ in range(3):
            req = mock.job_status("gb-term", job_ids=[1])
            result = mock.get(req)
        assert result[1].is_terminal() is True

    def test_non_terminal_status_has_is_terminal_false(self):
        mock = MockSky(default_scenario=Scenario.happy_path())
        mock.launch(mock.Task(run="x"), cluster_name="gb-nonterm")
        req = mock.job_status("gb-nonterm", job_ids=[1])
        result = mock.get(req)
        assert result[1].is_terminal() is False

    def test_scenario_exhausted_repeats_last_step(self):
        mock = MockSky(default_scenario=Scenario.happy_path())
        mock.launch(mock.Task(run="x"), cluster_name="gb-exhaust")
        # Advance past all 3 steps
        for _ in range(5):
            req = mock.job_status("gb-exhaust", job_ids=[1])
            result = mock.get(req)
        assert result[1] == MockJobStatus("SUCCEEDED", is_terminal=True)

    def test_unknown_cluster_uses_default_scenario(self):
        mock = MockSky(default_scenario=Scenario.happy_path())
        # launch auto-assigns default scenario
        mock.launch(mock.Task(run="x"), cluster_name="gb-unknown")
        req = mock.job_status("gb-unknown", job_ids=[1])
        result = mock.get(req)
        assert result[1] == MockJobStatus("PENDING", is_terminal=False)


class TestMockSkyCleanup:
    def test_download_logs_returns_configured_path(self):
        scenario = Scenario(
            cloud="aws",
            steps=[
                ScenarioStep(status="RUNNING", is_terminal=False),
                ScenarioStep(
                    status="SUCCEEDED",
                    is_terminal=True,
                    logs={"1": "/tmp/sky_logs/job-1"},
                ),
            ],
        )
        mock = MockSky()
        mock.set_scenario("gb-logs", scenario)
        mock.launch(mock.Task(run="x"), cluster_name="gb-logs")
        # Advance through both steps
        for _ in range(2):
            req = mock.job_status("gb-logs", job_ids=[1])
            mock.get(req)
        result = mock.download_logs("gb-logs", job_ids=["1"])
        assert result == {"1": "/tmp/sky_logs/job-1"}

    def test_download_logs_returns_empty_when_no_logs(self):
        mock = MockSky(default_scenario=Scenario.happy_path())
        mock.launch(mock.Task(run="x"), cluster_name="gb-nologs")
        result = mock.download_logs("gb-nologs", job_ids=["1"])
        assert result == {}

    def test_down_returns_request_id(self):
        mock = MockSky(default_scenario=Scenario.happy_path())
        mock.launch(mock.Task(run="x"), cluster_name="gb-down")
        req_id = mock.down("gb-down", purge=True)
        assert isinstance(req_id, str)
        result = mock.get(req_id)
        assert result is None
