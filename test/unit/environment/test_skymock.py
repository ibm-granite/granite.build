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
