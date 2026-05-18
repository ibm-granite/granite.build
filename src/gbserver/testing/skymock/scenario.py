from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScenarioStep:
    status: str
    is_terminal: bool
    error: Optional[str] = None
    logs: Optional[dict] = None


@dataclass
class Scenario:
    steps: list[ScenarioStep]
    cloud: str = "aws"

    @classmethod
    def happy_path(cls, cloud: str = "aws") -> Scenario:
        return cls(
            cloud=cloud,
            steps=[
                ScenarioStep(status="PENDING", is_terminal=False),
                ScenarioStep(status="RUNNING", is_terminal=False),
                ScenarioStep(status="SUCCEEDED", is_terminal=True),
            ],
        )

    @classmethod
    def failure(cls, cloud: str = "aws", error: Optional[str] = None) -> Scenario:
        if error is None:
            error = f"{cloud.upper()} ResourceExhausted: capacity unavailable"
        return cls(
            cloud=cloud,
            steps=[
                ScenarioStep(status="PENDING", is_terminal=False),
                ScenarioStep(status="RUNNING", is_terminal=False),
                ScenarioStep(status="FAILED", is_terminal=True, error=error),
            ],
        )

    @classmethod
    def preemption_then_recovery(cls, cloud: str = "aws") -> Scenario:
        return cls(
            cloud=cloud,
            steps=[
                ScenarioStep(status="PENDING", is_terminal=False),
                ScenarioStep(status="RUNNING", is_terminal=False),
                ScenarioStep(status="PREEMPTED", is_terminal=False),
                ScenarioStep(status="PENDING", is_terminal=False),
                ScenarioStep(status="RUNNING", is_terminal=False),
                ScenarioStep(status="SUCCEEDED", is_terminal=True),
            ],
        )

    @classmethod
    def cross_cloud_failover(
        cls, primary: str = "aws", fallback: str = "gcp"
    ) -> tuple[Scenario, Scenario]:
        primary_scenario = cls(
            cloud=primary,
            steps=[
                ScenarioStep(status="PENDING", is_terminal=False),
                ScenarioStep(
                    status="FAILED",
                    is_terminal=True,
                    error=f"{primary.upper()} QuotaExceeded: no capacity in region",
                ),
            ],
        )
        fallback_scenario = cls.happy_path(cloud=fallback)
        return primary_scenario, fallback_scenario
