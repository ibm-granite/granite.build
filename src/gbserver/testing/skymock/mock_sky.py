from __future__ import annotations

import uuid
from typing import Optional
from unittest.mock import MagicMock

from gbserver.testing.skymock.scenario import Scenario


class MockJobStatus:
    def __init__(self, name: str, is_terminal: bool):
        self._name = name
        self._is_terminal = is_terminal

    def is_terminal(self) -> bool:
        return self._is_terminal

    def __str__(self) -> str:
        return f"JobStatus.{self._name}"

    def __repr__(self) -> str:
        return f"MockJobStatus({self._name!r}, is_terminal={self._is_terminal})"

    def __eq__(self, other) -> bool:
        if isinstance(other, MockJobStatus):
            return self._name == other._name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._name)


class _StorageMode:
    """Mock for sky.StorageMode with attribute and item access."""

    MOUNT = "MOUNT"
    COPY = "COPY"

    def __getitem__(self, key: str) -> str:
        if key in ("MOUNT", "COPY"):
            return key
        raise KeyError(f"Invalid StorageMode: {key}")


class MockSky:
    """Drop-in mock replacement for the sky module."""

    def __init__(self, default_scenario: Optional[Scenario] = None):
        self._default_scenario = default_scenario
        self._scenarios: dict[str, Scenario] = {}
        self._positions: dict[str, int] = {}
        self._request_results: dict[str, tuple] = {}
        self._clusters: dict[str, dict] = {}
        self.StorageMode = _StorageMode()

    def set_scenario(self, cluster_name: str, scenario: Scenario) -> None:
        """Associate a scenario with a cluster name."""
        self._scenarios[cluster_name] = scenario

    def _get_scenario(self, cluster_name: str) -> Scenario:
        """Get scenario for cluster, assigning default if needed."""
        if cluster_name not in self._scenarios:
            if self._default_scenario is None:
                raise ValueError(
                    f"No scenario for cluster {cluster_name!r} and no default_scenario set"
                )
            self._scenarios[cluster_name] = self._default_scenario
        return self._scenarios[cluster_name]

    def Resources(self, **kwargs) -> MagicMock:  # noqa: N802
        """Mock sky.Resources constructor."""
        mock = MagicMock()
        for key, value in kwargs.items():
            setattr(mock, key, value)
        return mock

    def Task(self, **kwargs) -> MagicMock:  # noqa: N802
        """Mock sky.Task constructor."""
        mock = MagicMock()
        for key, value in kwargs.items():
            setattr(mock, key, value)
        mock.set_file_mounts = MagicMock()
        mock.set_storage_mounts = MagicMock()
        return mock

    def Storage(self, **kwargs) -> MagicMock:  # noqa: N802
        """Mock sky.Storage constructor."""
        mock = MagicMock()
        for key, value in kwargs.items():
            setattr(mock, key, value)
        return mock

    def launch(self, task, cluster_name: Optional[str] = None, **kwargs) -> str:
        """Mock sky.launch — stores cluster and returns a request_id."""
        if cluster_name is None:
            cluster_name = f"gb-{uuid.uuid4().hex[:8]}"
        self._get_scenario(cluster_name)
        self._clusters[cluster_name] = {"task": task, **kwargs}
        if cluster_name not in self._positions:
            self._positions[cluster_name] = 0
        request_id = str(uuid.uuid4())
        self._request_results[request_id] = ("launch", cluster_name)
        return request_id

    def stream_and_get(self, request_id: str) -> tuple[int, MagicMock]:
        """Mock sky.stream_and_get — returns (job_id, handle)."""
        self._request_results.pop(request_id, None)
        return (1, MagicMock())

    def job_status(self, cluster_name: str, job_ids=None) -> str:
        """Mock sky.job_status — returns a request_id for later get()."""
        scenario = self._get_scenario(cluster_name)
        position = self._positions.get(cluster_name, 0)
        # Cap at last step
        position = min(position, len(scenario.steps) - 1)
        step = scenario.steps[position]

        job_id = job_ids[0] if job_ids else 1
        status = MockJobStatus(step.status, is_terminal=step.is_terminal)
        result_dict = {job_id: status}

        request_id = str(uuid.uuid4())
        self._request_results[request_id] = ("job_status", result_dict, cluster_name)
        return request_id

    def get(self, request_id: str):
        """Mock sky.get — retrieves stored result and advances scenario."""
        entry = self._request_results.pop(request_id)
        if entry[0] == "job_status":
            _, result_dict, cluster_name = entry
            scenario = self._scenarios[cluster_name]
            position = self._positions.get(cluster_name, 0)
            # Advance position, capping at last step
            if position < len(scenario.steps) - 1:
                self._positions[cluster_name] = position + 1
            return result_dict
        return entry[1]
