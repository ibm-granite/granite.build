#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from typing import Dict, List, Tuple
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from gbserver.lineage.openlineage_models import (
    LineageEvent,
    PaginatedResponse,
    RunState,
)
from gbserver.lineage.openlineage_service import LineageService, LineageServiceFactory

pytestmark = pytest.mark.g4os

_TEST_API_KEY = "test-lineage-key-12345"
_AUTH_ENV = {
    "GBSERVER_AUTH_MODE": "apikey",
    "GBSERVER_API_KEY": _TEST_API_KEY,
}
_AUTH_HEADERS = {"Authorization": f"Bearer {_TEST_API_KEY}"}


# ---------------------------------------------------------------------------
# In-memory mock service implementing the LineageService ABC
# ---------------------------------------------------------------------------
class MockLineageService(LineageService):
    def __init__(self):
        self.events: Dict[str, Dict] = {}

    def emit_event(self, event: Dict) -> None:
        run_id = event["run"]["runId"]
        self.events[run_id] = event

    def get_run_lineage(self, run_id: str) -> Dict:
        return self.events.get(run_id)

    def search_lineage_by_tags(
        self, tags: List[str], limit: int = 10, offset: int = 0
    ) -> Tuple[int, List[Dict]]:
        if not tags:
            all_events = list(self.events.values())
        else:
            tag_set = set(tags)
            all_events = []
            for ev in self.events.values():
                run_facets = ev.get("run", {}).get("facets", {})
                ev_tags = run_facets.get("tags", {})
                ev_tag_strings = {
                    f"{k}={v}" for k, v in ev_tags.items() if not k.startswith("_")
                }
                if tag_set & ev_tag_strings:
                    all_events.append(ev)
        total = len(all_events)
        return total, all_events[offset : offset + limit]

    def search_runs_by_artifact(
        self, repo_id: str, limit: int = 10, offset: int = 0
    ) -> Tuple[int, List[Dict]]:
        matching = []
        for ev in self.events.values():
            for ds in ev.get("inputs", []) + ev.get("outputs", []):
                if repo_id == ds.get("facets", {}).get("repo_id", ""):
                    matching.append(ev)
                    break
        total = len(matching)
        return total, matching[offset : offset + limit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SAMPLE_EVENT = {
    "eventType": "START",
    "eventTime": "2024-04-15T10:30:00.000Z",
    "run": {
        "runId": "test-run-001",
        "facets": {"tags": {"env": "dev", "team": "ml"}},
    },
    "job": {"namespace": "granite-ml", "name": "train_model", "facets": {}},
    "inputs": [
        {
            "namespace": "s3://data",
            "name": "training.parquet",
            "facets": {"repo_id": "org/input-data"},
        }
    ],
    "outputs": [
        {
            "namespace": "huggingface://models",
            "name": "org/granite-model",
            "facets": {"repo_id": "org/granite-model"},
        }
    ],
    "producer": "https://github.com/granite-lineage/producer",
}


def _make_sample_event(run_id: str = "test-run-001", **overrides) -> dict:
    ev = {**_SAMPLE_EVENT, "run": {**_SAMPLE_EVENT["run"], "runId": run_id}}
    ev.update(overrides)
    return ev


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------
class TestLineageServiceFactory:
    def test_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported lineage provider"):
            LineageServiceFactory.create("nonexistent")


# ---------------------------------------------------------------------------
# API endpoint tests with mocked service
# ---------------------------------------------------------------------------
class TestOpenLineageAPI:
    @pytest.fixture(autouse=True)
    def _setup_client(self):
        self.mock_service = MockLineageService()
        with (
            patch.dict(os.environ, _AUTH_ENV, clear=False),
            patch(
                "gbserver.api.lineage._get_openlineage_service",
                return_value=self.mock_service,
            ),
        ):
            from gbserver.api.root_api import root_api

            self.client = TestClient(root_api, headers=_AUTH_HEADERS)
            yield

    def test_ingest_lineage_event(self):
        response = self.client.post("api/v1/lineage/", json=_SAMPLE_EVENT)
        assert response.status_code == 200
        assert response.json() == {"status": "accepted"}
        assert "test-run-001" in self.mock_service.events

    def test_ingest_lineage_event_invalid_body(self):
        response = self.client.post("api/v1/lineage/", json={"invalid": "data"})
        assert response.status_code == 422

    def test_get_lineage_event_found(self):
        self.mock_service.emit_event(_make_sample_event("run-abc"))
        response = self.client.get("api/v1/lineage/run-abc")
        assert response.status_code == 200
        body = response.json()
        assert body["run"]["runId"] == "run-abc"
        assert body["job"]["name"] == "train_model"

    def test_get_lineage_event_not_found(self):
        response = self.client.get("api/v1/lineage/nonexistent-id")
        assert response.status_code == 404

    def test_search_lineage_by_tags(self):
        self.mock_service.emit_event(_make_sample_event("run-1"))
        response = self.client.post("api/v1/lineage/search", json={"tags": ["env=dev"]})
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["count"] == 1
        assert len(body["runs"]) == 1

    def test_search_lineage_by_tags_empty(self):
        response = self.client.post(
            "api/v1/lineage/search", json={"tags": ["no=match"]}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["runs"] == []

    def test_search_lineage_by_tags_pagination(self):
        for i in range(5):
            self.mock_service.emit_event(
                _make_sample_event(
                    f"run-{i}",
                    run={
                        "runId": f"run-{i}",
                        "facets": {"tags": {"env": "dev"}},
                    },
                )
            )

        response = self.client.post(
            "api/v1/lineage/search",
            json={"tags": ["env=dev"], "limit": 2, "offset": 1},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 5
        assert body["count"] == 2
        assert body["limit"] == 2
        assert body["offset"] == 1

    def test_get_lineage_by_artifact(self):
        self.mock_service.emit_event(_make_sample_event("run-art"))
        response = self.client.post(
            "api/v1/lineage/artifact/runs",
            json={"repo_id": "org/granite-model"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert len(body["runs"]) == 1

    def test_get_lineage_by_artifact_no_match(self):
        self.mock_service.emit_event(_make_sample_event("run-x"))
        response = self.client.post(
            "api/v1/lineage/artifact/runs",
            json={"repo_id": "org/nonexistent"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["runs"] == []

    def test_existing_build_endpoint_still_works(self):
        response = self.client.get("api/v1/lineage/build/non-existent-uuid")
        assert response.status_code == 404

    def test_existing_target_endpoint_still_works(self):
        response = self.client.get("api/v1/lineage/target/non-existent-uuid")
        assert response.status_code == 404
