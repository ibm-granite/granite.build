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
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from gbserver.lineage.openlineage_service import LineageService, LineageServiceFactory

pytestmark = pytest.mark.ibm

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

    def get_artifact_graph(
        self,
        artifact_name: Optional[str] = None,
        artifact_url: Optional[str] = None,
        max_depth: int = 10,
        direction: str = "downstream",
    ) -> Optional[Dict]:
        if artifact_name == "not-found:v0":
            return None
        if artifact_url == "https://huggingface.co/org/not-found":
            return None
        if not artifact_name and not artifact_url:
            return None

        display_name = artifact_name or artifact_url
        root_id = f"entity/project/{display_name}"
        root_node = {
            "id": root_id,
            "node_type": "artifact",
            "name": display_name,
            "artifact_type": "model",
            "is_root": True,
            "metadata": {},
        }
        nodes = [root_node]
        edges = []

        if max_depth > 1 and direction in ("downstream", "both"):
            run_id = "entity/project/run-123"
            nodes.append(
                {
                    "id": run_id,
                    "node_type": "run",
                    "name": "tunedmodel",
                    "artifact_type": None,
                    "is_root": False,
                    "metadata": {
                        "run_id": "run-123",
                        "state": "finished",
                        "created_at": "2025-03-19T18:00:00",
                    },
                }
            )
            edges.append({"source": root_id, "target": run_id})

            output_id = "entity/project/output-model:v0"
            nodes.append(
                {
                    "id": output_id,
                    "node_type": "artifact",
                    "name": "output-model:v0",
                    "artifact_type": "model",
                    "is_root": False,
                    "metadata": {},
                }
            )
            edges.append({"source": run_id, "target": output_id})

        if max_depth > 1 and direction in ("upstream", "both"):
            producer_run_id = "entity/project/run-000"
            nodes.append(
                {
                    "id": producer_run_id,
                    "node_type": "run",
                    "name": "base-training",
                    "artifact_type": None,
                    "is_root": False,
                    "metadata": {
                        "run_id": "run-000",
                        "state": "finished",
                        "created_at": "2025-03-18T10:00:00",
                    },
                }
            )
            edges.append({"source": root_id, "target": producer_run_id})

            input_id = "entity/project/raw-data:v0"
            nodes.append(
                {
                    "id": input_id,
                    "node_type": "artifact",
                    "name": "raw-data:v0",
                    "artifact_type": "dataset",
                    "is_root": False,
                    "metadata": {},
                }
            )
            edges.append({"source": producer_run_id, "target": input_id})

        return {
            "root_id": root_id,
            "nodes": nodes,
            "edges": edges,
            "truncated": max_depth <= 1,
        }


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

    def test_existing_build_endpoint_still_works(self):
        response = self.client.get("api/v1/lineage/build/non-existent-uuid")
        assert response.status_code == 404

    def test_existing_target_endpoint_still_works(self):
        response = self.client.get("api/v1/lineage/target/non-existent-uuid")
        assert response.status_code == 404

    # --- Artifact Graph Endpoint ---

    def test_get_artifact_graph(self):
        response = self.client.post(
            "api/v1/lineage/artifact",
            json={"artifact_name": "my-model:v0"},
        )
        assert response.status_code == 200
        body = response.json()
        assert "root_id" in body
        assert "nodes" in body
        assert "edges" in body
        assert body["nodes"][0]["is_root"] is True
        assert body["nodes"][0]["node_type"] == "artifact"
        assert len(body["nodes"]) == 3
        assert len(body["edges"]) == 2

    def test_get_artifact_graph_not_found(self):
        response = self.client.post(
            "api/v1/lineage/artifact",
            json={"artifact_name": "not-found:v0"},
        )
        assert response.status_code == 404

    def test_get_artifact_graph_invalid_direction(self):
        response = self.client.post(
            "api/v1/lineage/artifact",
            json={"artifact_name": "my-model:v0", "direction": "invalid"},
        )
        assert response.status_code == 400

    def test_get_artifact_graph_max_depth(self):
        response = self.client.post(
            "api/v1/lineage/artifact",
            json={"artifact_name": "my-model:v0", "max_depth": 1},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["truncated"] is True
        assert len(body["nodes"]) == 1

    def test_get_artifact_graph_upstream(self):
        response = self.client.post(
            "api/v1/lineage/artifact",
            json={"artifact_name": "my-model:v0", "direction": "upstream"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["nodes"][0]["is_root"] is True
        assert len(body["nodes"]) == 3
        assert any(n["name"] == "base-training" for n in body["nodes"])

    def test_get_artifact_graph_both_directions(self):
        response = self.client.post(
            "api/v1/lineage/artifact",
            json={"artifact_name": "my-model:v0", "direction": "both"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["nodes"][0]["is_root"] is True
        assert len(body["nodes"]) == 5
        assert len(body["edges"]) == 4
        node_names = {n["name"] for n in body["nodes"]}
        assert "tunedmodel" in node_names
        assert "base-training" in node_names

    def test_get_artifact_graph_by_url(self):
        response = self.client.post(
            "api/v1/lineage/artifact",
            json={
                "artifact_url": "https://huggingface.co/buckets/ibm-research/test-bucket"
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["nodes"][0]["is_root"] is True
        assert len(body["nodes"]) == 3

    def test_get_artifact_graph_by_url_not_found(self):
        response = self.client.post(
            "api/v1/lineage/artifact",
            json={"artifact_url": "https://huggingface.co/org/not-found"},
        )
        assert response.status_code == 404

    def test_get_artifact_graph_no_params(self):
        response = self.client.post(
            "api/v1/lineage/artifact",
            json={},
        )
        assert response.status_code == 400
