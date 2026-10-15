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

pytestmark = pytest.mark.standalone

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

    def count_events_by_tags(
        self, tags: List[str], required_tags: Optional[List[str]] = None
    ) -> int:
        total, _ = self.search_lineage_by_tags(tags, limit=len(self.events), offset=0)
        return total

    def count_runs_by_tags(
        self, tags: List[str], required_tags: Optional[List[str]] = None
    ) -> int:
        total, _ = self.search_lineage_by_tags(tags, limit=len(self.events), offset=0)
        return total

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

    def filter_unrecorded(
        self, target_ids: set[str], expected_counts=None, on_query_error=None
    ) -> set[str]:
        # Mirror the real service: count the distinct runs carrying each
        # ``target_id=<uuid>`` tag on emitted events, then treat a candidate as
        # recorded only when its run count meets its expected count. A ``None``
        # expected count (or a missing key) falls back to presence (>=1 run).
        run_counts: dict[str, int] = {}
        seen: set = set()  # (target_id, run_id) so each run counts once per target
        for ev in self.events.values():
            run = ev.get("run", {})
            run_id = run.get("runId", "")
            target_id = run.get("facets", {}).get("tags", {}).get("target_id")
            if not target_id or (target_id, run_id) in seen:
                continue
            seen.add((target_id, run_id))
            run_counts[target_id] = run_counts.get(target_id, 0) + 1
        recorded: set[str] = set()
        for tid in target_ids:
            count = run_counts.get(tid, 0)
            if count == 0:
                continue
            expected = (expected_counts or {}).get(tid)
            if expected is None or count >= expected:
                recorded.add(tid)
        return set(target_ids) - recorded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SAMPLE_EVENT = {
    "eventType": "START",
    "eventTime": "2024-04-15T10:30:00.000Z",
    "run": {
        "runId": "test-run-001",
        # owner matches the synthetic apikey user (GBSERVER_API_USER, default
        # "standalone") so the real has_space_member_access grants the caller
        # access via the owner path — exercising the gate rather than bypassing it.
        "facets": {
            "tags": {"env": "dev", "team": "ml"},
            "job_details": {"owner": "standalone"},
        },
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
    # Merge a ``run`` override into the base run dict rather than replacing it,
    # so the positional ``run_id`` is preserved unless the caller overrides
    # ``runId`` explicitly. A plain ``ev.update(overrides)`` would drop it.
    run_override = overrides.pop("run", None)
    if run_override is not None:
        ev["run"] = {**ev["run"], **run_override}
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
                        "facets": {
                            "tags": {"env": "dev"},
                            "job_details": {"owner": "standalone"},
                        },
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

    def test_search_lineage_owner_path_grants_access(self):
        # In standalone mode ``StandaloneSpaceAccessManager.is_space_admin``
        # always returns True, so the super-admin / space-member branches of
        # ``has_space_member_access`` are always-True and cannot exclude any
        # run. The only branch a standalone test can meaningfully exercise is
        # the *owner* short-circuit, which returns access before any admin
        # check. Both runs below are owned by the caller ("standalone"), so
        # both pass via the owner path and survive the tag filter.
        self.mock_service.emit_event(
            _make_sample_event(
                "run-mine",
                run={
                    "runId": "run-mine",
                    "facets": {
                        "tags": {"env": "dev"},
                        "job_details": {"owner": "standalone"},
                    },
                },
            )
        )
        self.mock_service.emit_event(
            _make_sample_event(
                "run-mine-2",
                run={
                    "runId": "run-mine-2",
                    "facets": {
                        "tags": {"env": "dev"},
                        "job_details": {"owner": "standalone"},
                    },
                },
            )
        )
        response = self.client.post("api/v1/lineage/search", json={"tags": ["env=dev"]})
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        returned_run_ids = {run["run"]["runId"] for run in body["runs"]}
        assert returned_run_ids == {"run-mine", "run-mine-2"}
