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

import pytest
from pydantic import ValidationError

from gbserver.lineage.openlineage_models import (
    ArtifactLineageRequest,
    Dataset,
    Job,
    LineageEvent,
    PaginatedResponse,
    Run,
    RunState,
    TagSearchRequest,
)

pytestmark = pytest.mark.g4os


class TestRunState:
    def test_enum_values(self):
        assert RunState.START == "START"
        assert RunState.RUNNING == "RUNNING"
        assert RunState.COMPLETE == "COMPLETE"
        assert RunState.ABORT == "ABORT"
        assert RunState.FAIL == "FAIL"
        assert RunState.OTHER == "OTHER"
        assert len(RunState) == 6


class TestLineageEvent:
    def test_valid_full_event(self):
        event = LineageEvent(
            eventType=RunState.START,
            eventTime="2024-04-15T10:30:00.000Z",
            run=Run(runId="test-run-id", facets={"tags": {"env": "dev"}}),
            job=Job(namespace="granite-ml", name="train_model", facets={}),
            inputs=[Dataset(namespace="s3://data", name="training.parquet", facets={})],
            outputs=[Dataset(namespace="s3://models", name="model.ckpt", facets={})],
            producer="https://github.com/granite-lineage/producer",
        )
        dumped = event.model_dump()
        roundtripped = LineageEvent.model_validate(dumped)
        assert roundtripped.eventType == RunState.START
        assert roundtripped.run.runId == "test-run-id"
        assert roundtripped.job.name == "train_model"
        assert len(roundtripped.inputs) == 1
        assert len(roundtripped.outputs) == 1

    def test_minimal_event(self):
        event = LineageEvent(
            eventType=RunState.OTHER,
            eventTime="2024-01-01T00:00:00Z",
            run=Run(runId="min-run"),
            job=Job(namespace="ns", name="job"),
            producer="test-producer",
        )
        assert event.inputs == []
        assert event.outputs == []
        assert "RunEvent" in event.schemaURL

    def test_invalid_event_type(self):
        with pytest.raises(ValidationError):
            LineageEvent(
                eventType="INVALID",
                eventTime="2024-01-01T00:00:00Z",
                run=Run(runId="run"),
                job=Job(namespace="ns", name="job"),
                producer="test",
            )


class TestTagSearchRequest:
    def test_defaults(self):
        req = TagSearchRequest()
        assert req.tags == []
        assert req.limit == 10
        assert req.offset == 0

    def test_custom_values(self):
        req = TagSearchRequest(tags=["env=prod", "team=ml"], limit=50, offset=20)
        assert req.tags == ["env=prod", "team=ml"]
        assert req.limit == 50
        assert req.offset == 20


class TestArtifactLineageRequest:
    def test_construction(self):
        req = ArtifactLineageRequest(repo_id="org/my-model")
        assert req.repo_id == "org/my-model"
        assert req.limit == 10
        assert req.offset == 0

    def test_custom_pagination(self):
        req = ArtifactLineageRequest(repo_id="org/data", limit=5, offset=10)
        assert req.limit == 5
        assert req.offset == 10


class TestPaginatedResponse:
    def test_construction(self):
        resp = PaginatedResponse(
            count=3, total=10, limit=5, offset=0, runs=["a", "b", "c"]
        )
        assert resp.count == 3
        assert resp.total == 10
        assert resp.limit == 5
        assert resp.offset == 0
        assert len(resp.runs) == 3
