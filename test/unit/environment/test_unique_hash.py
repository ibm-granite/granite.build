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

"""Tests for the ``{{ unique_hash }}`` output-URI template variable."""

import asyncio
from types import SimpleNamespace

import pytest

from gbserver.environment.environment import Environment
from gbserver.resilience import RetryHandler
from gbserver.types.buildevent import (
    ArtifactEventPayload,
    BuildEvent,
    BuildEventType,
    EntityRunMetadata,
    MultiArtifactEventPayload,
)
from gbserver.utils.template import fill_template
from gbserver.utils.utils import short_alphanumeric_lower_hash

URI_TEMPLATE = "file:///tmp/out/eval_{{ unique_hash }}"


class _Rendered(Exception):
    """Raised by the stub store lookup to capture the rendered push URI."""


def _render_push_uri(run_metadata: EntityRunMetadata) -> str:
    def _capture(uri, **kwargs):
        raise _Rendered(str(uri))

    stub = SimpleNamespace(_get_storeconfig=_capture)
    with pytest.raises(_Rendered) as exc:
        Environment.pushasset(
            stub,  # type: ignore[arg-type]
            task_group=None,  # type: ignore[arg-type]
            binding={"path": "/tmp/out"},
            uristr=URI_TEMPLATE,
            run_metadata=run_metadata,
        )
    return str(exc.value)


def test_unique_hash_definition():
    rm = EntityRunMetadata(targetsteprun_id="tsr-1", attempt=2)
    assert rm.unique_hash() == short_alphanumeric_lower_hash("tsr-1:2")


def test_unique_hash_changes_per_attempt_and_targetsteprun():
    base = EntityRunMetadata(targetsteprun_id="tsr-1")
    assert base.attempt == 0
    assert (
        base.unique_hash() == EntityRunMetadata(targetsteprun_id="tsr-1").unique_hash()
    )
    assert (
        base.unique_hash()
        != EntityRunMetadata(targetsteprun_id="tsr-1", attempt=1).unique_hash()
    )
    assert (
        base.unique_hash() != EntityRunMetadata(targetsteprun_id="tsr-2").unique_hash()
    )


def test_attempt_round_trips_through_dict():
    rm = EntityRunMetadata(targetsteprun_id="tsr-1", attempt=3)
    assert EntityRunMetadata.from_dict(rm.to_dict()).attempt == 3
    assert EntityRunMetadata.from_dict({}).attempt == 0


def test_pushasset_renders_unique_hash():
    rm = EntityRunMetadata(targetsteprun_id="tsr-1", attempt=1)
    rendered = _render_push_uri(rm)
    assert rendered.endswith(f"eval_{rm.unique_hash()}")
    # Deterministic: a second render of the same attempt agrees.
    assert _render_push_uri(rm) == rendered


def test_non_strict_render_preserves_unique_hash():
    assert fill_template(URI_TEMPLATE, {"run_metadata": {}}) == URI_TEMPLATE


def _artifact_event(rm: EntityRunMetadata) -> BuildEvent:
    return BuildEvent(
        run_metadata=rm,
        type=BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT,
        payload=ArtifactEventPayload(binding={"path": "/tmp/out"}),
    )


def test_retry_handler_stamps_attempt_on_new_artifact():
    handler = RetryHandler(
        launch_id="l1",
        downstream_queue=asyncio.Queue(),
        environment=SimpleNamespace(),  # type: ignore[arg-type]
        max_retries=3,
    )
    shared = EntityRunMetadata(targetsteprun_id="tsr-1")
    first = _artifact_event(shared)
    handler._stamp_attempt(first)
    assert first.run_metadata.attempt == 0

    handler.retry_count = 2
    second = _artifact_event(shared)
    handler._stamp_attempt(second)
    assert second.run_metadata.attempt == 2
    # The shared monitor metadata (and earlier events) are not mutated.
    assert shared.attempt == 0
    assert first.run_metadata.attempt == 0


def test_retry_handler_stamps_multiartifact_event():
    handler = RetryHandler(
        launch_id="l1",
        downstream_queue=asyncio.Queue(),
        environment=SimpleNamespace(),  # type: ignore[arg-type]
        max_retries=3,
    )
    handler.retry_count = 1
    event = BuildEvent(
        run_metadata=EntityRunMetadata(targetsteprun_id="tsr-1"),
        type=BuildEventType.NEW_MULTIARTIFACT_IN_ENVIRONMENT_EVENT,
        payload=MultiArtifactEventPayload(artifacts=[]),
    )
    handler._stamp_attempt(event)
    assert event.run_metadata.attempt == 1


def test_retry_handler_does_not_stamp_other_events():
    handler = RetryHandler(
        launch_id="l1",
        downstream_queue=asyncio.Queue(),
        environment=SimpleNamespace(),  # type: ignore[arg-type]
        max_retries=3,
    )
    handler.retry_count = 1
    rm = EntityRunMetadata(targetsteprun_id="tsr-1")
    event = BuildEvent(run_metadata=rm, type=BuildEventType.MESSAGE_EVENT)
    handler._stamp_attempt(event)
    assert event.run_metadata is rm
    assert rm.attempt == 0
