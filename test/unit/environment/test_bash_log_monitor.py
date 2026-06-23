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

"""Unit tests for the Bash environment's log_monitor.

Regression coverage for the bug where the Bash log_monitor read the launched
process's (already drained) stdout pipe instead of the job.log file the wrapper
tees workload output to. The artifact marker (LLMB_ARTIFACT_ID) is written to
job.log, so the monitor must tail that file and emit a
NEWARTIFACT_IN_ENVIRONMENT_EVENT.
"""

import asyncio
from unittest.mock import patch

import pytest

from gbserver.types.buildevent import BuildEventType, EntityRunMetadata

# Mirrors the lora-finetune step.yaml log_monitor config: matches
# "LLMB_ARTIFACT_ID:<id> LLMB_ARTIFACT_PATH:<path>" lines.
ARTIFACT_EVENT_CONFIGS = [
    {
        "event_type": "NEWARTIFACT_IN_ENVIRONMENT_EVENT",
        "line_regex": "LLMB_ARTIFACT_ID:.* LLMB_ARTIFACT_PATH:.*",
        "is_json": False,
        "event_fields": [
            {"field_name": "binding_id", "field_regex": "(?<=LLMB_ARTIFACT_ID:)[^ ]+"},
            {
                "field_name": "path",
                "field_regex": "(?<=LLMB_ARTIFACT_PATH:).*",
                "is_data": True,
            },
            {
                "field_name": "binding",
                "field_value_template": '{ "path": "{{ fields.data.path }}" }',
                "is_json": True,
            },
        ],
    },
]


def _make_bash():
    from gbserver.environment.bash import Bash

    return Bash(event_q=asyncio.Queue())


@pytest.mark.standalone
@pytest.mark.asyncio
async def test_monitor_log_monitor_emits_newartifact_event_from_job_log(tmp_path):
    """monitor_log_monitor tails job.log and emits a NEWARTIFACT event for the
    artifact marker line written there (not to the process stdout pipe)."""
    bash = _make_bash()
    launch_id = "launch-test-1"

    artifact_path = tmp_path / "outputs" / "adapter"
    job_log = tmp_path / "job.log"
    job_log.write_text(
        "some training output\n"
        f"LLMB_ARTIFACT_ID:adapter LLMB_ARTIFACT_PATH:{artifact_path}\n"
        "workload script finished successfully\n"
    )
    bash.log_paths[launch_id] = str(job_log)

    event_q: asyncio.Queue = asyncio.Queue()
    # The launch task normally sets this when the workload exits; set it up front
    # so the stream reads the existing (complete) file and then ends.
    bash._get_launch_stopped_event(launch_id).set()

    await asyncio.wait_for(
        bash.monitor_log_monitor(
            launch_id=launch_id,
            event_q=event_q,
            entityrun_metadata=EntityRunMetadata(build_id="b1", target_name="t1"),
            event_configs=ARTIFACT_EVENT_CONFIGS,
        ),
        timeout=30,
    )

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())

    artifact_events = [
        e for e in events if e.type is BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT
    ]
    assert len(artifact_events) == 1, f"expected one NEWARTIFACT event, got {events}"
    assert artifact_events[0].payload.binding_id == "adapter"
    assert artifact_events[0].payload.binding == {"path": str(artifact_path)}


@pytest.mark.standalone
@pytest.mark.asyncio
async def test_monitor_log_monitor_no_event_without_marker(tmp_path):
    """A job.log with no artifact marker yields no NEWARTIFACT event."""
    bash = _make_bash()
    launch_id = "launch-test-2"

    job_log = tmp_path / "job.log"
    job_log.write_text("just some output\nno markers here\n")
    bash.log_paths[launch_id] = str(job_log)

    event_q: asyncio.Queue = asyncio.Queue()
    bash._get_launch_stopped_event(launch_id).set()

    await asyncio.wait_for(
        bash.monitor_log_monitor(
            launch_id=launch_id,
            event_q=event_q,
            entityrun_metadata=EntityRunMetadata(build_id="b2", target_name="t2"),
            event_configs=ARTIFACT_EVENT_CONFIGS,
        ),
        timeout=30,
    )

    events = []
    while not event_q.empty():
        events.append(event_q.get_nowait())
    artifact_events = [
        e for e in events if e.type is BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT
    ]
    assert artifact_events == []


@pytest.mark.standalone
@pytest.mark.asyncio
async def test_pushasset_filestore_copies_binding_path_to_uri(tmp_path):
    """A {"path": ...} binding has its path (not the dict) copied to the file
    URI destination. For a directory artifact, its CONTENTS land directly at the
    output URI (no extra nesting of the source dir under it)."""
    from gbcommon.uri.uri import URI

    bash = _make_bash()

    src_dir = tmp_path / "step-outputs" / "adapter"
    src_dir.mkdir(parents=True)
    (src_dir / "adapter_model.safetensors").write_text("weights")
    dest_dir = tmp_path / "outputs" / "lora-finetune" / "adapter_abcd1234"

    binding = {"path": str(src_dir)}
    uri = URI.get_uri(f"file:{dest_dir}")

    result = await bash.pushasset_filestore(binding=binding, uri=uri)

    assert result is uri
    # The source dir's CONTENTS were copied into dest (not nested as dest/adapter/).
    assert (dest_dir / "adapter_model.safetensors").read_text() == "weights"
    assert not (dest_dir / "adapter").exists()


@pytest.mark.standalone
@pytest.mark.asyncio
async def test_pushasset_filestore_raises_on_copy_failure():
    """A failed copy raises (instead of silently marking the artifact pushed)."""
    from gbcommon.uri.uri import URI

    bash = _make_bash()
    binding = {"path": "/some/source/adapter"}
    uri = URI.get_uri("file:outputs/lora-finetune/adapter_abcd1234/")

    # The push goes through FileURI.pull(), which calls sync_or_copy in the
    # gbcommon.uri.file module — patch it there.
    with patch(
        "gbcommon.uri.file.sync_or_copy",
        side_effect=ValueError("rsync failed"),
    ) as mock_copy:
        with pytest.raises(ValueError, match="rsync failed"):
            await bash.pushasset_filestore(binding=binding, uri=uri)

    # The path (not the dict) is passed as the rsync source, with raise_errors=True.
    # (No trailing slash is appended because the path doesn't exist on disk.)
    args, kwargs = mock_copy.call_args
    assert args[0] == "/some/source/adapter"
    assert kwargs.get("raise_errors") is True
