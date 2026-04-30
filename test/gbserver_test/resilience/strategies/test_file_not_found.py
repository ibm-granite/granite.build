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

"""
Tests for FileNotFoundRetryStrategy.

Related Issues:
- #1829: vLLM eval FileNotFoundError on files that exist on NFS mount
"""

import json
from typing import Self

import pytest

from gbserver.resilience.strategies.file_not_found import FileNotFoundRetryStrategy
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventType,
    EntityRunMetadata,
)


def _make_event(msg: str) -> BuildEvent:
    """Helper to create a MESSAGE_EVENT with the given message."""
    return BuildEvent(
        run_metadata=EntityRunMetadata(build_id="test-build-id"),
        type=BuildEventType.MESSAGE_EVENT,
        payload=BuildEventMessagePayload(msg=msg),
    )


def _wrap_json(data: dict) -> str:
    """Wrap dict as markdown JSON code block."""
    return f"```json\n{json.dumps(data)}\n```"


class TestFileNotFoundRetryStrategy:
    """Tests for FileNotFoundError detection and node extraction."""

    # ── Detection tests ─────────────────────────────────────────────

    def test_detects_file_not_found_no_such_file(self: Self) -> None:
        """Test detection of standard FileNotFoundError."""
        strategy = FileNotFoundRetryStrategy()
        event = _make_event(
            "FileNotFoundError: No such file or directory: "
            "'/gb-lakehouse-prod-read-only/models/model-00011-of-00016.safetensors'"
        )
        assert strategy.should_retry(event)

    def test_detects_file_not_found_errno2(self: Self) -> None:
        """Test detection of FileNotFoundError with errno format."""
        strategy = FileNotFoundRetryStrategy()
        event = _make_event(
            "FileNotFoundError: [Errno 2] No such file or directory: '/path/to/file'"
        )
        assert strategy.should_retry(event)

    def test_detects_vllm_safetensors_error(self: Self) -> None:
        """Test detection of the exact vLLM error from issue #1829."""
        strategy = FileNotFoundRetryStrategy()
        event = _make_event(
            "ERROR 03-01 15:44:06 [multiproc_executor.py:743] "
            "FileNotFoundError: No such file or directory: "
            "/gb-lakehouse-prod-read-only/models/granite_dot_build/public/shared/"
            "gb_intermediate_checkpoint_pggnv0tu_safetensors-53792/"
            "granite-dot-build/model-00011-of-00016.safetensors"
        )
        assert strategy.should_retry(event)

    def test_detects_file_not_found_in_traceback(self: Self) -> None:
        """Test detection within a Python traceback."""
        strategy = FileNotFoundRetryStrategy()
        event = _make_event(
            "Traceback (most recent call last):\n"
            '  File "/step_venv/lib64/python3.12/site-packages/vllm/model_executor/'
            'model_loader/weight_utils.py", line 626, in safetensors_weights_iterator\n'
            '    with safe_open(st_file, framework="pt") as f:\n'
            "FileNotFoundError: No such file or directory: '/path/to/model.safetensors'"
        )
        assert strategy.should_retry(event)

    # ── Non-matching tests ──────────────────────────────────────────

    def test_does_not_retry_on_normal_message(self: Self) -> None:
        """Test that normal messages don't trigger retry."""
        strategy = FileNotFoundRetryStrategy()
        event = _make_event("All pods are ready and running")
        assert not strategy.should_retry(event)

    def test_does_not_retry_on_other_errors(self: Self) -> None:
        """Test that non-FileNotFoundError exceptions don't trigger retry."""
        strategy = FileNotFoundRetryStrategy()
        event = _make_event("PermissionError: [Errno 13] Permission denied: '/path'")
        assert not strategy.should_retry(event)

    def test_does_not_retry_on_status_event(self: Self) -> None:
        """Test that STATUS_EVENTs are not processed."""
        strategy = FileNotFoundRetryStrategy()
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventMessagePayload(msg="FileNotFoundError: No such file or directory"),
        )
        assert not strategy.should_retry(event)

    # ── Node extraction tests ───────────────────────────────────────

    def test_extract_node_from_json_payload(self: Self) -> None:
        """Test node extraction from JSON in message payload."""
        strategy = FileNotFoundRetryStrategy()
        msg_data = {
            "node_name": "gpu-worker-3",
            "error": "FileNotFoundError: No such file or directory",
        }
        event = _make_event(f"FileNotFoundError: No such file or directory\n{_wrap_json(msg_data)}")
        nodes = strategy.extract_nodes_to_avoid(event)
        assert nodes == {"gpu-worker-3"}

    def test_extract_node_from_pod_placement(self: Self) -> None:
        """Test node extraction from pod_placement field."""
        strategy = FileNotFoundRetryStrategy()
        msg_data = {
            "state": "Failed",
            "pod_placement": {
                "vllm-pod-1": "gpu-worker-3",
                "vllm-pod-2": "gpu-worker-5",
            },
        }
        event = _make_event(_wrap_json(msg_data))
        nodes = strategy.extract_nodes_to_avoid(event)
        assert nodes == {"gpu-worker-3", "gpu-worker-5"}

    def test_extract_no_nodes_when_no_json(self: Self) -> None:
        """Test that empty set returned when no node info available."""
        strategy = FileNotFoundRetryStrategy()
        event = _make_event("FileNotFoundError: No such file or directory: '/path/to/file'")
        nodes = strategy.extract_nodes_to_avoid(event)
        assert nodes == set()

    def test_accepts_object_types_is_false(self: Self) -> None:
        """Test that this strategy doesn't filter by object type."""
        assert FileNotFoundRetryStrategy.accepts_object_types is False
