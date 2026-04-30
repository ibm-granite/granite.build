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
Tests for NCCLErrorRetryStrategy.
"""

import json
from typing import Self

import pytest

from gbserver.resilience.strategies.nccl_error import NCCLErrorRetryStrategy
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventType,
    EntityRunMetadata,
)


class TestNCCLErrorRetryStrategy:
    """Tests for NCCL error detection strategy."""

    def test_detects_nccl_error_3_internal(self: Self) -> None:
        """Test detection of NCCL Error 3: internal error."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="RuntimeError: NCCL Error 3: internal error - "
                "please report this issue to the NCCL developers"
            ),
        )

        assert strategy.should_retry(event)

    def test_detects_nccl_error_2_unhandled(self: Self) -> None:
        """Test detection of NCCL Error 2: unhandled system error."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="RuntimeError: NCCL Error 2: unhandled system error"
            ),
        )

        assert strategy.should_retry(event)

    def test_detects_nccl_error_in_internal_error(self: Self) -> None:
        """Test detection of NCCL error in format with internal error."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="NCCL error in: ncclCommInitRank: internal error - failed to initialize communicator"
            ),
        )

        assert strategy.should_retry(event)

    def test_detects_nccl_failure_internal_error(self: Self) -> None:
        """Test detection of NCCL failure with internal error."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="NCCL failure detected: internal error in collective operation"
            ),
        )

        assert strategy.should_retry(event)

    def test_detects_cuda_illegal_memory(self: Self) -> None:
        """Test detection of CUDA illegal memory access."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="RuntimeError: CUDA error: an illegal memory access was encountered"
            ),
        )

        assert strategy.should_retry(event)

    def test_does_not_retry_cuda_out_of_memory(self: Self) -> None:
        """Test that CUDA out of memory errors do NOT trigger retry.

        OOM errors indicate configuration issues (model/batch too large)
        rather than transient hardware failures, so retry won't help.
        """
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg="CUDA out of memory. Tried to allocate 2.00 GiB"),
        )

        # OOM should NOT trigger retry
        assert not strategy.should_retry(event)

    def test_detects_cudnn_internal_error(self: Self) -> None:
        """Test detection of CUDNN internal error."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg="CUDNN_STATUS_INTERNAL_ERROR"),
        )

        assert strategy.should_retry(event)

    def test_detects_cuda_device_assert(self: Self) -> None:
        """Test detection of CUDA device-side assert."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="RuntimeError: CUDA error: device-side assert triggered"
            ),
        )

        assert strategy.should_retry(event)

    def test_detects_vllm_illegal_address_error(self: Self) -> None:
        """Test detection of vLLM CUDA illegal address error (Issue #1691)."""
        strategy = NCCLErrorRetryStrategy()

        # Actual error from production
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="RuntimeError: Worker failed with error 'CUDA error: an illegal memory access was encountered"
            ),
        )

        assert strategy.should_retry(event)

    def test_detects_cuda_error_code_illegal_address(self: Self) -> None:
        """Test detection of CUDA error code names (Issue #1691)."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/"
            ),
        )

        assert strategy.should_retry(event)

    def test_detects_torch_accelerator_error_cuda(self: Self) -> None:
        """Test detection of torch.AcceleratorError CUDA illegal memory access.

        PyTorch may raise torch.AcceleratorError instead of RuntimeError
        for CUDA errors, particularly in newer versions.
        From production build b72ec9c4 on node dmf-nnnqh-gpu-worker-3-kvjm6.
        """
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="torch.AcceleratorError: CUDA error: an illegal memory access was encountered"
            ),
        )

        assert strategy.should_retry(event)

    def test_detects_triton_cuda_illegal_memory(self: Self) -> None:
        """Test detection of Triton-wrapped CUDA illegal memory access (vLLM format).

        vLLM workers report CUDA errors wrapped in Triton error format:
        "Triton Error [CUDA]: an illegal memory access was encountered"
        rather than the direct "CUDA error:" prefix.
        """
        strategy = NCCLErrorRetryStrategy()

        # Actual error from production (buildid ab9138ec-12b7-4bc0-9127-5c7268f3a6ec)
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="RuntimeError: Worker failed with error "
                "'Triton Error [CUDA]: an illegal memory access was encountered'"
            ),
        )

        assert strategy.should_retry(event)

    def test_detects_triton_cuda_in_json_payload(self: Self) -> None:
        """Test detection of Triton CUDA error embedded in AppWrapper JSON payload.

        The AppWrapperMonitor publishes failed pod logs inside a JSON structure
        wrapped in markdown code blocks.
        """
        strategy = NCCLErrorRetryStrategy()

        payload_data = {
            "appwrapper": "gb8ilnjfa8",
            "state": "Failed",
            "pod_placement": {"gb8ilnjfa8-job-fxr44": "dmf-nnnqh-gpu-worker-3-kvjm6"},
            "failed_pods": {
                "gb8ilnjfa8-job-fxr44": {
                    "failure-reason": "container failed with exit code non-zero",
                    "logs": {
                        "step-main": [
                            "RuntimeError: Worker failed with error "
                            "'Triton Error [CUDA]: an illegal memory access was encountered'"
                        ]
                    },
                }
            },
        }
        msg = f"\n```json\n{json.dumps(payload_data, indent=4)}\n```\n"

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg=msg),
        )

        assert strategy.should_retry(event)

        # Also verify node extraction from pod_placement
        nodes = strategy.extract_nodes_to_avoid(event)
        assert "dmf-nnnqh-gpu-worker-3-kvjm6" in nodes

    def test_detects_worker_failed_illegal_memory(self: Self) -> None:
        """Test detection of generic worker wrapper with illegal memory access."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="Worker failed with error: illegal memory access on device 0"
            ),
        )

        assert strategy.should_retry(event)

    def test_ignores_nccl_warnings(self: Self) -> None:
        """Test that NCCL warnings don't trigger retry."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="[W525] Warning: WARNING: process group has NOT been destroyed "
                "before we destruct ProcessGroupNCCL."
            ),
        )

        assert not strategy.should_retry(event)

    def test_ignores_non_gpu_errors(self: Self) -> None:
        """Test that non-GPU errors don't trigger retry."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg="RuntimeError: Connection refused"),
        )

        assert not strategy.should_retry(event)

    def test_ignores_cuda_launch_failure(self: Self) -> None:
        """Test that CUDA launch failures do NOT trigger retry (software bug)."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="cudaErrorLaunchFailure: too many resources requested for launch"
            ),
        )

        # Launch failures are software bugs, not hardware issues
        assert not strategy.should_retry(event)

    def test_ignores_cuda_illegal_instruction(self: Self) -> None:
        """Test that CUDA illegal instruction does NOT trigger retry (software bug)."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(
                msg="cudaErrorIllegalInstruction: illegal instruction encountered"
            ),
        )

        # Illegal instruction is compilation issue, not hardware
        assert not strategy.should_retry(event)

    def test_ignores_cuda_misaligned_address(self: Self) -> None:
        """Test that CUDA misaligned address does NOT trigger retry (software bug)."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg="cudaErrorMisalignedAddress: misaligned address"),
        )

        # Misaligned address is pointer bug, not hardware
        assert not strategy.should_retry(event)

    def test_ignores_non_message_events(self: Self) -> None:
        """Test that non-MESSAGE_EVENT types are ignored."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventMessagePayload(msg="RuntimeError: NCCL Error 3: internal error"),
        )

        assert not strategy.should_retry(event)

    def test_extracts_node_from_metadata(self: Self) -> None:
        """Test extraction of node name from event metadata."""
        strategy = NCCLErrorRetryStrategy()

        # Embed node name in payload message as JSON (how K8s monitors format it)
        payload_data = {
            "error": "RuntimeError: NCCL Error 3: internal error",
            "node_name": "dmf-nnnqh-gpu-worker-3-kvjm6",
        }
        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg=json.dumps(payload_data)),
        )

        nodes = strategy.extract_nodes_to_avoid(event)
        assert "dmf-nnnqh-gpu-worker-3-kvjm6" in nodes

    def test_extracts_node_from_json_payload(self: Self) -> None:
        """Test extraction of node name from JSON payload."""
        strategy = NCCLErrorRetryStrategy()

        payload_data = {
            "node_name": "worker-gpu-node-2",
            "pod_name": "training-pod-123",
            "error": "NCCL Error 3",
        }

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg=json.dumps(payload_data)),
        )

        nodes = strategy.extract_nodes_to_avoid(event)
        assert "worker-gpu-node-2" in nodes

    def test_extracts_node_from_json_markdown(self: Self) -> None:
        """Test extraction of node name from markdown-wrapped JSON."""
        strategy = NCCLErrorRetryStrategy()

        payload_data = {"nodeName": "worker-gpu-node-3"}
        msg = f"Error occurred:\n```json\n{json.dumps(payload_data)}\n```"

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg=msg),
        )

        nodes = strategy.extract_nodes_to_avoid(event)
        assert "worker-gpu-node-3" in nodes

    def test_handles_nested_node_name(self: Self) -> None:
        """Test extraction of node name from nested JSON structure."""
        strategy = NCCLErrorRetryStrategy()

        payload_data = {"spec": {"nodeName": "worker-gpu-node-4"}}

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg=json.dumps(payload_data)),
        )

        nodes = strategy.extract_nodes_to_avoid(event)
        assert "worker-gpu-node-4" in nodes

    def test_extracts_node_from_pod_placement(self: Self) -> None:
        """Test extraction of node name from pod_placement in AppWrapper payload."""
        strategy = NCCLErrorRetryStrategy()

        payload_data = {
            "appwrapper": "gb8ilnjfa8",
            "state": "Failed",
            "pod_placement": {"gb8ilnjfa8-job-fxr44": "dmf-nnnqh-gpu-worker-3-kvjm6"},
        }
        msg = f"\n```json\n{json.dumps(payload_data)}\n```\n"

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg=msg),
        )

        nodes = strategy.extract_nodes_to_avoid(event)
        assert "dmf-nnnqh-gpu-worker-3-kvjm6" in nodes

    def test_extracts_multiple_nodes_from_pod_placement(self: Self) -> None:
        """Test extraction of multiple nodes from pod_placement."""
        strategy = NCCLErrorRetryStrategy()

        payload_data = {
            "pod_placement": {
                "pod-1": "gpu-worker-1",
                "pod-2": "gpu-worker-2",
            },
        }
        msg = f"\n```json\n{json.dumps(payload_data)}\n```\n"

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg=msg),
        )

        nodes = strategy.extract_nodes_to_avoid(event)
        assert "gpu-worker-1" in nodes
        assert "gpu-worker-2" in nodes

    def test_returns_empty_set_when_no_node_found(self: Self) -> None:
        """Test that empty set is returned when node name cannot be extracted."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg="RuntimeError: NCCL Error 3: internal error"),
        )

        nodes = strategy.extract_nodes_to_avoid(event)
        assert len(nodes) == 0

    def test_handles_malformed_json(self: Self) -> None:
        """Test handling of malformed JSON in payload."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg="{ invalid json }"),
        )

        # Should not crash
        nodes = strategy.extract_nodes_to_avoid(event)
        assert len(nodes) == 0

    def test_handles_missing_payload(self: Self) -> None:
        """Test handling of events without payload."""
        strategy = NCCLErrorRetryStrategy()

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=None,  # type: ignore[arg-type]
        )

        assert not strategy.should_retry(event)

    def test_real_world_nccl_error_message(self: Self) -> None:
        """Test detection of real-world NCCL error message from issue #1609."""
        strategy = NCCLErrorRetryStrategy()

        # Actual error message from the issue
        real_error = """
        Traceback (most recent call last):
          File "/opt/app-root/src/trainer.py", line 245, in <module>
            train()
          File "/opt/app-root/src/trainer.py", line 180, in train
            trainer.train()
          File "/usr/local/lib/python3.11/site-packages/transformers/trainer.py", line 1555
            return inner_training_loop(
          File "/usr/local/lib/python3.11/site-packages/transformers/trainer.py", line 1860
            tr_loss_step = self.training_step(model, inputs)
        RuntimeError: NCCL Error 3: internal error - please report this issue to the NCCL developers
        """

        event = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg=real_error),
        )

        assert strategy.should_retry(event)

        # For node extraction test, wrap error with node info in JSON
        # (simulating how K8s monitor would format it)
        payload_with_node = {
            "error": real_error,
            "node_name": "dmf-nnnqh-gpu-worker-3-kvjm6",
        }
        event_with_node = BuildEvent(
            run_metadata=EntityRunMetadata(build_id="test-build-id"),
            type=BuildEventType.MESSAGE_EVENT,
            payload=BuildEventMessagePayload(msg=json.dumps(payload_with_node)),
        )

        nodes = strategy.extract_nodes_to_avoid(event_with_node)
        assert "dmf-nnnqh-gpu-worker-3-kvjm6" in nodes
