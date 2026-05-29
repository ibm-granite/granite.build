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

"""Tests that BuildRunnerJob deletes the K8s job/pod when stop is requested."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.types.status import Status

pytestmark = pytest.mark.ibm


class TestBuildRunnerJobCancelCleanup:
    """Verify that when is_stop_requested is True, the K8s job and pod are deleted."""

    @pytest.mark.asyncio
    async def test_cancel_deletes_job_and_pod(self):
        """When is_stop_requested exits the monitoring loop, __delete_job_and_pod_with_retry must be called."""
        from gbserver.buildrunnerjob.buildrunnerjob import BuildRunnerJob

        # Create a mock StoredBuild
        stored_build = MagicMock()
        stored_build.uuid = "test-build-uuid"
        stored_build.status = Status.RUNNING

        # Create a mock storage
        mock_storage = MagicMock()
        mock_storage.build_storage.get_by_uuid.return_value = stored_build
        mock_storage.table_name_prefix = ""

        # Patch the constructor dependencies
        with patch.object(BuildRunnerJob, "__init__", lambda self, *a, **kw: None):
            runner = BuildRunnerJob.__new__(BuildRunnerJob)
            runner.stored_build = stored_build
            runner.storage = mock_storage
            runner.namespace = "test-ns"
            runner.monitoring_interval = 0.01
            runner.is_stop_requested = False
            runner.is_running = False

            # Mock the K8s API methods
            mock_batchv1 = AsyncMock()
            runner._BuildRunnerJob__create_namespaced_job_with_retry = AsyncMock(
                return_value=MagicMock()
            )
            runner._BuildRunnerJob__delete_job_and_pod_with_retry = AsyncMock()
            runner._BuildRunnerJob__get_batchv1job_body = MagicMock(
                return_value=("test-job", {})
            )

            # Set stop after first read iteration
            async def read_then_stop(*args, **kwargs):
                runner.is_stop_requested = True
                return MagicMock()

            runner._BuildRunnerJob__read_namespaced_job_with_retry = AsyncMock(
                side_effect=read_then_stop
            )

            # Mock get_admin_storage to avoid singleton access
            mock_admin_storage = MagicMock()
            mock_admin_storage.build_storage.update_fields = MagicMock()

            with (
                patch(
                    "gbserver.buildrunnerjob.buildrunnerjob.get_admin_storage",
                    return_value=mock_admin_storage,
                ),
                patch(
                    "gbserver.buildrunnerjob.buildrunnerjob.AtomicApiClient"
                ) as mock_api_cls,
            ):
                # Set up the async context manager for the API client
                mock_api = AsyncMock()
                mock_api_cls.create_api_client = AsyncMock(return_value=mock_api)
                mock_api.__aenter__ = AsyncMock(return_value=mock_api)
                mock_api.__aexit__ = AsyncMock(return_value=False)

                with patch(
                    "gbserver.buildrunnerjob.buildrunnerjob.client.BatchV1Api",
                    return_value=mock_batchv1,
                ):
                    await runner._BuildRunnerJob__start_job()

            # ASSERT: delete was called with correct args when cancel was requested
            runner._BuildRunnerJob__delete_job_and_pod_with_retry.assert_awaited_once_with(
                mock_batchv1, "test-job"
            )
