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

"""Tests that BuildWatcher handles CANCEL_REQUESTED builds not in its tracking dict."""

import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.storage.stored_build import StoredBuild
from gbserver.types.status import Status

pytestmark = pytest.mark.ibm


class TestBuildWatcherOrphanCancel:
    """Verify that orphaned CANCEL_REQUESTED builds are transitioned to CANCELLED."""

    def test_orphan_cancel_requested_transitions_to_cancelled(self):
        """A CANCEL_REQUESTED build not in build_runners should be transitioned to CANCELLED."""
        from gbserver.buildwatcher.buildwatcher import BuildWatcher

        with patch.object(BuildWatcher, "__init__", lambda self, *a, **kw: None):
            watcher = BuildWatcher.__new__(BuildWatcher)
            watcher.build_runners = {}
            watcher.build_threads = {}
            watcher.build_pr_threads = {}
            watcher._builds_lock = threading.Lock()

            mock_build = MagicMock(spec=StoredBuild)
            mock_build.uuid = "orphan-build-uuid"
            mock_build.status = Status.CANCEL_REQUESTED

            mock_admin_storage = MagicMock()
            with (
                patch(
                    "gbserver.buildwatcher.buildwatcher.get_admin_storage",
                    return_value=mock_admin_storage,
                ),
                patch.object(
                    watcher,
                    "_BuildWatcher__cleanup_runner_resources",
                    create=True,
                ) as mock_cleanup,
            ):
                watcher._BuildWatcher__process_cancel_requested_build(mock_build)

            # Verify runner-resource cleanup was called
            mock_cleanup.assert_called_once_with("orphan-build-uuid")
            # Verify status was updated to CANCELLED
            mock_admin_storage.build_storage.update_fields.assert_called_once()
            call_args = mock_admin_storage.build_storage.update_fields.call_args
            assert call_args[0][0] == "orphan-build-uuid"
            assert call_args[1]["fields"] == {"status": Status.CANCELLED}

    def test_tracked_build_still_calls_stop(self):
        """A CANCEL_REQUESTED build in build_runners should still call stop() as before."""
        from gbserver.buildwatcher.buildwatcher import BuildWatcher

        with patch.object(BuildWatcher, "__init__", lambda self, *a, **kw: None):
            watcher = BuildWatcher.__new__(BuildWatcher)
            watcher.build_threads = {}
            watcher.build_pr_threads = {}
            watcher._builds_lock = threading.Lock()

            mock_runner = MagicMock()
            watcher.build_runners = {"tracked-build-uuid": mock_runner}

            mock_build = MagicMock(spec=StoredBuild)
            mock_build.uuid = "tracked-build-uuid"

            mock_admin_storage = MagicMock()
            with patch(
                "gbserver.buildwatcher.buildwatcher.get_admin_storage",
                return_value=mock_admin_storage,
            ):
                watcher._BuildWatcher__process_cancel_requested_build(mock_build)

            # Verify stop() was called on the runner
            mock_runner.stop.assert_called_once()
            # Verify status was NOT updated (the runner handles that)
            mock_admin_storage.build_storage.update_fields.assert_not_called()

    def test_cleanup_resources_deletes_aw_and_rc(self):
        """BuildRunnerJob.cleanup_resources should delete AppWrappers and RayClusters by label."""
        from gbserver.buildrunnerjob.buildrunnerjob import BuildRunnerJob

        mock_custom_api = AsyncMock()
        mock_custom_api.list_namespaced_custom_object = AsyncMock(
            side_effect=[
                # First call: AppWrappers
                {"items": [{"metadata": {"name": "gb-aw-orphan"}}]},
                # Second call: RayClusters
                {"items": [{"metadata": {"name": "r-orphan-ray-cluster"}}]},
            ]
        )
        mock_custom_api.delete_namespaced_custom_object = AsyncMock()

        with (
            patch(
                "gbserver.buildrunnerjob.buildrunnerjob.AtomicApiClient.create_api_client"
            ) as mock_api_cls,
            patch(
                "gbserver.buildrunnerjob.buildrunnerjob.BUILDRUNNERJOB_NAMESPACE",
                "test-ns",
            ),
        ):
            mock_api = AsyncMock()
            mock_api_cls.return_value = mock_api
            mock_api.__aenter__ = AsyncMock(return_value=mock_api)
            mock_api.__aexit__ = AsyncMock(return_value=False)

            with patch(
                "gbserver.buildrunnerjob.buildrunnerjob.client.CustomObjectsApi",
                return_value=mock_custom_api,
            ):
                BuildRunnerJob.cleanup_resources("test-build-id")

        # Should have deleted both an AppWrapper and a RayCluster
        assert mock_custom_api.delete_namespaced_custom_object.await_count == 2

    def test_cleanup_resources_swallows_exceptions(self):
        """BuildRunnerJob.cleanup_resources should not raise on K8s connection failure."""
        from gbserver.buildrunnerjob.buildrunnerjob import BuildRunnerJob

        with patch(
            "gbserver.buildrunnerjob.buildrunnerjob.AtomicApiClient.create_api_client",
            side_effect=ConnectionError("unreachable"),
        ):
            # Should not raise
            BuildRunnerJob.cleanup_resources("test-build-id")
