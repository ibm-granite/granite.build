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

"""Tests that cleanup_helm() deletes orphaned RayClusters after helm uninstall."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gbserver.environment.k8s import K8s

pytestmark = pytest.mark.ibm


class TestRayClusterCleanup:
    """Verify that RayClusters are explicitly deleted after helm uninstall."""

    @pytest.mark.asyncio
    async def test_raycluster_deleted_after_helm_uninstall(self):
        """After helm uninstall, _delete_rayclusters_for_release should be called."""
        with patch.object(K8s, "__init__", lambda self, *a, **kw: None):
            k8s = K8s.__new__(K8s)
            k8s._launched_releases = {"launch-1": "gb-test-release"}
            k8s.kube_config = None
            k8s.kube_context = None
            k8s.ssl_verification = True
            k8s.config = MagicMock()
            k8s.config.config = {"namespace": "test-ns"}

            # Mock helm uninstall subprocess
            with patch(
                "gbserver.environment.k8s.launch_command_and_raise_errors",
                new_callable=AsyncMock,
                return_value=(MagicMock(), "", ""),
            ):
                # Mock the new RayCluster deletion method
                k8s._delete_rayclusters_for_release = AsyncMock()

                await k8s.cleanup_helm(launch_id="launch-1")

                k8s._delete_rayclusters_for_release.assert_awaited_once_with(
                    "gb-test-release", "launch-1"
                )

    @pytest.mark.asyncio
    async def test_delete_rayclusters_deletes_existing_raycluster(self):
        """_delete_rayclusters_for_release should delete a RayCluster by name."""
        with patch.object(K8s, "__init__", lambda self, *a, **kw: None):
            k8s = K8s.__new__(K8s)
            k8s.kube_config = None
            k8s.kube_context = None
            k8s.ssl_verification = True
            k8s.config = MagicMock()
            k8s.config.config = {"namespace": "test-ns"}

            mock_custom_api = AsyncMock()
            mock_custom_api.get_namespaced_custom_object = AsyncMock(
                return_value={"metadata": {"name": "gb-test-release-ray-cluster"}}
            )
            mock_custom_api.delete_namespaced_custom_object = AsyncMock()

            with patch(
                "gbserver.environment.k8s.AtomicApiClient.create_api_client"
            ) as mock_api_cls:
                mock_api = AsyncMock()
                mock_api_cls.return_value = mock_api
                mock_api.__aenter__ = AsyncMock(return_value=mock_api)
                mock_api.__aexit__ = AsyncMock(return_value=False)

                with patch(
                    "gbserver.environment.k8s.client.CustomObjectsApi",
                    return_value=mock_custom_api,
                ):
                    await k8s._delete_rayclusters_for_release(
                        "gb-test-release", "launch-1"
                    )

            mock_custom_api.delete_namespaced_custom_object.assert_awaited_once_with(
                group="ray.io",
                version="v1",
                namespace="test-ns",
                plural="rayclusters",
                name="gb-test-release-ray-cluster",
            )

    @pytest.mark.asyncio
    async def test_delete_rayclusters_noop_when_not_found(self):
        """_delete_rayclusters_for_release should handle 404 gracefully."""
        from kubernetes_asyncio.client import ApiException

        with patch.object(K8s, "__init__", lambda self, *a, **kw: None):
            k8s = K8s.__new__(K8s)
            k8s.kube_config = None
            k8s.kube_context = None
            k8s.ssl_verification = True
            k8s.config = MagicMock()
            k8s.config.config = {"namespace": "test-ns"}

            mock_custom_api = AsyncMock()
            mock_custom_api.get_namespaced_custom_object = AsyncMock(
                side_effect=ApiException(status=404)
            )

            with patch(
                "gbserver.environment.k8s.AtomicApiClient.create_api_client"
            ) as mock_api_cls:
                mock_api = AsyncMock()
                mock_api_cls.return_value = mock_api
                mock_api.__aenter__ = AsyncMock(return_value=mock_api)
                mock_api.__aexit__ = AsyncMock(return_value=False)

                with patch(
                    "gbserver.environment.k8s.client.CustomObjectsApi",
                    return_value=mock_custom_api,
                ):
                    # Should not raise
                    await k8s._delete_rayclusters_for_release(
                        "gb-test-release", "launch-1"
                    )

            mock_custom_api.delete_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_rayclusters_swallows_non_404_api_exception(self):
        """Non-404 ApiException (e.g. 403 Forbidden) should be logged, not re-raised."""
        from kubernetes_asyncio.client import ApiException

        with patch.object(K8s, "__init__", lambda self, *a, **kw: None):
            k8s = K8s.__new__(K8s)
            k8s.kube_config = None
            k8s.kube_context = None
            k8s.ssl_verification = True
            k8s.config = MagicMock()
            k8s.config.config = {"namespace": "test-ns"}

            mock_custom_api = AsyncMock()
            mock_custom_api.get_namespaced_custom_object = AsyncMock(
                side_effect=ApiException(status=403)
            )

            with patch(
                "gbserver.environment.k8s.AtomicApiClient.create_api_client"
            ) as mock_api_cls:
                mock_api = AsyncMock()
                mock_api_cls.return_value = mock_api
                mock_api.__aenter__ = AsyncMock(return_value=mock_api)
                mock_api.__aexit__ = AsyncMock(return_value=False)

                with patch(
                    "gbserver.environment.k8s.client.CustomObjectsApi",
                    return_value=mock_custom_api,
                ):
                    # Should not raise despite 403
                    await k8s._delete_rayclusters_for_release(
                        "gb-test-release", "launch-1"
                    )

            mock_custom_api.delete_namespaced_custom_object.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_rayclusters_swallows_generic_exception(self):
        """Generic exceptions (e.g. connection errors) should be logged, not re-raised."""
        with patch.object(K8s, "__init__", lambda self, *a, **kw: None):
            k8s = K8s.__new__(K8s)
            k8s.kube_config = None
            k8s.kube_context = None
            k8s.ssl_verification = True
            k8s.config = MagicMock()
            k8s.config.config = {"namespace": "test-ns"}

            with patch(
                "gbserver.environment.k8s.AtomicApiClient.create_api_client",
                side_effect=ConnectionError("network unreachable"),
            ):
                # Should not raise despite connection error
                await k8s._delete_rayclusters_for_release("gb-test-release", "launch-1")
