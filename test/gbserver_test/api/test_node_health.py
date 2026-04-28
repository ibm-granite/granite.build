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
API tests for node health endpoints.

Tests authentication, authorization, and functionality of all node health
REST API endpoints. The API queries node_failure_storage directly.
"""

from typing import Self
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.testclient import TestClient
from gbserver_test.test_utils import AbstractSingletonStorageUsingTest

from gbserver.api.node_health import (
    get_failure_summary,
    get_node_failures,
    get_problematic_nodes,
    health_check,
    node_health_api,
    resolve_node_failures,
)
from gbserver.storage.stored_node_failure import StoredNodeFailure


class TestNodeHealthAPIEndpoints(AbstractSingletonStorageUsingTest):
    """Direct tests for node health API endpoint functions."""

    @pytest.fixture
    def mock_request(self: Self) -> Mock:
        """Create a mock request object."""
        return Mock(spec=Request)

    def _add_failure(
        self: Self,
        node_name: str = "test-node-1",
        build_id: str = "build-123",
        launch_id: str = "launch-456",
        failure_type: str = "FailedMount",
        namespace: str = "",
        cluster: str = "",
    ) -> StoredNodeFailure:
        """Helper to add a failure directly to storage."""
        metadata = {}
        if namespace:
            metadata["namespace"] = namespace
        if cluster:
            metadata["cluster"] = cluster

        failure = StoredNodeFailure(
            node_name=node_name,
            build_id=build_id,
            launch_id=launch_id,
            failure_type=failure_type,
            metadata=metadata,
        )
        self.storage.node_failure_storage.add(failure)
        return failure

    @pytest.mark.asyncio
    async def test_get_failure_summary(self: Self, mock_request: Mock) -> None:
        """Test get_failure_summary endpoint function."""
        self._add_failure(
            node_name="test-node-1",
            namespace="granite-build",
            cluster="test-cluster",
        )

        result = await get_failure_summary(mock_request)

        assert "test-node-1" in result
        assert result["test-node-1"]["total_failures"] == 1
        assert result["test-node-1"]["namespaces"] == ["granite-build"]
        assert result["test-node-1"]["clusters"] == ["test-cluster"]

    @pytest.mark.asyncio
    async def test_get_problematic_nodes(self: Self, mock_request: Mock) -> None:
        """Test get_problematic_nodes endpoint function."""
        for i in range(3):
            self._add_failure(
                node_name="problematic-node",
                build_id=f"build-{i}",
                launch_id=f"launch-{i}",
            )

        result = await get_problematic_nodes(mock_request, threshold=3, minutes=None)

        assert "problematic_nodes" in result
        assert "problematic-node" in result["problematic_nodes"]

    @pytest.mark.asyncio
    async def test_get_node_failures(self: Self, mock_request: Mock) -> None:
        """Test get_node_failures endpoint function."""
        self._add_failure(
            node_name="test-node-2",
            build_id="build-abc",
            launch_id="launch-xyz",
            failure_type="FailedAttachVolume",
            namespace="granite-build-staging",
            cluster="staging-cluster",
        )

        result = await get_node_failures(mock_request, "test-node-2", minutes=None)

        assert result["node_name"] == "test-node-2"
        assert result["failure_count"] == 1
        assert result["failures"][0]["failure_type"] == "FailedAttachVolume"
        assert result["failures"][0]["namespace"] == "granite-build-staging"
        assert result["failures"][0]["cluster"] == "staging-cluster"

    @pytest.mark.asyncio
    async def test_resolve_requires_admin(self: Self, mock_request: Mock) -> None:
        """Test that resolve_node_failures requires super admin."""
        self._add_failure(node_name="test-node-3")

        with patch("gbserver.api.node_health.is_super_admin", return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await resolve_node_failures(mock_request, "test-node-3")

            assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
            assert "Super admin privileges required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_resolve_with_admin(self: Self, mock_request: Mock) -> None:
        """Test that admin can resolve node failures."""
        self._add_failure(node_name="test-node-4")

        with patch("gbserver.api.node_health.is_super_admin", return_value=True):
            result = await resolve_node_failures(mock_request, "test-node-4")

            assert result["status"] == "success"
            assert result["resolved_count"] == 1

        # Verify resolved (should not appear in recent unresolved failures)
        failures = self.storage.node_failure_storage.get_recent_failures("test-node-4")
        assert len(failures) == 0

    @pytest.mark.asyncio
    async def test_health_check_with_storage(self: Self) -> None:
        """Test health_check endpoint with storage set."""
        result = await health_check()

        assert result["status"] == "healthy"
        assert result["service"] == "node-health-tracker"

    @pytest.mark.asyncio
    async def test_health_check_without_storage(self: Self) -> None:
        """Test health_check returns 503 when storage is unavailable."""
        with patch(
            "gbserver.storage.singleton_storage.get_admin_storage",
            side_effect=RuntimeError("no storage"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await health_check()
            assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_read_endpoints_return_503_without_storage(
        self: Self, mock_request: Mock
    ) -> None:
        """Test that read endpoints return 503 when storage is unavailable."""
        with patch(
            "gbserver.storage.singleton_storage.get_admin_storage",
            side_effect=RuntimeError("no storage"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_failure_summary(mock_request)
            assert exc_info.value.status_code == 503

            with pytest.raises(HTTPException) as exc_info:
                await get_problematic_nodes(mock_request)
            assert exc_info.value.status_code == 503

            with pytest.raises(HTTPException) as exc_info:
                await get_node_failures(mock_request, "test-node")
            assert exc_info.value.status_code == 503


class TestNodeHealthAPIIntegration(AbstractSingletonStorageUsingTest):
    """Integration tests using TestClient."""

    @pytest.fixture
    def app(self: Self) -> FastAPI:
        """Create a minimal FastAPI app with node health API mounted."""
        app = FastAPI()
        app.mount("/node-health", node_health_api)
        return app

    @pytest.fixture
    def client(self: Self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint_returns_200(self: Self, client: TestClient) -> None:
        """Test health endpoint via HTTP client."""
        response = client.get("/node-health/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"

    def test_summary_aggregates_data(self: Self, client: TestClient) -> None:
        """Test summary endpoint returns node failure data."""
        failure = StoredNodeFailure(
            node_name="integration-test-node",
            build_id="build-999",
            launch_id="launch-999",
            failure_type="FailedMount",
        )
        self.storage.node_failure_storage.add(failure)

        response = client.get("/node-health/summary")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "integration-test-node" in data
