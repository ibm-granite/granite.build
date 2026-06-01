#!/usr/bin/env python3

"""Unit tests for cleanup_helm retry with backoff."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def k8s_env():
    """Create a minimal K8s instance for testing cleanup_helm."""
    # Import here to avoid issues if kubernetes_asyncio is not installed
    from gbserver.environment.k8s import K8s

    k8s = K8s.__new__(K8s)
    k8s._launched_releases = {"launch-1": "release-1"}
    k8s.kube_config = None
    k8s.kube_context = None
    k8s.ssl_verification = True
    k8s.config = MagicMock()
    k8s.config.config = {"namespace": "test-ns"}
    return k8s


class TestCleanupRetry:
    """Tests for cleanup_helm retry logic."""

    @pytest.mark.asyncio
    async def test_cleanup_succeeds_first_attempt(self, k8s_env):
        """Cleanup succeeds on first try - no retries needed."""
        with patch(
            "gbserver.environment.k8s.launch_command_and_raise_errors",
            new_callable=AsyncMock,
            return_value=(MagicMock(), "", ""),
        ) as mock_cmd:
            with patch.object(
                k8s_env, "_delete_rayclusters_for_release", new_callable=AsyncMock
            ):
                await k8s_env.cleanup_helm(launch_id="launch-1")
            assert mock_cmd.call_count == 1

    @pytest.mark.asyncio
    async def test_cleanup_retries_on_failure(self, k8s_env):
        """Cleanup retries with backoff when helm uninstall fails."""
        with patch(
            "gbserver.environment.k8s.launch_command_and_raise_errors",
            new_callable=AsyncMock,
            side_effect=[
                Exception("api down"),
                Exception("api down"),
                (MagicMock(), "", ""),
            ],
        ) as mock_cmd:
            with patch.object(
                k8s_env, "_delete_rayclusters_for_release", new_callable=AsyncMock
            ):
                with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                    await k8s_env.cleanup_helm(launch_id="launch-1")
            assert mock_cmd.call_count == 3
            # Verify backoff delays: 10*2^0=10, 10*2^1=20
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(10)
            mock_sleep.assert_any_call(20)

    @pytest.mark.asyncio
    async def test_cleanup_gives_up_after_max_retries(self, k8s_env):
        """Cleanup raises after exhausting all retries."""
        with patch(
            "gbserver.environment.k8s.launch_command_and_raise_errors",
            new_callable=AsyncMock,
            side_effect=Exception("api down"),
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(ValueError, match="helm uninstall failed"):
                    await k8s_env.cleanup_helm(launch_id="launch-1")

    @pytest.mark.asyncio
    async def test_cleanup_skips_unknown_launch_id(self, k8s_env):
        """Cleanup returns immediately for unknown launch IDs."""
        with patch(
            "gbserver.environment.k8s.launch_command_and_raise_errors",
            new_callable=AsyncMock,
        ) as mock_cmd:
            await k8s_env.cleanup_helm(launch_id="unknown-id")
            assert mock_cmd.call_count == 0

    @pytest.mark.asyncio
    async def test_raycluster_timeout_increased_to_30s(self, k8s_env):
        """RayCluster deletion uses 30s timeout."""
        with patch(
            "gbserver.environment.k8s.launch_command_and_raise_errors",
            new_callable=AsyncMock,
            return_value=(MagicMock(), "", ""),
        ):
            with patch.object(
                k8s_env,
                "_delete_rayclusters_for_release",
                new_callable=AsyncMock,
            ) as mock_rc:
                with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait:
                    # We need to actually let the helm uninstall succeed first
                    # then check wait_for is called with timeout=30
                    mock_wait.return_value = None
                    await k8s_env.cleanup_helm(launch_id="launch-1")
                    # Find the call to wait_for with _delete_rayclusters_for_release
                    mock_wait.assert_called_once()
                    _, kwargs = mock_wait.call_args
                    assert kwargs.get("timeout") == 30
