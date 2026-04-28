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
Test suite for HybridSpaceSecretManager.

This module contains comprehensive tests for the hybrid secret manager that chains
multiple secret managers with priority-based fallback, covering priority order,
fallback mechanisms, error handling, edge cases, and integration scenarios.
"""

from unittest.mock import Mock, patch

import pytest

from gbserver.spacesecretmanager.hybridspacesecretmanager import (
    HybridSpaceSecretManager,
)
from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager

pytestmark = pytest.mark.g4os


def test_priority_order_first_manager_wins():
    """Test that the first manager in the chain takes precedence.

    Verifies that when multiple managers have the same secret, the value
    from the first manager is returned and subsequent managers are not queried.
    """
    # Create mock managers
    manager1 = Mock(spec=SpaceSecretManager)
    manager2 = Mock(spec=SpaceSecretManager)

    # Configure mock behavior: manager1 returns a value
    manager1.get_secret.return_value = {"value": "val1"}
    manager2.get_secret.return_value = {"value": "val2"}

    # Create hybrid manager with mocked initialization
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = [manager1, manager2]

        managers_config = [
            {"type": "env", "config": {}},
            {"type": "local", "config": {}},
        ]
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Test get_secret
    result = hybrid.get_secret("test_key")

    # Verify manager1 was called and returned its value
    assert result == {"value": "val1"}
    manager1.get_secret.assert_called_once_with(
        secret_name="test_key", secret_type="arbitrary", secret_group_name=""
    )

    # Verify manager2 was NOT called (stops at first success)
    manager2.get_secret.assert_not_called()


def test_fallback_when_first_returns_empty():
    """Test that fallback to second manager works when first returns empty.

    Verifies that when the first manager returns an empty dict (secret not found),
    the hybrid manager tries the next manager in the chain.
    """
    # Create mock managers
    manager1 = Mock(spec=SpaceSecretManager)
    manager2 = Mock(spec=SpaceSecretManager)

    # Configure mock behavior: manager1 returns empty, manager2 has the value
    manager1.get_secret.return_value = {}
    manager2.get_secret.return_value = {"value": "val2"}

    # Create hybrid manager with mocked initialization
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = [manager1, manager2]

        managers_config = [
            {"type": "env", "config": {}},
            {"type": "local", "config": {}},
        ]
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Test get_secret
    result = hybrid.get_secret("test_key")

    # Verify both managers were called
    assert result == {"value": "val2"}
    manager1.get_secret.assert_called_once()
    manager2.get_secret.assert_called_once()


def test_all_managers_fail_returns_empty():
    """Test that when all managers return empty, hybrid returns empty.

    Verifies that if no manager in the chain has the requested secret,
    an empty dict is returned rather than raising an exception.
    """
    # Create mock managers
    manager1 = Mock(spec=SpaceSecretManager)
    manager2 = Mock(spec=SpaceSecretManager)

    # Configure mock behavior: both return empty
    manager1.get_secret.return_value = {}
    manager2.get_secret.return_value = {}

    # Create hybrid manager with mocked initialization
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = [manager1, manager2]

        managers_config = [
            {"type": "env", "config": {}},
            {"type": "local", "config": {}},
        ]
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Test get_secret
    result = hybrid.get_secret("test_key")

    # Verify empty dict returned
    assert result == {}
    manager1.get_secret.assert_called_once()
    manager2.get_secret.assert_called_once()


def test_get_secrets_merge_with_precedence():
    """Test that get_secrets() merges all managers with first-wins precedence.

    Verifies that get_secrets() combines secrets from all managers, with earlier
    managers taking precedence for duplicate keys.
    """
    # Create mock managers
    manager1 = Mock(spec=SpaceSecretManager)
    manager2 = Mock(spec=SpaceSecretManager)

    # Configure mock behavior: overlapping keys with different values
    manager1.get_secrets.return_value = {"key1": "val1", "key3": "val3"}
    manager2.get_secrets.return_value = {"key1": "val2", "key2": "val2"}

    # Create hybrid manager with mocked initialization
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = [manager1, manager2]

        managers_config = [
            {"type": "env", "config": {}},
            {"type": "local", "config": {}},
        ]
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Test get_secrets
    result = hybrid.get_secrets()

    # Verify merged result with first-wins precedence
    assert result == {"key1": "val1", "key2": "val2", "key3": "val3"}
    manager1.get_secrets.assert_called_once()
    manager2.get_secrets.assert_called_once()


def test_create_secret_writes_to_first_writable():
    """Test that create_secret() writes to the first writable manager.

    Verifies that create_secret() delegates to the first manager that successfully
    creates the secret without raising NotImplementedError.
    """
    # Create mock managers
    manager1 = Mock(spec=SpaceSecretManager)
    manager2 = Mock(spec=SpaceSecretManager)

    # Configure mock behavior: both can write
    manager1.create_secret.return_value = None  # Success
    manager2.create_secret.return_value = None  # Success

    # Create hybrid manager with mocked initialization
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = [manager1, manager2]

        managers_config = [
            {"type": "env", "config": {}},
            {"type": "local", "config": {}},
        ]
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Test create_secret
    hybrid.create_secret("test_key", "test_value")

    # Verify only manager1 was called (first writable)
    manager1.create_secret.assert_called_once_with(
        secret_name="test_key",
        secret_value="test_value",
        secret_type="arbitrary",
        secret_group_name="",
    )
    manager2.create_secret.assert_not_called()


def test_create_secret_error_when_all_readonly():
    """Test that create_secret() raises NotImplementedError when all managers are read-only.

    Verifies that if all managers in the chain raise NotImplementedError (read-only),
    the hybrid manager also raises NotImplementedError with an appropriate message.
    """
    # Create mock managers
    manager1 = Mock(spec=SpaceSecretManager)
    manager2 = Mock(spec=SpaceSecretManager)

    # Configure mock behavior: both are read-only
    manager1.create_secret.side_effect = NotImplementedError("Manager 1 is read-only")
    manager2.create_secret.side_effect = NotImplementedError("Manager 2 is read-only")

    # Create hybrid manager with mocked initialization
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = [manager1, manager2]

        managers_config = [
            {"type": "env", "config": {}},
            {"type": "local", "config": {}},
        ]
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Test create_secret raises NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        hybrid.create_secret("test_key", "test_value")

    # Verify error message mentions read-only
    error_message = str(exc_info.value)
    assert "read-only" in error_message.lower()

    # Verify both managers were tried
    manager1.create_secret.assert_called_once()
    manager2.create_secret.assert_called_once()


def test_manager_initialization_failure_handling(caplog):
    """Test that manager initialization failures are handled gracefully.

    Verifies that if a manager fails to initialize, the hybrid manager logs
    an error but continues initializing remaining managers.
    """
    # Create one successful manager
    manager1 = Mock(spec=SpaceSecretManager)

    # Mock factory to simulate initialization failure for second manager
    def factory_side_effect(secret_manager_type, uri, **kwargs):
        if secret_manager_type == "env":
            return manager1
        elif secret_manager_type == "local":
            raise RuntimeError("Failed to initialize local manager")
        raise ValueError("Unknown type")

    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = factory_side_effect

        managers_config = [
            {"type": "env", "config": {}},
            {"type": "local", "config": {}},  # This one will fail
        ]

        # Create hybrid manager (should not raise exception)
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Verify hybrid manager was created with only one manager
    assert len(hybrid.managers) == 1
    assert hybrid.managers[0] == manager1

    # Verify error was logged
    assert "Failed to initialize manager" in caplog.text


def test_manager_runtime_exception_handling():
    """Test that runtime exceptions from managers are handled gracefully.

    Verifies that if a manager raises an exception during get_secret(),
    the hybrid manager catches it, logs a warning, and continues to the next manager.
    """
    # Create mock managers
    manager1 = Mock(spec=SpaceSecretManager)
    manager2 = Mock(spec=SpaceSecretManager)

    # Configure mock behavior: manager1 raises exception, manager2 succeeds
    manager1.get_secret.side_effect = RuntimeError("Connection failed")
    manager2.get_secret.return_value = {"value": "val2"}

    # Create hybrid manager with mocked initialization
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = [manager1, manager2]

        managers_config = [
            {"type": "env", "config": {}},
            {"type": "local", "config": {}},
        ]
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Test get_secret (should fallback to manager2)
    result = hybrid.get_secret("test_key")

    # Verify fallback worked
    assert result == {"value": "val2"}
    manager1.get_secret.assert_called_once()
    manager2.get_secret.assert_called_once()


def test_empty_managers_list_edge_case():
    """Test that hybrid manager handles empty managers list gracefully.

    Verifies that when initialized with an empty managers list, all operations
    return empty results or raise appropriate errors.
    """
    # Create hybrid manager with empty list
    hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=[])

    # Test get_secret returns empty
    result = hybrid.get_secret("test_key")
    assert result == {}

    # Test get_secrets returns empty
    result = hybrid.get_secrets()
    assert result == {}

    # Test create_secret raises NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        hybrid.create_secret("test_key", "test_value")

    error_message = str(exc_info.value)
    assert "no managers available" in error_message.lower()


def test_multiple_instances_same_manager_type():
    """Test that multiple instances of the same manager type can be used.

    Verifies that the hybrid manager can chain multiple instances of the same
    manager type (e.g., two env managers with different prefixes).
    """
    # Create mock managers
    manager1 = Mock(spec=SpaceSecretManager)
    manager2 = Mock(spec=SpaceSecretManager)

    # Configure mock behavior: different values from each
    manager1.get_secret.return_value = {}
    manager2.get_secret.return_value = {"value": "val2"}

    # Create hybrid manager with two env managers
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = [manager1, manager2]

        managers_config = [
            {"type": "env", "config": {"prefix": "APP1_"}},
            {"type": "env", "config": {"prefix": "APP2_"}},
        ]
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Verify both managers were initialized
    assert len(hybrid.managers) == 2

    # Test that both are used correctly
    result = hybrid.get_secret("test_key")
    assert result == {"value": "val2"}
    manager1.get_secret.assert_called_once()
    manager2.get_secret.assert_called_once()


def test_full_integration_env_local_chain(tmp_path, monkeypatch):
    """Test full integration with real EnvSpaceSecretManager and LocalSpaceSecretManager.

    Verifies that the hybrid manager works correctly with real manager instances,
    demonstrating priority and fallback with actual implementations.
    """
    # Set up environment variable for env manager
    monkeypatch.setenv("GBSERVER_SECRET_ENV_KEY", "env_value")

    # Set up local secrets file for local manager
    import base64

    secrets_file = tmp_path / "secrets.yaml"
    # Use base64 encoding as LocalSpaceSecretManager expects
    encoded_value = base64.b64encode(b"local_value").decode("utf-8")
    secrets_file.write_text(f"LOCAL_KEY: {encoded_value}\n")

    # Load secret managers (needed for factory to work)
    SpaceSecretManager.load_spacesecretmanagers()

    # Create hybrid manager with real managers
    managers_config = [
        {"type": "env", "config": {"prefix": "GBSERVER_SECRET_"}},
        {"type": "local", "config": {"secrets_dir": str(secrets_file)}},
    ]
    hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Verify both managers were initialized
    assert len(hybrid.managers) == 2

    # Test 1: Get secret from env manager (first in chain)
    result = hybrid.get_secret("env_key")
    assert result == {"value": "env_value"}

    # Test 2: Get secret from local manager (fallback when env doesn't have it)
    # LocalSpaceSecretManager stores keys as-is from the YAML file (uppercase)
    result = hybrid.get_secret("LOCAL_KEY")
    assert result == {"value": "local_value"}

    # Test 3: Get non-existent secret (both return empty)
    result = hybrid.get_secret("nonexistent")
    assert result == {}

    # Test 4: Get all secrets (merge from both managers)
    all_secrets = hybrid.get_secrets()
    assert all_secrets is not None
    assert "ENV_KEY" in all_secrets
    assert all_secrets["ENV_KEY"] == "env_value"
    assert "LOCAL_KEY" in all_secrets
    assert all_secrets["LOCAL_KEY"] == "local_value"

    # Test 5: Create secret (should fail on env, succeed on local)
    hybrid.create_secret("new_key", "new_value")

    # Verify secret was created in local manager
    import yaml

    updated_data = yaml.safe_load(secrets_file.read_text())
    # create_secret stores the key as-is (no uppercasing)
    assert "new_key" in updated_data
    # Verify it's the base64 encoded value
    decoded = base64.b64decode(updated_data["new_key"]).decode("utf-8")
    assert decoded == "new_value"


def test_nested_hybrid_prevention(caplog):
    """Test that nested hybrid managers are prevented.

    Verifies that attempting to nest a hybrid manager within another hybrid
    manager is detected and prevented with an appropriate error message.
    """
    managers_config = [
        {"type": "env", "config": {}},
        {
            "type": "hybrid",
            "config": {"managers": []},
        },  # Nested hybrid - should be rejected
    ]

    # Create hybrid manager (should skip nested hybrid)
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        # Only env manager should be created
        mock_env = Mock(spec=SpaceSecretManager)
        mock_factory.return_value = mock_env

        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Verify nested hybrid was rejected
    assert "Nested hybrid managers are not supported" in caplog.text
    # Only one manager should be initialized (env, not hybrid)
    assert len(hybrid.managers) == 1


def test_create_secret_fails_all_with_runtime_errors():
    """Test create_secret() when all managers fail with runtime errors (not NotImplementedError).

    Verifies that if all managers fail with runtime exceptions (not read-only),
    a RuntimeError is raised with details about the failure.
    """
    # Create mock managers
    manager1 = Mock(spec=SpaceSecretManager)
    manager2 = Mock(spec=SpaceSecretManager)

    # Configure mock behavior: both fail with runtime errors
    manager1.create_secret.side_effect = RuntimeError("Connection timeout")
    manager2.create_secret.side_effect = RuntimeError("Permission denied")

    # Create hybrid manager with mocked initialization
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = [manager1, manager2]

        managers_config = [
            {"type": "env", "config": {}},
            {"type": "local", "config": {}},
        ]
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Test create_secret raises RuntimeError (not NotImplementedError)
    with pytest.raises(RuntimeError) as exc_info:
        hybrid.create_secret("test_key", "test_value")

    # Verify error message mentions failure
    error_message = str(exc_info.value)
    assert "Failed to create secret" in error_message
    assert "Permission denied" in error_message  # Should mention last error

    # Verify both managers were tried
    manager1.create_secret.assert_called_once()
    manager2.create_secret.assert_called_once()


def test_get_secrets_with_exceptions():
    """Test that get_secrets() handles exceptions from individual managers gracefully.

    Verifies that if some managers raise exceptions during get_secrets(),
    the hybrid manager continues and merges results from successful managers.
    """
    # Create mock managers
    manager1 = Mock(spec=SpaceSecretManager)
    manager2 = Mock(spec=SpaceSecretManager)
    manager3 = Mock(spec=SpaceSecretManager)

    # Configure mock behavior: manager1 succeeds, manager2 fails, manager3 succeeds
    manager1.get_secrets.return_value = {"key1": "val1"}
    manager2.get_secrets.side_effect = RuntimeError("Database error")
    manager3.get_secrets.return_value = {"key2": "val2"}

    # Create hybrid manager with mocked initialization
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = [manager1, manager2, manager3]

        managers_config = [
            {"type": "env", "config": {}},
            {"type": "local", "config": {}},
            {"type": "ibmcloud", "config": {}},
        ]
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Test get_secrets (should merge successful managers)
    result = hybrid.get_secrets()

    # Verify merged result contains secrets from working managers
    assert result == {"key1": "val1", "key2": "val2"}
    manager1.get_secrets.assert_called_once()
    manager2.get_secrets.assert_called_once()
    manager3.get_secrets.assert_called_once()


def test_create_secret_mixed_readonly_and_failure():
    """Test create_secret() with mix of read-only and failing managers.

    Verifies proper error handling when some managers are read-only and others
    fail with runtime errors. Should raise RuntimeError (not NotImplementedError)
    since at least one manager attempted to write.
    """
    # Create mock managers
    manager1 = Mock(spec=SpaceSecretManager)
    manager2 = Mock(spec=SpaceSecretManager)
    manager3 = Mock(spec=SpaceSecretManager)

    # Configure mock behavior: readonly, runtime error, readonly
    manager1.create_secret.side_effect = NotImplementedError("Read-only")
    manager2.create_secret.side_effect = RuntimeError("Write failed")
    manager3.create_secret.side_effect = NotImplementedError("Read-only")

    # Create hybrid manager with mocked initialization
    with patch.object(SpaceSecretManager, "get_spacesecretmanager") as mock_factory:
        mock_factory.side_effect = [manager1, manager2, manager3]

        managers_config = [
            {"type": "env", "config": {}},
            {"type": "local", "config": {}},
            {"type": "ibmcloud", "config": {}},
        ]
        hybrid = HybridSpaceSecretManager(uri="hybrid://", managers=managers_config)

    # Test create_secret raises RuntimeError (not NotImplementedError)
    # because manager2 attempted to write but failed
    with pytest.raises(RuntimeError) as exc_info:
        hybrid.create_secret("test_key", "test_value")

    error_message = str(exc_info.value)
    assert "Failed to create secret" in error_message
    assert "Write failed" in error_message

    # Verify all managers were tried
    manager1.create_secret.assert_called_once()
    manager2.create_secret.assert_called_once()
    manager3.create_secret.assert_called_once()
