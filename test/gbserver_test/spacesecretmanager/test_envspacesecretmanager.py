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
Test suite for EnvSpaceSecretManager.

This module contains comprehensive tests for the environment variable-based
secret manager, covering exact matches, case-insensitive lookups, name
transformations, error handling, and edge cases.
"""

import pytest

from gbserver.spacesecretmanager.envspacesecretmanager import EnvSpaceSecretManager


def test_exact_match_lookup(monkeypatch):
    """Test that exact match lookup retrieves the correct secret value.

    Verifies that when an environment variable with the exact normalized name
    is set, get_secret() returns the correct value in the expected format.
    """
    monkeypatch.setenv("GBSERVER_SECRET_API_KEY", "test-value")
    manager = EnvSpaceSecretManager(uri="env://")
    result = manager.get_secret("api_key")
    assert result == {"value": "test-value"}


def test_case_insensitive_matching(monkeypatch):
    """Test that secret lookups are case-insensitive.

    Verifies that secret names can be provided in any case (uppercase, lowercase,
    mixed case) and will still match the environment variable correctly.
    """
    monkeypatch.setenv("GBSERVER_SECRET_API_KEY", "test-value")
    manager = EnvSpaceSecretManager(uri="env://")

    # Test uppercase
    result = manager.get_secret("API_KEY")
    assert result == {"value": "test-value"}

    # Test mixed case
    result = manager.get_secret("Api_Key")
    assert result == {"value": "test-value"}

    # Test lowercase
    result = manager.get_secret("api_key")
    assert result == {"value": "test-value"}


def test_name_transformation_variants(monkeypatch):
    """Test that name transformation handles dashes, dots, and underscores.

    Verifies that secret names with different separators (underscores, dashes, dots)
    are all normalized to match the same environment variable with underscores.
    """
    monkeypatch.setenv("GBSERVER_SECRET_API_KEY", "test-value")
    manager = EnvSpaceSecretManager(uri="env://")

    # Test with dashes
    result = manager.get_secret("api-key")
    assert result == {"value": "test-value"}

    # Test with dots
    result = manager.get_secret("api.key")
    assert result == {"value": "test-value"}

    # Test with underscores (original)
    result = manager.get_secret("api_key")
    assert result == {"value": "test-value"}

    # Test mixed separators
    monkeypatch.setenv("GBSERVER_SECRET_API_KEY_VALUE", "mixed-value")
    result = manager.get_secret("api-key.value")
    assert result == {"value": "mixed-value"}


def test_secret_not_found_returns_empty_dict():
    """Test that requesting a non-existent secret returns an empty dictionary.

    Verifies that when a secret is not found in the environment variables,
    the manager returns an empty dict rather than raising an exception.
    """
    manager = EnvSpaceSecretManager(uri="env://")
    result = manager.get_secret("nonexistent")
    assert result == {}


def test_get_secrets_returns_all_with_prefix(monkeypatch):
    """Test that get_secrets() returns all secrets with the configured prefix.

    Verifies that get_secrets() returns a dictionary of all environment variables
    matching the prefix, with prefix stripped from keys.
    """
    monkeypatch.setenv("GBSERVER_SECRET_KEY1", "value1")
    monkeypatch.setenv("GBSERVER_SECRET_KEY2", "value2")
    monkeypatch.setenv("OTHER_VAR", "value3")  # Different prefix, should be ignored

    manager = EnvSpaceSecretManager(uri="env://")
    secrets = manager.get_secrets()
    assert secrets is not None

    # Should only include secrets with the prefix, prefix stripped
    assert secrets == {"KEY1": "value1", "KEY2": "value2"}
    # Should not include OTHER_VAR
    assert "OTHER_VAR" not in secrets


def test_custom_prefix_configuration(monkeypatch):
    """Test that custom prefix can be configured during initialization.

    Verifies that the manager can be initialized with a custom prefix
    and correctly retrieves secrets using that prefix.
    """
    monkeypatch.setenv("MYAPP_SECRET_TOKEN", "custom-value")
    manager = EnvSpaceSecretManager(uri="env://", prefix="MYAPP_SECRET_")
    result = manager.get_secret("token")
    assert result == {"value": "custom-value"}


def test_create_secret_raises_not_implemented_error():
    """Test that create_secret() raises NotImplementedError.

    Verifies that attempting to create a secret raises an appropriate error,
    as environment variable secrets are read-only and must be set externally.
    """
    manager = EnvSpaceSecretManager(uri="env://")

    with pytest.raises(NotImplementedError) as exc_info:
        manager.create_secret("key", "value")

    # Verify error message mentions environment variables
    error_message = str(exc_info.value)
    assert "read-only" in error_message.lower()
    assert "environment variable" in error_message.lower()
    assert "GBSERVER_SECRET_KEY" in error_message


def test_special_characters_in_values(monkeypatch):
    """Test that special characters in secret values are handled correctly.

    Verifies that secrets containing unicode characters, newlines, quotes,
    and other special characters are retrieved without corruption.
    """
    # Unicode characters
    monkeypatch.setenv("GBSERVER_SECRET_UNICODE", "Hello 世界 🌍")
    manager = EnvSpaceSecretManager(uri="env://")
    result = manager.get_secret("unicode")
    assert result == {"value": "Hello 世界 🌍"}

    # Newlines
    monkeypatch.setenv("GBSERVER_SECRET_NEWLINES", "line1\nline2")
    result = manager.get_secret("newlines")
    assert result == {"value": "line1\nline2"}

    # Quotes
    monkeypatch.setenv("GBSERVER_SECRET_QUOTES", 'value with "quotes"')
    result = manager.get_secret("quotes")
    assert result == {"value": 'value with "quotes"'}


def test_empty_prefix_edge_case(monkeypatch):
    """Test that empty prefix allows matching any environment variable.

    Verifies that when prefix is set to empty string, the manager can
    retrieve secrets from environment variables without any prefix.
    """
    monkeypatch.setenv("API_KEY", "value")
    manager = EnvSpaceSecretManager(uri="env://", prefix="")
    result = manager.get_secret("api_key")
    assert result == {"value": "value"}


def test_multiple_calls_consistency(monkeypatch):
    """Test that multiple calls to get_secret return consistent results.

    Verifies that calling get_secret() multiple times for the same secret
    returns the same value consistently.
    """
    monkeypatch.setenv("GBSERVER_SECRET_TEST", "value")
    manager = EnvSpaceSecretManager(uri="env://")

    # Call multiple times
    result1 = manager.get_secret("test")
    result2 = manager.get_secret("test")
    result3 = manager.get_secret("test")

    # All should return the same value
    assert result1 == {"value": "value"}
    assert result2 == {"value": "value"}
    assert result3 == {"value": "value"}


def test_get_secrets_with_empty_prefix(monkeypatch):
    """Test that get_secrets() works correctly with an empty prefix.

    Verifies that when prefix is empty, get_secrets() returns all
    environment variables.
    """
    monkeypatch.setenv("KEY1", "value1")
    monkeypatch.setenv("KEY2", "value2")

    manager = EnvSpaceSecretManager(uri="env://", prefix="")
    secrets = manager.get_secrets()
    assert secrets is not None

    # Should include our test keys
    assert "KEY1" in secrets
    assert "KEY2" in secrets
    assert secrets["KEY1"] == "value1"
    assert secrets["KEY2"] == "value2"


def test_secret_with_empty_value(monkeypatch):
    """Test that secrets with empty string values are handled correctly.

    Verifies that environment variables set to empty strings are still
    retrieved correctly (not confused with missing variables).
    """
    monkeypatch.setenv("GBSERVER_SECRET_EMPTY", "")
    manager = EnvSpaceSecretManager(uri="env://")
    result = manager.get_secret("empty")
    assert result == {"value": ""}


def test_complex_secret_name_normalization(monkeypatch):
    """Test normalization of complex secret names with multiple special characters.

    Verifies that secret names with multiple dots, dashes, and mixed case
    are correctly normalized to match environment variable names.
    """
    monkeypatch.setenv("GBSERVER_SECRET_MY_COMPLEX_API_KEY_V2", "complex-value")
    manager = EnvSpaceSecretManager(uri="env://")

    # Test various formats that should all normalize to the same env var
    result = manager.get_secret("my-complex-api-key-v2")
    assert result == {"value": "complex-value"}

    result = manager.get_secret("my.complex.api.key.v2")
    assert result == {"value": "complex-value"}

    result = manager.get_secret("MY-COMPLEX-API-KEY-V2")
    assert result == {"value": "complex-value"}

    result = manager.get_secret("my_complex_api_key_v2")
    assert result == {"value": "complex-value"}


def test_get_secret_with_unused_parameters(monkeypatch):
    """Test that unused parameters (secret_type, secret_group_name) don't affect behavior.

    Verifies that the compatibility parameters secret_type and secret_group_name
    can be passed but don't affect the secret retrieval behavior.
    """
    monkeypatch.setenv("GBSERVER_SECRET_API_KEY", "test-value")
    manager = EnvSpaceSecretManager(uri="env://")

    # Call with all parameters
    result = manager.get_secret(
        "api_key", secret_type="custom_type", secret_group_name="custom_group"
    )
    assert result == {"value": "test-value"}


def test_get_secrets_with_username_parameter(monkeypatch):
    """Test that get_secrets() ignores the username parameter.

    Verifies that the username parameter (for compatibility) doesn't affect
    the results returned by get_secrets().
    """
    monkeypatch.setenv("GBSERVER_SECRET_KEY1", "value1")
    monkeypatch.setenv("GBSERVER_SECRET_KEY2", "value2")

    manager = EnvSpaceSecretManager(uri="env://")

    # Should return same results regardless of username
    secrets1 = manager.get_secrets(username=None)
    assert secrets1 is not None
    secrets2 = manager.get_secrets(username="testuser")
    assert secrets2 is not None

    assert secrets1 == secrets2
    assert secrets1 == {"KEY1": "value1", "KEY2": "value2"}
