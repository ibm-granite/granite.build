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

"""Tests for the editable file-backed LocalUserSecretManager."""

import pytest

from gbserver.usersecretmanager.localusersecretmanager import LocalUserSecretManager


@pytest.fixture
def manager(tmp_path):
    return LocalUserSecretManager(uri="local", dir=str(tmp_path / "user_secrets"))


def test_create_then_get_and_list(manager):
    manager.create_user_secret("alice", "API_KEY", "hunter2")
    assert manager.get_user_secret("alice", "API_KEY") == "hunter2"
    assert manager.list_user_secrets("alice") == ["API_KEY"]
    assert manager.get_user_secrets("alice") == {"API_KEY": "hunter2"}


def test_users_are_isolated(manager):
    manager.create_user_secret("alice", "K", "a-value")
    manager.create_user_secret("bob", "K", "b-value")
    assert manager.get_user_secret("alice", "K") == "a-value"
    assert manager.get_user_secret("bob", "K") == "b-value"
    assert manager.list_user_secrets("alice") == ["K"]


def test_update(manager):
    manager.create_user_secret("alice", "K", "v1")
    manager.update_user_secret("alice", "K", "v2")
    assert manager.get_user_secret("alice", "K") == "v2"


def test_update_missing_raises(manager):
    with pytest.raises(ValueError):
        manager.update_user_secret("alice", "missing", "v")


def test_delete(manager):
    manager.create_user_secret("alice", "K", "v")
    manager.delete_user_secret("alice", "K")
    assert manager.list_user_secrets("alice") == []


def test_delete_missing_raises(manager):
    with pytest.raises(ValueError):
        manager.delete_user_secret("alice", "missing")


def test_missing_user_returns_empty(manager):
    assert manager.list_user_secrets("nobody") == []
    assert manager.get_user_secret("nobody", "K") is None
    assert manager.get_user_secrets("nobody") == {}


def test_value_is_persisted_encoded(manager, tmp_path):
    """Values are base64-encoded on disk, not stored in plaintext."""
    manager.create_user_secret("alice", "K", "supersecret")
    user_file = tmp_path / "user_secrets" / "alice.yaml"
    assert user_file.exists()
    on_disk = user_file.read_text()
    assert "supersecret" not in on_disk


def test_requires_dir():
    with pytest.raises(ValueError):
        LocalUserSecretManager(uri="local", dir="")


@pytest.mark.parametrize("bad_user", ["../etc", "a/b", "", "with space"])
def test_rejects_unsafe_user_id(manager, bad_user):
    with pytest.raises(ValueError):
        manager.list_user_secrets(bad_user)


def test_not_read_only(manager):
    assert manager.read_only is False
