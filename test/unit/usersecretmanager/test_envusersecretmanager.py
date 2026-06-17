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

"""Tests for the read-only env-backed EnvUserSecretManager."""

import pytest

from gbserver.usersecretmanager.envusersecretmanager import EnvUserSecretManager


@pytest.fixture
def manager():
    return EnvUserSecretManager(uri="env")


def test_get_user_secret(manager, monkeypatch):
    monkeypatch.setenv("GBSERVER_USER_SECRET_ALICE_API_KEY", "envsecret")
    assert manager.get_user_secret("alice", "API_KEY") == "envsecret"


def test_name_normalization(manager, monkeypatch):
    monkeypatch.setenv("GBSERVER_USER_SECRET_ALICE_API_KEY", "envsecret")
    # dashes/dots normalize to underscores, case-insensitive
    assert manager.get_user_secret("alice", "api-key") == "envsecret"
    assert manager.get_user_secret("alice", "api.key") == "envsecret"
    assert manager.get_user_secret("ALICE", "API_KEY") == "envsecret"


def test_list_and_get_all(manager, monkeypatch):
    monkeypatch.setenv("GBSERVER_USER_SECRET_ALICE_A", "1")
    monkeypatch.setenv("GBSERVER_USER_SECRET_ALICE_B", "2")
    monkeypatch.setenv("GBSERVER_USER_SECRET_BOB_C", "3")
    assert sorted(manager.list_user_secrets("alice")) == ["A", "B"]
    assert manager.get_user_secrets("alice") == {"A": "1", "B": "2"}
    # bob's secrets do not leak into alice's view
    assert manager.list_user_secrets("bob") == ["C"]


def test_missing_secret_is_none(manager):
    assert manager.get_user_secret("alice", "NOPE") is None


def test_is_read_only(manager):
    assert manager.read_only is True


def test_write_operations_raise(manager):
    with pytest.raises(NotImplementedError):
        manager.create_user_secret("alice", "K", "v")
    with pytest.raises(NotImplementedError):
        manager.update_user_secret("alice", "K", "v")
    with pytest.raises(NotImplementedError):
        manager.delete_user_secret("alice", "K")


def test_custom_prefix(monkeypatch):
    monkeypatch.setenv("MYPREFIX_ALICE_K", "v")
    manager = EnvUserSecretManager(uri="env", prefix="MYPREFIX_")
    assert manager.get_user_secret("alice", "K") == "v"
