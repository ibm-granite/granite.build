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

"""Tests for UserSecretManager auto-discovery and the selection factory."""

import pytest

from gbserver.usersecretmanager.envusersecretmanager import EnvUserSecretManager
from gbserver.usersecretmanager.factory import get_user_secret_manager
from gbserver.usersecretmanager.localusersecretmanager import LocalUserSecretManager
from gbserver.usersecretmanager.usersecretmanager import UserSecretManager


def test_autodiscovery_registers_backends():
    UserSecretManager.load_usersecretmanagers()
    keys = UserSecretManager.usersecretmanagers
    # The helper module factory.py must NOT be registered as a backend.
    assert "factory" not in keys
    for expected in ("env", "local", "ibmcloud"):
        assert expected in keys, f"{expected} backend not auto-discovered"


def test_get_usersecretmanager_local(tmp_path):
    manager = UserSecretManager.get_usersecretmanager("local", dir=str(tmp_path / "us"))
    assert isinstance(manager, LocalUserSecretManager)


def test_get_usersecretmanager_env():
    manager = UserSecretManager.get_usersecretmanager("env")
    assert isinstance(manager, EnvUserSecretManager)


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        UserSecretManager.get_usersecretmanager("does-not-exist")


def test_factory_defaults_to_local_in_standalone(tmp_path, monkeypatch):
    """Standalone mode set at runtime must select the IBM-free local backend.

    Regression: the standalone default used to be frozen at import time, so a
    process that imported constants under a non-standalone env and only later
    entered standalone mode wrongly defaulted to the ibmcloud backend.
    """
    monkeypatch.delenv("GBSERVER_USER_SECRET_MANAGER", raising=False)
    monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")
    monkeypatch.setenv("GB_HOME_DIR", str(tmp_path / "gb_home"))
    assert isinstance(get_user_secret_manager(), LocalUserSecretManager)


def test_factory_invalid_config_json_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("GBSERVER_USER_SECRET_MANAGER", "local")
    monkeypatch.setenv("GBSERVER_USER_SECRET_DIR", str(tmp_path / "us"))
    monkeypatch.setenv("GBSERVER_USER_SECRET_MANAGER_CONFIG", "{not valid json")
    with pytest.raises(ValueError):
        get_user_secret_manager()
