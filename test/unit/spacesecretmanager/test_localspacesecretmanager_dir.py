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

"""Unit tests for LocalSpaceSecretManager secrets_dir resolution.

Covers the optional/defaulted/expanded `secrets_dir` behavior that lets a
standalone space.yaml use `type: local` with `config: {}` (no secrets_dir) and
land at <gb_home>/space_secrets.
"""

from pathlib import Path

from gbserver.spacesecretmanager.localspacesecretmanager import LocalSpaceSecretManager


def test_default_secrets_dir_uses_gb_home(tmp_path, monkeypatch):
    """No secrets_dir -> <gb_home>/space_secrets, resolved from GB_HOME_DIR at call time."""
    gb_home = tmp_path / "gbhome"
    monkeypatch.setenv("GB_HOME_DIR", str(gb_home))

    manager = LocalSpaceSecretManager(uri="local")

    assert manager.dir == gb_home / "space_secrets"


def test_explicit_secrets_dir_is_expanded(tmp_path, monkeypatch):
    """~ and ${ENV} in an explicit secrets_dir are expanded."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MY_SECRETS_BASE", str(tmp_path / "base"))

    tilde = LocalSpaceSecretManager(uri="local", secrets_dir="~/space_secrets")
    assert tilde.dir == tmp_path / "space_secrets"

    envvar = LocalSpaceSecretManager(uri="local", secrets_dir="${MY_SECRETS_BASE}/s")
    assert envvar.dir == tmp_path / "base" / "s"


def test_explicit_path_object_preserved(tmp_path):
    """A plain absolute Path is used as-is."""
    manager = LocalSpaceSecretManager(uri="local", secrets_dir=tmp_path / "secrets")
    assert manager.dir == tmp_path / "secrets"


def test_default_dir_is_writable_crud(tmp_path, monkeypatch):
    """The defaulted dir supports a create/read round-trip."""
    monkeypatch.setenv("GB_HOME_DIR", str(tmp_path / "gbhome"))
    manager = LocalSpaceSecretManager(uri="local")

    manager.create_secret(
        secret_name="API_KEY", secret_value="hunter2", secret_group_name="group1"
    )

    assert (Path(manager.dir) / "group1.yaml").exists()
    assert manager.get_secrets()["API_KEY"] == "hunter2"
