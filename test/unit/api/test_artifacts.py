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

"""Unit tests for GET /artifacts/decode?id= per-object authorization.

decode_uri(id=...) is an alternate read path to the same artifact
read_artifact protects (both load by uuid via get_admin_storage, which
bypasses row-level security). This exercises the real confirm_space_write_access
check directly, mocking only storage and the space-role lookups — no DB
required. The uri= mode never touches storage and must stay open to anyone.

test/conftest.py's autouse `_mock_space_access` fixture stubs
gbserver.api.artifacts.confirm_space_write_access to an unconditional no-op in
mock mode (so unrelated tests don't need real space setup), which would make
every test here trivially pass regardless of the fix under test. `_real_authz`
restores the real function for the duration of each test below.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from gbserver.api import artifacts as artifacts_module
from gbserver.api.artifacts import decode_uri, register_artifact
from gbserver.api.utils import (
    confirm_space_write_access as _real_confirm_space_write_access,
)
from gbserver.api.utils import has_space_write_access as _real_has_space_write_access
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.types.artifact import ArtifactType

VICTIM_OWNER = "victim_b"
VICTIM_SPACE = "space-B"
ATTACKER = "attacker_a"


@contextmanager
def _real_authz():
    """Restore the real confirm_space_write_access AND has_space_write_access.

    test/conftest.py's autouse `_mock_space_access` fixture stubs both out
    (confirm_space_write_access to an unconditional no-op, has_space_write_access
    to an unconditional (True, "standalone")) in mock mode. Restoring only one
    still leaves the other short-circuiting the real owner/admin decision.
    """
    with (
        patch(
            "gbserver.api.artifacts.confirm_space_write_access",
            side_effect=_real_confirm_space_write_access,
        ),
        patch(
            "gbserver.api.utils.has_space_write_access",
            side_effect=_real_has_space_write_access,
        ),
    ):
        yield


def _victim_artifact() -> ArtifactRegistration:
    art = ArtifactRegistration(
        type=ArtifactType.MODEL,
        uri="hf://huggingface.co/models/team-b/private-model",
        space_name=VICTIM_SPACE,
        username=VICTIM_OWNER,
    )
    art.uuid = "11111111-1111-1111-1111-111111111111"
    return art


class _FakeRegistry:
    def __init__(self, item):
        self._item = item

    def get_by_uuid(self, uuid):
        return self._item if uuid == self._item.uuid else None


def _fake_request(login: str, email: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(data={"user": SimpleNamespace(login=login, email=email)})
    )


def _patched_storage(artifact):
    fake_storage = SimpleNamespace(artifact_registry=_FakeRegistry(artifact))
    return patch.object(
        artifacts_module, "get_admin_storage", return_value=fake_storage
    )


def test_decode_uri_by_id_rejects_non_owner_non_admin():
    art = _victim_artifact()
    with (
        _patched_storage(art),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        with pytest.raises(HTTPException) as exc:
            decode_uri(_fake_request(ATTACKER, f"{ATTACKER}@example.com"), id=art.uuid)
        assert exc.value.status_code == 401


def test_decode_uri_by_id_allows_owner():
    art = _victim_artifact()
    with (
        _patched_storage(art),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        resp = decode_uri(
            _fake_request(VICTIM_OWNER, f"{VICTIM_OWNER}@example.com"), id=art.uuid
        )
    assert resp.uri == art.uri


def test_decode_uri_by_id_allows_space_admin():
    art = _victim_artifact()
    with (
        _patched_storage(art),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=True),
    ):
        resp = decode_uri(
            _fake_request("space_admin_x", "space_admin_x@example.com"), id=art.uuid
        )
    assert resp.uri == art.uri


def test_decode_uri_by_uri_requires_no_auth_and_touches_no_storage():
    """The uri= mode never loads a stored object, so it must stay open to any
    authenticated caller regardless of space membership."""
    with patch.object(artifacts_module, "get_admin_storage") as get_storage:
        resp = decode_uri(
            _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
            uri="hf://huggingface.co/models/anyone/anything",
        )
    get_storage.assert_not_called()
    assert resp.uri == "hf://huggingface.co/models/anyone/anything"


# ------------------------------------------------------------------ register_artifact


def _new_artifact(username: str) -> ArtifactRegistration:
    return ArtifactRegistration(
        type=ArtifactType.MODEL,
        uri="hf://huggingface.co/models/team-b/new-model",
        space_name=VICTIM_SPACE,
        username=username,
    )


def _registry_storage():
    fake_storage = SimpleNamespace(
        artifact_registry=SimpleNamespace(add=lambda a: None)
    )
    return patch.object(
        artifacts_module, "get_admin_storage", return_value=fake_storage
    )


def test_register_artifact_rejects_forged_username():
    with (
        _registry_storage(),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        with pytest.raises(HTTPException) as exc:
            register_artifact(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                _new_artifact(VICTIM_OWNER),
            )
        assert exc.value.status_code == 401


def test_register_artifact_allows_self_registration():
    with (
        _registry_storage(),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        resp = register_artifact(
            _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
            _new_artifact(ATTACKER),
        )
    assert resp.registered.username == ATTACKER


def test_register_artifact_allows_admin_impersonation():
    with (
        _registry_storage(),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=True),
        patch("gbserver.api.utils.is_space_admin", return_value=True),
    ):
        resp = register_artifact(
            _fake_request("admin_x", "admin_x@example.com"),
            _new_artifact(VICTIM_OWNER),
        )
    assert resp.registered.username == VICTIM_OWNER
