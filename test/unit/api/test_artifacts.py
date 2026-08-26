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

"""Unit tests for GET /artifacts/decode?id= and GET /artifacts/{id} per-object
authorization.

decode_uri(id=...) is an alternate read path to the same artifact
read_artifact protects (both load by uuid via get_admin_storage, which
bypasses row-level security) -- but the two are deliberately NOT at the same
access level. decode_uri(id=) stays at write access (confirm_space_write_access)
because it resolves additional metadata (e.g. resource_group_id for hf://
URIs) not present on the stored object; read_artifact was loosened to member
access (confirm_space_member_access) because list_artifacts() already returns
that exact object -- including its uri -- to any space member, so write
access there wasn't protecting anything the list didn't already expose. Both
checks are exercised directly here, mocking only storage and the space-role
lookups — no DB required. decode_uri's uri= mode never touches storage and
must stay open to anyone.

test/conftest.py's autouse `_mock_space_access` fixture stubs
gbserver.api.artifacts.confirm_space_write_access to an unconditional no-op,
and gbserver.api.utils.is_super_admin to an unconditional True, in mock mode
(so unrelated tests don't need real space setup) — either of which would make
every test here trivially pass regardless of the fix under test. `_real_authz`
restores confirm_space_write_access/has_space_write_access for the
decode_uri(id=)/register_artifact tests; the read_artifact tests patch
is_super_admin and space_access_check directly instead, since
confirm_space_member_access is never stubbed by conftest.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from unit.api._space_scoping_test_helpers import (
    ALICE,
    ALICE_SPACE,
    BOB_SPACE,
)
from unit.api._space_scoping_test_helpers import fake_request as _fake_request
from unit.api._space_scoping_test_helpers import row_matches as _row_matches
from unit.api._space_scoping_test_helpers import set_alice_access as _set_alice_access

from gbserver.api import artifacts as artifacts_module
from gbserver.api.artifacts import (
    decode_uri,
    list_artifact_tags,
    list_artifacts,
    read_artifact,
    register_artifact,
)
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


# ------------------------------------------------------------------ read_artifact


def test_read_artifact_allows_owner():
    art = _victim_artifact()
    with (
        _patched_storage(art),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.space_access_check", return_value=False),
    ):
        resp = read_artifact(
            _fake_request(VICTIM_OWNER, f"{VICTIM_OWNER}@example.com"), art.uuid
        )
    assert resp.artifact.uuid == art.uuid


def test_read_artifact_allows_non_owner_space_member():
    """The behavior change under review: a non-owner, non-admin member of the
    artifact's space is now allowed, matching list_artifacts()."""
    art = _victim_artifact()
    with (
        _patched_storage(art),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.space_access_check", return_value=True),
    ):
        resp = read_artifact(
            _fake_request("teammate_c", "teammate_c@example.com"), art.uuid
        )
    assert resp.artifact.uuid == art.uuid


def test_read_artifact_rejects_non_member():
    art = _victim_artifact()
    with (
        _patched_storage(art),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.space_access_check", return_value=False),
    ):
        with pytest.raises(HTTPException) as exc:
            read_artifact(_fake_request(ATTACKER, f"{ATTACKER}@example.com"), art.uuid)
        assert exc.value.status_code == 401


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


# ------------------------------------------------------------------ list_artifacts / list_artifact_tags
#
# Regression coverage for the cross-space list-endpoint disclosure: list_artifacts
# and list_artifact_tags built their row_filter straight from query params and
# called storage.artifact_registry.get_by_where() with no authorization check at
# all, so any authenticated user could enumerate artifacts (including internal
# HuggingFace/Lakehouse URIs) from every space regardless of membership.
# scope_space_name_filter() (api/utils.py) closes the gap.


def _artifact_in(
    space_name: str, owner: str, name: str, tags=None
) -> ArtifactRegistration:
    return ArtifactRegistration(
        type=ArtifactType.MODEL,
        uri=f"hf://huggingface.co/models/{space_name}/{name}",
        space_name=space_name,
        username=owner,
        name=name,
        tags=tags or [],
    )


def _patched_list_registry(artifacts: list):
    """A get_admin_storage() stand-in whose artifact_registry.get_by_where()
    actually applies the row_filter, so the tests prove the space_name value
    computed by scope_space_name_filter() is what narrows the result set."""
    fake_storage = SimpleNamespace(
        artifact_registry=SimpleNamespace(
            get_by_where=lambda where=None: [
                a for a in artifacts if _row_matches(a, where or {})
            ]
        )
    )
    return patch.object(
        artifacts_module, "get_admin_storage", return_value=fake_storage
    )


def test_list_artifacts_excludes_other_spaces_for_non_admin():
    alice_art = _artifact_in(ALICE_SPACE, ALICE, "alice-model")
    bob_art = _artifact_in(BOB_SPACE, "bob", "bob-dataset")
    with (
        _patched_list_registry([alice_art, bob_art]),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        _set_alice_access(manager)
        resp = list_artifacts(_fake_request(ALICE, f"{ALICE}@example.com"))
    assert {a.space_name for a in resp.artifacts} == {ALICE_SPACE}


def test_list_artifacts_rejects_explicit_cross_space_request():
    """Even an explicit space_name=space-B (bob's space) must return nothing,
    not bob's artifacts -- the PoC from the report leaked exactly this via
    internal HuggingFace/Lakehouse URIs."""
    bob_art = _artifact_in(BOB_SPACE, "bob", "bob-dataset")
    with (
        _patched_list_registry([bob_art]),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        _set_alice_access(manager)
        resp = list_artifacts(
            _fake_request(ALICE, f"{ALICE}@example.com"), space_name=BOB_SPACE
        )
    assert resp.artifacts == []


def test_list_artifacts_unrestricted_for_super_admin():
    alice_art = _artifact_in(ALICE_SPACE, ALICE, "alice-model")
    bob_art = _artifact_in(BOB_SPACE, "bob", "bob-dataset")
    with (
        _patched_list_registry([alice_art, bob_art]),
        patch("gbserver.api.utils.is_super_admin", return_value=True),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        resp = list_artifacts(_fake_request("admin_x", "admin_x@example.com"))
    assert {a.space_name for a in resp.artifacts} == {ALICE_SPACE, BOB_SPACE}
    manager.assert_not_called()


def test_list_artifact_tags_excludes_other_spaces_for_non_admin():
    alice_art = _artifact_in(ALICE_SPACE, ALICE, "alice-model", tags=["alpha"])
    bob_art = _artifact_in(BOB_SPACE, "bob", "bob-dataset", tags=["beta"])
    with (
        _patched_list_registry([alice_art, bob_art]),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.get_space_access_manager") as manager,
    ):
        _set_alice_access(manager)
        tags = list_artifact_tags(_fake_request(ALICE, f"{ALICE}@example.com"))
    assert tags == ["alpha"]
