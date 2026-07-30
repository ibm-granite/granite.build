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

"""Unit tests for POST /builds/ and POST /builds/validate identity binding.

req.username is the identity a submitted or validated build runs/resolves
secrets as (HackerOne 3875452 for submit_build; the same pattern was found
unfixed in validate_build during a follow-up audit — validate_build had no
Request param at all, so it couldn't check identity, and its space_uri path
bypasses space storage entirely). Both must reject a caller acting under a
DIFFERENT username unless the caller is a space/super admin explicitly
impersonating that user — the same confirm_space_write_access gate
PUT /builds/{id}/update already applies.

test/conftest.py's autouse `_mock_space_access` fixture stubs both
confirm_space_write_access (in this module) and has_space_write_access (in
utils) to an unconditional no-op/pass in mock mode, which would make every
test here trivially pass regardless of the fix under test. `_real_authz`
restores both real functions for the duration of each test below.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from gbserver.api import builds as builds_module
from gbserver.api.builds import (
    BuildSubmitRequest,
    BuildValidateRequest,
    BuildValidation,
    submit_build,
    validate_build,
)
from gbserver.api.utils import (
    confirm_space_write_access as _real_confirm_space_write_access,
)
from gbserver.api.utils import has_space_write_access as _real_has_space_write_access
from gbserver.storage.stored_space import StoredSpace

SPACE = "space-B"
VICTIM = "victim_b"
ATTACKER = "attacker_a"


@contextmanager
def _real_authz():
    """Restore the real confirm_space_write_access AND has_space_write_access,
    undoing the autouse `_mock_space_access` fixture's unconditional bypass."""
    with (
        patch(
            "gbserver.api.builds.confirm_space_write_access",
            side_effect=_real_confirm_space_write_access,
        ),
        patch(
            "gbserver.api.utils.has_space_write_access",
            side_effect=_real_has_space_write_access,
        ),
    ):
        yield


def _fake_request(login: str, email: str) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(data={"user": SimpleNamespace(login=login, email=email)})
    )


def _submit_req(username: str) -> BuildSubmitRequest:
    return BuildSubmitRequest(
        name="poc",
        build_archive="dGVzdA==",
        space_name=SPACE,
        username=username,
        tags=[],
    )


def _patched_storage():
    space = StoredSpace(name=SPACE, git_repo_uri="")
    fake_storage = SimpleNamespace(
        space_storage=SimpleNamespace(
            get_by_name=lambda name: space if name == SPACE else None
        ),
        build_storage=SimpleNamespace(add=lambda b: b.uuid),
    )
    return patch.object(builds_module, "get_admin_storage", return_value=fake_storage)


def test_submit_build_rejects_forged_username():
    with (
        _patched_storage(),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        with pytest.raises(HTTPException) as exc:
            submit_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                _submit_req(VICTIM),
            )
        assert exc.value.status_code == 401


def test_submit_build_allows_self_submission():
    with (
        _patched_storage(),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        resp = submit_build(
            _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
            _submit_req(ATTACKER),
        )
    assert resp.build_id


def test_submit_build_allows_admin_impersonation():
    with (
        _patched_storage(),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=True),
        patch("gbserver.api.utils.is_space_admin", return_value=True),
    ):
        resp = submit_build(
            _fake_request("admin_x", "admin_x@example.com"),
            _submit_req(VICTIM),
        )
    assert resp.build_id


# ------------------------------------------------------------------ validate_build

_NO_OP_VALIDATION = patch.object(
    BuildValidation,
    "validate_build_archive",
    return_value=MagicMock(is_valid=lambda: True, model_dump=lambda: {}),
)


def _validate_req(username: str, space_name: str = "", space_uri: str = ""):
    return BuildValidateRequest(
        build_archive="dGVzdA==",
        username=username,
        space_name=space_name,
        space_uri=space_uri,
    )


def test_validate_build_rejects_forged_username_via_space_name():
    with (
        _patched_storage(),
        _real_authz(),
        _NO_OP_VALIDATION,
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        with pytest.raises(HTTPException) as exc:
            validate_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                _validate_req(VICTIM, space_name=SPACE),
            )
        assert exc.value.status_code == 401


def test_validate_build_rejects_forged_username_via_space_uri():
    """space_uri bypasses space storage entirely, so there is no space to
    check admin-ness against — only super-admin can impersonate here. This
    path calls is_super_admin directly (bound into builds.py's own namespace
    at import time, not utils.py's), so that's what must be patched."""
    with (
        _patched_storage(),
        patch("gbserver.api.builds.is_super_admin", return_value=False),
        _NO_OP_VALIDATION,
    ):
        with pytest.raises(HTTPException) as exc:
            validate_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                _validate_req(VICTIM, space_uri="git://example/space.git"),
            )
        assert exc.value.status_code == 401


def test_validate_build_allows_self_validation_via_space_uri():
    with (
        _patched_storage(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        _NO_OP_VALIDATION,
    ):
        resp = validate_build(
            _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
            _validate_req(ATTACKER, space_uri="git://example/space.git"),
        )
    assert resp.status_code == 200


def test_validate_build_allows_admin_impersonation_via_space_name():
    with (
        _patched_storage(),
        _real_authz(),
        _NO_OP_VALIDATION,
        patch("gbserver.api.utils.is_super_admin", return_value=True),
        patch("gbserver.api.utils.is_space_admin", return_value=True),
    ):
        resp = validate_build(
            _fake_request("admin_x", "admin_x@example.com"),
            _validate_req(VICTIM, space_name=SPACE),
        )
    assert resp.status_code == 200
