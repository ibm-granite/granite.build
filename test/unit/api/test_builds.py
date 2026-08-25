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
    BuildRestartRequest,
    BuildSubmitRequest,
    BuildValidateRequest,
    BuildValidation,
    get_build_archive,
    read_build,
    request_cancellation,
    restart_build,
    submit_build,
    validate_build,
)
from gbserver.api.utils import (
    confirm_space_write_access as _real_confirm_space_write_access,
)
from gbserver.api.utils import has_space_write_access as _real_has_space_write_access
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_space import StoredSpace
from gbserver.types.status import Status

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


# ------------------------------------------------------------------ read_build / get_build_archive
#
# Regression coverage for the asymmetry fixed here: read_build and
# get_build_archive used to call the owner/admin-only authorize_build_access
# (via #201's confirm_space_write_access), while get_build_status/
# get_buildevents from that same PR correctly used the broader
# authorize_build_read_access (any space member). A build's owner is VICTIM
# throughout; MEMBER is a different user who is a member of SPACE but not its
# owner, and OUTSIDER is a user with no relationship to SPACE at all.
#
# Both endpoints' current (fixed) path is authorize_build_read_access ->
# confirm_space_member_access -> has_space_member_access, which only consults
# is_super_admin and space_access_check — not confirm_space_write_access,
# has_space_write_access, or is_space_admin. _real_authz() and the
# is_space_admin patch below don't affect that path today; they're kept as a
# tripwire so that if either endpoint ever regresses back onto the
# owner/admin-only authorize_build_access, the non_owner_space_member case
# (which _mock_space_access would otherwise mock into an unconditional pass)
# fails instead of silently succeeding.

MEMBER = "member_c"
OUTSIDER = "outsider_d"

# Base64 of an empty (but structurally valid) zip, so get_build_archive's
# zipfile.ZipFile() call succeeds and returns {"files": {}}.
_EMPTY_ZIP_B64 = "UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA=="


def _stored_build(owner: str) -> StoredBuild:
    return StoredBuild.create(
        name="poc",
        space_name=SPACE,
        source_uri="",
        username=owner,
        build_archive=_EMPTY_ZIP_B64,
    )


def _patched_build_storage(build: StoredBuild):
    fake_storage = SimpleNamespace(
        build_storage=SimpleNamespace(
            get_by_uuid=lambda uuid: build if uuid == build.uuid else None
        ),
    )
    return patch.object(builds_module, "get_admin_storage", return_value=fake_storage)


def _assert_read_build_ok(resp, build: StoredBuild) -> None:
    assert resp.build.uuid == build.uuid


def _assert_get_build_archive_ok(resp, build: StoredBuild) -> None:
    assert resp == {"files": {}}


@pytest.mark.parametrize(
    "endpoint,assert_ok",
    [
        (read_build, _assert_read_build_ok),
        (get_build_archive, _assert_get_build_archive_ok),
    ],
    ids=["read_build", "get_build_archive"],
)
@pytest.mark.parametrize(
    "caller,space_access,should_succeed",
    [
        (VICTIM, False, True),  # owner
        (MEMBER, True, True),  # non-owner space member — the fix
        (OUTSIDER, False, False),  # non-member
    ],
    ids=["owner", "non_owner_space_member", "non_member"],
)
def test_read_access_by_space_membership(
    endpoint, assert_ok, caller, space_access, should_succeed
):
    build = _stored_build(VICTIM)
    with (
        _patched_build_storage(build),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
        patch("gbserver.api.utils.space_access_check", return_value=space_access),
    ):
        if should_succeed:
            resp = endpoint(_fake_request(caller, f"{caller}@example.com"), build.uuid)
            assert_ok(resp, build)
        else:
            with pytest.raises(HTTPException) as exc:
                endpoint(_fake_request(caller, f"{caller}@example.com"), build.uuid)
            assert exc.value.status_code == 401


# --- request_cancellation: FAILED-with-retries window (regression) ------------
#
# After an attempt fails, the build sits in FAILED until the retry loop flips it
# back to RUNNING. A cancel landing in that window must be honored (set to
# CANCELLED) rather than rejected as "already finished". See
# gbserver.api.builds.request_cancellation.


def _cancel_build(status: Status, has_retries: bool, uuid: str = "build-1"):
    """A minimal stand-in for StoredBuild exposing only what request_cancellation
    reads: uuid, status, and has_retries_remaining()."""
    return SimpleNamespace(
        uuid=uuid,
        status=status,
        has_retries_remaining=lambda: has_retries,
    )


class _FakeBuildStorage:
    """Fake IStoredBuildStorage whose update_fields honors the should_update guard
    against the build as it currently exists in storage (stored_status)."""

    def __init__(self, stored_status: Status):
        self._stored_status = stored_status

    def update_fields(self, uuid, fields, should_update=None):
        # Emulate the atomic guard: check against the live stored status.
        current = SimpleNamespace(status=self._stored_status)
        if should_update is not None and not should_update(current):
            return None
        return SimpleNamespace(uuid=uuid, status=fields["status"])


def test_cancel_failed_with_retries_sets_cancelled():
    build = _cancel_build(Status.FAILED, has_retries=True)
    storage = _FakeBuildStorage(stored_status=Status.FAILED)
    result = request_cancellation(storage, build)  # type: ignore[arg-type]
    assert result.status == Status.CANCELLED


def test_cancel_failed_without_retries_rejected_412():
    build = _cancel_build(Status.FAILED, has_retries=False)
    storage = _FakeBuildStorage(stored_status=Status.FAILED)
    with pytest.raises(HTTPException) as exc:
        request_cancellation(storage, build)  # type: ignore[arg-type]
    assert exc.value.status_code == 412


def test_cancel_failed_with_retries_races_running_409():
    # The runner flipped FAILED -> RUNNING (re-running the retry in place) before
    # the cancel's write, so the should_update guard rejects the CANCELLED write
    # and the client gets a 409.
    build = _cancel_build(Status.FAILED, has_retries=True)
    storage = _FakeBuildStorage(stored_status=Status.RUNNING)
    with pytest.raises(HTTPException) as exc:
        request_cancellation(storage, build)  # type: ignore[arg-type]
    assert exc.value.status_code == 409


def test_cancel_running_sets_cancel_requested():
    build = _cancel_build(Status.RUNNING, has_retries=True)
    storage = _FakeBuildStorage(stored_status=Status.RUNNING)
    result = request_cancellation(storage, build)  # type: ignore[arg-type]
    assert result.status == Status.CANCEL_REQUESTED


# ------------------------------------------------------------------ restart_build
#
# Option A: a restart reuses the SAME build id. restart_build re-opens a
# finished build in place (status -> SUBMITTED, retry_count reset) so the
# BuildWatcher re-dispatches it onto a fresh runner. There is no retry chain and
# no new build id.


def _restart_build_storage(builds: dict):
    """A build_storage mock backed by a {uuid: StoredBuild} dict, supporting the
    get_by_uuid / update_fields surface restart_build + reopen_finished_build
    use. update_fields honors the should_update guard against the build as it
    currently exists in storage (so a race can be simulated by pre-seeding a
    non-finished status)."""

    def _update_fields(uuid, fields, should_update=None):
        current = builds.get(uuid)
        if current is None:
            return None
        if should_update is not None and not should_update(current):
            return None
        updated = current.model_copy(update=fields)
        builds[uuid] = updated
        return updated

    return SimpleNamespace(get_by_uuid=builds.get, update_fields=_update_fields)


def _patched_restart_storage(builds: dict):
    space = StoredSpace(name=SPACE, git_repo_uri="")
    fake_storage = SimpleNamespace(
        space_storage=SimpleNamespace(
            get_by_name=lambda name: space if name == SPACE else None
        ),
        build_storage=_restart_build_storage(builds),
    )
    return patch.object(builds_module, "get_admin_storage", return_value=fake_storage)


def _prior_build(username: str, status: Status = Status.FAILED) -> StoredBuild:
    return StoredBuild(
        name="poc",
        space_name=SPACE,
        source_uri="",
        username=username,
        build_archive="dGVzdA==",
        status=status,
        targets=["a", "b"],
        retry_count=3,
    )


def test_restart_build_missing_build_returns_404():
    with _patched_restart_storage({}):
        with pytest.raises(HTTPException) as exc:
            restart_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                BuildRestartRequest(build_id="does-not-exist"),
            )
    assert exc.value.status_code == 404


def test_restart_build_rejects_active_build_409():
    prior = _prior_build(ATTACKER, status=Status.RUNNING)
    with _patched_restart_storage({prior.uuid: prior}):
        with pytest.raises(HTTPException) as exc:
            restart_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                BuildRestartRequest(build_id=prior.uuid),
            )
    assert exc.value.status_code == 409


def test_restart_build_reopens_same_build_in_place():
    prior = _prior_build(ATTACKER, status=Status.FAILED)
    builds = {prior.uuid: prior}
    with (
        _patched_restart_storage(builds),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        resp = restart_build(
            _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
            BuildRestartRequest(build_id=prior.uuid),
        )
    # Same build id — no new build created.
    assert resp.build_id == prior.uuid
    reopened = builds[prior.uuid]
    # Re-opened in place: SUBMITTED for re-dispatch, fresh retry budget.
    assert reopened.status == Status.SUBMITTED
    assert reopened.retry_count == 0
    # Definition/targets are untouched (already on the build).
    assert reopened.build_archive == prior.build_archive
    assert reopened.targets == prior.targets


def test_restart_build_rejects_succeeded_build_409():
    """A fully-succeeded build has nothing to restart (every target already
    succeeded), so it is rejected with 409 and left untouched — even for an
    authorized owner. This is the SUCCESS carve-out on top of is_finished()."""
    prior = _prior_build(ATTACKER, status=Status.SUCCESS)
    builds = {prior.uuid: prior}
    with (
        _patched_restart_storage(builds),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        with pytest.raises(HTTPException) as exc:
            restart_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                BuildRestartRequest(build_id=prior.uuid),
            )
    assert exc.value.status_code == 409
    # Not re-opened: the build stays SUCCESS, no flip to SUBMITTED.
    assert builds[prior.uuid].status == Status.SUCCESS


def test_restart_build_reopen_race_returns_409():
    # The build read as FAILED, but a concurrent writer flipped it to RUNNING
    # before the guarded flip's write, so the should_update guard rejects it and
    # the client gets a 409 rather than a fresh runner attaching to a live build.
    prior = _prior_build(ATTACKER, status=Status.FAILED)

    class _RacingStorage:
        def get_by_uuid(self, uuid):
            return prior if uuid == prior.uuid else None

        def update_fields(self, uuid, fields, should_update=None):
            live = SimpleNamespace(status=Status.RUNNING)
            if should_update is not None and not should_update(live):
                return None
            return prior.model_copy(update=fields)

    space = StoredSpace(name=SPACE, git_repo_uri="")
    storage = SimpleNamespace(
        space_storage=SimpleNamespace(
            get_by_name=lambda name: space if name == SPACE else None
        ),
        build_storage=_RacingStorage(),
    )
    with (
        patch.object(builds_module, "get_admin_storage", return_value=storage),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        with pytest.raises(HTTPException) as exc:
            restart_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                BuildRestartRequest(build_id=prior.uuid),
            )
    assert exc.value.status_code == 409


def test_restart_build_rejects_forged_username_as_404():
    """An unauthorized caller gets 404, identical to a nonexistent build — not a
    401/409 that would confirm the id is real or leak its liveness. Collapsing
    them removes the id oracle across spaces the caller cannot reach."""
    prior = _prior_build(VICTIM, status=Status.FAILED)
    with (
        _patched_restart_storage({prior.uuid: prior}),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        with pytest.raises(HTTPException) as exc:
            restart_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                BuildRestartRequest(build_id=prior.uuid),
            )
    assert exc.value.status_code == 404
    assert exc.value.detail == f"Build {prior.uuid} not found"


def test_restart_build_authz_precedes_status_disclosure():
    """An unauthorized caller must not learn a build's liveness: authz is enforced
    BEFORE the is_finished() 409, so restarting another user's *active* build
    returns the not-found 404, not a 409 that would leak that it is live."""
    prior = _prior_build(VICTIM, status=Status.RUNNING)
    with (
        _patched_restart_storage({prior.uuid: prior}),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        with pytest.raises(HTTPException) as exc:
            restart_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                BuildRestartRequest(build_id=prior.uuid),
            )
    assert exc.value.status_code == 404
