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
    BuildContinueRequest,
    BuildSubmitRequest,
    BuildValidateRequest,
    BuildValidation,
    continue_build,
    submit_build,
    validate_build,
)
from gbserver.api.utils import (
    confirm_space_write_access as _real_confirm_space_write_access,
)
from gbserver.api.utils import has_space_write_access as _real_has_space_write_access
from gbserver.storage.stored_build import (
    StoredBuild,
    create_continuation_build,
    get_retry_chain_members,
)
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


# ------------------------------------------------------------------ continue_build


def _fake_build_storage(builds: dict):
    """A build_storage mock backed by a {uuid: StoredBuild} dict that supports
    the get_by_uuid / add / update surface create_continuation_build uses."""

    def _add(b):
        builds[b.uuid] = b
        return b.uuid

    def _update(b):
        builds[b.uuid] = b
        return b

    return SimpleNamespace(
        get_by_uuid=builds.get,
        add=_add,
        update=_update,
    )


def _patched_continue_storage(builds: dict):
    space = StoredSpace(name=SPACE, git_repo_uri="")
    fake_storage = SimpleNamespace(
        space_storage=SimpleNamespace(
            get_by_name=lambda name: space if name == SPACE else None
        ),
        build_storage=_fake_build_storage(builds),
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
    )


def test_continue_build_missing_build_returns_404():
    with _patched_continue_storage({}):
        with pytest.raises(HTTPException) as exc:
            continue_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                BuildContinueRequest(build_id="does-not-exist"),
            )
    assert exc.value.status_code == 404


def test_continue_build_rejects_active_build_409():
    prior = _prior_build(ATTACKER, status=Status.RUNNING)
    with _patched_continue_storage({prior.uuid: prior}):
        with pytest.raises(HTTPException) as exc:
            continue_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                BuildContinueRequest(build_id=prior.uuid),
            )
    assert exc.value.status_code == 409


def test_continue_build_rejects_when_chain_tip_still_active():
    """The finished-check applies to the chain TIP, not the passed-in member:
    continuing a finished root while a newer attempt is still RUNNING is a 409,
    so a fresh runner is never attached to a live tip."""
    root = _prior_build(ATTACKER, status=Status.FAILED)
    tip = _prior_build(ATTACKER, status=Status.RUNNING)
    tip.retry_of_build_id = root.uuid
    root.retry_build_id = tip.uuid
    with _patched_continue_storage({root.uuid: root, tip.uuid: tip}):
        with pytest.raises(HTTPException) as exc:
            # Pass the (finished) ROOT; the tip is still running.
            continue_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                BuildContinueRequest(build_id=root.uuid),
            )
    assert exc.value.status_code == 409
    # The 409 names the actual live attempt (the tip), not the passed-in build.
    assert tip.uuid in exc.value.detail


def test_continue_build_creates_linked_continuation():
    prior = _prior_build(ATTACKER, status=Status.FAILED)
    builds = {prior.uuid: prior}
    with (
        _patched_continue_storage(builds),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        resp = continue_build(
            _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
            BuildContinueRequest(build_id=prior.uuid),
        )
    assert resp.build_id and resp.build_id != prior.uuid
    # The response reports the resolved chain root (here the prior build itself).
    assert resp.root_build_id == prior.uuid
    continuation = builds[resp.build_id]
    # Fresh build, linked to the prior chain root, fresh retry budget, SUBMITTED.
    assert continuation.retry_of_build_id == prior.uuid
    assert continuation.retry_count == 0
    assert continuation.status == Status.SUBMITTED
    assert continuation.build_archive == prior.build_archive
    assert continuation.targets == prior.targets
    # Back-link set on the prior (chain tip) so the chain advances.
    assert builds[prior.uuid].retry_build_id == resp.build_id


def test_continue_build_accepts_mid_chain_member_and_links_to_root():
    # root -> tip (any member may be continued; continuation links to the root
    # and the back-link lands on the chain tip, not the passed-in member).
    root = _prior_build(ATTACKER, status=Status.FAILED)
    tip = _prior_build(ATTACKER, status=Status.FAILED)
    tip.retry_of_build_id = root.uuid
    root.retry_build_id = tip.uuid
    # Give the tip a distinct definition to prove the continuation is seeded from
    # the tip (latest attempt), not from the passed-in root.
    tip.targets = ["tip-target"]
    root.targets = ["root-target"]
    builds = {root.uuid: root, tip.uuid: tip}
    with (
        _patched_continue_storage(builds),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        resp = continue_build(
            _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
            BuildContinueRequest(build_id=root.uuid),
        )
    continuation = builds[resp.build_id]
    assert continuation.retry_of_build_id == root.uuid
    # The response reports the resolved root even though a mid-chain member was passed.
    assert resp.root_build_id == root.uuid
    # Definition is sourced from the tip (latest attempt), not the passed-in root.
    assert continuation.targets == ["tip-target"]
    # Tip gets the back-link; root's existing link is untouched.
    assert builds[tip.uuid].retry_build_id == resp.build_id
    assert builds[root.uuid].retry_build_id == tip.uuid


def test_repeated_continuations_linearize_into_one_chain():
    """Continuing the same root repeatedly appends to the current tip, so the
    chain stays linear (A -> B -> C) rather than branching into multiple chains
    sharing a root. `gb build status --follow-retries` then shows every
    continuation in a single walk."""
    root = _prior_build(ATTACKER, status=Status.FAILED)
    builds = {root.uuid: root}
    bs = _fake_build_storage(builds)

    # Continue root -> B, then continue root again -> C. Both attach to the tip.
    b = create_continuation_build(bs, builds[root.uuid])
    c = create_continuation_build(bs, builds[root.uuid])

    chain = [m.uuid for m in get_retry_chain_members(bs, builds[root.uuid])]
    assert chain == [root.uuid, b.uuid, c.uuid]
    # No back-link was overwritten: root -> B -> C, each single forward hop.
    assert builds[root.uuid].retry_build_id == b.uuid
    assert builds[b.uuid].retry_build_id == c.uuid
    # Every continuation links to the same resolved root.
    assert b.retry_of_build_id == root.uuid
    assert c.retry_of_build_id == root.uuid


def test_create_continuation_build_uses_passed_chain_without_rewalking():
    """When the caller (the /continue endpoint) has already resolved the chain, it
    passes it in and create_continuation_build must NOT walk it again — each member
    is an unindexed point read, so re-walking would double the reads."""
    root = _prior_build(ATTACKER, status=Status.FAILED)
    tip = _prior_build(ATTACKER, status=Status.FAILED)
    tip.retry_of_build_id = root.uuid
    root.retry_build_id = tip.uuid
    builds = {root.uuid: root, tip.uuid: tip}
    bs = _fake_build_storage(builds)
    chain = get_retry_chain_members(bs, root)

    with patch("gbserver.storage.stored_build.get_retry_chain_members") as walk:
        cont = create_continuation_build(bs, root, chain=chain)

    walk.assert_not_called()
    # Same linkage as the walk-it-yourself path: linked to root, back-link on tip.
    assert cont.retry_of_build_id == root.uuid
    assert builds[tip.uuid].retry_build_id == cont.uuid


def test_chain_walk_includes_intermediate_members_from_any_member():
    """get_retry_chain_members must return EVERY member of a flat-to-root chain
    (root -> mid -> tip), not just [self, root], regardless of which member it is
    called from. Target reuse (BuildRunner.__get_retry_chain_build_ids) relies on
    this: retry_of_build_id points every member at the root, so a backward walk
    would miss `mid` and re-run a target that first succeeded there."""
    root = _prior_build(ATTACKER, status=Status.FAILED)
    mid = _prior_build(ATTACKER, status=Status.FAILED)
    tip = _prior_build(ATTACKER, status=Status.FAILED)
    # Flat-to-root: mid and tip both point their retry_of_build_id at the root.
    mid.retry_of_build_id = root.uuid
    tip.retry_of_build_id = root.uuid
    root.retry_build_id = mid.uuid
    mid.retry_build_id = tip.uuid
    bs = _fake_build_storage({root.uuid: root, mid.uuid: mid, tip.uuid: tip})

    expected = [root.uuid, mid.uuid, tip.uuid]
    # The full chain is recovered from any starting member, and always includes mid.
    for member in (root, mid, tip):
        chain = [m.uuid for m in get_retry_chain_members(bs, member)]
        assert chain == expected
        assert mid.uuid in chain


def test_continue_build_rejects_forged_username_as_404():
    """An unauthorized caller gets 404, identical to a nonexistent build — not a
    401 that would confirm the id is real. Collapsing the two removes the id
    oracle: a caller without space access cannot tell a build id they may not
    reach from one that does not exist, so cannot enumerate ids across spaces."""
    prior = _prior_build(VICTIM, status=Status.FAILED)
    with (
        _patched_continue_storage({prior.uuid: prior}),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        with pytest.raises(HTTPException) as exc:
            continue_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                BuildContinueRequest(build_id=prior.uuid),
            )
    assert exc.value.status_code == 404
    # Same detail as the missing-build 404, so the two are indistinguishable.
    assert exc.value.detail == f"Build {prior.uuid} not found"


def test_continue_build_authz_precedes_status_disclosure():
    """An unauthorized caller must not learn a prior build's liveness: authz is
    enforced BEFORE the is_finished() 409, so continuing another user's *active*
    build returns the not-found 404, not a 409 that would leak that it is live."""
    prior = _prior_build(VICTIM, status=Status.RUNNING)
    with (
        _patched_continue_storage({prior.uuid: prior}),
        _real_authz(),
        patch("gbserver.api.utils.is_super_admin", return_value=False),
        patch("gbserver.api.utils.is_space_admin", return_value=False),
    ):
        with pytest.raises(HTTPException) as exc:
            continue_build(
                _fake_request(ATTACKER, f"{ATTACKER}@example.com"),
                BuildContinueRequest(build_id=prior.uuid),
            )
    assert exc.value.status_code == 404
