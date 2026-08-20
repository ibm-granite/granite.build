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

"""Tests for the BlueVela project-membership gate.

The gate **fails closed**, so most of this file is one assertion repeated over
every way the lookup can come back inconclusive: no such account, no groups, a
non-zero ``id``, unparseable output, a tunnel that raises. Each of those must
deny. A gate that allowed on error would be bypassable exactly when the login
nodes misbehave, which is precisely when someone would be trying.

The other half covers the rollout mode, which is load-bearing rather than
cosmetic: the gate assumes the BV account name equals the granite.build login,
and that assumption is unverified against a real login node. If it is wrong,
``enforce`` denies every build — so ``shadow`` must compute and log the same
decision while letting the build through, and ``off`` must not even issue the
lookups.
"""

from __future__ import annotations

import shlex
from unittest.mock import AsyncMock

import pytest

from gbserver.environment import bv_project_access as bpa
from gbserver.environment.bv_project_access import (
    BvProjectAccessDenied,
    confirm_any_project_membership,
    confirm_any_project_membership_without_tunnel,
    has_any_project_membership,
    parse_group_names,
    project_groups,
)

LOGIN = "koyfman"

# `id -Gn` output shapes, as observed on a login node: names on one line,
# space separated.
GROUPS_WITH_PROJECT = "users granite-build proj_guardian proj_data-eng\n"
GROUPS_NO_PROJECT = "users granite-build wheel\n"
PASSWD_LINE = (
    "koyfman:*:1001:2001:koyfman@us.ibm.com;123456;Yan Koyfman:/u/koyfman:/bin/bash\n"
)


class _Tunnel:
    """Tunnel double dispatching on the command string, recording call order."""

    def __init__(
        self,
        *,
        id_stdout: str = GROUPS_WITH_PROJECT,
        id_rc: int = 0,
        id_stderr: str = "",
        passwd_stdout: str = PASSWD_LINE,
        passwd_rc: int = 0,
        raises: Exception | None = None,
    ):
        self.commands: list[str] = []
        self._id_stdout = id_stdout
        self._id_rc = id_rc
        self._id_stderr = id_stderr
        self._passwd_stdout = passwd_stdout
        self._passwd_rc = passwd_rc
        self._raises = raises
        self.run_remote_with_retries = AsyncMock(side_effect=self._run)

    async def _run(self, cmd, raise_on_error=True):
        self.commands.append(cmd)
        if self._raises is not None:
            raise self._raises
        if cmd.startswith("id -Gn"):
            return (self._id_rc, self._id_stdout, self._id_stderr)
        if cmd.startswith("getent passwd"):
            return (self._passwd_rc, self._passwd_stdout, "")
        raise AssertionError(f"unexpected command: {cmd}")

    @property
    def id_calls(self) -> list[str]:
        return [c for c in self.commands if c.startswith("id -Gn")]


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setattr(bpa, "BV_PROJECT_ACCESS_MODE", "enforce")


@pytest.fixture
def shadow(monkeypatch):
    monkeypatch.setattr(bpa, "BV_PROJECT_ACCESS_MODE", "shadow")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parses_id_gn_output():
    assert parse_group_names(GROUPS_WITH_PROJECT) == [
        "users",
        "granite-build",
        "proj_guardian",
        "proj_data-eng",
    ]


@pytest.mark.parametrize("empty", ["", "   ", "\n", None])
def test_empty_id_output_yields_no_groups(empty):
    assert parse_group_names(empty) == []


def test_selects_only_project_groups():
    assert project_groups(parse_group_names(GROUPS_WITH_PROJECT)) == [
        "proj_guardian",
        "proj_data-eng",
    ]


def test_the_bare_prefix_is_not_a_project():
    # A group literally named "proj_" names no project; treating it as one would
    # hand membership to anyone who happens to be in it.
    assert project_groups(["proj_", "users"]) == []


def test_a_numeric_gid_is_not_a_project():
    # `id -Gn` prints the raw GID when a group has no name, and still exits 0.
    # An unresolvable GID is not evidence of project membership.
    assert project_groups(parse_group_names("users 4096 31337\n")) == []


# ---------------------------------------------------------------------------
# has_any_project_membership — the allow path, then every deny path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_member_of_a_project_is_allowed():
    tunnel = _Tunnel()
    allowed, _reason = await has_any_project_membership(tunnel, LOGIN)
    assert allowed is True


@pytest.mark.asyncio
async def test_lookup_is_keyed_and_quoted():
    """Only keyed lookups — never an enumeration of passwd or the group table.

    ``shlex.quote`` adds no quoting to a login that needs none, so the expected
    command is built the same way rather than hard-coding quotes. The charset
    guard rejects anything that *would* need quoting before it reaches the shell
    (see test_an_unusable_login_is_rejected_before_the_shell), so quoting here is
    the second layer, not the first.
    """
    tunnel = _Tunnel()
    await has_any_project_membership(tunnel, LOGIN)
    assert tunnel.id_calls == [f"id -Gn {shlex.quote(LOGIN)}"]
    joined = " ".join(tunnel.commands)
    assert "getent group" not in joined, "must not enumerate the group table"
    # An argument-less `getent passwd` would enumerate every account; every call
    # this module makes names exactly one key.
    assert "getent passwd\n" not in joined and "getent passwd |" not in joined


@pytest.mark.asyncio
async def test_no_project_group_is_denied():
    allowed, reason = await has_any_project_membership(
        _Tunnel(id_stdout=GROUPS_NO_PROJECT), LOGIN
    )
    assert allowed is False
    assert "none matching" in reason


@pytest.mark.asyncio
async def test_no_such_account_is_denied():
    # `id` exits non-zero for an unknown name.
    allowed, reason = await has_any_project_membership(
        _Tunnel(id_rc=1, id_stdout="", id_stderr="id: 'nobody': no such user"), LOGIN
    )
    assert allowed is False
    assert "exited 1" in reason


@pytest.mark.asyncio
async def test_no_groups_resolved_is_denied():
    allowed, reason = await has_any_project_membership(_Tunnel(id_stdout="\n"), LOGIN)
    assert allowed is False
    assert "no groups resolved" in reason


@pytest.mark.asyncio
async def test_a_tunnel_error_is_denied():
    """The fail-closed case that will actually occur in production: the tunnel
    does not retry asyncssh ConnectionLost, so a dropped connection lands here."""
    allowed, reason = await has_any_project_membership(
        _Tunnel(raises=RuntimeError("Connection lost")), LOGIN
    )
    assert allowed is False
    assert "could not read groups" in reason


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["", "   ", None])
async def test_a_missing_login_is_denied_without_any_lookup(bad):
    tunnel = _Tunnel()
    allowed, _reason = await has_any_project_membership(tunnel, bad)
    assert allowed is False
    assert tunnel.commands == [], "must not touch the login node for an empty login"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        "a b",  # whitespace
        "x;id",  # command separator
        "$(whoami)",  # substitution
        "a" * 65,  # over length
        "-flag",  # leading dash could parse as an option
        "../root",  # path traversal shape
    ],
)
async def test_an_unusable_login_is_rejected_before_the_shell(bad):
    tunnel = _Tunnel()
    allowed, _reason = await has_any_project_membership(tunnel, bad)
    assert allowed is False
    assert tunnel.commands == []


# ---------------------------------------------------------------------------
# Rollout mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_off_issues_no_lookups_at_all(monkeypatch):
    monkeypatch.setattr(bpa, "BV_PROJECT_ACCESS_MODE", "off")
    tunnel = _Tunnel(id_stdout=GROUPS_NO_PROJECT)
    await confirm_any_project_membership(tunnel, LOGIN)  # must not raise
    assert tunnel.commands == [], "off must not cost a round trip"


@pytest.mark.asyncio
async def test_enforce_denies_a_non_member(enforce):
    with pytest.raises(BvProjectAccessDenied) as ei:
        await confirm_any_project_membership(
            _Tunnel(id_stdout=GROUPS_NO_PROJECT), LOGIN
        )
    msg = str(ei.value)
    assert LOGIN in msg
    # The message reaches the build status and its PR comment, where the reader
    # does not care about our internals.
    for leak in ("getent", "id -Gn", "proj_", "POSIX", "group", "tunnel"):
        assert leak not in msg, f"user-facing message leaks {leak!r}: {msg!r}"


@pytest.mark.asyncio
async def test_enforce_allows_a_member(enforce):
    await confirm_any_project_membership(_Tunnel(), LOGIN)  # must not raise


@pytest.mark.asyncio
async def test_shadow_computes_the_decision_but_allows(shadow):
    tunnel = _Tunnel(id_stdout=GROUPS_NO_PROJECT)
    await confirm_any_project_membership(tunnel, LOGIN)  # must not raise
    # It has to actually run the check, otherwise shadow logs prove nothing about
    # what enforce would do.
    assert tunnel.id_calls, "shadow must still perform the lookup"


@pytest.mark.asyncio
async def test_shadow_resolves_the_account_email_for_the_log(shadow):
    """Shadow mode is only actionable if the log says which BV account matched."""
    tunnel = _Tunnel(id_stdout=GROUPS_NO_PROJECT)
    await confirm_any_project_membership(tunnel, LOGIN)
    assert any(c.startswith("getent passwd") for c in tunnel.commands)


@pytest.mark.asyncio
async def test_the_allow_path_costs_no_extra_round_trip(enforce):
    tunnel = _Tunnel()
    await confirm_any_project_membership(tunnel, LOGIN)
    assert not any(
        c.startswith("getent passwd") for c in tunnel.commands
    ), "the diagnostic passwd lookup is only worth paying for on a denial"


@pytest.mark.asyncio
async def test_a_broken_passwd_lookup_still_denies_cleanly(enforce):
    """The diagnostic lookup must never change the verdict, or mask it."""
    tunnel = _Tunnel(id_stdout=GROUPS_NO_PROJECT, passwd_rc=2, passwd_stdout="")
    with pytest.raises(BvProjectAccessDenied):
        await confirm_any_project_membership(tunnel, LOGIN)


@pytest.mark.asyncio
async def test_an_unrecognized_mode_falls_back_to_off_not_enforce(monkeypatch):
    """A typo in the rollout flag must not be what starts denying every build."""
    monkeypatch.setattr(bpa, "BV_PROJECT_ACCESS_MODE", "enfoce")
    tunnel = _Tunnel(id_stdout=GROUPS_NO_PROJECT)
    await confirm_any_project_membership(tunnel, LOGIN)  # must not raise
    assert tunnel.commands == []


# ---------------------------------------------------------------------------
# Tunnel-less path (use_ssh=false) — the review finding this section closes.
#
# The gate CANNOT allow through this path — every one of its lookups is a keyed
# getent/id over the tunnel it does not have — so `off` is the only mode that is
# a plain no-op. `shadow` still logs, and `enforce` still denies, matching the
# module's fail-closed rule.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_tunnel_off_is_a_noop(monkeypatch):
    monkeypatch.setattr(bpa, "BV_PROJECT_ACCESS_MODE", "off")
    await confirm_any_project_membership_without_tunnel(LOGIN)  # must not raise


@pytest.mark.asyncio
async def test_without_tunnel_enforce_denies_regardless_of_login(enforce):
    """The whole point: an ssh-less deployment must NOT be a bypass. Even a login
    that would pass with a tunnel is denied here, because the gate cannot verify
    membership without one."""
    with pytest.raises(BvProjectAccessDenied) as ei:
        await confirm_any_project_membership_without_tunnel(LOGIN)
    # Same scrubbed-message contract as the with-tunnel path — the reader is a
    # user looking at their PR, not an operator.
    for leak in ("getent", "id -Gn", "proj_", "POSIX", "group", "tunnel"):
        assert leak not in str(ei.value)


@pytest.mark.asyncio
async def test_without_tunnel_shadow_does_not_raise(shadow):
    await confirm_any_project_membership_without_tunnel(LOGIN)  # must not raise


# ---------------------------------------------------------------------------
# Traceback suppression (review finding #2)
# ---------------------------------------------------------------------------


def test_denial_hides_traceback_from_pr_comment():
    """The scrubbed message reaches the PR; the stack frames must not.

    Read by ``get_readable_error_message``: without this marker, the exception's
    frames (``bv_project_access.py`` / ``confirm_any_project_membership``) end up
    in the ``<details>`` block, leaking exactly the mechanism the message hides.
    """
    from gbserver.utils.unwrap_errors import get_readable_error_message

    err = BvProjectAccessDenied("Access to a BlueVela project is required...")
    assert getattr(err, "hide_traceback_from_pr", False) is True

    body = get_readable_error_message(
        err, err_stack="Traceback...\n  File bv_project_access.py, line 251, in ..."
    )
    assert "Access to a BlueVela project" in body
    for leak in (
        "bv_project_access.py",
        "confirm_any_project_membership",
        "<details>",
        "Full Stack Trace",
        "Traceback",
    ):
        assert leak not in body, f"PR body still leaks {leak!r}"


def test_ordinary_exceptions_still_carry_a_traceback():
    """The suppressor must be OPT-IN, not a blanket format change."""
    from gbserver.utils.unwrap_errors import get_readable_error_message

    err = RuntimeError("something else went wrong")
    body = get_readable_error_message(err, err_stack="Traceback... some frames")
    assert "Full Stack Trace" in body
    assert "some frames" in body


def test_hidden_traceback_survives_being_re_raised_from():
    """`raise X from denial` must not smuggle the frames back in."""
    from gbserver.utils.unwrap_errors import get_readable_error_message

    denial = BvProjectAccessDenied("Access to a BlueVela project is required...")
    try:
        try:
            raise denial
        except BvProjectAccessDenied as inner:
            raise RuntimeError("wrapped by something else") from inner
    except RuntimeError as outer:
        body = get_readable_error_message(outer, err_stack="Traceback... frames here")
    assert "Full Stack Trace" not in body
    assert "Traceback" not in body


# ---------------------------------------------------------------------------
# Default mode
#
# `BV_PROJECT_ACCESS_MODE` is baked at import time from an env var, so the
# effective default is the one asserted here — with the env var unset, the gate
# runs in enforce.
# ---------------------------------------------------------------------------


def test_default_mode_is_enforce(monkeypatch):
    """With the env var unset, the gate runs in enforce — the gate is on.

    Flipping this default is a security change, not a cosmetic one: an
    operator who never touches the config must land on a *denying* mode, not
    a permissive one. Explicit fixture teardown reloads constants.py back to
    match the ambient env, so downstream tests can't be perturbed.
    """
    import importlib

    from gbserver.types import constants as gbconstants

    monkeypatch.delenv("GBSERVER_BV_PROJECT_ACCESS_MODE", raising=False)
    reloaded = importlib.reload(gbconstants)
    try:
        assert reloaded.BV_PROJECT_ACCESS_MODE == "enforce", (
            "the gate defaults ON — flipping this default is a security change; "
            "'off' or 'shadow' must be selected explicitly by the operator"
        )
    finally:
        importlib.reload(gbconstants)
