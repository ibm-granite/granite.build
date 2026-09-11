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

"""Gate: may this user run a build on BlueVela at all?

A build is allowed onto a BV/LSF environment only if the submitting user belongs
to at least one **BV project** — a POSIX group named ``{prefix}{project}`` on the
login nodes (``proj_*`` by default). This is deliberately a coarser question than
the one ``api/environment_files_paths.authorize_folder_access`` answers: that one
asks "may you read ``/proj/{folder}``" for one named folder, this one asks "do you
have any project at all", as a precondition for consuming GPU time.

WHERE THIS RUNS, AND WHY IT MATTERS. Called from ``Lsf.setup_bsub`` once the SSH
tunnel is already open — not at build submit. Two reasons:

* The tunnel is **already a prerequisite** for the build to run, so checking here
  adds no new dependency and no new failure mode. If the login nodes are down the
  build was doomed regardless. Measured while designing this: with all three BV
  login nodes unreachable, opening a tunnel took **372s** before failing. A
  fail-closed check at submit time would have put that six-minute path in front of
  every submission, including ones that would otherwise have queued fine.
* At submit time there is no email to check anyway (``StoredBuild`` has no email
  field) and the submit handler never parses the build archive, so it cannot even
  tell that a build targets BlueVela.

IDENTITY. At run time the only identity available is the granite.build login
(``EntityRunMetadata.username``), reaching ``setup_bsub`` through
``Run._add_to_run_kwargs`` as ``runmetadata``. We therefore assume **the BV
account name equals that login** and verify the account exists with a keyed
``getent passwd``. That assumption is the weak point of this module — see the
shadow-mode note below.

Only **keyed** lookups are used: ``getent passwd <login>`` and ``id -Gn <login>``.
Nothing here enumerates passwd or the group table. That mirrors
``environment_files_paths``, which likewise only ever looks up names it already
has, and it keeps the check O(1) regardless of how many projects or users exist.

FAILS CLOSED. Every inconclusive outcome denies: no passwd entry, no groups,
non-zero rc, unparseable output, or a tunnel error. And no tunnel at all — a
``use_ssh=false`` deployment routes through
``confirm_any_project_membership_without_tunnel``, which cannot verify anything
and therefore denies under ``enforce`` rather than silently allow. A gate that
allowed on error, or on the absence of the very channel it needs, would be
bypassable exactly when the login nodes misbehave.

Rollout mode. Default is ``enforce`` — the gate is on. ``shadow`` runs the check
and logs what it would have done while allowing the build through, and ``off``
turns the gate off entirely. Because ``enforce`` on an unverified deployment can
lock out every legitimate user (the assumption above), the sequence for a fresh
environment is ``shadow`` first until the logs show sensible decisions, THEN
``enforce``. See ``BV_PROJECT_ACCESS_MODE`` in types/constants.py.
"""

import re
import shlex
from typing import List, Optional, Tuple

from gbserver.types.constants import (
    BV_PROJECT_ACCESS_MODE,
    BV_PROJECT_ACCESS_MODE_ENFORCE,
    BV_PROJECT_ACCESS_MODE_OFF,
    BV_PROJECT_ACCESS_MODE_SHADOW,
    BV_PROJECT_ACCESS_MODES,
    BV_PROJECT_GROUP_PREFIX,
    ENV_VAR_GBSERVER_BV_PROJECT_ACCESS_MODE,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# A login we are willing to interpolate into a remote command. Even though every
# value is shlex.quote()d before it reaches the shell, a login is also echoed
# into logs and error messages, so it is validated to a conservative charset
# first: rejecting outright beats quoting something that should never have got
# this far. POSIX portable usernames plus the '-' and '.' that GHE logins use.
_LOGIN_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class BvProjectAccessDenied(Exception):
    """The submitting user has no BlueVela project, so the build must not run.

    The message is user-facing: it reaches the build's status and its PR comment,
    where the reader is someone who wants to run code on GPUs and has no interest
    in POSIX groups, ``getent``, or granite.build's internals. Name the
    requirement and the remedy, never the mechanism.
    """

    # Contract read by ``utils.unwrap_errors.get_readable_error_message``: the PR
    # comment shows the message only, never the traceback. Otherwise the frames
    # (``confirm_any_project_membership``, ``bv_project_access.py``) would leak
    # exactly the mechanism the message above is written to hide, and the
    # scrubbing tests here would guarantee something the reader never actually
    # sees. Class-level rather than per-instance so every raise site inherits it.
    hide_traceback_from_pr = True


def parse_group_names(id_gn_stdout: str) -> List[str]:
    """Split ``id -Gn`` output into group names.

    ``id -Gn`` prints names separated by single spaces on one line. Group names
    cannot contain whitespace, so a plain split is exact rather than a heuristic.

    A group whose GID has no name resolves to the bare number instead (``id``
    still exits 0 in that case); such entries simply never match the project
    prefix, which is the correct outcome — an unresolvable GID is not evidence of
    project membership.
    """
    return (id_gn_stdout or "").split()


def project_groups(group_names: List[str]) -> List[str]:
    """The subset of ``group_names`` that name a BV project."""
    prefix = BV_PROJECT_GROUP_PREFIX
    # A group equal to the bare prefix ("proj_") names no project, so require at
    # least one character after it.
    return [g for g in group_names if g.startswith(prefix) and len(g) > len(prefix)]


def resolve_mode() -> str:
    """The effective rollout mode, validated.

    An unrecognized value degrades to ``off`` with a loud log rather than to
    ``enforce``. Fail-closed governs the membership lookup; a typo in our own
    rollout flag is a configuration error, and it must not be the thing that
    starts denying every build.
    """
    mode = BV_PROJECT_ACCESS_MODE
    if mode not in BV_PROJECT_ACCESS_MODES:
        logger.error(
            "%s=%r is not one of %s; treating the BV project gate as %r",
            ENV_VAR_GBSERVER_BV_PROJECT_ACCESS_MODE,
            mode,
            list(BV_PROJECT_ACCESS_MODES),
            BV_PROJECT_ACCESS_MODE_OFF,
        )
        return BV_PROJECT_ACCESS_MODE_OFF
    return mode


async def _bv_account_email(tunnel, login: str) -> Optional[str]:
    """GECOS email of the BV account named ``login``, or None if there is none.

    Purely diagnostic: it is what makes a shadow-mode log actionable, by showing
    *which* BV account a granite.build login resolved to. Never used to allow or
    deny — the account's existence is established by ``id`` below, and a site
    whose GECOS carries no email must not thereby fail the gate.
    """
    try:
        rc, stdout, _stderr = await tunnel.run_remote_with_retries(
            f"getent passwd {shlex.quote(login)}", raise_on_error=False
        )
    except Exception as e:  # diagnostic only; never fail the gate on this
        logger.info("BV project gate: getent passwd for %r failed: %s", login, e)
        return None
    if rc != 0 or not (stdout or "").strip():
        return None
    # Reuse the environment-files parser rather than re-deriving it: the GECOS
    # layout (email as ``;``-sub-field 0) is a login-node convention, not a
    # standard, so two copies would drift the day the site changes it.
    #
    # Imported lazily on purpose. This module sits in the environment layer and is
    # imported by lsf.py at process start, while environment_files_paths pulls in
    # FastAPI; a module-level import would make the build runner depend on the API
    # layer just to parse one string. Deferring it keeps that edge on the one
    # diagnostic path that actually needs it.
    # pylint: disable=import-outside-toplevel
    from gbserver.api.environment_files_paths import parse_gecos_email

    return parse_gecos_email((stdout or "").splitlines()[0])


async def has_any_project_membership(tunnel, login: str) -> Tuple[bool, str]:
    """Whether ``login`` belongs to at least one BV project.

    Returns ``(allowed, reason)``. ``reason`` is for logs and is never shown to
    the user; it distinguishes the failure branches that the caller deliberately
    collapses into one denial.

    Fails closed: anything other than a definitive "yes" returns False.
    """
    if not login or not _LOGIN_RE.match(login):
        return False, f"login {login!r} is empty or not a usable account name"

    try:
        rc, stdout, stderr = await tunnel.run_remote_with_retries(
            f"id -Gn {shlex.quote(login)}", raise_on_error=False
        )
    except Exception as e:
        # Tunnel/transport failure. Deny — see the fail-closed note in the module
        # docstring. NOTE run_remote_with_retries does not retry
        # asyncssh ConnectionLost (utils/ssh_tunnel.py), so a dropped connection
        # lands here and denies a legitimate build; shadow mode is how we find
        # out how often that happens before it can bite anyone.
        return False, f"could not read groups for {login!r}: {e}"

    if rc != 0:
        # No such account, or the name service could not answer. Indistinguishable
        # from here, and both deny.
        return False, (
            f"id -Gn exited {rc} for {login!r} "
            f"(stderr: {(stderr or '').strip()[:200]!r})"
        )

    groups = parse_group_names(stdout)
    if not groups:
        return False, f"no groups resolved for {login!r}"

    projects = project_groups(groups)
    if not projects:
        return False, (
            f"{login!r} is in {len(groups)} group(s), none matching "
            f"{BV_PROJECT_GROUP_PREFIX!r}"
        )

    return True, f"{login!r} is in {len(projects)} project group(s)"


def _dispatch_denial(mode: str, login: str, reason: str, extra: str = "") -> None:
    """Log the denial per mode, then raise iff ``enforce``.

    Extracted so ``confirm_any_project_membership`` and
    ``confirm_any_project_membership_without_tunnel`` cannot drift on what the
    log line and the user-facing message look like — both go through here.

    ``extra`` carries optional context for the log only (e.g. the diagnostic BV
    account email). It is never included in the exception message.
    """
    logger.warning(
        "BV project gate [%s]: %s %r — %s%s",
        mode,
        "DENY" if mode == BV_PROJECT_ACCESS_MODE_ENFORCE else "would deny",
        login,
        reason,
        f" ({extra})" if extra else "",
    )
    if mode == BV_PROJECT_ACCESS_MODE_SHADOW:
        return
    raise BvProjectAccessDenied(
        f"Access to a BlueVela project is required to run here, and the account "
        f"'{login}' does not currently have one. Ask the owner of the project "
        f"you are working on to add you, then start this run again."
    )


async def confirm_any_project_membership(tunnel, login: str) -> None:
    """Enforce the gate for ``login`` according to the rollout mode.

    Raises ``BvProjectAccessDenied`` only in ``enforce`` mode. In ``shadow`` the
    same decision is computed and logged but the build proceeds; in ``off``
    nothing runs at all, not even the lookups.
    """
    mode = resolve_mode()
    if mode == BV_PROJECT_ACCESS_MODE_OFF:
        return

    allowed, reason = await has_any_project_membership(tunnel, login)
    if allowed:
        logger.info("BV project gate [%s]: allow %r — %s", mode, login, reason)
        return

    # Resolved only for the log, and only when there is something to explain: on
    # the allow path nobody needs it, and it costs a round trip. Never touched by
    # the denial verdict — the account's existence is decided by ``id -Gn``, and
    # a site whose GECOS carries no email must not thereby fail the gate.
    account_email = await _bv_account_email(tunnel, login)
    _dispatch_denial(
        mode, login, reason, extra=f"BV account email: {account_email or 'unknown'}"
    )


async def confirm_any_project_membership_without_tunnel(login: str) -> None:
    """Enforce the gate when no SSH tunnel is available (``use_ssh=false``).

    The gate cannot make a positive decision without the tunnel — every lookup
    it does is a keyed ``getent``/``id`` over that connection. So the choice is
    between denying and silently allowing everyone, and the whole module fails
    closed: ``enforce`` denies, ``shadow`` logs, ``off`` no-ops.

    Currently theoretical for BlueVela — every configured environment on this
    path uses SSH — but a future ``use_ssh=false`` LSF deployment would
    otherwise bypass the gate entirely, which is exactly the finding this
    guard exists to close.
    """
    mode = resolve_mode()
    if mode == BV_PROJECT_ACCESS_MODE_OFF:
        return
    _dispatch_denial(
        mode,
        login,
        "cannot verify BV project membership: no SSH tunnel is available "
        "(use_ssh=false on this environment)",
    )


__all__ = [
    "BvProjectAccessDenied",
    "confirm_any_project_membership",
    "confirm_any_project_membership_without_tunnel",
    "has_any_project_membership",
    "parse_group_names",
    "project_groups",
    "resolve_mode",
]
