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

"""Authorization + path helpers for the project-folder REST API.

A "project folder" is a GPFS directory ``PROJECTS_GPFS_BASE/{folder}`` (e.g.
``/proj/demo``) guarded by the POSIX group ``proj_{folder}``. Access is
governed entirely by that group membership — NOT by build ownership or space
membership — so this module replaces ``authorize_build_access`` with a
``getent``-based check.

The path-safety primitives (``validate_subpath``, ``resolve_and_check_real_path``)
are reused verbatim from ``build_files_paths``; they already take a root arg and
are agnostic to whether the root is a build root or a project-folder root.

SECURITY — this file is the access-control core of the project-files API:

* **Authorize before you read.** ``authorize_project_access`` runs ONLY
  ``getent`` (the authorization lookup itself). Callers MUST await it and let it
  return before issuing any data-touching command (``readlink``/``ls``/``find``/
  ``grep``/``stat``/SFTP) against ``/proj/{folder}``.
* **No existence leak.** Every authorization failure — non-member, missing
  ``proj_{folder}`` group, empty/malformed requester email, unparseable getent
  output, or a malformed folder name — raises the *same* ``ProjectAccessDenied``
  (HTTP 404, identical body). A requester must not be able to tell "you lack
  access" from "no such folder". Error surfaces never echo the resolved GPFS
  path, group members, or the group name.
* **Service identity, not the requester.** The getent calls (and all later data
  reads) run over the shared service-identity tunnel — never the requester's
  own login. gbserver owns the group check; GPFS perms are a backstop.
"""

import re
import shlex
from pathlib import PurePosixPath
from typing import List, Optional, Set

from fastapi import HTTPException, Request, status

# Re-exported so project_files.py imports both path primitives from one place.
from gbserver.api.build_files_paths import (  # noqa: F401
    resolve_and_check_real_path,
    validate_subpath,
)
from gbserver.types.constants import PROJECTS_GPFS_BASE
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# A project-folder name maps directly into a POSIX group name
# (``proj_{folder}``) that is interpolated into a ``getent group`` shell
# command. Restrict to a conservative token charset; anything else is
# rejected the same indistinguishable way as "not a member" so a
# well-formed-but-forbidden probe can't be told apart from a malformed one.
# We additionally reject the bare ``.`` / ``..`` names (which pass the charset
# but are not real folders) below.
_FOLDER_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ProjectAccessDenied(HTTPException):
    """Uniform 404 for every authorization/existence failure.

    The status code and body are FIXED and identical across all failure
    branches so the response never reveals whether the folder/group exists
    or merely that the requester lacks access. Do not add branch-specific
    detail here.
    """

    def __init__(self) -> None:
        super().__init__(status.HTTP_404_NOT_FOUND, "project folder not found")


def validate_folder_name(folder: str) -> str:
    """Return ``folder`` unchanged if it is a safe folder-name token.

    Raises ``ProjectAccessDenied`` (NOT a distinct 400) for anything that
    isn't — empty, wrong charset, or the ``.``/``..`` pseudo-names — so a
    malformed name is indistinguishable from "not a member".
    """
    if not folder or folder in (".", "..") or _FOLDER_RE.match(folder) is None:
        raise ProjectAccessDenied()
    return folder


def project_root(folder: str) -> PurePosixPath:
    """Absolute GPFS root for a project folder: ``PROJECTS_GPFS_BASE/{folder}``.

    ``folder`` must already have passed ``validate_folder_name``.
    """
    return PurePosixPath(PROJECTS_GPFS_BASE) / folder


def _group_name(folder: str) -> str:
    """POSIX group guarding a project folder: ``proj_{folder}``."""
    return f"proj_{folder}"


# --------------------------------------------------------------- getent parsers


def parse_group_members(getent_group_stdout: str) -> List[str]:
    """Parse ``getent group proj_{folder}`` output into a list of usernames.

    Format: ``name:passwd:gid:member1,member2,...``. We take the first
    non-empty line, split into at most 4 fields on ``:``, and split the
    member field on ``,`` (dropping empties). Returns ``[]`` for empty or
    malformed output.
    """
    for line in (getent_group_stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 3)
        if len(parts) < 4:
            return []
        return [m for m in parts[3].split(",") if m]
    return []


def parse_passwd_email(getent_passwd_line: str) -> Optional[str]:
    """Extract the email from one ``getent passwd`` line, or None.

    Expected format on the login node::

        alice:*:1001:2001:alice@example.com;NNNNNN;Alice Example:/u/alice:/bin/bash

    The GECOS field (``:``-index 4) is ``;``-delimited with the email as
    sub-field 0. Returns the email only if sub-field 0 looks like one (has
    an ``@``); otherwise None (missing email, malformed row, or a GECOS
    without an email in the first sub-field).
    """
    parts = (getent_passwd_line or "").split(":")
    if len(parts) < 5:
        return None
    gecos = parts[4]
    email = gecos.split(";")[0].strip()
    if "@" not in email:
        return None
    return email


async def _authorized_emails_for_group(tunnel, folder: str) -> Set[str]:
    """Return the lowercased set of emails authorized for ``proj_{folder}``.

    Runs exactly two ``getent`` round-trips over the service-identity tunnel:
    ``getent group`` to enumerate members, then a single batched
    ``getent passwd m1 m2 ... mN`` to resolve their emails. Returns an empty
    set if the group is missing/empty or nothing resolves — the caller turns
    that into ``ProjectAccessDenied``.

    Kept as one internal function so a future TTL cache, cron-refreshed map,
    or mapping-file fallback is a local change behind a stable signature.
    """
    group = _group_name(folder)
    rc, stdout, _stderr = await tunnel.run_remote(
        f"getent group {shlex.quote(group)}", raise_on_error=False
    )
    if rc != 0 or not (stdout or "").strip():
        return set()

    members = parse_group_members(stdout)
    if not members:
        return set()

    quoted = " ".join(shlex.quote(m) for m in members)
    rc, stdout, _stderr = await tunnel.run_remote(
        f"getent passwd {quoted}", raise_on_error=False
    )
    # rc=2 means "one or more keys not found" — partial output is still
    # usable; only bail if there is nothing to parse at all.
    if not (stdout or "").strip():
        return set()

    emails: Set[str] = set()
    for line in stdout.splitlines():
        email = parse_passwd_email(line)
        if email is not None:
            emails.add(email.lower())
    return emails


async def authorize_project_access(request: Request, tunnel, folder: str) -> None:
    """Confirm the requester may access ``proj_{folder}``; else deny uniformly.

    Runs ONLY ``getent`` (via ``_authorized_emails_for_group``) — no data-read
    against the folder. Raises ``ProjectAccessDenied`` on EVERY failure branch
    (empty requester email, missing/empty group, no resolvable emails, or
    requester not among them) so failures are indistinguishable. Callers MUST
    await this and let it return before any ``/proj/{folder}`` data command.
    """
    requester = (request.state.data["user"].email or "").strip().lower()
    if not requester:
        # User.email defaults to "" — treat an unpopulated identity as denied,
        # and short-circuit before issuing even the getent lookups.
        raise ProjectAccessDenied()

    emails = await _authorized_emails_for_group(tunnel, folder)
    if requester not in emails:
        raise ProjectAccessDenied()


__all__ = [
    "ProjectAccessDenied",
    "authorize_project_access",
    "parse_group_members",
    "parse_passwd_email",
    "project_root",
    "resolve_and_check_real_path",
    "validate_folder_name",
    "validate_subpath",
]
