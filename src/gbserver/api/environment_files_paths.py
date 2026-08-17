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

"""Environment + authorization + path helpers for the environment-files REST API.

The ``/files/{environment}/{folder}`` API browses a named folder on the login
nodes of a *supported environment*. An "environment folder" is the directory
``config.gpfs_base/{folder}`` (e.g. ``/proj/demo`` on ``bluevela``) guarded by
the POSIX group ``proj_{folder}``. Access is governed entirely by that group
membership — NOT by build ownership or space membership — so this module
replaces ``authorize_build_access`` with a ``getent``-based check.

The path-safety primitives (``validate_subpath``, ``resolve_and_check_real_path``)
are reused verbatim from ``build_files_paths``; they already take a root arg and
are agnostic to whether the root is a build root or an environment-folder root.

SECURITY — this file is the access-control core of the environment-files API:

* **Only supported environments.** ``resolve_environment`` maps the
  ``{environment}`` path segment to its ``EnvironmentFilesConfig`` via the
  registry. An unsupported value raises the SAME ``AccessDenied`` 404 as a
  missing folder — a caller cannot enumerate which environments exist. (A
  *known* environment that is merely unconfigured on this deployment is a 503,
  not a probe, so it is surfaced distinctly.)
* **Authorize before you read.** ``authorize_folder_access`` runs ONLY
  ``getent`` (the authorization lookup itself). Callers MUST await it and let it
  return before issuing any data-touching command (``readlink``/``ls``/``find``/
  ``grep``/``stat``/SFTP) against the folder.
* **No existence leak.** Every authorization failure — unsupported environment,
  non-member, missing ``proj_{folder}`` group, empty/malformed requester email,
  unparseable getent output, or a malformed folder name — raises the *same*
  ``AccessDenied`` (HTTP 404, identical body). A requester must not be able to
  tell "you lack access" from "no such folder/environment". Error surfaces never
  echo the resolved path, group members, or the group name.
* **Service identity, not the requester.** The getent calls (and all later data
  reads) run over the shared service-identity tunnel — never the requester's
  own login. gbserver owns the group check; filesystem perms are a backstop.
"""

import re
import shlex
from dataclasses import replace
from pathlib import PurePosixPath
from typing import List, Optional, Set

from fastapi import HTTPException, Request, status

# Re-exported so environment_files.py imports both path primitives from one place.
from gbserver.api.build_files_paths import (  # noqa: F401
    resolve_and_check_real_path,
    validate_subpath,
)
from gbserver.types.constants import (
    ENV_FILES_GETENT_BATCH_MAX,
    ENVIRONMENT_FILES_REGISTRY,
    SUPPORTED_ENV_FOR_FILES,
    EnvironmentFilesConfig,
    get_supported_env_for_files_uri,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# A folder name maps directly into a POSIX group name (``proj_{folder}``) that
# is interpolated into a ``getent group`` shell command. Restrict to a
# conservative token charset; anything else is rejected the same
# indistinguishable way as "not a member" so a well-formed-but-forbidden probe
# can't be told apart from a malformed one. We additionally reject the bare
# ``.`` / ``..`` names (which pass the charset but are not real folders) below.
_FOLDER_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class AccessDenied(HTTPException):
    """Uniform 404 for every authorization/existence failure.

    The status code and body are FIXED and identical across all failure
    branches (unsupported environment, missing folder, non-member, malformed
    name) so the response never reveals whether the environment/folder/group
    exists or merely that the requester lacks access. Do not add branch-specific
    detail here.
    """

    def __init__(self) -> None:
        super().__init__(status.HTTP_404_NOT_FOUND, "not found")


class EnvironmentNotConfigured(HTTPException):
    """503 for a *known* environment that is unconfigured on this deployment.

    Distinct from ``AccessDenied``: reaching this means the ``{environment}`` is
    supported (in the registry) but its ``environment_uri`` could not be derived
    from the public space config on this deployment, which is a deployment-config
    problem — not something a caller can probe for, so it does not need to hide
    behind the uniform 404.
    """

    def __init__(self, environment: str) -> None:
        super().__init__(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"environment '{environment}' is not configured for this deployment",
        )


def resolve_environment(environment: str) -> EnvironmentFilesConfig:
    """Map an ``{environment}`` path segment to its config, or deny.

    Raises ``AccessDenied`` (the uniform 404) for any unsupported environment —
    identical to a missing folder, so the supported set can't be enumerated.
    Raises ``EnvironmentNotConfigured`` (503) for a supported environment whose
    environment URI can't be derived from the public space config repo on this
    deployment. Otherwise returns the ``EnvironmentFilesConfig`` the handlers
    thread into tunnel + path resolution, with ``environment_uri`` set to the
    derived value.
    """
    config = ENVIRONMENT_FILES_REGISTRY.get(environment)
    if config is None:
        raise AccessDenied()
    # bluevela is the only supported environment today, so its derivation is
    # called directly. A second environment would generalize this to a per-entry
    # resolver; not abstracted prematurely.
    environment_uri = (
        get_supported_env_for_files_uri()
        if environment == SUPPORTED_ENV_FOR_FILES
        else config.environment_uri
    )
    if not environment_uri:
        raise EnvironmentNotConfigured(environment)
    return replace(config, environment_uri=environment_uri)


def validate_folder_name(folder: str) -> str:
    """Return ``folder`` unchanged if it is a safe folder-name token.

    Raises ``AccessDenied`` (NOT a distinct 400) for anything that isn't —
    empty, wrong charset, or the ``.``/``..`` pseudo-names — so a malformed
    name is indistinguishable from "not a member".
    """
    if not folder or folder in (".", "..") or _FOLDER_RE.match(folder) is None:
        raise AccessDenied()
    return folder


def folder_root(config: EnvironmentFilesConfig, folder: str) -> PurePosixPath:
    """Absolute root for a folder on an environment: ``gpfs_base/{folder}``.

    ``folder`` must already have passed ``validate_folder_name``.
    """
    return PurePosixPath(config.gpfs_base) / folder


def _group_name(folder: str) -> str:
    """POSIX group guarding a folder: ``proj_{folder}``.

    The ``proj_`` prefix is the real on-disk group-naming convention on the
    login nodes (independent of the ``{environment}`` URL segment), so it stays
    fixed even though the API noun no longer says "project".
    """
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


def parse_gecos_email(getent_passwd_line: str) -> Optional[str]:
    """Extract the email from one ``getent passwd`` line's GECOS field, or None.

    Expected format on the login node::

        alice:*:1001:2001:alice@example.com;NNNNNN;Alice Example:/u/alice:/bin/bash

    NOTE this is a login-node *convention*, not a POSIX/getent standard: the
    GECOS field (``:``-index 4) is ``;``-delimited with the email as sub-field 0.
    This function is named for the field it parses (GECOS) rather than the
    meaning it hopes to find, precisely because that meaning is site-specific.
    Returns the email only if sub-field 0 looks like one (has an ``@``);
    otherwise None (missing email, malformed row, or a GECOS without an email in
    the first sub-field).
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

    Runs one ``getent group`` round-trip over the service-identity tunnel to
    enumerate members, then resolves their emails with ``getent passwd`` in
    chunks of ``ENV_FILES_GETENT_BATCH_MAX`` (usually a single call). Returns
    an empty set if the group is missing/empty or nothing resolves — the caller
    turns that into ``AccessDenied``.

    Chunking bounds each ``getent passwd`` command line so a large
    ``proj_{folder}`` group can't overflow ARG_MAX / the login shell's arg
    limit and fail authz for a legitimate member.

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

    emails: Set[str] = set()
    for start in range(0, len(members), ENV_FILES_GETENT_BATCH_MAX):
        chunk = members[start : start + ENV_FILES_GETENT_BATCH_MAX]
        quoted = " ".join(shlex.quote(m) for m in chunk)
        # rc=2 means "one or more keys not found" — partial output is still
        # usable; we ignore rc and parse whatever came back for each chunk.
        _rc, stdout, _stderr = await tunnel.run_remote(
            f"getent passwd {quoted}", raise_on_error=False
        )
        for line in (stdout or "").splitlines():
            email = parse_gecos_email(line)
            if email is not None:
                emails.add(email.lower())
    return emails


async def authorize_folder_access(request: Request, tunnel, folder: str) -> None:
    """Confirm the requester may access ``proj_{folder}``; else deny uniformly.

    Runs ONLY ``getent`` (via ``_authorized_emails_for_group``) — no data-read
    against the folder. Raises ``AccessDenied`` on EVERY failure branch (empty
    requester email, missing/empty group, no resolvable emails, or requester not
    among them) so failures are indistinguishable. Callers MUST await this and
    let it return before any folder data command.
    """
    requester = (request.state.data["user"].email or "").strip().lower()
    if not requester:
        # User.email defaults to "" — treat an unpopulated identity as denied,
        # and short-circuit before issuing even the getent lookups.
        raise AccessDenied()

    emails = await _authorized_emails_for_group(tunnel, folder)
    if requester not in emails:
        raise AccessDenied()


__all__ = [
    "AccessDenied",
    "EnvironmentNotConfigured",
    "authorize_folder_access",
    "folder_root",
    "parse_gecos_email",
    "parse_group_members",
    "resolve_and_check_real_path",
    "resolve_environment",
    "validate_folder_name",
    "validate_subpath",
]
