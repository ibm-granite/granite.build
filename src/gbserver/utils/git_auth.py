#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""
Per-host git transport and token selection.

Each GitHub instance (github.com or an enterprise host)
has its own token, looked up by a slug of its hostname, so a token is only ever
sent to the host it belongs to:

- service token (deployment env): ``GBSERVER_GIT_TOKEN_<SLUG>``; for
  ``DEFAULT_GH_DOMAIN`` also the legacy ``GBSERVER_GITHUB_TOKEN`` (or
  ``GITHUB_TOKEN``).
- build token (user secret, else space secret): ``GIT_TOKEN_<SLUG>``.

``GBSERVER_GIT_TRANSPORT`` picks how a ``git+ssh`` URI is cloned: ``ssh_then_https``
(default: SSH, then HTTPS if SSH fails), ``https`` (skip SSH) or ``ssh``.
"""

import base64
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional

from gbcommon.types.constants import DEFAULT_GH_DOMAIN
from gbserver.types.constants import ENV_VAR_DEFAULT_GITHUB_TOKEN
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

ENV_VAR_GIT_TRANSPORT = "GBSERVER_GIT_TRANSPORT"
SERVICE_TOKEN_ENV_PREFIX = "GBSERVER_GIT_TOKEN_"
SECRET_TOKEN_PREFIX = "GIT_TOKEN_"


class GitTransport(str, Enum):
    """How a ``git+ssh`` URI is cloned."""

    SSH = "ssh"
    SSH_THEN_HTTPS = "ssh_then_https"
    HTTPS = "https"


DEFAULT_GIT_TRANSPORT = GitTransport.SSH_THEN_HTTPS


def get_git_transport() -> GitTransport:
    """The transport from ``GBSERVER_GIT_TRANSPORT``, read at call time."""
    value = os.getenv(ENV_VAR_GIT_TRANSPORT, "").strip().lower()
    if not value:
        return DEFAULT_GIT_TRANSPORT
    try:
        return GitTransport(value)
    except ValueError:
        logger.warning(
            "ignoring invalid %s=%r (expected one of %s); using %s",
            ENV_VAR_GIT_TRANSPORT,
            value,
            ", ".join(t.value for t in GitTransport),
            DEFAULT_GIT_TRANSPORT.value,
        )
        return DEFAULT_GIT_TRANSPORT


def host_slug(host: str) -> str:
    """``git.example.com`` -> ``GIT_EXAMPLE_COM``."""
    return re.sub(r"[^a-z0-9]", "_", host.strip().lower()).upper()


@dataclass(frozen=True)
class GitToken:
    """A token plus a label naming where it came from (safe to log)."""

    label: str
    token: str = field(repr=False)


def service_tokens(host: str) -> List[GitToken]:
    """Deployment tokens for ``host``, most specific name first."""
    names = [SERVICE_TOKEN_ENV_PREFIX + host_slug(host)]
    if host.lower() == DEFAULT_GH_DOMAIN.lower():
        names += [ENV_VAR_DEFAULT_GITHUB_TOKEN, "GITHUB_TOKEN"]
    for name in names:
        value = os.getenv(name, "")
        if value:
            # Only the first one set: GITHUB_TOKEN is just a fallback name.
            return [GitToken(label=f"service:{name}", token=value)]
    return []


def secret_tokens(host: str, secrets: Optional[Mapping[str, str]]) -> List[GitToken]:
    """The build's token for ``host``: user secrets already override space ones."""
    name = SECRET_TOKEN_PREFIX + host_slug(host)
    value = (secrets or {}).get(name) or ""
    return [GitToken(label=f"secret:{name}", token=value)] if value else []


def get_token_candidates(
    host: str, secrets: Optional[Mapping[str, str]] = None
) -> List[GitToken]:
    """Tokens to try for ``host`` on the server: service first, then the build's.

    The service token is the HTTPS stand-in for the server's SSH identity; the
    build's token covers repos the service account can't read. Duplicates are
    dropped.
    """
    seen = set()
    result = []
    for candidate in service_tokens(host) + secret_tokens(host, secrets):
        if candidate.token not in seen:
            seen.add(candidate.token)
            result.append(candidate)
    return result


def get_service_token(host: str) -> str:
    """The deployment token for ``host``, or ``""``."""
    tokens = service_tokens(host)
    return tokens[0].token if tokens else ""


def git_auth_env(
    host: str, token: Optional[str], base_env: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """Env vars that make git send ``token`` to ``https://host/`` only.

    The token goes into an ``http.<url>.extraHeader`` set through
    ``GIT_CONFIG_*``, so it stays out of the URL, argv and ``.git/config``.
    Entries already in ``base_env`` (default: ``os.environ``) are kept.
    """
    env = {"GIT_TERMINAL_PROMPT": "0"}
    if not token:
        return env
    base = os.environ if base_env is None else base_env
    try:
        count = int(base.get("GIT_CONFIG_COUNT", "0") or "0")
    except ValueError:
        count = 0
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env.update(
        {
            "GIT_CONFIG_COUNT": str(count + 1),
            f"GIT_CONFIG_KEY_{count}": f"http.https://{host}/.extraHeader",
            f"GIT_CONFIG_VALUE_{count}": f"Authorization: Basic {basic}",
        }
    )
    return env
