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

"""Materialize inline SkyPilot config from a Skypilot ``environment.yaml``.

Write/merge-only. Three destinations are supported:

  * ``cluster_ssh_configs`` -> ``~/.<cloud>/config`` (OpenSSH ``Host`` blocks,
    merged by alias under a cross-process file lock). A **foreign** (non-gbserver)
    entry with differing content raises ``SkypilotConfigCollisionError`` —
    gbserver never clobbers a user's own hosts. A prior **gbserver-managed** block
    owned by the *same* environment with differing content is overwritten,
    self-healing a re-keyed entry; identical content is a no-op; a differing block
    owned by a *different* environment raises (a cross-environment alias clash is
    surfaced, not silently overwritten — the owner is recorded per alias). SkyPilot re-reads the single
    ``~/.<cloud>/config`` for a cluster's whole lifetime, and it caches SSH
    ControlMaster sockets keyed on (host, port, user) — NOT the key — so a
    re-keyed config is masked by a live socket until it expires. Clearing that
    socket so a credential change actually takes effect is a test-only concern,
    gated by ``GBTEST_SKY_SSH_RESET`` (see ``gbserver.environment.skypilot``);
    production leaves SkyPilot's socket management untouched. Single-host by
    design (the config files are host-local).
  * ``cloud_config`` -> ``~/.sky/config.yaml`` (deep-merged into the global
    SkyPilot config the API server / optimizer reads directly; the env's values
    win, unrelated keys are preserved). Last writer wins: the keys gbserver last
    wrote are recorded next to the file, and the next write removes them before
    merging its own, so one environment's settings (a Kubernetes context pin, say)
    never outlive it into another's launches.
  * ``aws_credentials`` -> ``~/.aws/credentials`` (INI, mode 0600, merged by
    profile under a cross-process file lock).

Connection/credential field values are resolved by exact-name lookup against
the environment's secrets, falling back to the literal value. The module is
pure-filesystem (no ``sky`` import) so it is unit-testable without the SDK.
"""

import configparser
import copy
import hashlib
import io
import os
import random
import socket
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import psutil
import yaml
from filelock import FileLock

from gbserver.types.environmentconfig import (
    AwsCredentialProfile,
    ClusterSshConfigs,
    KubeConfigEntry,
)
from gbserver.types.errors import SkypilotConfigCollisionError
from gbserver.utils.logger import get_logger
from gbserver.utils.ssh_keys import write_private_key_file

logger = get_logger(__name__)

# Markers delimiting the gbserver-managed region of an SSH config file. Valid
# OpenSSH comments, so any foreign content outside the region is left untouched.
MANAGED_BEGIN = "# BEGIN gbserver-managed (cluster config)"
MANAGED_END = "# END gbserver-managed"
# Per-alias comment recording the contributing environment, so a later
# collision can name the owner even across processes (the file is the source
# of truth). Excluded from idempotency comparison.
_OWNER_PREFIX = "# gbserver-owner="

# Serializes file read-merge-write across threads in one process; the FileLock
# adds cross-process safety (filelock alone does not serialize same-process
# threads).
_THREAD_LOCK = threading.RLock()


# --------------------------------------------------------------------------- #
# Secret resolution
# --------------------------------------------------------------------------- #
def _resolve(value, secrets: Dict[str, str]):
    """Resolve a field value: a matching secret name wins, else the literal.

    :param value: The configured value (a secret name or a literal).
    :param secrets: Mapping of secret name -> value.
    :returns: The secret value if ``value`` is a known secret name, else
        ``value`` unchanged. ``None`` passes through.
    """
    if value is None:
        return None
    key = str(value)
    return secrets.get(key, key)


def _home(home: Optional[Path]) -> Path:
    """Return the home directory to materialize into (injectable for tests)."""
    return home if home is not None else Path.home()


def _raise_collision(kind: str, key: str, env_a: str, env_b: str, dest: str) -> None:
    """Raise a ``SkypilotConfigCollisionError`` with an actionable reason.

    :param kind: Human label for the colliding unit (e.g. ``"SSH Host"``).
    :param key: The conflicting alias / dotted key / profile name.
    :param env_a: The environment requesting the change.
    :param env_b: The owner of the existing, differing value.
    :param dest: The destination file / scope the conflict is in.
    :raises SkypilotConfigCollisionError: Always.
    """
    raise SkypilotConfigCollisionError(
        f"{kind} '{key}' in {dest} is defined differently by environment "
        f"'{env_a}' and {env_b}. Concurrent Skypilot environments must agree on "
        f"shared config; align the values or use distinct names."
    )


# --------------------------------------------------------------------------- #
# SSH config rendering
# --------------------------------------------------------------------------- #
def _render_value(value, secrets: Dict[str, str]) -> str:
    """Render one OpenSSH directive value (booleans -> ``yes``/``no``, else secret-resolved)."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(_resolve(value, secrets))


def _looks_like_secret_name(value: str) -> bool:
    """Heuristic: does ``value`` look like a secret name rather than a literal?

    The secret resolver falls back to the literal when a name is not found, so an
    unresolved secret silently becomes a bogus literal (e.g. an ``IdentityFile``
    pointing at a path named ``BV_SSH_PRIVATE_KEY``). This flags the common
    secret-identifier shape — an UPPER_SNAKE token containing at least one letter
    and no path/whitespace/punctuation chars — so callers can warn. Real literals
    like ``~/.ssh/id_ed25519``, ``login.example.com`` or ``yes`` do not match.

    :param value: The configured (pre-resolution) directive value as a string.
    :returns: True if ``value`` resembles a secret name.
    """
    return (
        len(value) >= 3
        and value == value.upper()
        and value.replace("_", "").isalnum()
        and any(ch.isalpha() for ch in value)
    )


def render_ssh_host(host: Dict[str, Any], secrets: Dict[str, str]) -> str:
    """Render one host mapping (OpenSSH directives) to a ``Host`` block.

    The ``Host`` key is the literal alias; every other key is emitted as an
    OpenSSH directive verbatim, with its value secret-resolved.

    :param host: Mapping of OpenSSH directive name -> value (must include ``Host``).
    :param secrets: Secret name -> value mapping for value resolution.
    :returns: The multi-line ``Host`` block text (resolved values).
    """
    alias = host["Host"]
    lines = [f"Host {alias}"]
    for key, raw in host.items():
        # Skip the alias itself and any unset directive.
        if key == "Host" or raw is None:
            continue
        raw_str = str(raw)
        in_secrets = raw_str in secrets
        logger.debug(
            "ssh directive %s for host %s resolved %s",
            key,
            alias,
            "from-secret" if in_secrets else "literal",
        )
        if not in_secrets and _looks_like_secret_name(raw_str):
            logger.warning(
                "SSH directive %s for host %s has value %r, which looks like a "
                "secret name but was not found in this environment's secrets; "
                "using it as a literal. If it should resolve to a secret, make "
                "sure that secret is available in the environment's secret store.",
                key,
                alias,
                raw_str,
            )
        lines.append(f"    {key} {_render_value(raw, secrets)}")
    return "\n".join(lines)


def render_ssh_hosts(
    hosts: List[Dict[str, Any]], secrets: Dict[str, str]
) -> Dict[str, str]:
    """Render hosts to an ``{alias: block}`` map (keyed for per-alias merge)."""
    return {h["Host"]: render_ssh_host(h, secrets) for h in hosts}


# Directive that supplies the private key *contents* (secret-resolved) instead of
# a path; materialized to a managed key file and rewritten as ``IdentityFile``.
IDENTITY_KEY_DIRECTIVE = "IdentityKey"


def _identity_key_path(cloud: str, alias: str, contents: str, home_path: Path) -> Path:
    """Return the managed key-file path for ``contents`` (content-addressed).

    The filename embeds a short hash of the key contents so identical keys reuse a
    stable path (idempotent, no false collisions) while a different key for the
    same alias yields a different path — surfacing a real conflict via the normal
    block-diff collision check.

    :param cloud: Cloud name (``lsf``/``slurm``).
    :param alias: SSH ``Host`` alias.
    :param contents: Resolved private-key material.
    :param home_path: Home directory to materialize under.
    :returns: ``<home>/.sky/keys/<cloud>__<alias>__<hash>.key``.
    """
    digest = hashlib.sha256(contents.encode("utf-8")).hexdigest()[:12]
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in f"{cloud}__{alias}")
    return home_path / ".sky" / "keys" / f"{safe}__{digest}.key"


def _materialize_identity_keys(
    hosts: List[Dict[str, Any]],
    cloud: str,
    secrets: Dict[str, str],
    home_path: Path,
) -> List[Dict[str, Any]]:
    """Resolve any ``IdentityKey`` directive to a managed key file + ``IdentityFile``.

    For each host declaring ``IdentityKey`` (a secret name or inline key material),
    the resolved key contents are written to a ``0600`` managed file and the host is
    rewritten to reference that file via ``IdentityFile`` (the ``IdentityKey``
    directive is dropped). Hosts without ``IdentityKey`` pass through unchanged. Key
    contents are never logged.

    :param hosts: Host directive mappings for one cloud.
    :param cloud: Cloud name (``lsf``/``slurm``) — used in the key-file path.
    :param secrets: Secret name -> value mapping for resolution.
    :param home_path: Home directory to materialize key files under.
    :returns: A new host list with ``IdentityKey`` resolved to ``IdentityFile``.
    :raises ValueError: If a host sets both ``IdentityKey`` and ``IdentityFile``, or
        ``IdentityKey`` resolves to empty.
    """
    result: List[Dict[str, Any]] = []
    for host in hosts:
        if IDENTITY_KEY_DIRECTIVE not in host:
            result.append(host)
            continue
        alias = host.get("Host")
        if "IdentityFile" in host:
            raise ValueError(
                f"SSH host {alias!r}: specify either IdentityFile or "
                f"{IDENTITY_KEY_DIRECTIVE}, not both."
            )
        raw = host[IDENTITY_KEY_DIRECTIVE]
        raw_str = str(raw) if raw is not None else ""
        if raw_str and raw_str not in secrets and _looks_like_secret_name(raw_str):
            logger.warning(
                "%s for host %s has value %r, which looks like a secret name but "
                "was not found in this environment's secrets; the key file would be "
                "invalid. Ensure the secret is available in the environment's "
                "secret store.",
                IDENTITY_KEY_DIRECTIVE,
                alias,
                raw_str,
            )
        contents = _resolve(raw, secrets)
        if not contents or not str(contents).strip():
            raise ValueError(
                f"SSH host {alias!r}: {IDENTITY_KEY_DIRECTIVE} resolved to an empty "
                "value; expected private key contents."
            )
        key_path = _identity_key_path(cloud, str(alias), str(contents), home_path)
        write_private_key_file(str(contents), key_path)
        new_host = {k: v for k, v in host.items() if k != IDENTITY_KEY_DIRECTIVE}
        new_host["IdentityFile"] = str(key_path)
        result.append(new_host)
    return result


# --------------------------------------------------------------------------- #
# HostName selection (multiple login nodes)
# --------------------------------------------------------------------------- #
def _expand_hostname_candidates(
    host: Dict[str, Any],
    sticky: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Expand a host's scalar/list ``HostName`` into shuffled per-candidate host dicts.

    Pure and I/O-free. A cluster's candidate ``HostName`` values are interchangeable
    login nodes of the *same* scheduler, so the launch picks one at random to spread
    load across them, and the launch path (see the ``_LoginNodeRotator`` in
    ``gbserver.environment.skypilot``) can fail over to the next candidate in this
    order when SkyPilot's own provisioning hits a transient SSH control-plane error.

    A scalar ``HostName`` yields a one-element list. Candidates are returned in random
    order; each carries every other directive of ``host`` with ``HostName`` collapsed
    to that one scalar, ready for :func:`render_ssh_host`.

    When ``sticky`` names a candidate already written for this alias (by an earlier
    launch in the same build), it is moved to the front so the shuffled order starts
    there: a failover another launch applied persists instead of being re-randomized
    back onto the node that just failed. ``sticky`` that is not a candidate is ignored.

    :param host: One host directive mapping (scalar or list ``HostName``).
    :param sticky: A previously-written ``HostName`` to keep first when it is still a
        candidate; ``None`` (or a non-candidate) leaves the random order unchanged.
    :returns: The per-candidate host dicts (sticky-first then shuffled), or ``None``
        when ``host`` has no ``HostName`` at all (the caller uses ``host`` unchanged).
    """
    candidates = host.get("HostName")
    if candidates is None:
        return None
    if not isinstance(candidates, list):
        candidates = [candidates]  # scalar: a one-element list
    shuffled = list(candidates)
    random.shuffle(shuffled)
    expanded = [{**host, "HostName": candidate} for candidate in shuffled]
    for i, cand in enumerate(expanded):
        if cand["HostName"] == sticky:
            expanded.insert(0, expanded.pop(i))  # keep the written node in front
            break
    return expanded


def _collapse_to_random_hostname(host: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse a scalar/list ``HostName`` to a single randomly-chosen candidate.

    Load-spreading, not reachability: the candidate login nodes reach the same
    scheduler, so any one is a valid target and a random pick distributes launches
    across them. A host without a ``HostName`` is returned unchanged.

    :param host: One host directive mapping (scalar or list ``HostName``).
    :returns: A host mapping with a single scalar ``HostName`` (or ``host`` unchanged
        when it declares no ``HostName``).
    """
    candidates = _expand_hostname_candidates(host)
    return host if candidates is None else candidates[0]


def _normalize(block: str) -> str:
    """Normalize a block for idempotency comparison (drop blanks/comments/ws)."""
    return "\n".join(
        ln.strip()
        for ln in block.strip().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    )


# --------------------------------------------------------------------------- #
# SSH managed-region parse / serialize (pure, no I/O)
# --------------------------------------------------------------------------- #
def _parse_managed(text: str) -> Tuple[str, Dict[str, Tuple[str, str]]]:
    """Split a config file into foreign text and the managed ``{alias: (block, owner)}``.

    :param text: Full current file contents.
    :returns: ``(foreign_text, blocks)`` where ``foreign_text`` is everything
        outside the managed region and ``blocks`` maps alias -> (block, owner).
    """
    if MANAGED_BEGIN not in text:
        return text, {}
    begin = text.index(MANAGED_BEGIN)
    after = begin + len(MANAGED_BEGIN)
    end = text.find(MANAGED_END, after)
    if end == -1:
        return text[:begin], _parse_host_blocks(text[after:])
    region = text[after:end]
    foreign = text[:begin] + text[end + len(MANAGED_END) :]
    return foreign, _parse_host_blocks(region)


def _parse_host_blocks(region: str) -> Dict[str, Tuple[str, str]]:
    """Parse a managed region's text into ``{alias: (block, owner)}``."""
    blocks: Dict[str, Tuple[str, str]] = {}
    alias: Optional[str] = None
    owner = ""
    pending_owner = ""
    lines: List[str] = []
    for raw in region.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(_OWNER_PREFIX):
            pending_owner = s[len(_OWNER_PREFIX) :].strip()
            continue
        if s.lower().startswith("host "):
            if alias is not None:
                blocks[alias] = ("\n".join(lines), owner)
            alias = s.split(None, 1)[1].strip()
            owner, pending_owner = pending_owner, ""
            lines = [f"Host {alias}"]
        elif alias is not None:
            lines.append(f"    {s}")
    if alias is not None:
        blocks[alias] = ("\n".join(lines), owner)
    return blocks


def _serialize_managed(blocks: Dict[str, Tuple[str, str]]) -> str:
    """Serialize ``{alias: (block, owner)}`` to a managed region (sorted, stable)."""
    parts = []
    for alias in sorted(blocks):
        block, owner = blocks[alias]
        parts.append(f"{_OWNER_PREFIX}{owner}\n{block}" if owner else block)
    return f"{MANAGED_BEGIN}\n" + "\n\n".join(parts) + f"\n{MANAGED_END}\n"


def _compose(foreign: str, region: str) -> str:
    """Recombine foreign content with the (re-serialized) managed region."""
    head = foreign.strip("\n")
    return f"{head}\n\n{region}" if head.strip() else region


def _blocks_equivalent(a: str, b: str) -> bool:
    """Return True if two SSH ``Host`` blocks are equivalent.

    Compares the normalized directive lines order-independently (ignoring blank
    lines, comments, and whitespace), so a hand-written entry that lists the same
    directives in a different order still matches what the env renders.
    """
    return sorted(_normalize(a).splitlines()) == sorted(_normalize(b).splitlines())


def _without_hostname(block: str) -> List[str]:
    """Return a block's normalized directive lines with any ``HostName`` line dropped.

    Helper for :func:`_blocks_differ_only_in_hostname`. Matches ``HostName``
    case-insensitively (OpenSSH keywords are case-insensitive) after normalization.

    :param block: An SSH ``Host`` block.
    :returns: Sorted normalized lines excluding the ``HostName`` directive.
    """
    return sorted(
        ln
        for ln in _normalize(block).splitlines()
        if not ln.lower().startswith("hostname ")
    )


def _block_hostname(block: str) -> Optional[str]:
    """Return a rendered block's ``HostName`` value, or ``None`` if it has none.

    Matches ``HostName`` case-insensitively (OpenSSH keywords are case-insensitive)
    after normalization. Used to check whether an existing on-disk login node is one
    of an incoming environment's candidates (see :func:`_merge_ssh`) and to keep a
    launch's pick sticky (see :func:`_read_managed_hostnames`).

    :param block: An SSH ``Host`` block.
    :returns: The ``HostName`` directive's value, or ``None`` when absent.
    """
    for ln in _normalize(block).splitlines():
        if ln.lower().startswith("hostname "):
            return ln.split(None, 1)[1].strip()
    return None


def _is_interchangeable_login_node(
    existing: str, incoming: str, candidate_hostnames: Optional[Set[str]]
) -> bool:
    """Return True when ``existing`` is a sibling login node of ``incoming``'s cluster.

    The last-writer-wins relaxation (see :func:`_merge_ssh`) applies only when two
    environments genuinely reference the *same* cluster through interchangeable login
    nodes: the blocks must differ **only** in ``HostName`` AND the already-written
    ``HostName`` must be one of the incoming environment's own candidate login nodes.
    Two environments that merely reuse an alias for *different* clusters (same user and
    key, unrelated hostnames) fail the candidate check and stay a genuine collision —
    the wrong-cluster case the owner check exists to catch.

    :param existing: The block already written for the alias.
    :param incoming: The block the current environment wants to write.
    :param candidate_hostnames: The incoming env's candidate ``HostName`` values for
        this alias; ``None`` or empty (candidates unknown) is treated as a clash.
    :returns: True only when a HostName-only difference is safe to overwrite.
    """
    if not candidate_hostnames:
        return False
    return (
        _blocks_differ_only_in_hostname(existing, incoming)
        and _block_hostname(existing) in candidate_hostnames
    )


def _blocks_differ_only_in_hostname(a: str, b: str) -> bool:
    """Return True if two blocks are identical except for their ``HostName`` value.

    The candidate login nodes of one cluster are interchangeable entry points to the
    *same* scheduler, differing only in ``HostName``; every credential/route directive
    (``User``/``Port``/``IdentityFile``/``ProxyJump``/…) is shared. So two environments
    that name the same ``Host`` alias but pick different login nodes are NOT a genuine
    clash — one can safely win. A difference in any *other* directive is a real
    conflict and is not covered here.

    :param a: One SSH ``Host`` block.
    :param b: The other SSH ``Host`` block.
    :returns: True when the two blocks match once ``HostName`` is ignored AND they
        actually differ (an exact match is handled by :func:`_blocks_equivalent`).
    """
    return not _blocks_equivalent(a, b) and _without_hostname(a) == _without_hostname(b)


def _merge_ssh(
    existing: Dict[str, Tuple[str, str]],
    incoming: Dict[str, str],
    foreign_blocks: Dict[str, Tuple[str, str]],
    env_name: str,
    dest: str,
    candidate_hostnames: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Tuple[str, str]]:
    """Merge incoming alias blocks into existing; raise only on a real conflict.

    Content-aware and owner-aware: a pre-existing entry for the same alias with
    identical content is a no-op. A **foreign** (non-gbserver) entry with differing
    content always conflicts — gbserver never overwrites user-owned entries. A prior
    **gbserver-managed** block with differing content is overwritten only when it is
    owned by *this* environment (self-healing a stale or re-keyed entry); a block
    owned by a *different* environment raises a collision naming both owners, so two
    environments on the same cloud that declare the same ``Host`` alias with
    conflicting content are surfaced rather than silently clobbering each other. A
    managed block with no recorded owner (e.g. written before owner tracking) is
    treated as self-healable rather than raising an unattributable collision.

    One narrow cross-environment difference is *not* a conflict: a block differing from
    another environment's only in its ``HostName`` **when the already-written hostname
    is one of this environment's own candidate login nodes** (see
    :func:`_is_interchangeable_login_node`). That is the genuine same-cluster,
    different-login-node case, so it may safely last-writer-win. Two environments that
    merely reuse an alias for *different* clusters (unrelated hostnames) still raise.

    :param existing: Current ``{alias: (block, owner)}`` from the managed region.
    :param incoming: New ``{alias: block}`` to merge in.
    :param foreign_blocks: ``{alias: (block, owner)}`` parsed from non-managed content.
    :param env_name: The contributing environment name.
    :param dest: Destination file path (for messages).
    :param candidate_hostnames: ``{alias: {candidate HostName, …}}`` for the incoming
        environment, gating the HostName-only relaxation; ``None`` (unknown) keeps the
        strict owner check (any cross-env difference raises).
    :returns: The merged ``{alias: (block, owner)}`` (foreign-equivalent aliases omitted).
    :raises SkypilotConfigCollisionError: On a foreign clash, or a differing block
        owned by another gbserver environment.
    """
    candidate_hostnames = candidate_hostnames or {}
    merged = dict(existing)
    for alias, block in incoming.items():
        if alias in foreign_blocks:
            if not _blocks_equivalent(foreign_blocks[alias][0], block):
                _raise_collision(
                    "SSH Host",
                    alias,
                    env_name,
                    "a pre-existing (non-gbserver) entry",
                    dest,
                )
            # Identical foreign entry already provides this host — leave it as-is.
            continue
        if alias in merged:
            prev_block, prev_owner = merged[alias]
            if _blocks_equivalent(prev_block, block):
                # Identical managed block already present — no-op (avoids a
                # rewrite and preserves the recorded owner).
                continue
            if (
                prev_owner
                and prev_owner != env_name
                and not _is_interchangeable_login_node(
                    prev_block, block, candidate_hostnames.get(alias)
                )
            ):
                # A *different* environment already manages this alias and this is
                # not the same-cluster/different-login-node case: a cross-environment
                # clash, not a re-key of our own entry. Refuse and name both owners.
                # A HostName-only difference whose existing node is one of our
                # candidates falls through to last-writer-wins below.
                _raise_collision(
                    "SSH Host", alias, env_name, f"environment '{prev_owner}'", dest
                )
        # New alias, our own re-key, or an unowned managed block: (over)write it
        # (self-heal).
        merged[alias] = (block, env_name)
    return merged


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #
def _lock_for(home_path: Path, name: str) -> FileLock:
    """Return a cross-process ``FileLock`` under ``~/.sky/locks``."""
    lock_dir = home_path / ".sky" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_dir / name))


def _write_atomic(path: Path, text: str, mode: Optional[int] = None) -> None:
    """Atomically write ``text`` to ``path`` (``*.gbtmp`` + ``os.replace``).

    With ``mode``, the temporary file is *created* with it, so content such as
    credentials is never readable by others, not even before a ``chmod``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".gbtmp")
    if mode is None:
        tmp.write_text(text, encoding="utf-8")
    else:
        tmp.unlink(missing_ok=True)  # O_CREAT's mode applies only to a new file
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, mode)  # O_CREAT's mode is masked by the umask
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Merge entry points (file I/O under locks)
# --------------------------------------------------------------------------- #
def _read_managed_hostnames(
    cloud: str, *, home: Optional[Path] = None
) -> Dict[str, str]:
    """Return ``{alias: HostName}`` for the gbserver-managed blocks in a cloud's config.

    Reads ``~/.<cloud>/config`` and extracts each managed block's ``HostName``. Used to
    keep a build's login-node pick **sticky**: an earlier launch may have failed over
    to a healthy candidate, and a later launch sharing this file should start from the
    node already written rather than re-randomizing (see
    :func:`_expand_hostname_candidates`). Lock-free best-effort — a concurrent write
    only costs a fallback to the random pick, and the subsequent merge takes the lock.

    :param cloud: ``"slurm"`` or ``"lsf"`` — whose ``~/.<cloud>/config`` to read.
    :param home: Home dir override (tests).
    :returns: ``{alias: HostName}`` for managed blocks that declare a ``HostName``
        (empty when the file is absent or has no managed region).
    """
    dest = _home(home) / f".{cloud}" / "config"
    if not dest.exists():
        return {}
    _foreign, managed = _parse_managed(dest.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for alias, (block, _owner) in managed.items():
        hostname = _block_hostname(block)
        if hostname is not None:
            out[alias] = hostname
    return out


def merge_ssh_blocks(
    cloud: str,
    alias_blocks: Dict[str, str],
    env_name: str,
    home: Optional[Path] = None,
    *,
    candidate_hostnames: Optional[Dict[str, Set[str]]] = None,
) -> None:
    """Merge rendered SSH ``Host`` blocks into ``~/.<cloud>/config``.

    Idempotent, owner-aware last-writer-wins: an identical managed block is a
    no-op; a differing block owned by the *same* environment is overwritten
    (self-heals a re-keyed entry); a differing block owned by a *different*
    environment, or a differing **foreign** (non-gbserver) entry, raises — except
    the narrow same-cluster/different-login-node case gated by ``candidate_hostnames``
    (see :func:`_merge_ssh`). Serialized across threads and processes by the per-cloud
    file lock.

    :param cloud: Cloud name (``slurm``/``lsf``) -> ``~/.<cloud>/config``.
    :param alias_blocks: ``{alias: block}`` to merge.
    :param env_name: The contributing environment name.
    :param home: Home dir override (tests).
    :param candidate_hostnames: ``{alias: {candidate HostName, …}}`` for this env,
        gating the HostName-only cross-env relaxation; ``None`` keeps it strict.
    :raises SkypilotConfigCollisionError: On a foreign clash, or a differing block
        for the same alias owned by another gbserver environment.
    """
    if not alias_blocks:
        return
    home_path = _home(home)
    dest = home_path / f".{cloud}" / "config"
    with _THREAD_LOCK, _lock_for(home_path, f"gbserver-{cloud}.lock"):
        text = dest.read_text(encoding="utf-8") if dest.exists() else ""
        foreign, existing = _parse_managed(text)
        merged = _merge_ssh(
            existing,
            alias_blocks,
            _parse_host_blocks(foreign),
            env_name,
            str(dest),
            candidate_hostnames,
        )
        if merged == existing:
            # Nothing new to manage (e.g. every incoming alias already exists as an
            # identical foreign/managed entry) — leave the file untouched.
            return
        _write_atomic(dest, _compose(foreign, _serialize_managed(merged)))


def _deep_merge_overwrite(base: Dict, overlay: Dict) -> None:
    """Recursively merge ``overlay`` into ``base`` in place; ``overlay`` wins.

    Nested dicts are merged key-by-key; a non-dict value (or a type change)
    replaces whatever is in ``base``. Keys present only in ``base`` are
    preserved. This is the "written from the env" semantics: the env's
    ``cloud_config`` is the source of truth and overwrites differing values in
    ``~/.sky/config.yaml`` while leaving unrelated keys untouched.

    :param base: The existing config (mutated in place).
    :param overlay: The env's cloud_config to layer on top.
    """
    for key, val in overlay.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge_overwrite(base[key], val)
        else:
            base[key] = val


#: Record of the cloud_config gbserver last wrote into ``~/.sky/config.yaml``
#: (environment name + the block), so the next write can take it back out.
_CLOUD_CONFIG_RECORD = ".gbserver-cloud-config.yaml"


def _remove_recorded(base: Dict, recorded: Dict) -> None:
    """Remove from ``base`` (in place) each value that still equals ``recorded``'s.

    Walks ``recorded`` like :func:`_deep_merge_overwrite` walks its overlay: dicts
    recurse, and any other value (lists included, which SkyPilot's config replaces
    whole) is removed only if ``base`` still holds exactly that value, so a key
    the user has since edited is kept. A dict that this leaves empty is removed
    too, so a removed section does not linger as ``{}``.

    :param base: The current config (mutated in place).
    :param recorded: The cloud_config gbserver wrote last.
    """
    for key, val in recorded.items():
        if key not in base:
            continue
        if isinstance(val, dict) and isinstance(base[key], dict):
            _remove_recorded(base[key], val)
            if not base[key]:
                del base[key]
        elif base[key] == val:
            del base[key]


def merge_cloud_config(
    cloud_config: Optional[Dict], env_name: str, home: Optional[Path] = None
) -> bool:
    """Make the env's ``cloud_config`` the gbserver part of ``~/.sky/config.yaml``.

    Last writer wins, the way SSH config blocks overwrite: first removes what the
    previous write recorded (:func:`_remove_recorded`, so keys the user edited
    since are kept), then deep-merges ``cloud_config`` with the env's values
    taking precedence, and records it for the next write. Keys gbserver never
    wrote are preserved. An empty or ``None`` ``cloud_config`` only removes the
    previous one, so an environment without one launches without, e.g., another
    environment's Kubernetes context pin. A key the env overwrote is removed, not
    restored, when the env's turn ends: its earlier value was not kept.

    Done under a cross-process file lock (the file is host-shared). The file alone
    does NOT reach a running gbserver's requests: its in-process SkyPilot client
    read the file at import and sends that copy with every request, so the caller
    must reload it -- see
    ``gbserver.environment.skypilot._reload_skypilot_client_config``.

    :param cloud_config: The behavioral SkyPilot config block (e.g. an ``lsf:``
        block) to write, sourced from the environment.yaml; ``None`` for none.
    :param env_name: The contributing environment name (recorded and logged).
    :param home: Home dir override (tests).
    :returns: Whether ``~/.sky/config.yaml`` changed.
    """
    cloud_config = cloud_config or {}
    home_path = _home(home)
    dest = home_path / ".sky" / "config.yaml"
    record = home_path / ".sky" / _CLOUD_CONFIG_RECORD
    with _THREAD_LOCK, _lock_for(home_path, "gbserver-sky-config.lock"):
        existing: Dict = {}
        if dest.exists():
            existing = yaml.safe_load(dest.read_text(encoding="utf-8")) or {}
        previous: Dict = {}
        if record.exists():
            previous = yaml.safe_load(record.read_text(encoding="utf-8")) or {}
        merged = copy.deepcopy(existing)
        _remove_recorded(merged, previous.get("cloud_config") or {})
        _deep_merge_overwrite(merged, copy.deepcopy(cloud_config))
        changed = merged != existing
        if changed:
            logger.info(
                "Writing cloud_config from environment '%s' into %s (replacing "
                "environment '%s')",
                env_name,
                dest,
                previous.get("environment"),
            )
            _write_atomic(
                dest,
                yaml.safe_dump(merged, default_flow_style=False, sort_keys=False),
            )
        if cloud_config:
            _write_atomic(
                record,
                yaml.safe_dump(
                    {"environment": env_name, "cloud_config": cloud_config},
                    default_flow_style=False,
                    sort_keys=False,
                ),
            )
        elif record.exists():
            record.unlink()
        return changed


def release_cloud_config(home: Optional[Path] = None) -> bool:
    """Remove the cloud_config gbserver last wrote from ``~/.sky/config.yaml``.

    For teardown (``make kube-teardown``): leaves only what gbserver never wrote,
    plus anything the user edited since. A later build writes its own again.

    :param home: Home dir override (tests).
    :returns: Whether ``~/.sky/config.yaml`` changed.
    """
    return merge_cloud_config(None, "(released)", home=home)


#: The gbserver-managed kubeconfig (under ``~/.sky``) that ``kube_configs``
#: contexts are merged into, added last on ``KUBECONFIG`` by
#: :func:`activate_kubeconfig`. The user's own kubeconfig is never written.
GBSERVER_KUBECONFIG = "gbserver-kubeconfig"

#: Default kubeconfig when ``KUBECONFIG`` is unset (as kubectl and SkyPilot use).
_DEFAULT_KUBECONFIG = "~/.kube/config"

#: Key in the managed kubeconfig recording which environment wrote each context.
_KUBE_OWNERS_KEY = "x-gbserver-owners"


def _start_time(proc: psutil.Process) -> str:
    """``proc``'s start time, as recorded in owner keys.

    Boot-relative where psutil offers it (Linux, macOS): the wall-clock value
    moves when the system clock is stepped (NTP, a VM resume), which would make
    a live owner look stale and get its context pruned mid-build.
    """
    try:
        # pylint: disable-next=protected-access
        started = proc._proc.create_time(monotonic=True)
    except (AttributeError, TypeError):
        started = proc.create_time()
    return f"{started:.2f}"


_PROCESS_TAG: Optional[str] = None


def kube_process_tag() -> str:
    """This gbserver process's tag for owner keys: ``@<host>:<pid>:<start time>``.

    The managed file outlives the process, so an owner recorded by a process
    that died mid-build (no teardown) would otherwise stay for good, and its
    context could never be rotated or removed. The start time tells a live
    process from a later one that reused the pid. Computed once per process,
    so an instance's owner key never changes.
    """
    global _PROCESS_TAG  # pylint: disable=global-statement
    if _PROCESS_TAG is None or not _PROCESS_TAG.split(":")[1] == str(os.getpid()):
        me = psutil.Process()
        _PROCESS_TAG = f"@{socket.gethostname()}:{me.pid}:{_start_time(me)}"
    return _PROCESS_TAG


def _owner_is_stale(owner: str) -> bool:
    """Whether ``owner`` was recorded by a gbserver process that no longer runs.

    Only owners tagged by :func:`kube_process_tag` *on this host* can be
    checked; any other owner (another host sharing the home dir, or an
    untagged one) is assumed live.
    """
    _, sep, tag = owner.rpartition("@")
    if not sep:
        return False
    host, _, rest = tag.partition(":")
    pid, _, started = rest.partition(":")
    if host != socket.gethostname():
        return False
    try:
        return _start_time(psutil.Process(int(pid))) != started
    except (ValueError, psutil.NoSuchProcess, psutil.ZombieProcess):
        return True
    except psutil.Error:
        return False


def kube_owner_label(owner: str) -> str:
    """``<space>/<environment>`` of an owner key, for messages and logs.

    Owner keys also carry an instance id and process tag
    (``<space>/<environment>#<id>@<host>:<pid>:<start>``), which mean nothing
    to a user.
    """
    return owner.split("#", 1)[0]


def kubeconfig_path(home: Optional[Path] = None) -> Path:
    """Path of the gbserver-managed kubeconfig."""
    return _home(home) / ".sky" / GBSERVER_KUBECONFIG


#: A placeholder context gbserver's kubeconfig always carries and names as its
#: ``current-context``. The kubernetes client rejects a merged ``KUBECONFIG``
#: with no current-context (and SkyPilot then sees no contexts at all), but a
#: real managed context there would become the process-wide *default* when the
#: user's own kubeconfig sets none, so anything relying on the default would
#: silently use an environment's secret credentials. The placeholder points at
#: an unroutable address with no credentials, so such a request fails loudly, as
#: it would without kube_configs. A user's own current-context still wins.
_KUBE_PLACEHOLDER = "gbserver-no-default-context"
_KUBE_PLACEHOLDER_ENTRIES = {
    "contexts": {
        "name": _KUBE_PLACEHOLDER,
        "context": {"cluster": _KUBE_PLACEHOLDER, "user": _KUBE_PLACEHOLDER},
    },
    # 192.0.2.0/24 is TEST-NET-1 (RFC 5737): never routed.
    "clusters": {
        "name": _KUBE_PLACEHOLDER,
        "cluster": {"server": "https://192.0.2.1:1"},
    },
    "users": {"name": _KUBE_PLACEHOLDER, "user": {}},
}


def _named(entries: Any, name: Any) -> Optional[Dict]:
    """The entry called ``name`` in a kubeconfig list (``contexts``, ...), or None.

    A section that is not a list holds no entries.
    """
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    return None


#: Prefix for the cluster and user names gbserver gives each managed context
#: (``gbserver/<context>``), so they never collide with a same-named cluster or
#: user in the user's kubeconfig, another environment's, or another context's.
_KUBE_NAME_PREFIX = "gbserver/"


#: The only kubeconfig ``user`` fields a ``kube_configs`` secret may carry:
#: static credentials, embedded as data. ``exec`` and ``auth-provider`` are
#: refused because they make every kubectl/SkyPilot use of the context run a
#: command on the gbserver host, so anyone able to write a space secret could
#: run arbitrary code there. File paths (``client-certificate``, ``client-key``,
#: ``tokenFile``) are refused because they would resolve on the gbserver host,
#: relative to the merged file, not to wherever the secret was written.
_KUBE_USER_FIELDS = frozenset(
    {
        "token",
        "client-certificate-data",
        "client-key-data",
        "username",
        "password",
    }
)
#: Same rule for the ``cluster`` entry: no ``certificate-authority`` file path.
_KUBE_CLUSTER_FIELDS = frozenset(
    {
        "server",
        "certificate-authority-data",
        "insecure-skip-tls-verify",
        "tls-server-name",
        "proxy-url",
        "disable-compression",
    }
)


#: The context entry may only name its cluster and user (rewritten by gbserver)
#: and a namespace.
_KUBE_CONTEXT_FIELDS = frozenset({"cluster", "user", "namespace"})
#: Fields whose value is a boolean; every other allowed field is a string.
_KUBE_BOOL_FIELDS = frozenset({"insecure-skip-tls-verify", "disable-compression"})


def _check_fields(kind: str, context: str, body: Any, allowed: frozenset) -> None:
    """Refuse a cluster/user/context body with a field outside ``allowed``.

    Values must be strings, or booleans for :data:`_KUBE_BOOL_FIELDS`: kubectl
    refuses to load the *whole* merged ``KUBECONFIG`` when one entry has a
    value of the wrong type, which would break every environment and the
    user's own kubectl. Only field *names* appear in the error, never values.

    :raises ValueError: If ``body`` is not a mapping, has a disallowed field,
        or a value of the wrong type.
    """
    if not isinstance(body, dict) or not all(isinstance(k, str) for k in body):
        raise ValueError(f"kubeconfig context '{context}': its {kind} is malformed")
    refused = sorted(k for k in body if k not in allowed)
    if refused:
        raise ValueError(
            f"kubeconfig context '{context}': {kind} fields {refused} are not "
            f"allowed in kube_configs (allowed: {sorted(allowed)}). exec and "
            "auth-provider plugins and file paths are refused; embed static "
            "credentials as *-data fields or a token."
        )
    wrong = sorted(
        k
        for k, v in body.items()
        if not isinstance(v, bool if k in _KUBE_BOOL_FIELDS else str)
    )
    if wrong:
        raise ValueError(
            f"kubeconfig context '{context}': {kind} fields {wrong} have values "
            "of the wrong type (expected strings; true/false for "
            f"{sorted(_KUBE_BOOL_FIELDS & set(wrong)) or 'flags'})"
        )


def _extract_kube_context(kubeconfig_text: str, context: str) -> Dict[str, Dict]:
    """Pull one context, with the cluster and user it refers to, out of a kubeconfig.

    The cluster and user are renamed ``gbserver/<context>`` and the context is
    pointed at the renamed pair. The kubernetes client keeps the first
    definition of each cluster/user *name* across ``KUBECONFIG``, so keeping a
    secret's own (often generic, e.g. ``default``) names could silently swap the
    server or credentials of an unrelated context.

    :param kubeconfig_text: A kubeconfig document (YAML), e.g. a secret's value.
    :param context: The context to extract.
    :returns: ``{"context": ..., "cluster": ..., "user": ...}``, each a named
        kubeconfig list entry.
    :raises ValueError: If the text is not a kubeconfig or lacks the context, or
        the cluster/user it names. The message never includes the text.
    """
    if not isinstance(kubeconfig_text, (str, bytes)):
        raise ValueError(f"kubeconfig for context '{context}' is not text")
    try:
        doc = yaml.safe_load(kubeconfig_text) or {}
    except (yaml.YAMLError, RecursionError):
        # `from None`: PyYAML's error echoes the offending line, which may hold
        # a token or key.
        raise ValueError(
            f"kubeconfig for context '{context}' is not valid YAML"
        ) from None
    if not isinstance(doc, dict):
        raise ValueError(f"kubeconfig for context '{context}' is not a mapping")
    for section in ("contexts", "clusters", "users"):
        if not isinstance(doc.get(section) or [], list):
            raise ValueError(
                f"kubeconfig for context '{context}': '{section}' is not a list"
            )
    ctx = _named(doc.get("contexts"), context)
    if ctx is None:
        raise ValueError(f"kubeconfig does not define context '{context}'")
    refs = ctx.get("context") or {}
    if not isinstance(refs, dict):
        raise ValueError(f"kubeconfig context '{context}' is malformed")
    _check_fields("context", context, refs, _KUBE_CONTEXT_FIELDS)
    if not refs.get("cluster") or not refs.get("user"):
        raise ValueError(
            f"kubeconfig context '{context}' must name both a cluster and a user"
        )
    cluster = _named(doc.get("clusters"), refs.get("cluster"))
    user = _named(doc.get("users"), refs.get("user"))
    if cluster is None or user is None:
        raise ValueError(
            f"kubeconfig context '{context}' names a cluster or user it does not define"
        )
    _check_fields("cluster", context, cluster.get("cluster"), _KUBE_CLUSTER_FIELDS)
    _check_fields("user", context, user.get("user"), _KUBE_USER_FIELDS)
    managed_name = _KUBE_NAME_PREFIX + context
    return {
        "context": {
            "name": context,
            "context": {
                **({"namespace": refs["namespace"]} if "namespace" in refs else {}),
                "cluster": managed_name,
                "user": managed_name,
            },
        },
        # Only the checked bodies: sibling keys of the list entry (an ``exec``
        # next to ``user:``, say) are dropped, not copied into the file.
        "cluster": {"name": managed_name, "cluster": cluster["cluster"]},
        "user": {"name": managed_name, "user": user["user"]},
    }


def _kubeconfig_paths(raw: Optional[str] = None) -> List[str]:
    """``KUBECONFIG`` (or ``~/.kube/config``) as absolute, ``~``-expanded paths.

    kubectl does not expand ``~`` in ``KUBECONFIG`` (the Python client does), so
    a literal ``~/.kube/config`` there would hide the user's file from the
    kubectl calls SkyPilot makes.
    """
    raw = raw if raw is not None else os.environ.get("KUBECONFIG")
    raw = raw or _DEFAULT_KUBECONFIG
    return [
        os.path.abspath(os.path.expanduser(part))
        for part in raw.split(os.pathsep)
        if part
    ]


class _UserKubeconfig:
    """What the user's own kubeconfig files (all but the managed one) define."""

    def __init__(self: "_UserKubeconfig") -> None:
        self.contexts: Set[str] = set()
        self.clusters: Set[str] = set()
        self.users: Set[str] = set()
        # The first and the last non-empty current-context across the files:
        # kubectl uses the first, the kubernetes Python client the last.
        self.first_current: Optional[str] = None
        self.last_current: Optional[str] = None

    @property
    def has_default(self: "_UserKubeconfig") -> bool:
        """Whether both tools' default is one of the user's own contexts.

        If either the first or the last current-context set is missing or names
        a context only the managed file defines, the tool following that rule
        would otherwise default to an environment's secret context.
        """
        return all(
            c is not None and c in self.contexts
            for c in (self.first_current, self.last_current)
        )


def _user_kubeconfig(managed: Path) -> _UserKubeconfig:
    """Read the user's kubeconfig files: :func:`_kubeconfig_paths` minus ``managed``.

    Unreadable files are skipped, as kubectl does.
    """
    found = _UserKubeconfig()
    for part in _kubeconfig_paths():
        path = Path(part)
        if path == managed or not path.is_file():
            continue
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(doc, dict):
            continue
        for section, names in (
            ("contexts", found.contexts),
            ("clusters", found.clusters),
            ("users", found.users),
        ):
            entries = doc.get(section)
            for entry in entries if isinstance(entries, list) else []:
                if isinstance(entry, dict) and entry.get("name"):
                    names.add(str(entry["name"]))
        if doc.get("current-context"):
            current = str(doc["current-context"])
            if found.first_current is None:
                found.first_current = current
            found.last_current = current
    return found


def _without(doc: Dict, section: str, name: str) -> None:
    """Drop the entry called ``name`` from ``doc[section]``, in place.

    Entries that are not mappings are dropped too, so a malformed managed file
    can be repaired by the next write instead of crashing it.
    """
    entries = doc.get(section)
    doc[section] = [
        e
        for e in (entries if isinstance(entries, list) else [])
        if isinstance(e, dict) and e.get("name") != name
    ]


def _set_current_context(doc: Dict, user_has_default: bool) -> None:
    """Give the merged ``KUBECONFIG`` a safe ``current-context``.

    The kubernetes client SkyPilot uses rejects a merged ``KUBECONFIG`` with no
    current-context (SkyPilot then sees no contexts at all), but a real managed
    context there would become the process-wide *default* and silently serve an
    environment's secret credentials to anything relying on the default. So:

    * the user's kubeconfig files set a usable current-context for both tools
      (the first one set, which kubectl uses, and the last, which the Python
      client uses, each name a user context): this file sets none, and both
      use the user's;
    * it does not: this file carries the unreachable, credential-less
      :data:`_KUBE_PLACEHOLDER` as its current-context, so such a request fails
      loudly, as it would without ``kube_configs``.

    Re-evaluated on each write; without managed contexts, nothing is added.

    :param doc: The managed kubeconfig (mutated in place).
    :param user_has_default: Whether the user's kubeconfig sets a usable one.
    """
    for section in _KUBE_PLACEHOLDER_ENTRIES:
        _without(doc, section, _KUBE_PLACEHOLDER)
    doc.pop("current-context", None)
    if doc.get("contexts") and not user_has_default:
        for section, entry in _KUBE_PLACEHOLDER_ENTRIES.items():
            doc.setdefault(section, []).append(copy.deepcopy(entry))
        doc["current-context"] = _KUBE_PLACEHOLDER


def _read_managed(dest: Path) -> Dict[str, Any]:
    """The managed kubeconfig, or ``{}``.

    :raises ValueError: If it is not a YAML mapping. PyYAML's own error, which
        echoes the offending line (possibly a token), is not chained.
    """
    if not dest.exists():
        return {}
    try:
        doc = yaml.safe_load(dest.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        raise ValueError(f"{dest} is not valid YAML; remove it to reset") from None
    if not isinstance(doc, dict):
        raise ValueError(f"{dest} is not a kubeconfig mapping; remove it to reset")
    return doc


#: Seconds to wait for the managed kubeconfig's cross-process lock. Bounded so
#: a stuck holder surfaces as an error (a TimeoutError, i.e. OSError) instead of
#: blocking the caller indefinitely.
_KUBE_LOCK_TIMEOUT_S = 60.0


def _kube_owners(doc: Dict) -> Dict[str, Set[str]]:
    """The managed file's owner record as ``{context: {owner, ...}}``.

    Older files recorded one owner per context as a string; that reads as a
    one-member set.
    """
    raw = doc.get(_KUBE_OWNERS_KEY) or {}
    owners: Dict[str, Set[str]] = {}
    if not isinstance(raw, dict):
        return owners
    for name, value in raw.items():
        if isinstance(value, str):
            owners[str(name)] = {value}
        elif isinstance(value, list):
            owners[str(name)] = {str(v) for v in value}
    return owners


def _store_owners(doc: Dict, owners: Dict[str, Set[str]]) -> None:
    doc[_KUBE_OWNERS_KEY] = {n: sorted(o) for n, o in sorted(owners.items()) if o}


def _drop_context(doc: Dict, name: str) -> None:
    """Remove a managed context with its ``gbserver/<context>`` cluster and user."""
    _without(doc, "contexts", name)
    _without(doc, "clusters", _KUBE_NAME_PREFIX + name)
    _without(doc, "users", _KUBE_NAME_PREFIX + name)


def _prune_stale_owners(doc: Dict, owners: Dict[str, Set[str]]) -> None:
    """Drop owners whose gbserver process is gone, and contexts left unowned.

    See :func:`_owner_is_stale`.
    """
    for name in list(owners):
        stale = {o for o in owners[name] if _owner_is_stale(o)}
        if not stale:
            continue
        logger.info(
            "Dropping kube context '%s' owners from gbserver processes that "
            "no longer run: %s",
            name,
            sorted(kube_owner_label(o) for o in stale),
        )
        owners[name] -= stale
        if not owners[name]:
            del owners[name]
            _drop_context(doc, name)


def _update_managed(
    home: Optional[Path], change: Callable[[Dict, Path], Any]
) -> Tuple[Any, bool]:
    """Read, change and (if changed) rewrite the managed kubeconfig, under lock.

    Shared by :func:`merge_kube_configs` and :func:`release_kube_configs`:
    after ``change(doc, dest)`` the default context is re-evaluated
    (:func:`_set_current_context`) and the file is written 0600 only when it
    differs.

    :param home: Home dir override (tests).
    :param change: Mutates the document in place; its return value is passed
        back.
    :returns: ``(change's result, whether the file changed)``.
    :raises TimeoutError: If the lock is not acquired within
        :data:`_KUBE_LOCK_TIMEOUT_S`.
    """
    dest = kubeconfig_path(home)
    lock = _lock_for(_home(home), "gbserver-kubeconfig.lock")
    # The in-process lock is timed too: a wedged thread must not hang every
    # kube build's event-loop worker forever.
    if not _THREAD_LOCK.acquire(timeout=_KUBE_LOCK_TIMEOUT_S):
        raise TimeoutError(
            f"timed out after {_KUBE_LOCK_TIMEOUT_S:.0f}s waiting for the "
            "gbserver kubeconfig lock"
        )
    try:
        with lock.acquire(timeout=_KUBE_LOCK_TIMEOUT_S):
            return _change_managed(dest, change)
    finally:
        _THREAD_LOCK.release()


def _change_managed(
    dest: Path, change: Callable[[Dict, Path], Any]
) -> Tuple[Any, bool]:
    """The body of :func:`_update_managed`, run with both locks held."""
    doc = _read_managed(dest)
    before = copy.deepcopy(doc)
    result = change(doc, dest)
    if doc.get("contexts") or before:
        doc.setdefault("apiVersion", "v1")
        doc.setdefault("kind", "Config")
        _set_current_context(doc, _user_kubeconfig(dest).has_default)
    changed = doc != before
    if changed:
        _write_atomic(dest, yaml.safe_dump(doc, default_flow_style=False), mode=0o600)
    return result, changed


def _validated_incoming(
    entries: List[KubeConfigEntry], secrets: Dict[str, str], env_name: str
) -> List[Dict[str, Dict]]:
    """Each entry's context/cluster/user, extracted from its secret and checked.

    :raises ValueError: On a reserved or repeated context name, a missing
        secret (whose configured value is not echoed: it might be a pasted
        kubeconfig), or a secret that fails :func:`_extract_kube_context`.
    """
    incoming = []
    seen: Set[str] = set()
    for entry in entries:
        if entry.context == _KUBE_PLACEHOLDER:
            raise ValueError(f"kube_configs context name '{entry.context}' is reserved")
        if entry.context in seen:
            raise ValueError(
                f"environment '{kube_owner_label(env_name)}': kube_configs lists context "
                f"'{entry.context}' more than once"
            )
        seen.add(entry.context)
        if entry.kubeconfig not in secrets:
            raise ValueError(
                f"environment '{kube_owner_label(env_name)}': kube_configs context "
                f"'{entry.context}' "
                "names a secret (its 'kubeconfig' value) that is not in the "
                "space's secrets; set it to the secret's name"
            )
        incoming.append(_extract_kube_context(secrets[entry.kubeconfig], entry.context))
    return incoming


def _check_user_collisions(
    parts: Dict[str, Dict], user: "_UserKubeconfig", env_name: str, dest: Path
) -> None:
    """Refuse an entry that would shadow, or default over, the user's own config.

    :raises SkypilotConfigCollisionError: On a same-named context, cluster or
        user in the user's kubeconfig, or a user current-context naming it.
    """
    # A clash on any of the three names would let the user's entry (it comes
    # first) or ours shadow the other in one of the tools. Checked first: a user
    # context of the same name is reported as such, even when also selected.
    for kind, item, theirs in (
        ("kube context", parts["context"], user.contexts),
        ("kube cluster", parts["cluster"], user.clusters),
        ("kube user", parts["user"], user.users),
    ):
        if item["name"] in theirs:
            _raise_collision(
                kind,
                item["name"],
                kube_owner_label(env_name),
                "your own kubeconfig",
                str(dest),
            )
    # The user's own default must never resolve to a secret context. kubectl
    # takes the first current-context set and nothing later on KUBECONFIG (our
    # placeholder included) overrides it, so a user current-context naming one
    # of our contexts is refused outright, including one set by
    # `kubectl config use-context <name>`.
    name = parts["context"]["name"]
    if name in (user.first_current, user.last_current):
        raise SkypilotConfigCollisionError(
            f"kube context '{name}' from environment "
            f"'{kube_owner_label(env_name)}' is the "
            "current-context of your own kubeconfig, so it would become "
            "the default for every tool. Set your current-context to one "
            "of your own contexts (and use --context to reach this one), "
            "or rename the context in the environment's kubeconfig secret."
        )


def _apply_entry(
    doc: Dict,
    owners: Dict[str, Set[str]],
    parts: Dict[str, Dict],
    env_name: str,
    dest: Path,
) -> None:
    """Add or update one context in ``doc``, recording ``env_name`` as an owner.

    * Not present: added, owned by ``env_name``.
    * Identical to what is there: ``env_name`` joins its owners.
    * Different, and ``env_name`` is its *only* owner: replaced (a rotated
      token).
    * Different otherwise: refused. Replacing it would change the credentials
      another environment uses.

    :raises SkypilotConfigCollisionError: In the last case.
    """
    name = parts["context"]["name"]
    current = owners.get(name, set())
    existing = [
        _named(doc.get(s), n)
        for s, n in (
            ("contexts", name),
            ("clusters", parts["cluster"]["name"]),
            ("users", parts["user"]["name"]),
        )
    ]
    identical = existing == [parts["context"], parts["cluster"], parts["user"]]
    if current and not identical and current != {env_name}:
        others = sorted({kube_owner_label(o) for o in current - {env_name}})
        _raise_collision(
            "kube context",
            name,
            kube_owner_label(env_name),
            "environment(s) " + ", ".join(f"'{o}'" for o in others),
            str(dest),
        )
    if not identical:
        for section, key in (
            ("contexts", "context"),
            ("clusters", "cluster"),
            ("users", "user"),
        ):
            _without(doc, section, parts[key]["name"])
            doc.setdefault(section, []).append(parts[key])
    owners[name] = current | {env_name}


def merge_kube_configs(
    entries: List[KubeConfigEntry],
    secrets: Dict[str, str],
    env_name: str,
    home: Optional[Path] = None,
) -> bool:
    """Merge an environment's ``kube_configs`` contexts into gbserver's kubeconfig.

    Owner-aware per context, like the SSH blocks: each context, with its own
    ``gbserver/<context>`` cluster and user, records the *set* of environments
    using it (see :func:`_apply_entry`). An identical entry adds an owner; a
    sole owner's differing entry replaces it (a rotated token); a context shared
    with, or owned by, another environment cannot be changed; and a context, a
    cluster or a user of the same name in the user's own kubeconfig, or a user
    current-context naming one, is refused. Contexts are removed only by
    :func:`release_kube_configs`, once no owner is left, so environments sharing
    one keep it while any of them builds. Written 0600 under a cross-process
    lock.

    :param entries: The environment's ``kube_configs`` entries.
    :param secrets: Secret name -> value; each entry's ``kubeconfig`` must name one.
    :param env_name: The contributing owner (``<space>/<environment>``).
    :param home: Home dir override (tests).
    :returns: Whether the managed kubeconfig changed.
    :raises ValueError: If a named secret is missing or invalid, or one
        environment lists a context twice.
    :raises SkypilotConfigCollisionError: On a collision (above).
    :raises TimeoutError: If the file's lock is not acquired in time.
    """
    if not entries:
        return False
    incoming = _validated_incoming(entries, secrets, env_name)

    def change(doc: Dict, dest: Path) -> None:
        owners = _kube_owners(doc)
        _prune_stale_owners(doc, owners)
        user = _user_kubeconfig(dest)
        for parts in incoming:
            _check_user_collisions(parts, user, env_name, dest)
            _apply_entry(doc, owners, parts, env_name, dest)
        _store_owners(doc, owners)

    _, changed = _update_managed(home, change)
    if changed:
        logger.info(
            "Wrote kube contexts %s for environment '%s' into %s",
            [p["context"]["name"] for p in incoming],
            kube_owner_label(env_name),
            kubeconfig_path(home),
        )
    return changed


def release_kube_configs(
    env_name: str,
    contexts: Optional[List[str]] = None,
    home: Optional[Path] = None,
) -> bool:
    """Drop ``env_name`` as an owner, removing contexts nobody else owns.

    :param env_name: The owner to release (``<space>/<environment>``).
    :param contexts: Only these contexts; all of the owner's when ``None``.
    :param home: Home dir override (tests).
    :returns: Whether the managed kubeconfig changed.
    :raises TimeoutError: If the file's lock is not acquired in time.
    """

    def change(doc: Dict, dest: Path) -> None:
        owners = _kube_owners(doc)
        _prune_stale_owners(doc, owners)
        for name in list(owners):
            if contexts is not None and name not in contexts:
                continue
            owners[name].discard(env_name)
            if not owners[name]:
                del owners[name]
                _drop_context(doc, name)
        if doc:
            _store_owners(doc, owners)

    _, changed = _update_managed(home, change)
    return changed


def host_sky_config(home: Optional[Path] = None) -> Dict[str, Any]:
    """The host's own part of ``~/.sky/config.yaml``: what gbserver did not write.

    The file minus the keys :func:`merge_cloud_config` recorded for the env
    whose ``cloud_config`` it last wrote. Unreadable or malformed files read as
    ``{}``.

    :param home: Home dir override (tests).
    """
    home_path = _home(home)
    try:
        doc = yaml.safe_load(
            (home_path / ".sky" / "config.yaml").read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError, RecursionError):
        return {}
    if not isinstance(doc, dict):
        return {}
    try:
        record = yaml.safe_load(
            (home_path / ".sky" / _CLOUD_CONFIG_RECORD).read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError, RecursionError):
        record = None
    recorded = record.get("cloud_config") if isinstance(record, dict) else None
    if isinstance(recorded, dict):
        _remove_recorded(doc, recorded)
    return doc


def user_kube_context_names(home: Optional[Path] = None) -> Set[str]:
    """Context names the user's own kubeconfig files (not gbserver's) define.

    :param home: Home dir override (tests).
    """
    return _user_kubeconfig(kubeconfig_path(home)).contexts


def managed_kube_contexts(home: Optional[Path] = None) -> Dict[str, Set[str]]:
    """The contexts gbserver's kubeconfig holds, with their owners.

    Read without the lock: a reader only checks membership, and the file is
    replaced atomically. Owners from gbserver processes that no longer run are
    left out (the next merge or release removes them from the file).

    :param home: Home dir override (tests).
    :returns: ``{context: {owner, ...}}``; empty when there is no file.
    :raises ValueError: If the file is not a valid kubeconfig mapping.
    """
    owners = _kube_owners(_read_managed(kubeconfig_path(home)))
    live = {n: {o for o in os_ if not _owner_is_stale(o)} for n, os_ in owners.items()}
    return {n: os_ for n, os_ in live.items() if os_}


def activate_kubeconfig(home: Optional[Path] = None) -> None:
    """Add gbserver's kubeconfig *last* on this process's ``KUBECONFIG``.

    Keeps whatever kubeconfig was in effect (``KUBECONFIG``, else
    ``~/.kube/config``) ahead of it, with every path made absolute and
    ``~``-expanded (kubectl does not expand ``~``). Last, so the user's own
    entries win any name clash in both kubectl and the Python client, and their
    ``current-context`` is the default when set (see
    :func:`_set_current_context`). Idempotent. SkyPilot re-reads ``KUBECONFIG``
    on every load, and a local API server receives the client's value with every
    request.

    :param home: Home dir override (tests).
    """
    managed = os.path.abspath(str(kubeconfig_path(home)))
    parts = [p for p in _kubeconfig_paths() if p != managed]
    value = os.pathsep.join(parts + [managed])
    if os.environ.get("KUBECONFIG") != value:
        os.environ["KUBECONFIG"] = value


def render_aws_profile(
    profile: AwsCredentialProfile, secrets: Dict[str, str]
) -> Tuple[str, Dict[str, str]]:
    """Render an ``AwsCredentialProfile`` to ``(section_name, {key: value})``.

    Fields configured as ``None`` — and fields naming a secret that resolves to
    ``None`` — are omitted, so ``configparser`` (which rejects non-string option
    values) never receives a ``None``.
    """
    fields = [
        ("aws_access_key_id", profile.aws_access_key_id),
        ("aws_secret_access_key", profile.aws_secret_access_key),
        ("aws_session_token", profile.aws_session_token),
    ]
    kv: Dict[str, str] = {}
    for key, raw in fields:
        if raw is None:
            continue
        resolved = _resolve(raw, secrets)
        if resolved is None:
            continue  # named secret resolved to None — omit rather than crash
        kv[key] = resolved
    return profile.profile, kv


def merge_aws_credentials(
    profiles: List[AwsCredentialProfile],
    secrets: Dict[str, str],
    env_name: str,
    home: Optional[Path] = None,
) -> None:
    """Merge AWS credential profiles into ``~/.aws/credentials`` (mode 0600).

    Refuses on conflict: an existing profile with different values raises rather
    than overwriting it (never clobbers a user's real credentials); identical
    values are an idempotent no-op.

    :param profiles: Profiles to materialize (values secret-resolved).
    :param secrets: Secret name -> value mapping.
    :param env_name: The contributing environment name.
    :param home: Home dir override (tests).
    :raises SkypilotConfigCollisionError: On same profile, different values.
    """
    if not profiles:
        return
    home_path = _home(home)
    dest = home_path / ".aws" / "credentials"
    with _THREAD_LOCK, _lock_for(home_path, "gbserver-aws.lock"):
        parser = configparser.ConfigParser()
        if dest.exists():
            parser.read(dest)
        for profile in profiles:
            section, kv = render_aws_profile(profile, secrets)
            if parser.has_section(section):
                existing = {k: parser[section].get(k) for k in kv}
                if existing != kv:
                    _raise_collision(
                        "AWS profile",
                        section,
                        env_name,
                        "an existing profile",
                        str(dest),
                    )
                continue
            parser.add_section(section)
            for k, v in kv.items():
                parser.set(section, k, v)
        buf = io.StringIO()
        parser.write(buf)
        _write_atomic(dest, buf.getvalue(), mode=0o600)


def _candidate_hostnames(hosts: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Map each host alias to the set of its candidate ``HostName`` values.

    Captured *before* a scalar/list ``HostName`` is collapsed to one pick, so the
    merge can tell a genuine same-cluster/different-login-node overwrite from a
    different cluster reusing the alias (see :func:`_merge_ssh`).

    :param hosts: Host dicts with scalar-or-list ``HostName`` (pre-collapse).
    :returns: ``{alias: {candidate HostName, …}}``; aliases without a ``HostName``
        are omitted.
    """
    out: Dict[str, Set[str]] = {}
    for host in hosts:
        hostname = host.get("HostName")
        if hostname is None:
            continue
        values = hostname if isinstance(hostname, list) else [hostname]
        out[str(host.get("Host"))] = {str(v) for v in values}
    return out


def materialize_ssh_for_cloud(
    env_name: str,
    ssh: ClusterSshConfigs,
    secrets: Dict[str, str],
    cloud: str,
    *,
    home: Optional[Path] = None,
) -> None:
    """Merge one cloud's inline SSH ``Host`` blocks into ``~/.<cloud>/config``.

    Extracted from :func:`materialize` so a launch can (re-)merge just the cloud
    it is provisioning. No-op when the config defines no hosts for ``cloud``. A
    list-valued ``HostName`` is collapsed to one randomly-chosen candidate for load
    spreading (the launch path additionally fails over between candidates on a
    transient SSH error — see ``gbserver.environment.skypilot._LoginNodeRotator``).

    :param env_name: The environment name (used in messages).
    :param ssh: Inline cluster SSH configs.
    :param secrets: Secret name -> value mapping for field resolution.
    :param cloud: ``"slurm"`` or ``"lsf"`` — the cloud whose hosts to merge.
    :param home: Home dir override (tests).
    :raises SkypilotConfigCollisionError: On a foreign clash (see
        :func:`merge_ssh_blocks`).
    """
    hosts = _resolve_cloud_hosts(ssh, secrets, cloud, home=home)
    if not hosts:
        return
    # Capture candidate hostnames before collapsing, so the merge can gate its
    # HostName-only cross-env relaxation on them (see _merge_ssh).
    candidates = _candidate_hostnames(hosts)
    # Collapse any list-valued HostName to a single (random) login node.
    hosts = [_collapse_to_random_hostname(h) for h in hosts]
    _merge_selected_hosts(
        cloud, hosts, secrets, env_name, candidate_hostnames=candidates, home=home
    )


def _resolve_cloud_hosts(
    ssh: ClusterSshConfigs,
    secrets: Dict[str, str],
    cloud: str,
    *,
    home: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return one cloud's host dicts with ``IdentityKey`` resolved to ``IdentityFile``.

    The first step of launch-time SSH materialization (see
    :func:`materialize_ssh_for_cloud`), split out so the launch path can resolve keys
    off the loop, then pick one login node per alias, then merge. Writes managed key
    files but performs no network I/O, so it is safe to run off the event loop via
    ``asyncio.to_thread``.

    :param ssh: Inline cluster SSH configs.
    :param secrets: Secret name -> value mapping for field resolution.
    :param cloud: ``"slurm"`` or ``"lsf"`` — the cloud whose hosts to resolve.
    :param home: Home dir override (tests).
    :returns: The host dicts for ``cloud`` (``[]`` if none), each with any
        ``IdentityKey`` rewritten to a managed ``IdentityFile`` and ``HostName`` still
        scalar-or-list (candidate selection has not run yet).
    :raises ValueError: If a host sets both ``IdentityKey`` and ``IdentityFile``, or
        ``IdentityKey`` resolves to empty (see :func:`_materialize_identity_keys`).
    """
    hosts = {"slurm": ssh.slurm, "lsf": ssh.lsf}.get(cloud)
    if not hosts:
        return []
    return _materialize_identity_keys(hosts, cloud, secrets, _home(home))


def _merge_selected_hosts(
    cloud: str,
    hosts: List[Dict[str, Any]],
    secrets: Dict[str, str],
    env_name: str,
    *,
    candidate_hostnames: Optional[Dict[str, Set[str]]] = None,
    home: Optional[Path] = None,
) -> None:
    """Render already-selected scalar-``HostName`` hosts and merge them into config.

    The tail of :func:`materialize_ssh_for_cloud`, callable on its own once the launch
    path has resolved keys (:func:`_resolve_cloud_hosts`) and picked one login node per
    host — the shared write step used both by that helper and by the launch-time
    ``_LoginNodeRotator`` each time it fails over. Local file I/O only.

    :param cloud: ``"slurm"`` or ``"lsf"`` — the cloud whose config to merge into.
    :param hosts: Host dicts with a single scalar ``HostName`` each.
    :param secrets: Secret name -> value mapping for directive resolution.
    :param env_name: The environment name (used in messages).
    :param candidate_hostnames: ``{alias: {candidate HostName, …}}`` for this env,
        forwarded to gate the merge's HostName-only cross-env relaxation.
    :param home: Home dir override (tests).
    :raises SkypilotConfigCollisionError: On a foreign clash (see
        :func:`merge_ssh_blocks`).
    """
    merge_ssh_blocks(
        cloud,
        render_ssh_hosts(hosts, secrets),
        env_name,
        home=home,
        candidate_hostnames=candidate_hostnames,
    )


def materialize(
    env_name: str,
    ssh: Optional[ClusterSshConfigs],
    cloud_config: Optional[Dict],
    aws_credentials: Optional[List[AwsCredentialProfile]],
    secrets: Dict[str, str],
    *,
    home: Optional[Path] = None,
) -> None:
    """Materialize all inline config sections for one environment.

    Write/merge-only; no cleanup. SSH/AWS sections raise
    ``SkypilotConfigCollisionError`` on a foreign/conflicting entry;
    ``cloud_config`` is written from the env into ``~/.sky/config.yaml``
    (env values win, last writer wins; see :func:`merge_cloud_config`). The
    Skypilot environment writes ``cloud_config`` itself, per launch, so it passes
    ``None`` here.

    :param env_name: The environment name (used in messages).
    :param ssh: Inline cluster SSH configs, or ``None``.
    :param cloud_config: Inline behavioral SkyPilot config, or ``None``.
    :param aws_credentials: Inline AWS credential profiles, or ``None``.
    :param secrets: Secret name -> value mapping for field resolution.
    :param home: Home dir override (tests).
    """
    if ssh:
        for cloud in ("slurm", "lsf"):
            materialize_ssh_for_cloud(env_name, ssh, secrets, cloud, home=home)
    if cloud_config:
        merge_cloud_config(cloud_config, env_name, home=home)
    if aws_credentials:
        merge_aws_credentials(aws_credentials, secrets, env_name, home=home)
