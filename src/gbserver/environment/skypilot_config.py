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

Write/merge-only — no refcount, no teardown. Three destinations are supported:

  * ``cluster_ssh_configs`` -> ``~/.<cloud>/config`` (OpenSSH ``Host`` blocks,
    merged by alias under a cross-process file lock; a foreign or differing
    entry raises ``SkypilotConfigCollisionError``).
  * ``cloud_config`` -> ``~/.sky/config.yaml`` (deep-merged into the global
    SkyPilot config the API server / optimizer reads directly; the env's values
    win, unrelated keys are preserved).
  * ``aws_credentials`` -> ``~/.aws/credentials`` (INI, mode 0600, merged by
    profile under a cross-process file lock).

Connection/credential field values are resolved by exact-name lookup against
the environment's secrets, falling back to the literal value. The module is
pure-filesystem (no ``sky`` import) so it is unit-testable without the SDK.
"""

import configparser
import hashlib
import io
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from filelock import FileLock

from gbserver.types.environmentconfig import (
    AwsCredentialProfile,
    ClusterSshConfigs,
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


def _merge_ssh(
    existing: Dict[str, Tuple[str, str]],
    incoming: Dict[str, str],
    foreign_blocks: Dict[str, Tuple[str, str]],
    env_name: str,
    dest: str,
) -> Dict[str, Tuple[str, str]]:
    """Merge incoming alias blocks into existing; raise only on a real conflict.

    Content-aware refuse-on-conflict: a pre-existing entry (foreign/non-gbserver
    or a prior gbserver-managed block) for the same alias is a conflict **only if
    its content differs**. An identical pre-existing entry is a no-op — the env
    and the on-disk config agree, so nothing is written and nothing is raised.

    :param existing: Current ``{alias: (block, owner)}`` from the managed region.
    :param incoming: New ``{alias: block}`` to merge in.
    :param foreign_blocks: ``{alias: (block, owner)}`` parsed from non-managed content.
    :param env_name: The contributing environment name.
    :param dest: Destination file path (for messages).
    :returns: The merged ``{alias: (block, owner)}`` (foreign-equivalent aliases omitted).
    :raises SkypilotConfigCollisionError: On a same-alias, differing-content clash.
    """
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
            old_block, old_owner = merged[alias]
            if not _blocks_equivalent(old_block, block):
                _raise_collision(
                    "SSH Host",
                    alias,
                    env_name,
                    f"'{old_owner or 'an existing entry'}'",
                    dest,
                )
        else:
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
    """Atomically write ``text`` to ``path`` (``*.gbtmp`` + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".gbtmp")
    tmp.write_text(text, encoding="utf-8")
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Merge entry points (file I/O under locks)
# --------------------------------------------------------------------------- #
def merge_ssh_blocks(
    cloud: str,
    alias_blocks: Dict[str, str],
    env_name: str,
    home: Optional[Path] = None,
) -> None:
    """Merge rendered SSH ``Host`` blocks into ``~/.<cloud>/config``.

    :param cloud: Cloud name (``slurm``/``lsf``) -> ``~/.<cloud>/config``.
    :param alias_blocks: ``{alias: block}`` to merge.
    :param env_name: The contributing environment name.
    :param home: Home dir override (tests).
    :raises SkypilotConfigCollisionError: On a true clash.
    """
    if not alias_blocks:
        return
    home_path = _home(home)
    dest = home_path / f".{cloud}" / "config"
    with _THREAD_LOCK, _lock_for(home_path, f"gbserver-{cloud}.lock"):
        text = dest.read_text(encoding="utf-8") if dest.exists() else ""
        foreign, existing = _parse_managed(text)
        merged = _merge_ssh(
            existing, alias_blocks, _parse_host_blocks(foreign), env_name, str(dest)
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


def merge_cloud_config(
    cloud_config: Dict, env_name: str, home: Optional[Path] = None
) -> None:
    """Write the env's ``cloud_config`` into ``~/.sky/config.yaml``.

    Deep-merges ``cloud_config`` into the global SkyPilot config (the file the
    API server / optimizer reads directly), with the env's values taking
    precedence and any unrelated keys preserved. Done under a cross-process file
    lock (the file is host-shared). gbserver materializes this before starting
    the API server, so the server picks it up.

    :param cloud_config: The behavioral SkyPilot config block (e.g. an ``lsf:``
        block) to write, sourced from the environment.yaml.
    :param env_name: The contributing environment name (for logging).
    :param home: Home dir override (tests).
    """
    if not cloud_config:
        return
    home_path = _home(home)
    dest = home_path / ".sky" / "config.yaml"
    with _THREAD_LOCK, _lock_for(home_path, "gbserver-sky-config.lock"):
        existing: Dict = {}
        if dest.exists():
            existing = yaml.safe_load(dest.read_text(encoding="utf-8")) or {}
        _deep_merge_overwrite(existing, cloud_config)
        logger.info(
            "Writing cloud_config from environment '%s' into %s", env_name, dest
        )
        _write_atomic(
            dest, yaml.safe_dump(existing, default_flow_style=False, sort_keys=False)
        )


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
    (env values win).

    :param env_name: The environment name (used in messages).
    :param ssh: Inline cluster SSH configs, or ``None``.
    :param cloud_config: Inline behavioral SkyPilot config, or ``None``.
    :param aws_credentials: Inline AWS credential profiles, or ``None``.
    :param secrets: Secret name -> value mapping for field resolution.
    :param home: Home dir override (tests).
    """
    if ssh:
        for cloud, hosts in (("slurm", ssh.slurm), ("lsf", ssh.lsf)):
            if hosts:
                # Resolve any IdentityKey directive to a managed key file +
                # IdentityFile before rendering (keeps render_ssh_host pure).
                hosts = _materialize_identity_keys(hosts, cloud, secrets, _home(home))
                merge_ssh_blocks(
                    cloud, render_ssh_hosts(hosts, secrets), env_name, home=home
                )
    if cloud_config:
        merge_cloud_config(cloud_config, env_name, home=home)
    if aws_credentials:
        merge_aws_credentials(aws_credentials, secrets, env_name, home=home)
