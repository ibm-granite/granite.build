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

Write/merge-only — no refcount, no teardown. Three destinations are supported,
all keyed so different clusters/profiles coexist and only a *true* clash raises
``SkypilotConfigCollisionError``:

  * ``cluster_ssh_configs`` -> ``~/.<cloud>/config`` (OpenSSH ``Host`` blocks,
    merged by alias under a cross-process file lock).
  * ``cloud_config`` -> a per-process temp YAML file pointed at by
    ``SKYPILOT_PROJECT_CONFIG`` (deep-merged by nested key), which SkyPilot
    sends to the API server as a per-request override.
  * ``aws_credentials`` -> ``~/.aws/credentials`` (INI, mode 0600, merged by
    profile under a cross-process file lock).

Connection/credential field values are resolved by exact-name lookup against
the environment's secrets, falling back to the literal value. The module is
pure-filesystem (no ``sky`` import) so it is unit-testable without the SDK.
"""

import configparser
import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from filelock import FileLock

from gbserver.types.environmentconfig import (
    AwsCredentialProfile,
    ClusterSshConfigs,
    ClusterSshHost,
)
from gbserver.types.errors import SkypilotConfigCollisionError
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Markers delimiting the gbserver-managed region of an SSH config file. Valid
# OpenSSH comments, so any foreign content outside the region is left untouched.
MANAGED_BEGIN = "# BEGIN gbserver-managed (cluster config)"
MANAGED_END = "# END gbserver-managed"
# Per-alias comment recording the contributing environment, so a later
# collision can name the owner even across processes (the file is the source
# of truth). Excluded from idempotency comparison.
_OWNER_PREFIX = "# gbserver-owner="

# Env var SkyPilot reads as the client "project" config and merges over the
# user config; its resolved content is sent to the server per request.
ENV_VAR_PROJECT_CONFIG = "SKYPILOT_PROJECT_CONFIG"

# Serializes file read-merge-write across threads in one process; the FileLock
# adds cross-process safety (filelock alone does not serialize same-process
# threads). cloud_config accumulation also lives under this lock.
_THREAD_LOCK = threading.RLock()


class _CloudConfigState:
    """Per-process accumulator for merged ``cloud_config`` (guarded by ``_THREAD_LOCK``).

    Held in a single module instance (rather than module globals) so the merge
    helpers mutate attributes without ``global`` statements.

    Attributes:
        config: The deep-merged config sent via ``SKYPILOT_PROJECT_CONFIG``.
        owners: Dotted-key -> contributing environment (for collision messages).
        path: The temp file the merged config is written to (stable per process).
    """

    def __init__(self) -> None:
        self.config: Dict = {}
        self.owners: Dict[str, str] = {}
        self.path: Optional[str] = None

    def reset(self) -> None:
        """Clear accumulated state (test helper)."""
        self.config = {}
        self.owners = {}
        self.path = None


_CLOUD_STATE = _CloudConfigState()


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
def render_ssh_host(host: ClusterSshHost, secrets: Dict[str, str]) -> str:
    """Render one ``ClusterSshHost`` to an OpenSSH ``Host`` block.

    :param host: The host stanza to render.
    :param secrets: Secret name -> value mapping for field resolution.
    :returns: The multi-line ``Host`` block text (resolved values).
    """
    lines = [f"Host {host.host}"]
    pairs = [
        ("HostName", host.hostname),
        ("User", host.user),
        ("Port", host.port),
        ("IdentityFile", host.identity_file),
    ]
    for key, raw in pairs:
        if raw is not None:
            logger.debug(
                "ssh field %s for host %s resolved %s",
                key,
                host.host,
                "from-secret" if str(raw) in secrets else "literal",
            )
            lines.append(f"    {key} {_resolve(raw, secrets)}")
    for opt_key, opt_raw in host.options.items():
        lines.append(f"    {opt_key} {_resolve(opt_raw, secrets)}")
    return "\n".join(lines)


def render_ssh_hosts(
    hosts: List[ClusterSshHost], secrets: Dict[str, str]
) -> Dict[str, str]:
    """Render hosts to an ``{alias: block}`` map (keyed for per-alias merge)."""
    return {h.host: render_ssh_host(h, secrets) for h in hosts}


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


def _foreign_aliases(foreign: str) -> set:
    """Return the set of ``Host`` aliases defined in non-managed (foreign) text."""
    out = set()
    for raw in foreign.splitlines():
        s = raw.strip()
        if s.lower().startswith("host ") and not s.startswith("#"):
            out.add(s.split(None, 1)[1].strip())
    return out


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


def _merge_ssh(
    existing: Dict[str, Tuple[str, str]],
    incoming: Dict[str, str],
    foreign: set,
    env_name: str,
    dest: str,
) -> Dict[str, Tuple[str, str]]:
    """Merge incoming alias blocks into existing; raise on a true clash.

    Refuses on any conflict: a foreign (non-gbserver) entry for the same alias, or
    a gbserver-managed alias whose body differs, raises so a misconfiguration is
    never silently resolved. Re-applying an identical gbserver-managed block is an
    idempotent no-op.

    :param existing: Current ``{alias: (block, owner)}`` from the managed region.
    :param incoming: New ``{alias: block}`` to merge in.
    :param foreign: Aliases already defined in non-managed content.
    :param env_name: The contributing environment name.
    :param dest: Destination file path (for messages).
    :returns: The merged ``{alias: (block, owner)}``.
    :raises SkypilotConfigCollisionError: On a foreign or differing-body clash.
    """
    merged = dict(existing)
    for alias, block in incoming.items():
        if alias in foreign:
            _raise_collision(
                "SSH Host", alias, env_name, "a pre-existing (non-gbserver) entry", dest
            )
        if alias in merged:
            old_block, old_owner = merged[alias]
            if _normalize(old_block) != _normalize(block):
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
            existing, alias_blocks, _foreign_aliases(foreign), env_name, str(dest)
        )
        _write_atomic(dest, _compose(foreign, _serialize_managed(merged)))


def _deep_merge_into(
    base: Dict, overlay: Dict, env_name: str, owners: Dict[str, str], prefix: tuple
) -> None:
    """Recursively merge ``overlay`` into ``base``; raise on a leaf clash.

    :param base: Accumulated config (mutated in place).
    :param overlay: New config to merge in.
    :param env_name: The contributing environment name.
    :param owners: Dotted-key -> owning environment (for messages).
    :param prefix: Current key path (for dotted-key messages).
    :raises SkypilotConfigCollisionError: On same leaf key, different value.
    """
    for key, val in overlay.items():
        dotted = ".".join(prefix + (str(key),))
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge_into(base[key], val, env_name, owners, prefix + (str(key),))
        elif key in base and base[key] != val:
            _raise_collision(
                "cloud_config key",
                dotted,
                env_name,
                f"'{owners.get(dotted, 'an existing value')}'",
                "skypilot config",
            )
        else:
            base[key] = val
            owners[dotted] = env_name


def _project_config_path(tmp_root: Optional[Path]) -> Path:
    """Return the per-process project-config temp file path.

    The filename is scoped to the PID so separate buildrunner *processes* on a
    shared host each own a distinct file (and set their own
    ``SKYPILOT_PROJECT_CONFIG``). That makes cloud_config genuinely per-process,
    so the in-process ``_THREAD_LOCK`` is sufficient and no cross-process file
    lock is needed for it (unlike the host-shared SSH / AWS files).
    """
    root = Path(tmp_root) if tmp_root else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return root / f"gb_sky_project_config_{os.getpid()}.yaml"


def merge_cloud_config(
    cloud_config: Dict, env_name: str, tmp_root: Optional[Path] = None
) -> None:
    """Deep-merge ``cloud_config`` into the per-process project config + env var.

    Writes the merged YAML to a temp file *before* setting
    ``SKYPILOT_PROJECT_CONFIG`` (SkyPilot errors if the env var points at a
    missing file).

    :param cloud_config: The behavioral SkyPilot config block to merge.
    :param env_name: The contributing environment name.
    :param tmp_root: Temp dir override (tests).
    :raises SkypilotConfigCollisionError: On a leaf-key clash.
    """
    if not cloud_config:
        return
    with _THREAD_LOCK:
        _deep_merge_into(
            _CLOUD_STATE.config, cloud_config, env_name, _CLOUD_STATE.owners, ()
        )
        path = (
            Path(_CLOUD_STATE.path)
            if _CLOUD_STATE.path
            else _project_config_path(tmp_root)
        )
        _write_atomic(
            path, yaml.safe_dump(_CLOUD_STATE.config, default_flow_style=False)
        )
        _CLOUD_STATE.path = str(path)
        os.environ[ENV_VAR_PROJECT_CONFIG] = str(path)


def render_aws_profile(
    profile: AwsCredentialProfile, secrets: Dict[str, str]
) -> Tuple[str, Dict[str, str]]:
    """Render an ``AwsCredentialProfile`` to ``(section_name, {key: value})``."""
    fields = [
        ("aws_access_key_id", profile.aws_access_key_id),
        ("aws_secret_access_key", profile.aws_secret_access_key),
        ("aws_session_token", profile.aws_session_token),
    ]
    kv = {key: _resolve(raw, secrets) for key, raw in fields if raw is not None}
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
    tmp_root: Optional[Path] = None,
) -> None:
    """Materialize all inline config sections for one environment.

    Write/merge-only; no cleanup. Raises ``SkypilotConfigCollisionError`` if any
    section clashes with config already materialized by another environment.

    :param env_name: The environment name (used in collision messages).
    :param ssh: Inline cluster SSH configs, or ``None``.
    :param cloud_config: Inline behavioral SkyPilot config, or ``None``.
    :param aws_credentials: Inline AWS credential profiles, or ``None``.
    :param secrets: Secret name -> value mapping for field resolution.
    :param home: Home dir override (tests).
    :param tmp_root: Temp dir override (tests).
    """
    if ssh:
        for cloud, hosts in (("slurm", ssh.slurm), ("lsf", ssh.lsf)):
            if hosts:
                merge_ssh_blocks(
                    cloud, render_ssh_hosts(hosts, secrets), env_name, home=home
                )
    if cloud_config:
        merge_cloud_config(cloud_config, env_name, tmp_root=tmp_root)
    if aws_credentials:
        merge_aws_credentials(aws_credentials, secrets, env_name, home=home)


def _reset_for_tests() -> None:
    """Reset module-level cloud_config state (test helper)."""
    _CLOUD_STATE.reset()
    os.environ.pop(ENV_VAR_PROJECT_CONFIG, None)
