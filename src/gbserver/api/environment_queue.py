#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""``GET /environments/{space}/{env_path}/queue``: a Slurm queue read over SSH.

The environment is addressed as builds address it: a space and the asset's path
under its ``environments/``. Login details are read from that asset, so nothing
is configured on the server. The caller must have access to the space, and the
asset must be a Slurm environment; anything else is a 404 that does not say which.
The remote commands are fixed and every filter runs in Python.

Trust assumption: only gbserver and its operators submit jobs as the asset's
``User``. Jobs are listed for that user only, so the free text in the output (job
name and comment) was written by gbserver, which slugifies the cluster name and
sanitizes every comment value. Other users' jobs contribute only fixed fields to
the queue-wide ``counts``.

Each job's granite.build tag (``_skypilot_metadata``) is parsed into ``gb``. A job
is shown in full only when its tag's space is one of the caller's spaces (super
admins see all). Every other job, including one without a readable tag, keeps
scheduler fields only.
"""

import asyncio
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel

from gbcommon.uri.uri import URI
from gbserver.api.login_tunnel import fetch_space_ssh_key, open_login_tunnel
from gbserver.api.utils import NO_ACCESSIBLE_SPACE, scope_space_name_filter
from gbserver.environment import _skypilot_metadata as tagmeta
from gbserver.environment.environment import Environment, UnknownEnvironmentType
from gbserver.spaces import db_space
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.constants import ENV_VAR_PREFIX
from gbserver.types.environmentconfig import ClusterSshConfigs
from gbserver.utils.logger import get_logger
from gbserver.utils.ssh_tunnel import SshTunnelError

logger = get_logger(__name__)

queue_api = FastAPI()

_P = ENV_VAR_PREFIX + "_ENV_QUEUE_"
GBSERVER_ENV_QUEUE_CACHE_TTL_S = float(os.getenv(_P + "CACHE_TTL_S", "30"))
GBSERVER_ENV_QUEUE_MAX_JOBS = int(os.getenv(_P + "MAX_JOBS", "5000"))
GBSERVER_ENV_QUEUE_SSH_BUDGET_S = float(os.getenv(_P + "SSH_BUDGET_S", "20"))
GBSERVER_ENV_QUEUE_COMMAND_TIMEOUT_S = float(os.getenv(_P + "COMMAND_TIMEOUT_S", "20"))
# Asset load and key fetch hit git and the Secret Manager; reuse them across refreshes.
_CONNECTION_TTL_S = 600.0
# Cache keys come from request paths: bound each cache (stale entries go first).
_MAX_CACHED_ENVIRONMENTS = 256


def _bound(cache: dict, is_stale) -> None:
    """Make room for one more entry: drop stale ones, then the oldest inserted."""
    if len(cache) < _MAX_CACHED_ENVIRONMENTS:
        return
    for k in [k for k, v in cache.items() if is_stale(v)]:
        del cache[k]
    while len(cache) >= _MAX_CACHED_ENVIRONMENTS:
        del cache[next(iter(cache))]


# ------------------------------------------------------------- environments

# Segments of an asset path under environments/: no '.', '..' or empty segment.
_ENV_PATH_RE = re.compile(
    r"[A-Za-z0-9_-][A-Za-z0-9_.-]*(/[A-Za-z0-9_-][A-Za-z0-9_.-]*)*"
)
_ENV_PATH_MAX = 256


@dataclass(frozen=True)
class QueueEnvironment:
    """A Slurm environment asset: the space holding it, and its path in there."""

    space: str
    env_path: str

    @property
    def key(self) -> str:
        return f"{self.space}/{self.env_path}"


def _not_found() -> HTTPException:
    # One body for every reason, so a caller can't probe spaces or asset paths.
    return HTTPException(status.HTTP_404_NOT_FOUND, "environment not found")


def parse_queue_environment(space_name: str, env_path: str) -> QueueEnvironment:
    """Validate the URL's space and asset path; 404 when malformed."""
    env_path = env_path.strip("/")
    if (
        not space_name
        or len(env_path) > _ENV_PATH_MAX
        or not _ENV_PATH_RE.fullmatch(env_path)
    ):
        raise _not_found()
    return QueueEnvironment(space=space_name, env_path=env_path)


def _unavailable(environment: str, why: str) -> HTTPException:
    """503 with a generic body; the reason (URIs, loader errors) goes to the log."""
    logger.warning("[env-queue] %s unavailable: %s", environment, why)
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        f"queue for {environment!r} is not available on this server",
    )


@dataclass(frozen=True)
class QueueConnection:
    login_nodes: Tuple[str, ...]
    username: str
    key_secret_name: str


def connection_from_env_config(environment: str, env_config) -> QueueConnection:
    """Login details from a Slurm environment's ``cluster_ssh_configs.slurm``.

    The host block is the one named by the asset's ``cluster``, or the only one.
    ``IdentityKey`` must name a secret; ``HostName``/``User`` must be literals.
    A non-Slurm asset is a 404: this API has no such environment.
    """
    if (env_config.subtype or "").lower() != "slurm":
        logger.info("[env-queue] %s: subtype %r", environment, env_config.subtype)
        raise _not_found()
    cfg = env_config.config or {}
    hosts = (
        ClusterSshConfigs.model_validate(cfg.get("cluster_ssh_configs") or {}).slurm
        or []
    )
    alias = cfg.get("cluster")
    host = next((h for h in hosts if h.get("Host") == alias), None)
    if host is None and len(hosts) == 1:
        host = hosts[0]
    if host is None:
        raise _unavailable(environment, "no matching cluster_ssh_configs.slurm host")
    nodes = host.get("HostName") or []
    if isinstance(nodes, str):
        nodes = [nodes]
    conn = QueueConnection(
        login_nodes=tuple(str(n).strip() for n in nodes if str(n).strip()),
        username=str(host.get("User") or "").strip(),
        key_secret_name=str(host.get("IdentityKey") or "").strip(),
    )
    missing = [
        k
        for k, v in (
            ("HostName", conn.login_nodes),
            ("User", conn.username),
            ("IdentityKey", conn.key_secret_name),
        )
        if not v
    ]
    if missing:
        raise _unavailable(environment, f"host is missing {', '.join(missing)}")
    return conn


def space_environment_uri(row, env_path: str) -> str:
    """URI of ``environments/<env_path>`` in a stored space's definitions.

    A db-backed space gives a ``db://`` URI, read through ``db_space.local_uri``.
    """
    base = db_space.get_space_config_uri(row)
    if db_space.is_db_uri(base):
        return f"{base}#subdirectory=environments/{env_path}"
    uri = URI.get_uri(base)
    uri.append_path(f"environments/{env_path}")
    return str(uri)


def _load_asset(environment: str, record: QueueEnvironment):
    """Space lookup, URI derivation (maybe a GitHub probe) and asset load: blocking.

    An unknown space or a path with no environment.yaml is a 404; anything else
    that fails (git, GitHub, the database) is a 503.
    """
    row = get_admin_storage().space_storage.get_by_name(record.space)
    if row is None:
        raise _not_found()
    try:
        uri = space_environment_uri(row, record.env_path)
        with db_space.local_uri(uri, row) as local:
            env_config, _ = Environment.load_environment_config(local)
    except UnknownEnvironmentType as e:
        logger.info("[env-queue] %s: %s", environment, e)
        raise _not_found() from e
    except Exception as e:  # pylint: disable=broad-exception-caught
        raise _unavailable(environment, f"failed to load the asset: {e}") from e
    return env_config


@dataclass
class _ResolvedConnection:
    conn: QueueConnection
    key_material: str
    expires: float


_connections: Dict[str, _ResolvedConnection] = {}


async def resolve_connection(
    environment: str, record: QueueEnvironment
) -> Tuple[QueueConnection, str]:
    """Login details and service key for ``environment``, cached for a few minutes."""
    hit = _connections.get(environment)
    if hit and hit.expires > time.monotonic():
        return hit.conn, hit.key_material
    env_config = await asyncio.to_thread(_load_asset, environment, record)
    conn = connection_from_env_config(environment, env_config)
    try:
        key = await fetch_space_ssh_key(record.space, conn.key_secret_name)
    except HTTPException as e:
        # A missing secret is misconfiguration, not an unknown environment.
        raise _unavailable(environment, str(e.detail)) from e
    _bound(_connections, lambda v: v.expires <= time.monotonic())
    _connections[environment] = _ResolvedConnection(
        conn, key, time.monotonic() + _CONNECTION_TTL_S
    )
    return conn, key


# ---------------------------------------------------------- squeue contract

# The service account's jobs. %k must stay last: the comment writer keeps "|", so
# lines are split with maxsplit. %j is a slugified cluster name.
JOB_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("job_id", "%i"),
    ("user", "%u"),
    ("account", "%a"),
    ("partition", "%P"),
    ("qos", "%q"),
    ("state", "%T"),
    ("reason", "%r"),
    ("submit_time", "%V"),
    ("start_time", "%S"),
    ("time_limit", "%l"),
    ("time_used", "%M"),
    ("nodes", "%D"),
    ("cpus", "%C"),
    ("min_memory", "%m"),
    ("nodelist", "%N"),
    ("gres", "%b"),
    ("name", "%j"),
    ("comment", "%k"),
)
assert JOB_FIELDS[-1][0] == "comment", "comment must be the last squeue field"
_JOB_FORMAT = "|".join(code for _, code in JOB_FIELDS)
# Whole-queue load from fixed fields only (state, node count, tres-per-node).
SQUEUE_LOAD = "squeue -h -a -t RUNNING,PENDING -o '%T|%D|%b'"


def squeue_jobs_command(username: str) -> str:
    return (
        f"squeue -h -a -t RUNNING,PENDING -u {shlex.quote(username)} "
        f"-o '{_JOB_FORMAT}'"
    )


_STATES = frozenset({"RUNNING", "PENDING"})
_EMPTY = frozenset({"", "N/A", "(null)", "None", "n/a"})
_JOB_ID_RE = re.compile(r"^\d+(?:_(?:\d+|\[[^\]|]*\]))?$")
# %b is tres-per-node: "gres/gpu:8", "gres:gpu:8", "gres/gpu:h200:4", "gpu:h100:1".
_GPU_GRES_RE = re.compile(r"(?:^|,)(?:gres[/:])?gpu(?::([A-Za-z0-9_.-]+))?:(\d+)")
_GB_COMMENT_KEYS = frozenset(tagmeta.COMMENT_KEYS)


class GbJobTag(BaseModel):
    """granite.build ids parsed from a job's comment tag."""

    build_id: str
    gb_env: Optional[str] = None
    space: Optional[str] = None
    username: Optional[str] = None
    build_name: Optional[str] = None
    target: Optional[str] = None
    step_uri: Optional[str] = None
    target_id: Optional[str] = None
    step_id: Optional[str] = None


class QueueJob(BaseModel):
    job_id: str
    array_job_id: Optional[str] = None
    name: Optional[str] = None
    user: Optional[str] = None
    account: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    state: str
    reason: Optional[str] = None
    submit_time: Optional[str] = None
    start_time: Optional[str] = None
    time_limit: Optional[str] = None
    time_used: Optional[str] = None
    nodes: Optional[int] = None
    cpus: Optional[int] = None
    min_memory: Optional[str] = None
    nodelist: Optional[str] = None
    gres: Optional[str] = None
    gpus_per_node: Optional[int] = None
    gpus: Optional[int] = None
    gpu_model: Optional[str] = None
    comment: Optional[str] = None
    gb: Optional[GbJobTag] = None
    mine: bool = False
    redacted: bool = False


class QueueCounts(BaseModel):
    """Whole queue, every user."""

    running: int = 0
    pending: int = 0
    running_gpus: int = 0


class QueueResponse(BaseModel):
    environment: str
    scheduler: Literal["slurm"] = "slurm"
    as_of: str
    cache_age_s: float
    login_node: Optional[str] = None
    counts: QueueCounts
    jobs: List[QueueJob]
    truncated: bool = False
    unparsed_lines: int = 0


def _blank(value: str) -> Optional[str]:
    value = value.strip()
    return None if value in _EMPTY else value


def parse_gpu_gres(gres: Optional[str]) -> Tuple[Optional[int], Optional[str], bool]:
    """(GPUs per node, model, ok). ``ok`` is False for a gpu gres we can't read."""
    if not gres:
        return None, None, True
    matches = list(_GPU_GRES_RE.finditer(gres))
    if not matches:
        return None, None, "gpu" not in gres.lower()
    model = next((m.group(1) for m in matches if m.group(1)), None)
    return sum(int(m.group(2)) for m in matches), model, True


def parse_gb_comment(comment: Optional[str]) -> Optional[GbJobTag]:
    """Parse the ``key=value;…`` tag. None without a build_id or with a repeated key."""
    if not comment:
        return None
    values: Dict[str, str] = {}
    for part in comment.split(";"):
        key, sep, value = part.partition("=")
        if not (sep and key in _GB_COMMENT_KEYS and value):
            continue
        if key in values:
            return None
        values[key] = value
    return GbJobTag(**values) if "build_id" in values else None


def parse_job_line(line: str) -> Optional[QueueJob]:
    parts = line.rstrip("\n").split("|", len(JOB_FIELDS) - 1)
    if len(parts) != len(JOB_FIELDS):
        return None
    raw = {name: value for (name, _), value in zip(JOB_FIELDS, parts)}
    job_id, state = raw["job_id"].strip(), raw["state"].strip()
    nodes, cpus = raw["nodes"].strip(), raw["cpus"].strip()
    if not _JOB_ID_RE.match(job_id) or state not in _STATES:
        return None
    if not (nodes.isdigit() and cpus.isdigit()):
        return None
    gres = _blank(raw["gres"])
    per_node, model, ok = parse_gpu_gres(gres)
    if not ok:
        return None
    comment = _blank(raw["comment"])
    return QueueJob(
        job_id=job_id,
        array_job_id=job_id.split("_", 1)[0] if "_" in job_id else None,
        name=_blank(raw["name"]),
        user=_blank(raw["user"]),
        account=_blank(raw["account"]),
        partition=_blank(raw["partition"]),
        qos=_blank(raw["qos"]),
        state=state,
        reason=_blank(raw["reason"]),
        submit_time=_blank(raw["submit_time"]),
        start_time=_blank(raw["start_time"]),
        time_limit=_blank(raw["time_limit"]),
        time_used=_blank(raw["time_used"]),
        nodes=int(nodes),
        cpus=int(cpus),
        min_memory=_blank(raw["min_memory"]),
        nodelist=_blank(raw["nodelist"]),
        gres=gres,
        gpus_per_node=per_node,
        gpus=per_node * int(nodes) if per_node is not None else None,
        gpu_model=model,
        comment=comment,
        gb=parse_gb_comment(comment),
    )


def parse_jobs(stdout: str) -> Tuple[List[QueueJob], int]:
    """Parse the service account's jobs. Returns (jobs, unparsed lines)."""
    jobs: List[QueueJob] = []
    bad = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        job = parse_job_line(line)
        if job is None:
            bad += 1
        else:
            jobs.append(job)
    return jobs, bad


def parse_load(stdout: str) -> Tuple[QueueCounts, int]:
    """Whole-queue counts from ``%T|%D|%b`` lines. Returns (counts, unparsed lines)."""
    counts = QueueCounts()
    bad = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 3 or parts[0] not in _STATES or not parts[1].isdigit():
            bad += 1
            continue
        per_node, _, ok = parse_gpu_gres(_blank(parts[2]))
        if not ok:
            bad += 1
            continue
        if parts[0] == "RUNNING":
            counts.running += 1
            counts.running_gpus += (per_node or 0) * int(parts[1])
        else:
            counts.pending += 1
    return counts, bad


# -------------------------------------------------------------------- fetch


@dataclass
class _Snapshot:
    fetched_at: float
    as_of: str = ""
    login_node: Optional[str] = None
    jobs: List[QueueJob] = field(default_factory=list)
    counts: QueueCounts = field(default_factory=QueueCounts)
    unparsed: int = 0
    error: Optional[HTTPException] = None


_snapshots: Dict[str, _Snapshot] = {}
_locks: Dict[str, asyncio.Lock] = {}


class _SshFailure(Exception):
    """The login node couldn't run a command (as opposed to squeue failing)."""


async def _run(tunnel, environment: str, command: str) -> str:
    try:
        rc, stdout, stderr = await tunnel.run_remote(command, raise_on_error=False)
    except (SshTunnelError, TimeoutError) as e:
        raise _SshFailure(str(e)) from e
    if rc != 0:
        # Never degrade to an empty list: that would read as "no jobs".
        raise _unavailable(
            environment, f"squeue exit {rc}: {(stderr or '').strip()[:300]}"
        )
    return stdout or ""


async def _fetch(environment: str, record: QueueEnvironment) -> _Snapshot:
    conn, key = await resolve_connection(environment, record)
    opened = False
    try:
        async with open_login_tunnel(
            record.space,
            list(conn.login_nodes),
            conn.username,
            key_material=key,
            budget_s=GBSERVER_ENV_QUEUE_SSH_BUDGET_S,
            command_timeout_s=GBSERVER_ENV_QUEUE_COMMAND_TIMEOUT_S,
            log_tag="env-queue",
        ) as tunnel:
            opened = True
            jobs_out = await _run(
                tunnel, environment, squeue_jobs_command(conn.username)
            )
            load_out = await _run(tunnel, environment, SQUEUE_LOAD)
            login_node = getattr(tunnel, "host", None)
    except (_SshFailure, HTTPException) as e:
        if not opened or isinstance(e, _SshFailure):
            # A rotated key or changed login node must not stay cached.
            _connections.pop(environment, None)
        if isinstance(e, _SshFailure):
            raise _unavailable(environment, f"login node error: {e}") from e
        raise
    jobs, bad_jobs = parse_jobs(jobs_out)
    counts, bad_load = parse_load(load_out)
    unparsed = bad_jobs + bad_load
    if unparsed:
        logger.warning(
            "[env-queue] %s: %d unexpected squeue line(s)", environment, unparsed
        )
    return _Snapshot(
        fetched_at=time.monotonic(),
        as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        login_node=login_node,
        jobs=jobs,
        counts=counts,
        unparsed=unparsed,
    )


def _fresh(snap: _Snapshot) -> bool:
    return time.monotonic() - snap.fetched_at < GBSERVER_ENV_QUEUE_CACHE_TTL_S


async def get_queue_snapshot(environment: str, record: QueueEnvironment) -> _Snapshot:
    """One refresh per TTL per environment; failures are cached for the TTL too."""
    snap = _snapshots.get(environment)
    if snap is None or not _fresh(snap):
        if environment not in _locks:
            _bound(_locks, lambda lock: not lock.locked())
        async with _locks.setdefault(environment, asyncio.Lock()):
            snap = _snapshots.get(environment)
            if snap is None or not _fresh(snap):
                try:
                    snap = await _fetch(environment, record)
                except HTTPException as e:
                    snap = _Snapshot(fetched_at=time.monotonic(), error=e)
                _bound(_snapshots, lambda v: not _fresh(v))
                _snapshots[environment] = snap
    if snap.error is not None:
        raise snap.error
    return snap


# --------------------------------------------------------------------- route


async def caller_spaces(request: Request) -> Optional[Set[str]]:
    """The caller's space names; None means every space (super admin)."""
    scope = await asyncio.to_thread(scope_space_name_filter, request)
    if isinstance(scope, str):
        # Only a super admin gets the (empty) requested filter back.
        return None if scope == "" else {scope}
    if isinstance(scope, list):
        return set(scope)
    assert scope is NO_ACCESSIBLE_SPACE
    return set()


def tag_spaces(spaces: Optional[Set[str]]) -> Optional[Set[str]]:
    """Space names in the tag's spelling: the comment writer sanitizes values."""
    return None if spaces is None else {tagmeta.comment_value(n) for n in spaces}


def view_for(job: QueueJob, spaces: Optional[Set[str]], caller: str) -> QueueJob:
    """Full view if the job's tag space is one of the caller's; else scheduler fields."""
    tag = job.gb
    if tag is not None and (spaces is None or (tag.space and tag.space in spaces)):
        mine = bool(tag.username) and tag.username == tagmeta.comment_value(caller)
        return job.model_copy(update={"mine": mine})
    return job.model_copy(
        update={"name": None, "comment": None, "gb": None, "redacted": True}
    )


def _tag_matches(tag: Optional[GbJobTag], filters: Dict[str, str]) -> bool:
    if not filters:
        return True
    return tag is not None and all(getattr(tag, k) == v for k, v in filters.items())


_STATE_ARGS = {"running": "RUNNING", "pending": "PENDING"}
_TAG_MAX = tagmeta.MAX_COMMENT_VALUE_LEN


@queue_api.get("/{space_name}/{env_path:path}/queue", response_model=QueueResponse)
async def get_environment_queue(
    request: Request,
    space_name: str,
    env_path: str,
    state: str = Query(
        "running,pending", max_length=32, description="Subset of: running, pending"
    ),
    gb_env: Optional[str] = Query(None, max_length=_TAG_MAX),
    space: Optional[str] = Query(None, max_length=_TAG_MAX),
    username: Optional[str] = Query(
        None, max_length=_TAG_MAX, description="The tag's gbserver user"
    ),
    build_id: Optional[str] = Query(None, max_length=_TAG_MAX),
    mine: bool = Query(
        False,
        description="Only jobs whose tag's username is the caller's login, "
        "whichever deployment launched them",
    ),
) -> QueueResponse:
    """granite.build's running and pending jobs on the Slurm environment asset
    ``environments/<env_path>`` of ``space_name``. ``counts`` cover the whole
    queue; filters AND together and run on the caller's view. 404 unless the
    caller has access to the space and the asset is a Slurm environment; 503,
    never an empty list, when the queue can't be read."""
    record = parse_queue_environment(space_name, env_path)
    spaces = await caller_spaces(request)
    if spaces is not None and record.space not in spaces:
        raise _not_found()
    wanted = {s.strip().lower() for s in state.split(",") if s.strip()}
    if not wanted or not wanted <= _STATE_ARGS.keys():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"state must be a comma-separated subset of {sorted(_STATE_ARGS)}",
        )
    wanted_states = {_STATE_ARGS[s] for s in wanted}
    # Compare in the tag's spelling: the writer sanitizes values.
    tag_filters = {
        k: tagmeta.comment_value(v)
        for k, v in (
            ("gb_env", gb_env),
            ("space", space),
            ("username", username),
            ("build_id", build_id),
        )
        if v is not None
    }
    snap = await get_queue_snapshot(record.key, record)
    visible = tag_spaces(spaces)
    caller = request.state.data["user"].login

    views = (
        view_for(j, visible, caller) for j in snap.jobs if j.state in wanted_states
    )
    jobs = [
        j for j in views if (not mine or j.mine) and _tag_matches(j.gb, tag_filters)
    ]
    return QueueResponse(
        environment=record.key,
        as_of=snap.as_of,
        cache_age_s=round(time.monotonic() - snap.fetched_at, 1),
        login_node=snap.login_node,
        counts=snap.counts,
        jobs=jobs[:GBSERVER_ENV_QUEUE_MAX_JOBS],
        truncated=len(jobs) > GBSERVER_ENV_QUEUE_MAX_JOBS,
        unparsed_lines=snap.unparsed,
    )
