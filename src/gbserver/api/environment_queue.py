#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Slurm environments read over SSH: ``…/queue``, ``…/capacity`` and ``…/history``.

Routes live under ``/environments/{space}/{env_path}/``: an environment is
addressed as builds address it, by a space and the asset's path under its
``environments/``. Login details are read from that asset, so nothing is
configured on the server. The caller must have access to the space, and the
asset must be a Slurm environment; anything else is a 404 that does not say which.
The remote commands are fixed and every filter runs in Python.

Trust assumption: only gbserver and its operators submit jobs as the asset's
``User``. Jobs (``squeue``, ``sacct``) are listed for that user only, so the free
text in the output (job name and comment) was written by gbserver, which slugifies
the cluster name and sanitizes every comment value. Other users' jobs contribute
only fixed fields: the queue-wide ``counts`` and ``…/capacity``'s node totals.

Each job's granite.build tag (``_skypilot_metadata``) is parsed into ``gb``. A job
is shown in full only when its tag's space is one of the caller's spaces (super
admins see all). Every other job, including one without a readable tag, keeps
scheduler fields only.
"""

import asyncio
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from gbcommon.uri.uri import URI
from gbserver.api.login_tunnel import fetch_space_ssh_key, open_login_tunnel
from gbserver.api.slurm_output import (
    HISTORY_DAYS,
    SINFO_NODES,
    SQUEUE_LOAD,
    GbJobTag,
    HistoryJob,
    ModelCapacity,
    PartitionCapacity,
    QueueCounts,
    QueueJob,
    parse_history,
    parse_jobs,
    parse_load,
    parse_sinfo_nodes,
    sacct_history_command,
    squeue_jobs_command,
    summarize_capacity,
)
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


# -------------------------------------------------------------------- fetch


@dataclass
class _Snapshot:
    fetched_at: float
    as_of: str = ""
    login_node: Optional[str] = None
    data: Any = None
    unparsed: int = 0
    error: Optional[HTTPException] = None


# Keyed by (environment, kind), e.g. ("ops/slurm/cluster-a", "history:7").
_snapshots: Dict[Tuple[str, str], _Snapshot] = {}
_locks: Dict[Tuple[str, str], asyncio.Lock] = {}


class _SshFailure(Exception):
    """The login node couldn't run a command (as opposed to the command failing)."""


async def _run(tunnel, environment: str, command: str) -> str:
    try:
        rc, stdout, stderr = await tunnel.run_remote(command, raise_on_error=False)
    except (SshTunnelError, TimeoutError) as e:
        raise _SshFailure(str(e)) from e
    if rc != 0:
        # Never degrade to an empty list: that would read as "nothing there".
        raise _unavailable(
            environment,
            f"{command.split()[0]} exit {rc}: {(stderr or '').strip()[:300]}",
        )
    return stdout or ""


async def run_on_login_node(
    environment: str,
    record: QueueEnvironment,
    commands_for: Callable[[QueueConnection], List[str]],
) -> Tuple[List[str], Optional[str]]:
    """Run fixed commands in one SSH session. Returns (outputs, login node)."""
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
            outs = [await _run(tunnel, environment, c) for c in commands_for(conn)]
            return outs, getattr(tunnel, "host", None)
    except (_SshFailure, HTTPException) as e:
        if not opened or isinstance(e, _SshFailure):
            # A rotated key or changed login node must not stay cached.
            _connections.pop(environment, None)
        if isinstance(e, _SshFailure):
            raise _unavailable(environment, f"login node error: {e}") from e
        raise


async def cached_snapshot(
    environment: str,
    kind: str,
    ttl_s: float,
    fetch: Callable[[], Awaitable[Tuple[Any, int, Optional[str]]]],
) -> _Snapshot:
    """One fetch per TTL per (environment, kind); failures are cached for the TTL too.

    ``fetch`` returns (data, unparsed lines, login node).
    """
    key = (environment, kind)

    def fresh(snap: Optional[_Snapshot]) -> bool:
        return snap is not None and time.monotonic() - snap.fetched_at < ttl_s

    snap = _snapshots.get(key)
    if not fresh(snap):
        if key not in _locks:
            _bound(_locks, lambda lock: not lock.locked())
        async with _locks.setdefault(key, asyncio.Lock()):
            snap = _snapshots.get(key)
            if not fresh(snap):
                try:
                    data, unparsed, node = await fetch()
                    if unparsed:
                        logger.warning(
                            "[env-queue] %s %s: %d unexpected line(s)",
                            environment,
                            kind,
                            unparsed,
                        )
                    snap = _Snapshot(
                        fetched_at=time.monotonic(),
                        as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        login_node=node,
                        data=data,
                        unparsed=unparsed,
                    )
                except HTTPException as e:
                    snap = _Snapshot(fetched_at=time.monotonic(), error=e)
                # Stale for its own kind's TTL; at worst a kind refetches early.
                _bound(_snapshots, lambda v: time.monotonic() - v.fetched_at >= ttl_s)
                _snapshots[key] = snap
    assert snap is not None
    if snap.error is not None:
        raise snap.error
    return snap


async def get_queue_snapshot(environment: str, record: QueueEnvironment) -> _Snapshot:
    """``data`` is (jobs, counts)."""

    async def fetch():
        (jobs_out, load_out), node = await run_on_login_node(
            environment,
            record,
            lambda conn: [squeue_jobs_command(conn.username), SQUEUE_LOAD],
        )
        jobs, bad_jobs = parse_jobs(jobs_out)
        counts, bad_load = parse_load(load_out)
        return (jobs, counts), bad_jobs + bad_load, node

    return await cached_snapshot(
        environment, "queue", GBSERVER_ENV_QUEUE_CACHE_TTL_S, fetch
    )


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


async def authorize(
    request: Request, space_name: str, env_path: str
) -> Tuple[QueueEnvironment, Optional[Set[str]]]:
    """The environment and the caller's spaces; 404 unless the caller has the space."""
    record = parse_queue_environment(space_name, env_path)
    spaces = await caller_spaces(request)
    if spaces is not None and record.space not in spaces:
        raise _not_found()
    return record, spaces


def view_for(job, spaces: Optional[Set[str]], caller: str):
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


def _tag_filters(gb_env, space, username, build_id) -> Dict[str, str]:
    """Tag filters in the tag's spelling: the writer sanitizes values."""
    return {
        k: tagmeta.comment_value(v)
        for k, v in (
            ("gb_env", gb_env),
            ("space", space),
            ("username", username),
            ("build_id", build_id),
        )
        if v is not None
    }


_STATE_ARGS = {"running": "RUNNING", "pending": "PENDING"}
_TAG_MAX = tagmeta.MAX_COMMENT_VALUE_LEN


_ROUTE = "/{space_name}/{env_path:path}"


@queue_api.get(_ROUTE + "/queue", response_model=QueueResponse)
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
    record, spaces = await authorize(request, space_name, env_path)
    tag_filters = _tag_filters(gb_env, space, username, build_id)
    wanted = {s.strip().lower() for s in state.split(",") if s.strip()}
    if not wanted or not wanted <= _STATE_ARGS.keys():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"state must be a comma-separated subset of {sorted(_STATE_ARGS)}",
        )
    wanted_states = {_STATE_ARGS[s] for s in wanted}
    snap = await get_queue_snapshot(record.key, record)
    all_jobs, counts = snap.data
    visible = tag_spaces(spaces)
    caller = request.state.data["user"].login

    views = (view_for(j, visible, caller) for j in all_jobs if j.state in wanted_states)
    jobs = [
        j for j in views if (not mine or j.mine) and _tag_matches(j.gb, tag_filters)
    ]
    return QueueResponse(
        environment=record.key,
        as_of=snap.as_of,
        cache_age_s=round(time.monotonic() - snap.fetched_at, 1),
        login_node=snap.login_node,
        counts=counts,
        jobs=jobs[:GBSERVER_ENV_QUEUE_MAX_JOBS],
        truncated=len(jobs) > GBSERVER_ENV_QUEUE_MAX_JOBS,
        unparsed_lines=snap.unparsed,
    )


class CapacityResponse(BaseModel):
    environment: str
    scheduler: Literal["slurm"] = "slurm"
    as_of: str
    cache_age_s: float
    login_node: Optional[str] = None
    models: List[ModelCapacity]
    partitions: List[PartitionCapacity]
    unparsed_lines: int = 0


@queue_api.get(_ROUTE + "/capacity", response_model=CapacityResponse)
async def get_environment_capacity(
    request: Request, space_name: str, env_path: str
) -> CapacityResponse:
    """Cluster-wide GPU capacity from ``sinfo``: aggregates only, no node names or
    jobs. 404 as for ``…/queue``."""
    record, _ = await authorize(request, space_name, env_path)
    environment = record.key

    async def fetch():
        (out,), node = await run_on_login_node(
            environment, record, lambda _c: [SINFO_NODES]
        )
        nodes, bad = parse_sinfo_nodes(out)
        return summarize_capacity(nodes), bad, node

    snap = await cached_snapshot(
        environment, "capacity", GBSERVER_ENV_QUEUE_CACHE_TTL_S, fetch
    )
    models, partitions = snap.data
    return CapacityResponse(
        environment=environment,
        as_of=snap.as_of,
        cache_age_s=round(time.monotonic() - snap.fetched_at, 1),
        login_node=snap.login_node,
        models=models,
        partitions=partitions,
        unparsed_lines=snap.unparsed,
    )


_HISTORY_TTL_S = 300.0


class HistoryTotals(BaseModel):
    """Over the jobs in this response (after visibility and filters)."""

    jobs: int = 0
    gpu_hours: float = 0.0
    gpu_hours_by_model: Dict[str, float] = Field(default_factory=dict)


class HistoryResponse(BaseModel):
    environment: str
    scheduler: Literal["slurm"] = "slurm"
    days: int
    as_of: str
    cache_age_s: float
    login_node: Optional[str] = None
    # False when no job carries a comment: the cluster may not store job comments
    # in accounting (AccountingStoreFlags=job_comment), so nothing can be shown.
    comments_recorded: bool
    totals: HistoryTotals
    jobs: List[HistoryJob]
    truncated: bool = False
    unparsed_lines: int = 0


@queue_api.get(_ROUTE + "/history", response_model=HistoryResponse)
async def get_environment_history(
    request: Request,
    space_name: str,
    env_path: str,
    days: int = Query(7, description="Window in days: 1, 7 or 30"),
    gb_env: Optional[str] = Query(None, max_length=_TAG_MAX),
    space: Optional[str] = Query(None, max_length=_TAG_MAX),
    username: Optional[str] = Query(None, max_length=_TAG_MAX),
    build_id: Optional[str] = Query(None, max_length=_TAG_MAX),
    mine: bool = Query(
        False, description="Only jobs whose tag's username is the caller's login"
    ),
) -> HistoryResponse:
    """granite.build's finished jobs from ``sacct``, with measured GPU-hours. Same
    visibility and 404s as ``…/queue``; ``totals`` cover the returned jobs."""
    record, spaces = await authorize(request, space_name, env_path)
    environment = record.key
    if days not in HISTORY_DAYS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"days must be one of {list(HISTORY_DAYS)}",
        )
    tag_filters = _tag_filters(gb_env, space, username, build_id)

    async def fetch():
        (out,), node = await run_on_login_node(
            environment,
            record,
            lambda conn: [sacct_history_command(conn.username, days)],
        )
        jobs, bad = parse_history(out)
        return jobs, bad, node

    snap = await cached_snapshot(environment, f"history:{days}", _HISTORY_TTL_S, fetch)
    visible = tag_spaces(spaces)
    caller = request.state.data["user"].login
    jobs = [
        j
        for j in (view_for(j, visible, caller) for j in snap.data)
        if (not mine or j.mine) and _tag_matches(j.gb, tag_filters)
    ]
    totals = HistoryTotals(jobs=len(jobs))
    for j in jobs:
        totals.gpu_hours += j.gpu_hours
        key = j.gpu_model or "unspecified"
        totals.gpu_hours_by_model[key] = (
            totals.gpu_hours_by_model.get(key, 0.0) + j.gpu_hours
        )
    totals.gpu_hours = round(totals.gpu_hours, 3)
    totals.gpu_hours_by_model = {
        k: round(v, 3) for k, v in totals.gpu_hours_by_model.items()
    }
    return HistoryResponse(
        environment=environment,
        days=days,
        as_of=snap.as_of,
        cache_age_s=round(time.monotonic() - snap.fetched_at, 1),
        login_node=snap.login_node,
        comments_recorded=any(j.comment for j in snap.data),
        totals=totals,
        jobs=jobs[:GBSERVER_ENV_QUEUE_MAX_JOBS],
        truncated=len(jobs) > GBSERVER_ENV_QUEUE_MAX_JOBS,
        unparsed_lines=snap.unparsed,
    )
