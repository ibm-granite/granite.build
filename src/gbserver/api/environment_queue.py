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

"""``GET /environments/{environment}/queue``: a Slurm queue read over SSH.

Environments come from ``GBSERVER_ENV_QUEUE_ENVIRONMENTS``. Login details are read
from the environment asset builds launch with, so builds and this endpoint follow
the same edits. The remote commands are constants; every filter runs in Python.

Each job's granite.build tag (``_skypilot_metadata``) is parsed into ``gb``. A job
is the caller's when the tag's ``username`` is their login; the tag is trusted,
since forging it only shows one's own job to someone else. On other users' tagged
jobs, ``comment``, ``name`` (the cluster name embeds the build id and target) and
every ``gb`` field but ``build_name`` are removed.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel

from gbserver.api.login_tunnel import fetch_space_ssh_key, open_login_tunnel
from gbserver.api.utils import is_super_admin
from gbserver.environment import _skypilot_metadata as tagmeta
from gbserver.environment.environment import Environment
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.types.constants import ENV_VAR_PREFIX, get_space_environment_uri
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


# ----------------------------------------------------------------- registry


@dataclass(frozen=True)
class QueueEnvironment:
    """A queue-capable environment: which space holds it, and its asset path."""

    space: str
    env_path: str


def _load_registry(raw: str) -> Dict[str, QueueEnvironment]:
    """Parse ``{"<name>": {"space": "...", "environment": "<path>"}}``; bad input → {}."""
    try:
        data = json.loads(raw or "{}")
        return {
            name: QueueEnvironment(space=v["space"], env_path=v["environment"])
            for name, v in data.items()
        }
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        logger.error("ignoring malformed %sENVIRONMENTS: %s", _P, e)
        return {}


QUEUE_ENVIRONMENT_REGISTRY = _load_registry(os.getenv(_P + "ENVIRONMENTS", ""))


def lookup_queue_environment(environment: str) -> QueueEnvironment:
    record = QUEUE_ENVIRONMENT_REGISTRY.get(environment)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "environment not found")
    return record


def _not_configured(environment: str, why: str) -> HTTPException:
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        f"queue for {environment!r} is not configured on this server ({why})",
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
    """
    if (env_config.subtype or "").lower() != "slurm":
        raise _not_configured(
            environment, f"subtype {env_config.subtype!r} is not slurm"
        )
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
        raise _not_configured(environment, "no matching cluster_ssh_configs.slurm host")
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
        raise _not_configured(environment, f"host is missing {', '.join(missing)}")
    return conn


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
    space = get_admin_storage().space_storage.get_by_name(record.space)
    if space is None:
        raise _not_configured(environment, f"space {record.space!r} not found")
    uri = get_space_environment_uri(space.git_repo_uri, record.env_path)
    if not uri:
        raise _not_configured(environment, "environment asset URI unavailable")
    try:
        env_config, _ = await asyncio.to_thread(
            Environment.load_environment_config, uri
        )
    except Exception as e:
        raise _not_configured(environment, f"failed to load {uri!r}: {e}") from e
    conn = connection_from_env_config(environment, env_config)
    try:
        key = await fetch_space_ssh_key(record.space, conn.key_secret_name)
    except HTTPException as e:
        # A missing secret is misconfiguration, not an unknown environment.
        raise _not_configured(environment, e.detail) from e
    _connections[environment] = _ResolvedConnection(
        conn, key, time.monotonic() + _CONNECTION_TTL_S
    )
    return conn, key


# ---------------------------------------------------------- squeue contract

# Fixed-format fields carry no user-settable text, so a strict field count is safe.
# The free-text fields (%j job name, %k comment) are each read in a separate call
# as "<job id>|<text>" and joined on job id, so a "|" in them can't shift columns.
FIXED_FIELDS: Tuple[Tuple[str, str], ...] = (
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
)
_SQUEUE = "squeue -h -a -t RUNNING,PENDING -o '{}'"
SQUEUE_FIXED = _SQUEUE.format("|".join(code for _, code in FIXED_FIELDS))
SQUEUE_NAME = _SQUEUE.format("%i|%j")
SQUEUE_COMMENT = _SQUEUE.format("%i|%k")

_STATES = frozenset({"RUNNING", "PENDING"})
_EMPTY = frozenset({"", "N/A", "(null)", "None", "n/a"})
_JOB_ID_RE = re.compile(r"^\d+(?:_(?:\d+|\[[^\]|]*\]))?$")
# %b is tres-per-node: "gres/gpu:8", "gres:gpu:8", "gres/gpu:h200:4", "gpu:h100:1".
_GPU_GRES_RE = re.compile(r"(?:^|,)(?:gres[/:])?gpu(?::([A-Za-z0-9_.-]+))?:(\d+)")
_GB_COMMENT_KEYS = frozenset(tagmeta.COMMENT_KEYS)


class GbJobTag(BaseModel):
    """granite.build ids from a job's comment; on a redacted tag only build_name."""

    build_id: Optional[str] = None
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
    running: int = 0
    pending: int = 0


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
    """Parse the ``key=value;…`` tag; None unless it carries a build_id.

    Values are delimiter-safe at write time, so a plain split is unambiguous.
    """
    if not comment:
        return None
    values: Dict[str, str] = {}
    for part in comment.split(";"):
        key, sep, value = part.partition("=")
        if sep and key in _GB_COMMENT_KEYS and value:
            values[key] = value
    return GbJobTag(**values) if "build_id" in values else None


def parse_fixed_line(line: str) -> Optional[QueueJob]:
    parts = line.rstrip("\n").split("|")
    if len(parts) != len(FIXED_FIELDS):
        return None
    raw = {name: value.strip() for (name, _), value in zip(FIXED_FIELDS, parts)}
    if not _JOB_ID_RE.match(raw["job_id"]) or raw["state"] not in _STATES:
        return None
    if not (raw["nodes"].isdigit() and raw["cpus"].isdigit()):
        return None
    gres = _blank(raw["gres"])
    per_node, model, ok = parse_gpu_gres(gres)
    if not ok:
        return None
    nodes = int(raw["nodes"])
    return QueueJob(
        job_id=raw["job_id"],
        array_job_id=raw["job_id"].split("_", 1)[0] if "_" in raw["job_id"] else None,
        user=_blank(raw["user"]),
        account=_blank(raw["account"]),
        partition=_blank(raw["partition"]),
        qos=_blank(raw["qos"]),
        state=raw["state"],
        reason=_blank(raw["reason"]),
        submit_time=_blank(raw["submit_time"]),
        start_time=_blank(raw["start_time"]),
        time_limit=_blank(raw["time_limit"]),
        time_used=_blank(raw["time_used"]),
        nodes=nodes,
        cpus=int(raw["cpus"]),
        min_memory=_blank(raw["min_memory"]),
        nodelist=_blank(raw["nodelist"]),
        gres=gres,
        gpus_per_node=per_node,
        gpus=per_node * nodes if per_node is not None else None,
        gpu_model=model,
    )


def parse_text_column(stdout: str) -> Tuple[Dict[str, Optional[str]], int]:
    """``<job id>|<text>`` lines → {job id: text}. A repeated id is unparsed for both."""
    out: Dict[str, Optional[str]] = {}
    dupes: set = set()
    bad = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        job_id, sep, text = line.partition("|")
        if not sep or not _JOB_ID_RE.match(job_id):
            bad += 1
            continue
        if job_id in out:
            dupes.add(job_id)
        out[job_id] = _blank(text)
    for job_id in dupes:
        out.pop(job_id, None)
    return out, bad + 2 * len(dupes)


def parse_queue(
    fixed_out: str, name_out: str, comment_out: str
) -> Tuple[List[QueueJob], int]:
    """Join the three squeue outputs on job id. Returns (jobs, unparsed lines)."""
    jobs: List[QueueJob] = []
    bad = 0
    for line in fixed_out.splitlines():
        if not line.strip():
            continue
        job = parse_fixed_line(line)
        if job is None:
            bad += 1
        else:
            jobs.append(job)
    names, bad_names = parse_text_column(name_out)
    comments, bad_comments = parse_text_column(comment_out)
    for job in jobs:
        job.name = names.get(job.job_id)
        job.comment = comments.get(job.job_id)
        job.gb = parse_gb_comment(job.comment)
    return jobs, bad + bad_names + bad_comments


# -------------------------------------------------------------------- fetch


@dataclass
class _Snapshot:
    fetched_at: float
    as_of: str = ""
    login_node: Optional[str] = None
    jobs: List[QueueJob] = field(default_factory=list)
    unparsed: int = 0
    error: Optional[HTTPException] = None


_snapshots: Dict[str, _Snapshot] = {}
_locks: Dict[str, asyncio.Lock] = {}


async def _run(tunnel, environment: str, command: str) -> str:
    try:
        rc, stdout, stderr = await tunnel.run_remote(command, raise_on_error=False)
    except (SshTunnelError, TimeoutError) as e:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "login node is slow or unreachable; please retry",
        ) from e
    if rc != 0:
        # Never degrade to an empty list: that would read as "no jobs".
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"squeue failed on {environment!r} (exit {rc}): "
            f"{(stderr or '').strip()[:300] or 'no error output'}",
        )
    return stdout or ""


async def _fetch(environment: str, record: QueueEnvironment) -> _Snapshot:
    conn, key = await resolve_connection(environment, record)
    async with open_login_tunnel(
        record.space,
        list(conn.login_nodes),
        conn.username,
        key_material=key,
        budget_s=GBSERVER_ENV_QUEUE_SSH_BUDGET_S,
        command_timeout_s=GBSERVER_ENV_QUEUE_COMMAND_TIMEOUT_S,
        log_tag="env-queue",
    ) as tunnel:
        outs = [
            await _run(tunnel, environment, c)
            for c in (SQUEUE_FIXED, SQUEUE_NAME, SQUEUE_COMMENT)
        ]
        jobs, unparsed = await asyncio.to_thread(parse_queue, *outs)
        if unparsed:
            logger.warning(
                "[env-queue] %s: %d unexpected squeue line(s)", environment, unparsed
            )
        return _Snapshot(
            fetched_at=time.monotonic(),
            as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            login_node=getattr(tunnel, "host", None),
            jobs=jobs,
            unparsed=unparsed,
        )


def _fresh(snap: _Snapshot) -> bool:
    return time.monotonic() - snap.fetched_at < GBSERVER_ENV_QUEUE_CACHE_TTL_S


async def get_queue_snapshot(environment: str, record: QueueEnvironment) -> _Snapshot:
    """One refresh per TTL per environment; failures are cached for the TTL too."""
    snap = _snapshots.get(environment)
    if snap is None or not _fresh(snap):
        async with _locks.setdefault(environment, asyncio.Lock()):
            snap = _snapshots.get(environment)
            if snap is None or not _fresh(snap):
                try:
                    snap = await _fetch(environment, record)
                except HTTPException as e:
                    snap = _Snapshot(fetched_at=time.monotonic(), error=e)
                _snapshots[environment] = snap
    if snap.error is not None:
        raise snap.error
    return snap


# --------------------------------------------------------------------- route


def view_for(job: QueueJob, caller: str, admin: bool) -> QueueJob:
    """The caller's view of a cached job: full if theirs, an admin's, or untagged."""
    tag_user = job.gb.username if job.gb else None
    mine = bool(tag_user) and tag_user == tagmeta.comment_value(caller)
    if mine or admin:
        return job.model_copy(update={"mine": mine})
    if job.gb is None:
        return job.model_copy(update={"comment": None, "redacted": True})
    return job.model_copy(
        update={
            "comment": None,
            "name": None,
            "gb": GbJobTag(build_name=job.gb.build_name),
            "redacted": True,
        }
    )


def _tag_matches(tag: Optional[GbJobTag], filters: Dict[str, str]) -> bool:
    if not filters:
        return True
    return tag is not None and all(getattr(tag, k) == v for k, v in filters.items())


_STATE_ARGS = {"running": "RUNNING", "pending": "PENDING"}
_TAG_MAX = tagmeta.MAX_COMMENT_VALUE_LEN


@queue_api.get("/{environment}/queue", response_model=QueueResponse)
async def get_environment_queue(
    request: Request,
    environment: str,
    state: str = Query(
        "running,pending", max_length=32, description="Subset of: running, pending"
    ),
    user: Optional[str] = Query(None, max_length=64, description="Slurm user"),
    gb_only: bool = Query(False, description="Only jobs with a granite.build tag"),
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
    """Running and pending jobs. ``counts`` cover the whole queue; filters AND
    together and tag filters run on the caller's (redacted) view. 503, never an
    empty list, when the queue can't be read."""
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
    record = lookup_queue_environment(environment)
    snap = await get_queue_snapshot(environment, record)
    caller = request.state.data["user"].login
    admin = is_super_admin(request)

    views = (
        view_for(j, caller, admin)
        for j in snap.jobs
        if j.state in wanted_states and (user is None or j.user == user)
    )
    jobs = [
        j
        for j in views
        if (not gb_only or j.gb is not None)
        and (not mine or j.mine)
        and _tag_matches(j.gb, tag_filters)
    ]
    return QueueResponse(
        environment=environment,
        as_of=snap.as_of,
        cache_age_s=round(time.monotonic() - snap.fetched_at, 1),
        login_node=snap.login_node,
        counts=QueueCounts(
            running=sum(j.state == "RUNNING" for j in snap.jobs),
            pending=sum(j.state == "PENDING" for j in snap.jobs),
        ),
        jobs=jobs[:GBSERVER_ENV_QUEUE_MAX_JOBS],
        truncated=len(jobs) > GBSERVER_ENV_QUEUE_MAX_JOBS,
        unparsed_lines=snap.unparsed,
    )
