#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""What Slurm prints, and how to read it: the fixed ``squeue``, ``sinfo`` and
``sacct`` commands the environment endpoints run, and their parsers.

Free text appears only in the service account's own jobs (job name and comment),
which gbserver writes; see ``environment_queue`` for the trust assumption.
"""

import re
import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from gbserver.environment import _skypilot_metadata as tagmeta

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
# Whole-queue load from fixed fields only (state, node count, tres-per-node, partition).
SQUEUE_LOAD = "squeue -h -a -t RUNNING,PENDING -o '%T|%D|%b|%P'"


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
    running_gpus_by_partition: Dict[str, int] = Field(default_factory=dict)


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
    """Whole-queue counts from ``%T|%D|%b|%P`` lines. Returns (counts, unparsed)."""
    counts = QueueCounts()
    bad = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 4 or parts[0] not in _STATES or not parts[1].isdigit():
            bad += 1
            continue
        per_node, _, ok = parse_gpu_gres(_blank(parts[2]))
        if not ok:
            bad += 1
            continue
        if parts[0] == "RUNNING":
            gpus = (per_node or 0) * int(parts[1])
            counts.running += 1
            counts.running_gpus += gpus
            part = parts[3].strip()
            counts.running_gpus_by_partition[part] = (
                counts.running_gpus_by_partition.get(part, 0) + gpus
            )
        else:
            counts.pending += 1
    return counts, bad


# ----------------------------------------------------------------- capacity

# One line per (node, partition). Fixed-width columns: none of these values
# contain spaces, and the widths are far beyond real node, gres and state names.
SINFO_NODES = (
    "sinfo -N -h -O 'NodeHost:128,StateLong:64,Gres:256,GresUsed:256,Partition:64'"
)
# Node states that can't take work; a node is also unusable with any of these
# state suffixes (not responding, powered/powering down or up, maintenance, reboot).
_UNUSABLE_STATES = (
    "drain",
    "down",
    "fail",
    "maint",
    "reboot",
    "power",
    "unk",
    "inval",
    "future",
    "reserved",
)
_UNUSABLE_FLAGS = frozenset("*~#!%$@^")


@dataclass
class _Node:
    state: str
    usable: bool
    gpus: int
    gpus_used: int
    model: Optional[str]
    partitions: Set[str] = field(default_factory=set)


def node_is_usable(state: str) -> bool:
    flags = set(state) & _UNUSABLE_FLAGS
    base = state.rstrip("*~#!%$@^-+").lower()
    return not flags and not base.startswith(_UNUSABLE_STATES)


def parse_sinfo_nodes(stdout: str) -> Tuple[Dict[str, _Node], int]:
    """Nodes from ``sinfo -N -O``, deduplicated across partitions."""
    nodes: Dict[str, _Node] = {}
    bad = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            bad += 1
            continue
        host, state, gres, used, partition = parts
        total, model, ok = parse_gpu_gres(_blank(gres))
        n_used, _, ok_used = parse_gpu_gres(_blank(used))
        if not (ok and ok_used):
            bad += 1
            continue
        node = nodes.get(host)
        if node is None:
            node = nodes[host] = _Node(
                state=state,
                usable=node_is_usable(state),
                gpus=total or 0,
                gpus_used=n_used or 0,
                model=model,
            )
        node.partitions.add(partition.rstrip("*"))  # "*" marks the default partition
    return nodes, bad


class ModelCapacity(BaseModel):
    gpu_model: Optional[str]
    nodes: int = 0
    usable_nodes: int = 0
    gpus: int = 0
    gpus_usable: int = 0
    gpus_used: int = 0
    gpus_free: int = 0


class PartitionCapacity(BaseModel):
    """Partitions can share nodes, so these totals overlap across partitions."""

    partition: str
    nodes: int = 0
    usable_nodes: int = 0
    gpus_free: int = 0


def summarize_capacity(
    nodes: Dict[str, _Node],
) -> Tuple[List[ModelCapacity], List[PartitionCapacity]]:
    """GPUs on usable nodes, by GPU model and by partition. Free = usable − used."""
    models: Dict[Optional[str], ModelCapacity] = {}
    parts: Dict[str, PartitionCapacity] = {}
    for node in nodes.values():
        free = max(node.gpus - node.gpus_used, 0) if node.usable else 0
        if node.gpus:  # CPU-only nodes count toward partitions, not GPU models
            m = models.setdefault(node.model, ModelCapacity(gpu_model=node.model))
            m.nodes += 1
            m.gpus += node.gpus
            if node.usable:
                m.usable_nodes += 1
                m.gpus_usable += node.gpus
                m.gpus_used += node.gpus_used
                m.gpus_free += free
        for name in node.partitions:
            p = parts.setdefault(name, PartitionCapacity(partition=name))
            p.nodes += 1
            if node.usable:
                p.usable_nodes += 1
                p.gpus_free += free
    return (
        sorted(models.values(), key=lambda m: (m.gpu_model is None, m.gpu_model or "")),
        sorted(parts.values(), key=lambda p: p.partition),
    )


# ------------------------------------------------------------------ history

# Finished jobs change slowly and sacct is heavier than squeue.
HISTORY_DAYS = (1, 7, 30)
HISTORY_FIELDS: Tuple[str, ...] = (
    "JobID",
    "JobName",
    "Partition",
    "QOS",
    "State",
    "Submit",
    "Start",
    "End",
    "ElapsedRaw",
    "NNodes",
    "AllocTRES",
    "ExitCode",
    "Comment",
)
assert HISTORY_FIELDS[-1] == "Comment", "Comment must be the last sacct field"
_FINISHED = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "TIMEOUT",
        "CANCELLED",
        "PREEMPTED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "BOOT_FAIL",
        "DEADLINE",
        "REQUEUED",
    }
)
# AllocTRES totals across nodes: "gres/gpu=8" and/or "gres/gpu:h200=8".
_ALLOC_GPU_RE = re.compile(r"(?:^|,)gres/gpu(?::([^=,]+))?=(\d+)")


def sacct_history_command(username: str, days: int) -> str:
    """``days`` is one of HISTORY_DAYS, so the command stays fixed per window."""
    assert days in HISTORY_DAYS
    return (
        f"sacct -X -n -P -u {shlex.quote(username)} -S now-{days}days -E now "
        f"--format={','.join(HISTORY_FIELDS)}"
    )


class HistoryJob(BaseModel):
    job_id: str
    array_job_id: Optional[str] = None
    name: Optional[str] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    state: str
    submit_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    elapsed_s: int = 0
    nodes: Optional[int] = None
    gpus: Optional[int] = None
    gpu_model: Optional[str] = None
    gpu_hours: float = 0.0
    exit_code: Optional[str] = None
    comment: Optional[str] = None
    gb: Optional[GbJobTag] = None
    mine: bool = False
    redacted: bool = False


def parse_alloc_gpus(tres: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    if not tres:
        return None, None
    total, model, by_model = None, None, 0
    for m in _ALLOC_GPU_RE.finditer(tres):
        if m.group(1):
            model = model or m.group(1)
            by_model += int(m.group(2))
        else:
            total = int(m.group(2))
    if total is None and by_model:
        total = by_model
    return total, model


def parse_history_line(line: str) -> Optional[HistoryJob]:
    parts = line.rstrip("\n").split("|", len(HISTORY_FIELDS) - 1)
    if len(parts) != len(HISTORY_FIELDS):
        return None
    raw = dict(zip(HISTORY_FIELDS, (p.strip() for p in parts[:-1]))) | {
        "Comment": parts[-1]
    }
    state = raw["State"].split(" ", 1)[0]  # "CANCELLED by 1234"
    if not _JOB_ID_RE.match(raw["JobID"]) or not raw["ElapsedRaw"].isdigit():
        return None
    if state not in _FINISHED:
        return None
    gpus, model = parse_alloc_gpus(_blank(raw["AllocTRES"]))
    elapsed = int(raw["ElapsedRaw"])
    comment = _blank(raw["Comment"])
    return HistoryJob(
        job_id=raw["JobID"],
        array_job_id=raw["JobID"].split("_", 1)[0] if "_" in raw["JobID"] else None,
        name=_blank(raw["JobName"]),
        partition=_blank(raw["Partition"]),
        qos=_blank(raw["QOS"]),
        state=state,
        submit_time=_blank(raw["Submit"]),
        start_time=_blank(raw["Start"]),
        end_time=_blank(raw["End"]),
        elapsed_s=elapsed,
        nodes=int(raw["NNodes"]) if raw["NNodes"].isdigit() else None,
        gpus=gpus,
        gpu_model=model,
        gpu_hours=round((gpus or 0) * elapsed / 3600, 3),
        exit_code=_blank(raw["ExitCode"]),
        comment=comment,
        gb=parse_gb_comment(comment),
    )


def parse_history(stdout: str) -> Tuple[List[HistoryJob], int]:
    """Finished jobs. Lines for jobs still running or pending are skipped, not
    counted as unparsed; anything else unexpected is."""
    jobs: List[HistoryJob] = []
    bad = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        job = parse_history_line(line)
        if job is not None:
            jobs.append(job)
            continue
        state = line.split("|")[4].split(" ", 1)[0] if line.count("|") >= 4 else ""
        if state not in {"RUNNING", "PENDING", "SUSPENDED"}:
            bad += 1
    return jobs, bad
