#
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

"""Write-time dedup for the lineage index: one copy per execution, whoever wrote it.

``gb_lineage_index`` is the source of truth, and more than one writer fills it: the
indexer (from ``gb_targets`` or a lineage store) and the Lakehouse importer. They can
describe the *same* execution -- a granite.build target run is pushed to Lakehouse
under ``job_id = targetrun.uuid``, the very id the indexer uses -- so a plain insert
that tolerates the unique collision would keep whichever copy landed first and throw
the other away, even when the second knew more (the namespace, the payloads, the
build id).

So every writer goes through these upserts instead:

- **job** (``gb_lineage_job``, unique ``job_id``) and **row** (``gb_lineage_index``,
  unique ``(job_id, input, output)``) are merged, not replaced: a value already
  stored is kept, a blank one is filled. Existing wins on a conflict, because a
  conflict between two sources about one execution has no right answer to pick and
  a stable one is better than a flapping one.
- The one exception is **provenance** in the attributes blob. The system that ran
  the job is the authority on it, so a granite.build write takes ``origin.system``
  over an imported copy's (see :data:`_SYSTEM_PRECEDENCE`), and the ids each source
  knew are unioned under ``origin.ids``.
- **tags** live in a row's ``attributes.job.tags`` map, so they merge like any
  other map: tags both sources know are kept once, new ones are added.

A terminal row -- ``TERMINAL -> X`` or ``X -> TERMINAL`` -- means "no recorded input"
(or output). It is superseded, and dropped, once the same job records a real edge on
that endpoint: a source that saw only the push of X must not leave X looking like a
creation next to a source that saw what X was made from.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.exc import IntegrityError

from gbserver.lineage.attributes import (
    JOB,
    JOB_COMPLETED_AT,
    JOB_STATUS,
    ORIGIN,
    ORIGIN_SYSTEM,
)
from gbserver.storage.lineage_job_storage import ILineageJobStorage
from gbserver.storage.lineage_row_storage import ILineageRowStorage
from gbserver.storage.stored_lineage_job import StoredLineageJob
from gbserver.storage.stored_lineage_row import (
    MAX_JOB_ID_LENGTH,
    TERMINAL,
    StoredLineageRow,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Outcomes, so a caller can count what a write did.
ADDED = "added"
UPDATED = "updated"
UNCHANGED = "unchanged"
SUPERSEDED = "superseded"
REJECTED = "rejected"

# Which producer's provenance wins when two describe one execution. Only
# granite.build is listed: it ran its own jobs, every other source is a copy.
_SYSTEM_PRECEDENCE = ("granite.build",)

# Job columns merged fill-blank. ``recorded_at`` is deliberately absent: it is when
# the index first wrote the record, and a merge is not a first write.
_JOB_COLUMNS = ("job_namespace", "space_name", "owner", "status", "started_at")

# Statuses a job moves on from. Fill-blank alone would freeze a record written
# while the job ran at RUNNING forever, so a finished status replaces one of these
# (column and ``job.status`` alike, with ``job.completed_at`` taken along). Matched
# case-insensitively: granite.build, W&B and Lakehouse spell them differently.
_IN_PROGRESS_STATUSES = frozenset(
    {"running", "pending", "queued", "started", "starting", "created", "submitted"}
)


def _moves_on(existing: Any, incoming: Any) -> bool:
    """Whether ``incoming`` is a finished status replacing an in-progress one."""
    return (
        str(existing or "").lower() in _IN_PROGRESS_STATUSES
        and not _is_blank(incoming)
        and str(incoming).lower() not in _IN_PROGRESS_STATUSES
    )


def _is_blank(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def merge_attributes(
    existing: Dict[str, Any], incoming: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep-merge two attributes blobs: fill blanks, keep what is already stored.

    Maps merge key by key; any other value is kept unless blank, except that a
    finished ``job.status`` replaces an in-progress one. ``origin.system``
    follows :func:`preferred_system`, and ``origin.ids`` is a union with existing
    winning per key -- the one place where both sides are expected to hold values.
    """
    merged = _fill_blanks(copy.deepcopy(existing or {}), incoming or {})
    old_job = (existing or {}).get(JOB) or {}
    new_job = (incoming or {}).get(JOB) or {}
    if _moves_on(old_job.get(JOB_STATUS), new_job.get(JOB_STATUS)):
        job = merged.setdefault(JOB, {})
        job[JOB_STATUS] = new_job[JOB_STATUS]
        if not _is_blank(new_job.get(JOB_COMPLETED_AT)):
            job[JOB_COMPLETED_AT] = new_job[JOB_COMPLETED_AT]
    old_origin = (existing or {}).get(ORIGIN) or {}
    new_origin = (incoming or {}).get(ORIGIN) or {}
    if old_origin or new_origin:
        origin = merged.setdefault(ORIGIN, {})
        system = preferred_system(
            old_origin.get(ORIGIN_SYSTEM, ""), new_origin.get(ORIGIN_SYSTEM, "")
        )
        if system:
            origin[ORIGIN_SYSTEM] = system
    return merged


def _fill_blanks(target: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in incoming.items():
        if _is_blank(value):
            continue
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _fill_blanks(current, value)
        elif _is_blank(current):
            target[key] = copy.deepcopy(value)
    return target


def preferred_system(existing: str, incoming: str) -> str:
    """The ``source_system`` to keep when two writers describe one execution."""
    if incoming in _SYSTEM_PRECEDENCE and existing not in _SYSTEM_PRECEDENCE:
        return incoming
    return existing or incoming


def merge_job(existing: StoredLineageJob, incoming: StoredLineageJob) -> Dict[str, Any]:
    """The fields to update on ``existing`` so it also carries ``incoming``.

    Empty when ``incoming`` adds nothing, so a re-run writes nothing.
    """
    fields: Dict[str, Any] = {}
    for column in _JOB_COLUMNS:
        if _is_blank(getattr(existing, column)) and not _is_blank(
            getattr(incoming, column)
        ):
            fields[column] = getattr(incoming, column)
    if _moves_on(existing.status, incoming.status):
        fields["status"] = incoming.status
    attributes = merge_attributes(existing.attributes, incoming.attributes)
    if attributes != existing.attributes:
        fields["attributes"] = attributes
    return fields


def upsert_job(storage: ILineageJobStorage, job: StoredLineageJob) -> str:
    """Add ``job``, or merge it into the record already stored under its ``job_id``."""
    if _job_id_too_long(job.job_id):
        return REJECTED
    existing = storage.get_job(job.job_id)
    if existing is None:
        try:
            storage.add(job)
            return ADDED
        except _LOST_RACE:
            # Lost a race with another writer: merge into what it wrote.
            existing = storage.get_job(job.job_id)
            if existing is None:
                raise
    fields = merge_job(existing, job)
    if not fields:
        return UNCHANGED
    storage.update_fields(existing.uuid, fields)
    return UPDATED


def upsert_row(storage: ILineageRowStorage, row: StoredLineageRow) -> str:
    """Add ``row``, or merge it into the stored row with the same identity.

    Also applies the terminal rule from the module docstring, in both directions: a
    terminal row is not added when the job already has a real edge on its endpoint,
    and a real edge removes the terminal rows it supersedes.

    The add and the delete of superseded terminals are one transaction on SQL
    storage (``add_replacing``). Where they are two writes, a crash between them
    leaves a stale terminal next to its real edge; the next write of that edge (an
    ``UNCHANGED`` re-run included) deletes it.
    """
    if _job_id_too_long(row.job_id):
        return REJECTED
    job_rows = storage.get_rows_by_job(row.job_id)
    stored = _same_identity(row, job_rows)
    if stored is None:
        if _superseded(row, job_rows):
            return SUPERSEDED
        stale = [other for other in job_rows if _superseded(other, [row])]
        # The terminal's attributes are a source's view of the same job; keep
        # them, merged before the add so the add is the only write of the row.
        for other in stale:
            row.attributes = merge_attributes(row.attributes, other.attributes)
        try:
            # One write where the backend allows: the row and the removal of the
            # terminals it supersedes land together or not at all.
            if stale:
                storage.add_replacing(row, [other.uuid for other in stale])
            else:
                storage.add(row)
        except _LOST_RACE:
            # Lost a race with another writer of the same (job, input, output):
            # merge into what it wrote, as upsert_job does.
            job_rows = storage.get_rows_by_job(row.job_id)
            stored = _same_identity(row, job_rows)
            if stored is None:
                raise
        else:
            return ADDED
    outcome = UNCHANGED
    attributes = merge_attributes(stored.attributes, row.attributes)
    if attributes != stored.attributes:
        storage.update_fields(stored.uuid, {"attributes": attributes})
        outcome = UPDATED
    _delete_stale(
        storage, [other for other in job_rows if _superseded(other, [stored])]
    )
    return outcome


# What ``add`` raises when another writer stored the same key first: the database's
# unique index (``IntegrityError``), or the storage's own uniqueness check
# (``ValueError``). Anything else -- an outage, a bad value -- propagates.
_LOST_RACE = (IntegrityError, ValueError)


def _same_identity(
    row: StoredLineageRow, job_rows: List[StoredLineageRow]
) -> Optional[StoredLineageRow]:
    for stored in job_rows:
        if stored.input == row.input and stored.output == row.output:
            return stored
    return None


def _delete_stale(storage: ILineageRowStorage, stale: List[StoredLineageRow]) -> None:
    if stale:
        storage.delete([stored.uuid for stored in stale])


def _job_id_too_long(job_id: str) -> bool:
    if len(job_id or "") <= MAX_JOB_ID_LENGTH:
        return False
    logger.warning(
        "Lineage job_id is %d chars, over the %d the column holds; not stored "
        "(job_id=%.80s...)",
        len(job_id),
        MAX_JOB_ID_LENGTH,
        job_id,
    )
    return True


def rows_to_add(rows: List[StoredLineageRow]) -> List[StoredLineageRow]:
    """The rows of jobs with nothing stored yet, deduplicated for one bulk add.

    The same rules :func:`upsert_row` applies one row at a time, applied within
    the batch: one row per ``(job_id, input, output)`` (attributes merged), and no
    terminal row that a real edge of the same job supersedes.
    """
    by_key: Dict[Tuple[str, str, str], StoredLineageRow] = {}
    for row in rows:
        if _job_id_too_long(row.job_id):
            continue
        key = (row.job_id, row.input, row.output)
        stored = by_key.get(key)
        if stored is None:
            by_key[key] = row
        else:
            stored.attributes = merge_attributes(stored.attributes, row.attributes)
    by_job: Dict[str, List[StoredLineageRow]] = {}
    for row in by_key.values():
        by_job.setdefault(row.job_id, []).append(row)
    return [
        row
        for job_rows in by_job.values()
        for row in job_rows
        if not _superseded(row, [other for other in job_rows if other is not row])
    ]


def _superseded(row: StoredLineageRow, others: List[StoredLineageRow]) -> bool:
    """Whether terminal ``row`` is covered by a real edge in ``others``."""
    if row.input == TERMINAL and row.output != TERMINAL:
        return any(o.output == row.output and o.input != TERMINAL for o in others)
    if row.output == TERMINAL and row.input != TERMINAL:
        return any(o.input == row.input and o.output != TERMINAL for o in others)
    return False
