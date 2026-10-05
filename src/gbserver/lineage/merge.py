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
- The one exception is **provenance**. The system that ran the job is the authority
  on it, so a granite.build write takes ``source_system`` over an imported copy's
  (see :data:`_SYSTEM_PRECEDENCE`), and the ids each source knew are unioned under
  ``origin.ids``.
- **tags** live in a row's ``attributes.job.tags`` map, so they merge like any
  other map: tags both sources know are kept once, new ones are added.

A terminal row -- ``TERMINAL -> X`` or ``X -> TERMINAL`` -- means "no recorded input"
(or output). It is superseded, and dropped, once the same job records a real edge on
that endpoint: a source that saw only the push of X must not leave X looking like a
creation next to a source that saw what X was made from.
"""

import copy
from typing import Any, Dict, List, Optional

from gbserver.lineage.attributes import ORIGIN, ORIGIN_IDS, ORIGIN_SYSTEM
from gbserver.storage.lineage_job_storage import ILineageJobStorage
from gbserver.storage.lineage_row_storage import ILineageRowStorage
from gbserver.storage.stored_lineage_job import StoredLineageJob
from gbserver.storage.stored_lineage_row import TERMINAL, StoredLineageRow
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Outcomes, so a caller can count what a write did.
ADDED = "added"
UPDATED = "updated"
UNCHANGED = "unchanged"
SUPERSEDED = "superseded"

# Which producer's provenance wins when two describe one execution. Only
# granite.build is listed: it ran its own jobs, every other source is a copy.
_SYSTEM_PRECEDENCE = ("granite.build",)

# Job columns merged fill-blank. ``recorded_at`` is deliberately absent: it is when
# the index first wrote the record, and a merge is not a first write.
_JOB_COLUMNS = ("job_namespace", "space_name", "owner", "status", "started_at")


def _is_blank(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def merge_attributes(
    existing: Dict[str, Any], incoming: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep-merge two attributes blobs: fill blanks, keep what is already stored.

    Maps merge key by key; any other value is kept unless blank. ``origin.system``
    follows :func:`preferred_system`, and ``origin.ids`` is a union with existing
    winning per key -- the one place where both sides are expected to hold values.
    """
    merged = _fill_blanks(copy.deepcopy(existing or {}), incoming or {})
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
    system = preferred_system(existing.source_system, incoming.source_system)
    if system != existing.source_system:
        fields["source_system"] = system
    attributes = merge_attributes(existing.attributes, incoming.attributes)
    if attributes != existing.attributes:
        fields["attributes"] = attributes
    return fields


def upsert_job(storage: ILineageJobStorage, job: StoredLineageJob) -> str:
    """Add ``job``, or merge it into the record already stored under its ``job_id``."""
    existing = storage.get_job(job.job_id)
    if existing is None:
        try:
            storage.add(job)
            return ADDED
        except Exception:
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
    """
    job_rows = storage.get_rows_by_job(row.job_id)
    for stored in job_rows:
        if stored.input == row.input and stored.output == row.output:
            attributes = merge_attributes(stored.attributes, row.attributes)
            if attributes == stored.attributes:
                return UNCHANGED
            storage.update_fields(stored.uuid, {"attributes": attributes})
            return UPDATED

    if _superseded(row, job_rows):
        return SUPERSEDED

    stale = [stored for stored in job_rows if _superseded(stored, [row])]
    storage.add(row)
    if stale:
        # The terminal's attributes are a source's view of the same job; keep them.
        attributes = row.attributes
        for stored in stale:
            attributes = merge_attributes(attributes, stored.attributes)
        if attributes != row.attributes:
            added = _find(storage, row)
            if added is not None:
                storage.update_fields(added.uuid, {"attributes": attributes})
        storage.delete([stored.uuid for stored in stale])
    return ADDED


def prune_superseded(storage: ILineageRowStorage, job_id: str) -> int:
    """Drop the terminal rows of one job that a real edge supersedes; return how many.

    For a bulk writer that adds rows in batches -- and so bypasses the check
    :func:`upsert_row` makes per row -- and runs this once per job it wrote a
    terminal for. The superseded rows' attributes are merged into the edge that
    replaces them, so nothing a source knew is lost with the row.
    """
    rows = storage.get_rows_by_job(job_id)
    stale = [row for row in rows if _superseded(row, rows)]
    if not stale:
        return 0
    for edge in rows:
        if edge in stale:
            continue
        covering = [row for row in stale if _superseded(row, [edge])]
        if not covering:
            continue
        attributes = edge.attributes
        for row in covering:
            attributes = merge_attributes(attributes, row.attributes)
        if attributes != edge.attributes:
            storage.update_fields(edge.uuid, {"attributes": attributes})
    storage.delete([row.uuid for row in stale])
    return len(stale)


def _superseded(row: StoredLineageRow, others: List[StoredLineageRow]) -> bool:
    """Whether terminal ``row`` is covered by a real edge in ``others``."""
    if row.input == TERMINAL and row.output != TERMINAL:
        return any(o.output == row.output and o.input != TERMINAL for o in others)
    if row.output == TERMINAL and row.input != TERMINAL:
        return any(o.input == row.input and o.output != TERMINAL for o in others)
    return False


def _find(
    storage: ILineageRowStorage, row: StoredLineageRow
) -> Optional[StoredLineageRow]:
    for stored in storage.get_rows_by_job(row.job_id):
        if stored.input == row.input and stored.output == row.output:
            return stored
    return None
