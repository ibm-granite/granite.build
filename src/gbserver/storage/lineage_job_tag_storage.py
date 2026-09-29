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

"""Base storage interface and implementation for lineage job tags.

Queries go through the indexed ``tag`` column: a filter by tag is one ``IN``
query returning job ids, never a scan of the job table.
"""

from typing import Dict, Iterable, List, Optional, Set

from gbserver.storage.storage import BaseItemStorage, IItemStorage
from gbserver.storage.stored_lineage_job_tag import StoredLineageJobTag
from gbserver.types.constants import GB_LINEAGE_JOB_TAG_TABLE_NAME


class ILineageJobTagStorage(IItemStorage[StoredLineageJobTag]):
    """Interface for lineage job tag storage implementations."""

    def get_job_ids_by_tags(
        self, any_of: List[str], all_of: Optional[List[str]] = None
    ) -> Set[str]:
        """Return the jobs carrying any of ``any_of`` and all of ``all_of``."""
        raise NotImplementedError

    def get_tags(self, job_ids: List[str]) -> Dict[str, List[str]]:
        """Return each job's tags, keyed by ``job_id``."""
        raise NotImplementedError


class BaseLineageJobTagStorage(
    BaseItemStorage[StoredLineageJobTag], ILineageJobTagStorage
):
    """Base storage implementation for lineage job tags."""

    def __init__(self, **kwargs) -> None:
        kwargs["item_class"] = StoredLineageJobTag
        if kwargs.get("table_name") is None:
            kwargs["table_name"] = GB_LINEAGE_JOB_TAG_TABLE_NAME
        super().__init__(**kwargs)

    def _get_column_values(self, item: StoredLineageJobTag) -> dict:
        """Every field is a column; the tag row has nothing that is only carried."""
        return item.model_dump(include={"job_id", "tag", "recorded_at"})

    @classmethod
    def _get_sample_item(cls) -> StoredLineageJobTag:
        """Return a sample record used to derive the table schema."""
        return StoredLineageJobTag(
            job_id="sample-job",
            tag="k=v",
            recorded_at="2000-01-01T00:00:00.000000+00:00",
        )

    def get_job_ids_by_tags(
        self, any_of: List[str], all_of: Optional[List[str]] = None
    ) -> Set[str]:
        """Return the jobs matching a tag filter, W&B's ``$in`` + required shape.

        Args:
            any_of: a job must carry at least one of these. Empty means no ``$in``
                constraint, in which case ``all_of`` alone decides.
            all_of: a job must carry every one of these.

        Returns:
            The matching job ids. Empty when both lists are empty: an unfiltered
            request is not a tag query, and answering it would list every job.
        """
        wanted_any = _batchable(any_of)
        wanted_all = _batchable(all_of or [])
        if not wanted_any and not wanted_all:
            return set()

        # One IN over the indexed column covers both lists; the AND is then
        # resolved over the (small) set of matching tag rows, not over jobs.
        tags_by_job: Dict[str, Set[str]] = {}
        for row in self.get_by_where(
            {"tag": sorted(set(wanted_any) | set(wanted_all))}
        ):
            tags_by_job.setdefault(row.job_id, set()).add(row.tag)

        required = set(wanted_all)
        return {
            job_id
            for job_id, tags in tags_by_job.items()
            if required.issubset(tags) and (not wanted_any or tags & set(wanted_any))
        }

    def get_tags(self, job_ids: List[str]) -> Dict[str, List[str]]:
        """Return each job's tags, sorted, keyed by ``job_id``, in one query.

        A job with no tags is absent from the result rather than mapped to ``[]``.
        """
        wanted = _batchable(job_ids)
        if not wanted:
            return {}
        tags_by_job: Dict[str, Set[str]] = {}
        for row in self.get_by_where({"job_id": wanted}):
            tags_by_job.setdefault(row.job_id, set()).add(row.tag)
        return {job_id: sorted(tags) for job_id, tags in tags_by_job.items()}


def _batchable(values: Iterable[str]) -> List[str]:
    """Drop empty values and duplicates; sorted so the query is stable."""
    return sorted({value for value in values if value})
