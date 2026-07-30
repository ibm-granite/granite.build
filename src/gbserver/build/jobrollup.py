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

"""Aggregate a build retry chain into a single job view.

A build-level retry creates a new StoredBuild linked to the original (see
docs/builds/build-retry.md). Each member keeps its own per-attempt status, so no
single member answers "did the job specification complete?" — the original stays
FAILED even when a later attempt finished every remaining target.

These functions derive that answer on read. They are pure: the caller fetches the
chain members and their target runs and passes them in root-first order. Nothing
here touches storage, so the whole rule is unit-testable without a database, and
both the build-status and lineage endpoints can share it rather than growing two
copies that drift apart.

Every function here is total — there are no raising paths. The job summary is
additive to endpoints that already work without it, so a malformed chain must
degrade to a best-effort summary rather than fail the request.
"""

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

# Chain members paired with their target runs, ordered root first — exactly what
# get_retry_chain_members() returns, zipped with each member's runs.
ChainInput = Sequence[Tuple[StoredBuild, List[StoredTargetRun]]]


class TargetOutcome(BaseModel):
    """One spec target's best outcome across the whole chain."""

    name: str
    # None means no run exists for this spec target: it was never dispatched,
    # e.g. an upstream dependency failed. Modelled as None rather than a new
    # Status member so the shared Status vocabulary stays untouched.
    status: Optional[Status] = None
    build_id: str = ""
    target_run_id: str = ""
    reused_from_target_run_id: str = ""
    attempt: int = 0


class JobTargetCounts(BaseModel):
    total: int = 0
    succeeded: int = 0
    # Has a run that did not succeed. While the job is RUNNING this includes
    # attempts still in flight.
    failed: int = 0
    not_run: int = 0


class JobSummary(BaseModel):
    job_id: str = ""
    status: Status = Status.PENDING
    attempts: int = 0
    build_ids: List[str] = Field(default_factory=list)
    targets: List[TargetOutcome] = Field(default_factory=list)
    counts: JobTargetCounts = Field(default_factory=JobTargetCounts)


class WinningRun(NamedTuple):
    """The run that actually produced a target's artifacts, plus how we got there."""

    build: StoredBuild
    run: StoredTargetRun
    reused_from_target_run_id: str = ""


def _run_sort_key(run: StoredTargetRun) -> Tuple[int, str, str]:
    """Order runs deterministically within one attempt.

    A skipped run has no started_at, so sort missing starts first instead of
    comparing None against a datetime. The UUID breaks remaining ties, keeping
    API responses and test assertions stable.
    """
    return (0 if run.started_at is None else 1, str(run.started_at or ""), run.uuid)


def _group_runs_by_name(
    chain: ChainInput,
) -> Dict[str, List[Tuple[int, StoredBuild, StoredTargetRun]]]:
    """Group every run in the chain by target name, earliest attempt first."""
    grouped: Dict[str, List[Tuple[int, StoredBuild, StoredTargetRun]]] = {}
    for index, (build, runs) in enumerate(chain):
        for run in sorted(runs, key=_run_sort_key):
            grouped.setdefault(run.name, []).append((index, build, run))
    return grouped


def resolve_spec_targets(root: StoredBuild, chain: ChainInput) -> List[str]:
    """The job's target names, in a stable order.

    ``root.targets`` records the requested target subset and is authoritative
    when set: it is the only way to know a target was never dispatched, since
    such a target has no StoredTargetRun at all. When it is unset the names are
    derived from observed runs, and never-dispatched targets are invisible —
    an accepted limitation that keeps build-config parsing off the read path.
    """
    if root.targets:
        return list(dict.fromkeys(root.targets))
    return sorted(_group_runs_by_name(chain))
