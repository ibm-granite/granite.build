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

"""
Abstract interface for lineage storage and singleton accessor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun


class ILineageStore(ABC):
    """Abstract interface for lineage storage backends."""

    @property
    def records_centralized_lineage(self) -> bool:
        """Whether this backend records lineage to a centralized store.

        True for real backends (e.g. WandB); False for the no-op backend used
        when lineage is disabled (standalone / GBSERVER_LINEAGE_PROVIDER=none),
        which records nothing. Callers/tests can use this to skip assertions that
        only make sense for a recording store. Defaults to True.
        """
        return True

    @abstractmethod
    def add_jobstats_for_build(
        self, storage: SingletonAdminStorage, build_id: str
    ) -> None: ...

    @abstractmethod
    def add_jobstats_for_build_target(
        self, storage: SingletonAdminStorage, build_id: str, target_id: str
    ) -> None: ...

    @abstractmethod
    def add_jobstats_for_original_artifact(
        self, artifact: ArtifactRegistration, sources: list[ArtifactRegistration]
    ) -> None: ...

    @abstractmethod
    def create_jobstats_for_target(
        self,
        storage: SingletonAdminStorage,
        targetrun: StoredTargetRun,
        build: Optional[StoredBuild] = None,
    ) -> Tuple: ...

    @abstractmethod
    def create_jobstats_for_original_artifact(
        self, artifact: ArtifactRegistration, sources: list[ArtifactRegistration]
    ): ...

    @abstractmethod
    def count_release_ids(
        self, release_id: str, target_id: Optional[str] = None
    ) -> int: ...

    @abstractmethod
    def does_release_id_exist(
        self, release_id: str, expected_count: int, target_id: Optional[str] = None
    ) -> bool: ...

    def filter_unrecorded(
        self,
        target_ids: set[str],
        expected_counts: Optional[dict[str, int]] = None,
        on_query_error: Optional[Callable[[Exception], None]] = None,
    ) -> set[str]:
        """Return the subset of ``target_ids`` not yet recorded in *this* store.

        Each sink owns its own record of what it has already recorded, so the
        shared admin DB can feed W&B and other sinks independently — no per-sink
        "recorded" bit lives in the admin schema. The reconciler passes the
        candidate target uuids selected by the time watermark and records only
        the returned subset, so a sink never re-emits a target it already has
        while a *different* sink can still record the same target.

        Bounded by ``target_ids`` (the candidates from this scan), so it never
        scans the sink's entire history. This is an efficiency optimization only
        — idempotent recording preserves correctness regardless — so
        implementations must never raise; on failure they return ``target_ids``
        unchanged, degrading to re-recording the candidates (harmless).

        That degradation is *safe* but not *free*, and it is indistinguishable
        from a genuine "none of these are recorded" answer. A caller that swept a
        wide candidate range and hit a sink timeout would re-record the entire
        range — turning one failed query into a write storm against the sink that
        just failed. Worse, a caller deciding something irreversible (e.g.
        retiring a build from a tracking set once its lineage is confirmed) must
        not read a failure as a verdict at all. ``on_query_error`` exists for
        that: when provided, implementations invoke it with the exception before
        returning the fail-open set, so the caller can tell "the sink says these
        are unrecorded" from "the sink did not answer". Callers that do not care
        omit it and keep the historical behavior.

        ``expected_counts`` maps a target uuid to the number of records the sink
        should hold for a *fully* recorded target (one W&B run per output
        artifact, or 1 for an output-less target). A single target can emit
        several records, and if a prior scan crashed part-way through it, the
        sink holds *some* of them — a "recorded" check that only tests presence
        would wrongly mark such a target complete and never re-record the
        missing records, leaving a permanent partial-lineage gap. So a target is
        treated as recorded only when its held-record count meets or exceeds its
        expected count. When ``expected_counts`` is ``None`` or lacks a target's
        key (e.g. the reconciler could not derive it), that target falls back to
        the presence check (recorded once >=1 record exists) — the pre-count
        behavior, which stays correct for older records that predate this check.

        Defaults to returning ``target_ids`` unchanged for backends that record
        no centralized lineage (e.g. the no-op store); such a store's recording
        leaf is itself a no-op, so the returned set is never actually recorded.
        """
        return target_ids


__JOBSTATS_STORAGE: Optional[ILineageStore] = None


def reset_lineage_store() -> None:
    """Reset the singleton so the next call to get_lineage_store() re-creates it."""
    global __JOBSTATS_STORAGE
    __JOBSTATS_STORAGE = None


def _resolve_lineage_provider() -> str:
    """Resolve the lineage provider at call time.

    GBSERVER_LINEAGE_PROVIDER wins if set; otherwise the default is "none" in
    standalone mode (no wandb dependency) and "wandb" elsewhere. Resolved
    dynamically — rather than read from a cached constant or written to os.environ
    at import — so standalone mode established at runtime is honored and the
    standalone default never leaks into the process environment.
    """
    import os

    from gbcommon.types.gbenvconfig import is_standalone
    from gbserver.types.constants import ENV_VAR_PREFIX

    default = "none" if is_standalone() else "wandb"
    return os.getenv(ENV_VAR_PREFIX + "_LINEAGE_PROVIDER", default)


def get_lineage_store() -> ILineageStore:
    """Get a singleton instance of the lineage storage backend."""
    global __JOBSTATS_STORAGE
    if __JOBSTATS_STORAGE is None:
        if _resolve_lineage_provider() == "none":
            from gbserver.lineage.noop_jobstats import NoopLineageStore

            __JOBSTATS_STORAGE = NoopLineageStore()
        else:
            from gbserver.lineage.wandb_jobstats import WandBLineageStore

            __JOBSTATS_STORAGE = WandBLineageStore()
    return __JOBSTATS_STORAGE
