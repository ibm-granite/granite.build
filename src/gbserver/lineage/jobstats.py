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
from typing import Optional, Tuple

from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun


class ILineageStore(ABC):
    """Abstract interface for lineage storage backends."""

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


__JOBSTATS_STORAGE: Optional[ILineageStore] = None


def reset_lineage_store() -> None:
    """Reset the singleton so the next call to get_lineage_store() re-creates it."""
    global __JOBSTATS_STORAGE
    __JOBSTATS_STORAGE = None


def get_lineage_store() -> ILineageStore:
    """Get a singleton instance of the lineage storage backend."""
    global __JOBSTATS_STORAGE
    if __JOBSTATS_STORAGE is None:
        from gbserver.lineage.wandb_jobstats import WandBLineageStore

        __JOBSTATS_STORAGE = WandBLineageStore()
    return __JOBSTATS_STORAGE
