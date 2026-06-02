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

from __future__ import annotations

from typing import Optional, Tuple

from gbserver.lineage.jobstats import ILineageStore
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun


class NoopLineageStore(ILineageStore):
    """Noop lineage store for external lineage providers (e.g. wandb).

    The internal Granite.Build admin storage still persists sufficient
    information to recover the full build lineage, which can be extracted
    and persisted to an external provider later.
    """

    def add_jobstats_for_build(
        self, storage: SingletonAdminStorage, build_id: str
    ) -> None:
        pass

    def add_jobstats_for_build_target(
        self, storage: SingletonAdminStorage, build_id: str, target_id: str
    ) -> None:
        pass

    def add_jobstats_for_original_artifact(
        self, artifact: ArtifactRegistration, sources: list[ArtifactRegistration]
    ) -> None:
        pass

    def create_jobstats_for_target(
        self,
        storage: SingletonAdminStorage,
        targetrun: StoredTargetRun,
        build: Optional[StoredBuild] = None,
    ) -> Tuple:
        return [], {}

    def create_jobstats_for_original_artifact(
        self, artifact: ArtifactRegistration, sources: list[ArtifactRegistration]
    ):
        return {}

    def count_release_ids(
        self, release_id: str, target_id: Optional[str] = None
    ) -> int:
        return 0

    def does_release_id_exist(
        self, release_id: str, expected_count: int, target_id: Optional[str] = None
    ) -> bool:
        return False
