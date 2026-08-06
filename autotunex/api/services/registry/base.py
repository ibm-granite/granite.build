# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""ModelRegistry seam contract.

Mirrors the historical services.dmf_service.Dmf public surface 1:1 so existing
call sites are untouched. Async/sync per-method matches Dmf exactly.
"""

from abc import ABC, abstractmethod


class ModelRegistry(ABC):
    def __init__(self, db):
        self.db = db

    @abstractmethod
    def get_checkpoints(self, artifact_url): ...

    @abstractmethod
    def pull_all_checkpoint_files(self, artifact_url): ...

    @abstractmethod
    def pull_checkpoint_file(self, artifact_url, file_paths): ...

    @abstractmethod
    async def get_models(self, user_id): ...

    @abstractmethod
    async def get_all_models(self): ...

    @abstractmethod
    async def publish_model(self, job_id, metadata, user_id): ...

    @abstractmethod
    async def delete_model(self, job_id, user_id): ...

    @abstractmethod
    def get_model_detail(
        self, model_label, namespace="base_training", table="model_shared"
    ): ...

    @abstractmethod
    def get_model_card(self, namespace, table, model_label, revision): ...

    @abstractmethod
    def search_models(self, query): ...
