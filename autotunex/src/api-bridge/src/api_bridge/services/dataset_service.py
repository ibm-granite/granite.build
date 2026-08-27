# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Dataset service for api-bridge (local copy from api/services/dataset_service.py)."""

import logging

from fastapi import HTTPException

from api_bridge import database, models

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, db: database.Database):
        self.db = db

    def push_dataset(self, dataset: models.DatasetInfo) -> models.DatasetInfo:
        """Create or update a dataset."""
        if dataset.id is not None and self.db.check_dataset_exists(dataset.id):
            return self.db.update_dataset(dataset=dataset)
        else:
            existing = self.db.get_dataset_by_name_and_user(
                dataset_name=dataset.name, user_id=dataset.user_id
            )
            if existing is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset '{dataset.name}' already exists for this user",
                )
            return self.db.insert_dataset(dataset=dataset)

    def get_datasets(self, user_id: str) -> list:
        """Get all datasets for a user."""
        return self.db.get_datasets(user_id)

    def find_or_create(self, name: str, artifact_uri: str, user_id: str) -> str:
        """Return a dataset id, reusing a (name, artifact_url) match or creating one.

        Mirrors the prior client behavior: match on name AND artifact_url; if no
        match, register the dataset and attach the artifact metadata.
        """
        for ds in self.db.get_datasets(user_id):
            if ds.get("name") == name and ds.get("artifact_url") == artifact_uri:
                logger.info("Dataset reused: %s (id=%s)", name, ds["id"])
                return str(ds["id"])

        info = models.DatasetInfo(user_id=user_id, name=name, description="")
        created = self.db.insert_dataset(dataset=info)
        new_id = str(created.id)
        if artifact_uri:
            self.db.update_dataset_metadata(
                id=new_id,
                user_id=user_id,
                metadata={
                    "train_records": None,
                    "train_file_size": None,
                    "validation_records": None,
                    "validation_file_size": None,
                    "artifact_id": None,
                    "artifact_url": artifact_uri,
                },
            )
        logger.info("Dataset created: %s (id=%s, artifact=%s)", name, new_id, artifact_uri)
        return new_id
