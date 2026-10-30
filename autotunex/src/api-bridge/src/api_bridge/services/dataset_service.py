# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Dataset service for api-bridge (local copy from api/services/dataset_service.py)."""

import logging
import re

from fastapi import HTTPException

from api_bridge import database, models

logger = logging.getLogger(__name__)

# The formats the main service accepts (DatasetCreate.data_format).
DATA_FORMATS = {"jsonl", "csv", "parquet"}


_HF_DATASET_PREFIXES = ("hf://huggingface.co/datasets/", "hf:///datasets/")
_APP_REPO_SUFFIX = re.compile(r"_([0-9a-f]{8})$")


def _hf_dataset_repo(uri: str | None) -> str | None:
    """``owner/repo`` of a huggingface.co dataset locator, either spelling, else None."""
    for prefix in _HF_DATASET_PREFIXES:
        if uri and uri.startswith(prefix):
            parts = uri[len(prefix) :].split("/")
            return "/".join(parts[:2]) if len(parts) >= 2 and all(parts[:2]) else None
    return None


def _app_pushed_row(datasets: list, artifact_uri: str) -> dict | None:
    """The row whose own push is ``artifact_uri``'s repo, if the caller can see it.

    The main service pushes a row's files to ``<name>_<id8>`` and stores the
    locator llmb prints (``hf://huggingface.co/...``), while a run names the repo
    from gbserver's cache path (``hf:///...``). A row matches only when its id
    starts with the suffix and its stored locator names the same repo.
    """
    repo = _hf_dataset_repo(artifact_uri)
    match = _APP_REPO_SUFFIX.search(repo) if repo else None
    if match is None:
        return None
    for ds in datasets:
        if str(ds["id"]).startswith(match.group(1)) and (
            _hf_dataset_repo(ds.get("artifact_url")) == repo
        ):
            return ds
    return None


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

    def find_or_create(
        self, name: str, artifact_uri: str, user_id: str, data_format: str | None = None
    ) -> str:
        """Return a dataset id, reusing a (name, artifact_url) match or creating one.

        Mirrors the prior client behavior: match on name AND artifact_url; if no
        match, register the dataset and attach the artifact metadata. A
        ``data_format`` outside :data:`DATA_FORMATS` is dropped (the column default,
        jsonl, applies), so the main service never reads a format it rejects. The
        format applies only to a new row: a reused one can be shared (SYSTEM_USER)
        or validated by the main service, so one run's file must not rewrite it.
        """
        fmt = data_format if data_format in DATA_FORMATS else None
        datasets = self.db.get_datasets(user_id)
        # First, so a run attaches to the row whose own push this is, not to a
        # duplicate an earlier run registered under the repo's name.
        pushed = _app_pushed_row(datasets, artifact_uri)
        if pushed is not None:
            logger.info("Dataset reused: %s is app-pushed (id=%s)", artifact_uri, pushed["id"])
            return str(pushed["id"])
        for ds in datasets:
            if ds.get("name") == name and ds.get("artifact_url") == artifact_uri:
                logger.info("Dataset reused: %s (id=%s)", name, ds["id"])
                return str(ds["id"])

        info = models.DatasetInfo(user_id=user_id, name=name, description="")
        created = self.db.insert_dataset(dataset=info, data_format=fmt)
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
