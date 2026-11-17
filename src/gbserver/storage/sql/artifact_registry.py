#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

from typing import Self

from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.artifact_registry import BaseArtifactRegistry, IArtifactRegistry
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class SQLArtifactRegistry(
    BaseSQLItemStorage[ArtifactRegistration], BaseArtifactRegistry, IArtifactRegistry
):
    # Some notes:

    def __init__(self: Self, **kwargs) -> None:
        kwargs["unique_columns"] = {
            ("uri", "space_name"): None,  # uri+space_name combo must be unique
            "checksum": "",  # checksum must be unique, but empty strings can be duplicated
        }
        # Tags are stored as a string, such as "tag1,tag2,tag3...", so we enable special handling for this column
        kwargs["exact_liked_list_columns"] = {"tags": "tags"}
        kwargs["indexed_columns"] = ["checksum"]

        super().__init__(**kwargs)
