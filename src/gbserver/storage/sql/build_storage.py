#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

from gbserver.storage.build_storage import BaseStoredBuildStorage, IStoredBuildStorage
from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.storage import UPDATED_TIME_FIELD_NAME
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import GB_BUILDS_TABLE_NAME


class SQLBuildStorage(
    BaseSQLItemStorage[StoredBuild], BaseStoredBuildStorage, IStoredBuildStorage
):

    def __init__(self, **kwargs) -> None:
        kwargs["default_pagination_sort_by_column"] = UPDATED_TIME_FIELD_NAME
        kwargs["exact_liked_list_columns"] = {"tags": "tags"}
        super().__init__(**kwargs)
