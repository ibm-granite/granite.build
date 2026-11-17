#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

from typing import Self

from gbserver.storage.sql.sql_storage import BaseSQLItemStorage
from gbserver.storage.steprun_storage import (
    BaseStoredStepRunStorage,
    IStoredStepRunStorage,
)
from gbserver.storage.stored_step_run import StoredStepRun


class SQLStepRunStorage(
    BaseSQLItemStorage[StoredStepRun], BaseStoredStepRunStorage, IStoredStepRunStorage
):

    def __init__(self: Self, **kwargs):
        super().__init__(**kwargs)
