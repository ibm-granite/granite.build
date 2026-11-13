# Copyright Granite.Build Authors
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

from libgbtest.storage.storage import (
    AbstractExistingDataReadTest,
    AbstractStorageTest,
    AbstractStorageTestSupport,
)

from gbserver.storage import singleton_storage
from gbserver.storage.storage import BaseItemStorage
from gbserver.storage.stored_step_run import StoredStepRun


class StepStorageTestSupport(AbstractStorageTestSupport):

    def __init__(self):
        super().__init__(sort_column="build_id")

    def _get_test_item(self, index):
        obj = StoredStepRun(
            build_id=f"buildid{index}",
            target_id=f"targetid{index}",
            definition_uri=f"http://definition{index}",
        )
        return obj


class BaseStepStorageTest(AbstractStorageTest):

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return StepStorageTestSupport()

    def _get_tested_storage(self) -> BaseItemStorage:
        return self.storage.step_storage


class BaseLegacyStoredStepTest(AbstractExistingDataReadTest):

    def _get_tested_readonly_storage(
        self, storage: singleton_storage.SingletonAdminStorage
    ):
        return storage.step_storage
