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

import datetime
import hashlib
import os
import random
from abc import abstractmethod
from typing import Optional

import pytest

from gbserver.storage import singleton_storage
from gbserver.storage.storage import BaseItemStorage, BaseStoredItem, IItemStorage
from gbserver.storage.storage_factory import StorageFactory
from gbserver.storage.stored_space import StoredSpace
from gbserver.types.constants import (
    GB_ARTIFACT_REGISTRY_TABLE_NAME,
    GB_BUILDS_TABLE_NAME,
    GB_ENVIRONMENT,
    GB_EVENTS_TABLE_NAME,
    GB_NODE_FAILURES_TABLE_NAME,
    GB_SPACES_TABLE_NAME,
    GB_STEP_RUNS_TABLE_NAME,
    GB_TARGET_RUNS_TABLE_NAME,
    GBSERVER_GITHUB_TOKEN,
    LAKEHOUSE_ENVIRONMENT,
    PUBLIC_SPACE_GIT_URI,
    PUBLIC_SPACE_LH_NAMESPACE,
    PUBLIC_SPACE_NAME,
)
from gbserver.utils.logger import get_logger

_SACRED_TABLE_NAMES = [
    GB_BUILDS_TABLE_NAME,
    GB_TARGET_RUNS_TABLE_NAME,
    GB_STEP_RUNS_TABLE_NAME,
    GB_ARTIFACT_REGISTRY_TABLE_NAME,
    GB_SPACES_TABLE_NAME,
    GB_EVENTS_TABLE_NAME,
    GB_NODE_FAILURES_TABLE_NAME,
]


_NOW = datetime.datetime.now()
_DATESTR = f"d{_NOW.month:02d}{_NOW.day:02d}"


def check_env_var_set(varname: str, msg: Optional[str] = None):
    val = os.environ.get(varname)
    assert val is not None, (
        msg if msg else f"{varname} environment variable must be set."
    )


logger = get_logger(__name__)


def check_test_config():
    """Verify a minimum set of configuration (generally in env vars) is set and fail assert if not.

    In mock mode (GBTEST_MODE != 'live'), this check is skipped since credentials
    are intentionally absent.
    """
    from libgbtest.mode import is_mock_mode

    if is_mock_mode():
        return

    assert (
        GB_ENVIRONMENT != "PROD"
    ), "GB_ENVIRONMENT env var setting is a problem. You need to be testing in either STAGING or DEV environment."
    assert (
        LAKEHOUSE_ENVIRONMENT != "PROD"
    ), f"LAKEHOUSE_ENVIRONMENT={LAKEHOUSE_ENVIRONMENT}, but should we one of STAGING or DEV"
    check_env_var_set("LAKEHOUSE_TOKEN")  # Needed for lakehouse
    check_env_var_set(
        "IBM_CLOUD_API_KEY"
    )  # Needed for ibm cloud secrets and maybe others
    assert GBSERVER_GITHUB_TOKEN != ""


class AbstractReadonlySingletonStorageUsingTest:
    """A helper class to enable running tests on a clean/empty set of admin tables that use a table name prefix to
    avoid wrecking the production/staging g.b tables.
    Sets the global singleton storage instances for each test_*() method.
    """

    @classmethod
    def _is_cloud_config_required(cls) -> bool:
        """Return True if this test class requires cloud configuration (IBM Cloud, GitHub tokens, etc.).
        Subclasses (e.g. SQLite tests) override to return False."""
        return True

    @classmethod
    def setup_class(cls):
        if cls._is_cloud_config_required():
            if GB_ENVIRONMENT not in ("DEV", "STAGING", "STANDALONE"):
                pytest.skip(
                    "Requires cloud configuration (GB_ENVIRONMENT=DEV, STAGING, or STANDALONE)"
                )
            check_test_config()
        f = cls._get_storage_factory()
        assert isinstance(
            f, StorageFactory
        ), "A StorageFactory instance was not provided"
        singleton_storage.set_storage_factory(f)
        if cls._is_cloud_config_required():
            assert (
                GB_ENVIRONMENT != "PROD"
            ), "GB_ENVIRONMENT env var setting is a problem. You need to be testing in either STAGING or DEV environment."

    @classmethod
    @abstractmethod
    def _get_storage_factory(cls) -> StorageFactory:
        """Get the storage factory to use in this test.  Sub-classes may override, for example, to test a specific factory."""
        return singleton_storage.get_storage_factory()

    def _verify_equal(self, result, expected):
        """Enables refined object comparison, for example, to allow ignoring updated_time field, when it is present."""
        is_equal = result == expected
        return is_equal

    def _verify_get_results(self, expected_results, results, ordered=True):
        if expected_results is None:
            assert (
                results is None
            ), f"Did not expect any results, but got {len(results)}"
        elif results is None:
            assert False, f"Expected {len(expected_results)}, but got none."
        elif isinstance(expected_results, BaseStoredItem):
            assert isinstance(results, BaseStoredItem), "Result is not a BaseStoredItem"
            assert self._verify_equal(
                results, expected_results
            ), f"Expected item {expected_results} is not equal to result {results}"
        elif isinstance(expected_results, list) and isinstance(results, list):
            assert len(expected_results) == len(
                results
            ), f"Expected {len(expected_results)} items, but got {len(results)}."
            index = 0
            for expected in expected_results:
                if ordered:
                    result = results[index]
                    if expected is None:
                        found = result is None
                    elif result is None:
                        found = expected is None
                    else:
                        found = self._verify_equal(result, expected)
                    index += 1
                else:
                    found = False
                    for result in results:
                        if expected is None:
                            match = result is None
                        elif result is None:
                            match = expected is None
                        else:
                            match = self._verify_equal(result, expected)
                        found = found or match
                        if found:
                            break
                assert found, f"Did not find expected item {expected}"
        else:
            assert False, f"Did not expected to get here."


class AbstractSingletonStorageUsingTest(AbstractReadonlySingletonStorageUsingTest):
    """A helper class to enable running tests on a clean/empty set of admin tables that use a table name prefix to
    avoid wrecking the production/staging g.b tables.
    Sets the global singleton storage instances for each test_*() method.
    """

    def setup_method(self, method):
        random_number = random.randint(
            0, 99999
        )  # So we have very low probablity of colliding with other test runs.
        # Build a prefix that fits within PostgreSQL's 63-char identifier limit.
        # A 6-char hex hash of the full method name ensures uniqueness even when the
        # name is truncated.  Format: d{MMDD}_{hex6}_{rand5}_{method_name}_
        # Fixed overhead: 5+1+6+1+5+1+1 = 20 chars; longest suffix "gb_space_users" = 14 chars.
        # That leaves 63 - 20 - 14 = 29 chars for the (possibly truncated) method name.
        name_hash = hashlib.sha1(method.__name__.encode()).hexdigest()[:6]
        short_method_name = method.__name__[:29]
        table_prefix = (
            f"{_DATESTR}_{name_hash}_{random_number:05d}_{short_method_name}_"
        )
        storage = getattr(self, "storage", None)
        assert (
            storage is None
        ), "This is not thread-safe.  Tests within a class must be run serially."
        self.storage = singleton_storage.set_storage_prefix(table_prefix)
        self._clear_storage()

    def teardown_method(self, method):
        self._clear_storage()
        self.storage = None

    # def get_testing_artifact_registry(self) -> LhArtifactRegistry:
    #     """Get the singleton instance of artifact registration storage that is cleaned before and after test_*() methods.  """
    #     return self.storage.artifact_registry

    # def get_testing_build_storage(self) -> LhBuildStorage:
    #     """Get the singleton instance of artifact registration storage that is cleaned before and after test_*() methods.  """
    #     return self.storage.build_storage

    # def get_testing_target_storage(self) -> LhTargetRunStorage:
    #     """Get the singleton instance of TargetRun storage that is cleaned before and after test_*() methods.  """
    #     return self.storage.target_storage

    # def get_testing_step_storage(self) -> LhStepRunStorage:
    #     """Get the singleton instance of StoredStepRun storage that is cleaned before and after test_*() methods.  """
    #     return self.storage.step_storage

    # def get_testing_space_storage(self) -> LhSpaceStorage:
    #     """Get the singleton instance of SpaceStorage storage that is cleaned before and after test_*() methods.  """
    #     return self.storage.space_storage

    def _get_storage_to_clear(self) -> list[BaseItemStorage]:
        """Get the instances in self.storage that should actually be cleaned before/after each test
        Sub-classes can override this to return only the storage instance(s) that they are using during testing to speed up the tests.
        """
        return [
            self.storage.build_storage,
            self.storage.target_storage,
            self.storage.step_storage,
            self.storage.space_storage,
            self.storage.space_user_storage,
            self.storage.artifact_registry,
            self.storage.event_storage,
            self.storage.node_failure_storage,
        ]

    def _clear_storage(self):
        to_clean = self._get_storage_to_clear()
        for storage in to_clean:
            self.__clear_storage(storage)

    def __clear_storage(self, storage: IItemStorage):
        assert isinstance(storage, IItemStorage)
        assert not storage.get_table_name() in _SACRED_TABLE_NAMES
        try:
            storage.delete_table()
        except Exception as e:
            logger.warning(
                f"Ignoring failed table deletion on table {storage.get_table_name()}: {e}"
            )


public_space = StoredSpace(
    name=PUBLIC_SPACE_NAME,
    git_repo_uri=PUBLIC_SPACE_GIT_URI,
    lakehouse_namespace=PUBLIC_SPACE_LH_NAMESPACE,
)


class AbstractSingletonStorageUsingPreloadedSpaceTest(
    AbstractSingletonStorageUsingTest
):
    """Extends the super class to preload the space admin table with public and testspace space definitions"""

    def setup_method(self, method):
        """Use the super class to initialize the admin storage and then create the "public" space.

        Args:
            method (_type_): _description_
        """
        super().setup_method(method)
        self.storage.space_storage.add([public_space])

    @classmethod
    def setup_class(cls):
        super().setup_class()


def find_unequal_attributes(obj1, obj2):
    unequal_attrs = {}
    # Get attributes common to both objects, excluding dunder methods/attributes
    attrs1 = {k: v for k, v in obj1.__dict__.items() if not k.startswith("__")}
    attrs2 = {k: v for k, v in obj2.__dict__.items() if not k.startswith("__")}

    # Check for differing values in common attributes
    for attr_name in attrs1:
        if attr_name in attrs2:
            if attrs1[attr_name] != attrs2[attr_name]:
                unequal_attrs[attr_name] = (attrs1[attr_name], attrs2[attr_name])
        else:
            # Attribute exists in obj1 but not obj2
            unequal_attrs[attr_name] = (attrs1[attr_name], "Attribute missing in obj2")

    # Check for attributes present in obj2 but not obj1
    for attr_name in attrs2:
        if attr_name not in attrs1:
            unequal_attrs[attr_name] = ("Attribute missing in obj1", attrs2[attr_name])

    return unequal_attrs


def is_pytest_running_parallel():
    return "PYTEST_XDIST_WORKER" in os.environ
