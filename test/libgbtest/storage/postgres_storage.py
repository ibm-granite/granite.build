#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Mixin that makes a storage test run against PostgreSQL."""

from gbserver.storage.sql.storage_factory import SQLStorageFactory
from gbserver.storage.storage_factory import StorageFactory


class PostgresStorageTestMixin:
    """Run a storage test class against PostgreSQL (the SQL storage factory).

    List it first among the bases, so its factory overrides the base test's:

        class SQLBuildStorageTests(PostgresStorageTestMixin, BaseBuildStorageTest): ...

    Which server is used comes from the GBSERVER_SQL_* connection settings: the
    local test PostgreSQL (test/unit/storage/postgres) or the hosted one
    (test/integration/ibm/storage/sql).
    """

    @classmethod
    def _get_storage_factory(cls) -> StorageFactory:
        """Use PostgreSQL storage, via the GBSERVER_SQL_* connection settings.

        Returns:
            A SQL storage factory.
        """
        return SQLStorageFactory()
