from typing import Self

from libgbtest.storage.storage import AbstractStorageTest, AbstractStorageTestSupport
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbserver.storage.storage import BaseItemStorage
from gbserver.storage.stored_status import StoredStatus
from gbserver.utils.utils import get_uuid


class StatusStorageTestSupport(AbstractStorageTestSupport):

    def __init__(self):
        # uuid is the only column exercised by _get_ascending_sorted_test_items
        # (see the "uuid-N" padding below): gb_status has no business columns
        # to sort by, by design — it's an opaque key/JSON store.
        super().__init__(sort_column="uuid")

    def _get_test_item(self, index):
        # uuid IS the status key, so (unlike most other stored items) it must
        # not be a deterministic function of index alone: the shared test
        # suite calls this twice with the same index expecting two distinct
        # items that still share a queryable column value. Zero-padded so
        # ascending string order matches ascending index order, for
        # test_sorting's use of _get_ascending_sorted_test_items().
        return StoredStatus(
            uuid=f"status-key-{index:04d}-{get_uuid()}",
            value={"index": index},
        )


class BaseStatusStorageTest(AbstractStorageTest):

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return StatusStorageTestSupport()

    def _get_tested_storage(self) -> BaseItemStorage:
        return self.storage.status_storage

    def test_count_with_where(self: Self) -> None:
        """gb_status exposes no business columns to filter by (only its key,
        ``uuid``, and ``created_time`` which the shared where-builder always
        excludes) — so this overrides the base test to filter by ``uuid``
        instead of the generic helper, which would build an empty/None where.
        """
        storage = self._get_tested_storage()
        item0 = self._get_test_item(0)
        item1 = self._get_test_item(1)
        item2 = self._get_test_item(2)
        storage.add([item0, item1, item2])

        assert storage.count() == 3
        assert storage.count(where={"uuid": item0.uuid}) == 1
        assert storage.count(where={"uuid": "no-such-key"}) == 0


class TestStatusValueMethods(AbstractSingletonStorageUsingTest):
    """Tests for the get_value()/set_value() convenience methods."""

    def test_get_value_missing_key_returns_none(self: Self) -> None:
        assert self.storage.status_storage.get_value("no-such-key") is None

    def test_set_then_get_value(self: Self) -> None:
        self.storage.status_storage.set_value("k1", {"build_id": "b1"})
        assert self.storage.status_storage.get_value("k1") == {"build_id": "b1"}

    def test_set_value_upserts_existing_key(self: Self) -> None:
        self.storage.status_storage.set_value("k1", {"build_id": "b1"})
        self.storage.status_storage.set_value("k1", {"build_id": "b2"})
        assert self.storage.status_storage.get_value("k1") == {"build_id": "b2"}
