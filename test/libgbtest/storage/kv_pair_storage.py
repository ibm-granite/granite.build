from typing import Any, Self

from libgbtest.storage.storage import AbstractStorageTest, AbstractStorageTestSupport
from libgbtest.utils import AbstractSingletonStorageUsingTest

from gbserver.storage.storage import BaseItemStorage, BaseStoredItem
from gbserver.storage.stored_kv_pair import StoredKeyValuePair
from gbserver.utils.utils import get_uuid


class KeyValuePairStorageTestSupport(AbstractStorageTestSupport):

    def __init__(self):
        super().__init__(sort_column="key")

    def _get_test_item(self, index):
        # `key` is uniquely indexed, so (unlike most other stored items) it must
        # not be a deterministic function of index alone: the shared test suite
        # calls this twice with the same index expecting two distinct items.
        # Zero-padded so ascending string order matches ascending index order,
        # for test_sorting's use of _get_ascending_sorted_test_items().
        return StoredKeyValuePair(
            key=f"status-key-{index:04d}-{get_uuid()}",
            value={"index": index},
        )


class BaseKeyValuePairStorageTest(AbstractStorageTest):

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return KeyValuePairStorageTestSupport()

    def _get_tested_storage(self) -> BaseItemStorage:
        return self.storage.kv_pair_storage

    def _get_where_search_columns(
        self, storage: BaseItemStorage, item: BaseStoredItem
    ) -> dict[str, Any]:
        columns = super()._get_where_search_columns(storage, item)
        # `key` is uniquely indexed, so it can never match the multiple
        # same-index items the shared where tests insert. Removing it leaves no
        # searchable column at all (created_time is dropped by the base), so
        # those tests skip themselves — gb_kv_pairs is an opaque key/JSON store
        # with no business columns to filter by, by design.
        del columns["key"]
        return columns

    def test_get_by_where_key_returns_the_single_row(self: Self) -> None:
        """``get_by_where`` on ``key`` is supported but unreachable from the
        shared suite.

        Its where tests need a column that can match several rows, and ``key``
        is unique, so ``_get_where_search_columns`` drops it and those tests
        skip themselves. That leaves the store's one real query untested, hence
        this.
        """
        storage = self._get_tested_storage()
        item0 = self._get_test_item(0)
        item1 = self._get_test_item(1)
        storage.add([item0, item1])

        found = storage.get_by_where({"key": item0.key})
        assert len(found) == 1
        assert found[0].key == item0.key
        assert found[0].uuid == item0.uuid
        assert storage.get_by_where({"key": "no-such-key"}) == []

    def test_count_with_where(self: Self) -> None:
        """Overridden because the generic helper now yields an empty where (see
        ``_get_where_search_columns``). Filtering by ``key`` is the real
        supported query, so exercise that directly.
        """
        storage = self._get_tested_storage()
        item0 = self._get_test_item(0)
        item1 = self._get_test_item(1)
        item2 = self._get_test_item(2)
        storage.add([item0, item1, item2])

        assert storage.count() == 3
        assert storage.count(where={"key": item0.key}) == 1
        assert storage.count(where={"key": "no-such-key"}) == 0


class TestKeyValuePairValueMethods(AbstractSingletonStorageUsingTest):
    """Tests for the get_value()/set_value() convenience methods."""

    def test_get_value_missing_key_returns_none(self: Self) -> None:
        assert self.storage.kv_pair_storage.get_value("no-such-key") is None

    def test_set_then_get_value(self: Self) -> None:
        self.storage.kv_pair_storage.set_value("k1", {"build_id": "b1"})
        assert self.storage.kv_pair_storage.get_value("k1") == {"build_id": "b1"}

    def test_set_value_upserts_existing_key(self: Self) -> None:
        self.storage.kv_pair_storage.set_value("k1", {"build_id": "b1"})
        self.storage.kv_pair_storage.set_value("k1", {"build_id": "b2"})
        assert self.storage.kv_pair_storage.get_value("k1") == {"build_id": "b2"}

    def test_set_value_upsert_does_not_create_a_second_row(self: Self) -> None:
        """The upsert resolves `key` to the existing row rather than adding."""
        storage = self.storage.kv_pair_storage
        storage.set_value("k1", {"build_id": "b1"})
        storage.set_value("k1", {"build_id": "b2"})
        assert storage.count(where={"key": "k1"}) == 1

    def test_uuid_is_a_generated_identifier_not_the_key(self: Self) -> None:
        """`uuid` no longer doubles as the lookup key."""
        storage = self.storage.kv_pair_storage
        storage.set_value("k1", {"build_id": "b1"})
        item = storage.get_by_key("k1")
        assert item is not None
        assert item.key == "k1"
        assert item.uuid != "k1"

    def test_set_value_preserves_created_time_across_updates(self: Self) -> None:
        """created_time means "when the key was first set"."""
        storage = self.storage.kv_pair_storage
        storage.set_value("k1", {"build_id": "b1"})
        first = storage.get_by_key("k1")
        assert first is not None
        storage.set_value("k1", {"build_id": "b2"})
        second = storage.get_by_key("k1")
        assert second is not None
        assert second.created_time == first.created_time
        assert second.uuid == first.uuid
