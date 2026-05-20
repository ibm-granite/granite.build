import uuid
from typing import Any

from libgbtest.storage.storage import (
    AbstractExistingDataReadTest,
    AbstractStorageTest,
    AbstractStorageTestSupport,
)

from gbcommon.uri.lh import LhURI
from gbserver.storage import singleton_storage
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.artifact_registry import (
    _BUILD_SCHEMA_VERSION1,
    _BUILD_SCHEMA_VERSION2,
    BaseArtifactRegistry,
    ChecksumConflictException,
    IArtifactRegistry,
)
from gbserver.storage.storage import BaseItemStorage, BaseStoredItem
from gbserver.types.artifact import ArtifactType
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class ArtifactStorageTestSupport(AbstractStorageTestSupport):

    def __init__(self):
        super().__init__(sort_column="name")

    def _get_test_item(self, index):
        # Create a good LH URI because the jobstats tests uses this and expects a valid uri
        uri = LhURI.get_table_uri(table_name=f"mytable{index}")
        return ArtifactRegistration(
            name=f"artifact{index}",
            uri=uri,
            type=ArtifactType.FILESET,
            produced_by_build_id=f"build_id{index}",
            space_name=f"spacename{index}",
            username=f"username{index}",
            created_by_build_id=f"some uuid{index}",
            created_by_target_id=f"some other uuid{index}",
            certified_no_restrictions=True,
            description=f"description{index}",
            tags=[f"tag{index}"],
            checksum=str(index),
        )


class BaseArtifactStorageTest(AbstractStorageTest):

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return ArtifactStorageTestSupport()

    def _get_tested_storage(self) -> BaseItemStorage:
        return self.storage.artifact_registry

    def _get_where_search_columns(
        self, storage: BaseItemStorage, item: BaseStoredItem
    ) -> dict[str, Any]:
        columns = super()._get_where_search_columns(storage, item)
        del columns[
            "uri"
        ]  # remove the values that must be unique, otherwise tests on searches of multiple items will fail.
        del columns[
            "checksum"
        ]  # remove the values that must be unique, otherwise tests on searches of multiple items will fail.
        return columns

    def _get_where_test_item(self, index: int):
        """Make sure the URI and checksum are always different for where testing, which inserts items with the same index more than once."""
        item = super()._get_where_test_item(index)
        assert isinstance(item, ArtifactRegistration)
        uuidstr = str(uuid.uuid1())
        item.uri = "http://foo.bar/" + uuidstr
        item.checksum = uuidstr
        return item

    def test_uniqueness_enforcement(self):
        # Make sure we can't use duplicate URIs
        self._duplication_test_helper(["uri", "space_name"])
        self._duplication_test_helper(["checksum"])

    def test_get_by_uri(self):
        storage = self._get_tested_storage()
        item1 = self._get_test_item(1)
        item2 = self._get_test_item(2)
        storage.add([item1, item2])
        assert isinstance(storage, IArtifactRegistry)
        assert isinstance(item1, ArtifactRegistration)
        item = storage.get_by_uri(uri=item1.uri, space_name=item1.space_name)
        assert item is not None, f"Did not find item by uri={item1.uri}"

    def test_get_by_uri_all_spaces(self):
        storage = self._get_tested_storage()
        item1 = self._get_test_item(1)
        item2 = self._get_test_item(2)
        assert isinstance(item1, ArtifactRegistration)
        common_uri = item1.uri
        item2.uri = common_uri
        storage.add([item1, item2])
        assert isinstance(storage, IArtifactRegistry)
        items = storage.get_by_uri(uri=common_uri)
        assert isinstance(items, list), f"expected a list, actual: {type(items)}"
        assert len(items) == 2, f"expected 2 artifacts, actual: {len(items)}"
        assert items[0].uri == items[1].uri
        assert items[0].space_name != items[1].space_name

    def test_get_by_uri_bad_space_name(self):
        storage = self._get_tested_storage()
        item1 = self._get_test_item(1)
        item2 = self._get_test_item(2)
        storage.add([item1, item2])
        assert isinstance(storage, IArtifactRegistry)
        assert isinstance(item1, ArtifactRegistration)
        bad_space_name = item1.space_name + "non-existent"
        item = storage.get_by_uri(uri=item1.uri, space_name=bad_space_name)
        assert (
            item is None
        ), f"expected 0 artifacts with bad_space_name={bad_space_name}, actual: {item}"

    def test_get_by_tag(self):
        storage = self._get_tested_storage()

        item1 = self._get_test_item(1)
        assert isinstance(item1, ArtifactRegistration)
        item1.tags.append("common")
        item1.tags.append("common12")
        item1.tags.append("item1")

        item2 = self._get_test_item(2)
        assert isinstance(item2, ArtifactRegistration)
        item2.tags.append("common")
        item2.tags.append("common12")
        item2.tags.append("item2")

        item3 = self._get_test_item(3)
        assert isinstance(item3, ArtifactRegistration)
        item3.tags.append("common")

        # Add the items
        storage.add([item1, item2, item3])

        # Test no matches
        items = storage.get_by_where({"tags": "comm"})
        self._verify_get_results(items, [], False)

        # Test matching 2 items
        items = storage.get_by_where({"tags": "common"})
        self._verify_get_results(items, [item1, item2, item3], False)

        # Test matching 1 items and uri
        items = storage.get_by_where({"tags": "item1", "uri": item1.uri})
        self._verify_get_results(items, [item1], False)

        # Test matching 2 tags  to get 1 items
        items = storage.get_by_where({"tags": ["common", "item2"]})
        self._verify_get_results(items, [item2], False)

        # Test matching 2 tags  to get 2 items
        items = storage.get_by_where({"tags": ["common", "common12"]})
        self._verify_get_results(items, [item1, item2], False)

        # Test matching 2 tags  to get 0 items
        items = storage.get_by_where({"tags": ["item1", "item2"]})
        self._verify_get_results(items, [], False)

    def test_checksum_column_addition(self):

        # Add an item using old schema
        storage1 = self.storage.artifact_registry  # Test db
        assert isinstance(storage1, BaseArtifactRegistry)
        storage1._schema_version = _BUILD_SCHEMA_VERSION1
        item1 = self._get_test_item(1)
        storage1.add(item1)
        old_columns = storage1.get_column_names()

        # Create a 2nd storage instance pointing to the same table, but using the new schema.
        storage2 = self._get_storage_factory().create_artifact_registry(
            table_name=storage1.get_table_name()
        )
        assert isinstance(storage2, BaseArtifactRegistry)
        storage2._schema_version = _BUILD_SCHEMA_VERSION2

        # Add an item using new schema
        item2 = self._get_test_item(2)
        storage2.add(item2)

        # Confirm that the version upgrade added the tags column
        new_columns = storage2.get_column_names()
        added_columns = [item for item in new_columns if item not in old_columns]
        assert len(added_columns) == 1
        assert "checksum" in added_columns

        # Now read both items with storage2
        items = storage2.get_by_uuid(None)
        self._verify_get_results(items, [item1, item2], ordered=False)

        # Now read both items with storage1
        items = storage1.get_by_uuid(None)
        self._verify_get_results(items, [item1, item2], ordered=False)

        items = storage2.get_by_where(where={"checksum": "2"})
        self._verify_get_results(items, [item2], ordered=False)

    def test_checksum_semantics(self):

        # Get empty test storage
        storage = self._get_tested_storage()

        # add items.
        item0 = self._get_test_item(0)  # checksum="0",
        item0.checksum = ""  # empty string means no checksum
        item1 = self._get_test_item(1)  # checksum="1"
        item2 = self._get_test_item(2)  # checksum="2"

        uuids = storage.add([item0, item1, item2])
        assert len(uuids) == 3, "Did not add 3 artifacts"

        # Make sure we can't use duplicate non-empty checksums
        try:
            item4 = self._get_test_item(4)
            item4.checksum = "2"
            storage.add(item4)  # diff uri, same checksum.
            assert False, "Was allowed to store the same item twice"
        except ChecksumConflictException as exc:
            logger.info(
                "Storage may have issued a WARNING or ERROR above for adding a duplicate uuid and can safely be ignored."
            )
            pass
        except Exception as exc:
            assert False, "Got unexpected exception adding duplicate checksum"

        # Make sure we can add duplicate checksums that are empty strings
        item3 = self._get_test_item(3)
        item3.checksum = ""  # Should not conflict with item0 above
        uuid = storage.add(item3)
        assert uuid == item3.uuid, "Did not add 3rd item that did not have a checksum"


class BaseLegacyArtifactStorageTest(AbstractExistingDataReadTest):

    def _get_tested_readonly_storage(
        self, storage: singleton_storage.SingletonAdminStorage
    ):
        return storage.artifact_registry
