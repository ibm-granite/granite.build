import datetime

from libgbtest.storage.storage import (
    AbstractExistingDataReadTest,
    AbstractStorageTest,
    AbstractStorageTestSupport,
)

from gbserver.storage import singleton_storage
from gbserver.storage.storage import BaseItemStorage
from gbserver.storage.stored_event import StoredEvent
from gbserver.types.buildevent import (
    ArtifactPushedEventPayload,
    BuildEvent,
    BuildEventMessagePayload,
    BuildEventMetricsPayload,
    BuildEventStatusPayload,
    BuildEventType,
    CreatedArtifactEventPayload,
    EntityRunMetadata,
)
from gbserver.types.metrics import Metric, MetricMetadata, MetricName
from gbserver.utils.utils import get_uuid


class EventStorageTestSupport(AbstractStorageTestSupport):

    def __init__(self):
        super().__init__(sort_column="build_id")

    def _get_test_item(self, index):
        build_id = f"build_id_{index}"
        username = f"user{index}"
        run_metadata = EntityRunMetadata(
            build_id=build_id, username=username, type=f"mtype{index}"
        )
        index_mod = index % 4
        event_data = {"somekey": f"somevalue{index}"}
        if index_mod == 0:
            type = BuildEventType.STATUS_EVENT
            payload = BuildEventStatusPayload(data=event_data, msg=f"msg{index}")
        elif index_mod == 1:
            type = BuildEventType.ARTIFACT_PUSHED_EVENT
            payload = ArtifactPushedEventPayload(
                data=event_data, uri=f"https://localhost/{index}"
            )
        elif index_mod == 2:
            type = BuildEventType.MESSAGE_EVENT
            payload = BuildEventMessagePayload(data=event_data, msg=f"msg{index}")
        elif index_mod == 3:
            type = BuildEventType.ARTIFACT_EVENT
            payload = CreatedArtifactEventPayload(
                data=event_data, uri=f"https://localhost/{index}"
            )
        elif index_mod == 4:
            type = BuildEventType.METRICS_EVENT
            metadata = MetricMetadata(username="user{index}")
            metric = Metric(
                name=MetricName.PROCESSING_DELAY, value=0.0, metadata=metadata
            )
            payload = BuildEventMetricsPayload(data=event_data, metrics=[metric])
        else:
            assert False, "should never reach this line"

        source = f"build-source{index}"
        timestamp = datetime.datetime(2023, 1, (index + 1) % 30)
        build_event = BuildEvent(
            run_metadata=run_metadata,
            type=type,
            payload=payload,
            timestamp=timestamp,
            source=source,
        )
        obj = StoredEvent(build_event=build_event)
        return obj


class BaseEventStorageTest(AbstractStorageTest):

    @classmethod
    def _get_test_config(cls) -> AbstractStorageTestSupport:
        return EventStorageTestSupport()

    def _get_tested_storage(self) -> BaseItemStorage:
        return self.storage.event_storage

    def test_build_sorting(self) -> None:
        """Test the get_sorted_build_events() method which is supposed to return builds sorted by the order they were inserted"""
        # TODO: this is not a terribly rigorous test.  Not sure how to make it harder.
        count = 10
        inserted = []
        build_id = get_uuid()
        for i in range(count):
            obj = self._get_test_item(
                0
            )  # Same index so we always get the same build id
            assert isinstance(obj, StoredEvent), f"obj is of type {obj.__class__}"
            assert isinstance(
                obj.build_event, BuildEvent
            ), f"obj is of type {obj.__class__}"
            obj.build_event.run_metadata.build_id = build_id
            self.storage.event_storage.add(obj)
            inserted.append(obj)

        queried_builds = self.storage.event_storage.get_sorted_build_events(build_id)
        assert inserted == queried_builds


class BaseLegacyEventStorageTest(AbstractExistingDataReadTest):

    def _get_tested_readonly_storage(
        self, storage: singleton_storage.SingletonAdminStorage
    ):
        return storage.event_storage
