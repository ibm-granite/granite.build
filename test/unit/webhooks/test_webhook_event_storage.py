"""Unit tests for webhook event persistence (write-ahead log)."""

import uuid

from gbserver.webhooks.event_models import StoredWebhookEvent


class TestWebhookEventStorage:
    """Tests for webhook event write-ahead storage."""

    @classmethod
    def setup_class(cls):
        from gbserver.storage.sql.webhook_event_storage import SQLWebhookEventStorage

        cls.table_name = f"test_whevt_{uuid.uuid4().hex[:8]}"
        cls.storage = SQLWebhookEventStorage(table_name=cls.table_name)

    @classmethod
    def teardown_class(cls):
        cls.storage.delete_table()

    def _make_event(self, **overrides) -> StoredWebhookEvent:
        defaults = {
            "subscription_id": str(uuid.uuid4()),
            "build_id": "build-001",
            "event_type": "STATUS_EVENT",
            "payload": {"status": "running", "message": "Build started"},
        }
        defaults.update(overrides)
        return StoredWebhookEvent(**defaults)

    def test_add_and_get(self):
        """Add an event and retrieve by UUID."""
        event = self._make_event()
        self.storage.add(event)

        retrieved = self.storage.get_by_uuid(event.uuid)
        assert retrieved is not None
        assert retrieved.subscription_id == event.subscription_id
        assert retrieved.build_id == "build-001"
        assert retrieved.event_type == "STATUS_EVENT"
        assert retrieved.payload == {"status": "running", "message": "Build started"}
        assert retrieved.delivered is False

    def test_get_pending_for_subscription(self):
        """Get undelivered events for a subscription."""
        sub_id = str(uuid.uuid4())
        other_sub_id = str(uuid.uuid4())

        e1 = self._make_event(subscription_id=sub_id)
        e2 = self._make_event(subscription_id=sub_id)
        e3 = self._make_event(subscription_id=other_sub_id)

        self.storage.add(e1)
        self.storage.add(e2)
        self.storage.add(e3)

        pending = self.storage.get_pending_for_subscription(sub_id)
        assert len(pending) == 2
        uuids = {e.uuid for e in pending}
        assert e1.uuid in uuids
        assert e2.uuid in uuids

    def test_mark_delivered(self):
        """Mark events as delivered."""
        sub_id = str(uuid.uuid4())
        e1 = self._make_event(subscription_id=sub_id)
        self.storage.add(e1)

        self.storage.mark_delivered([e1.uuid])

        retrieved = self.storage.get_by_uuid(e1.uuid)
        assert retrieved.delivered is True

        pending = self.storage.get_pending_for_subscription(sub_id)
        assert all(e.uuid != e1.uuid for e in pending)

    def test_get_pending_for_build(self):
        """Get undelivered events for a build across all subscriptions."""
        build_id = f"build-{uuid.uuid4().hex[:8]}"
        sub1 = str(uuid.uuid4())
        sub2 = str(uuid.uuid4())

        e1 = self._make_event(subscription_id=sub1, build_id=build_id)
        e2 = self._make_event(subscription_id=sub2, build_id=build_id)
        e3 = self._make_event(subscription_id=sub1, build_id="other-build")

        self.storage.add(e1)
        self.storage.add(e2)
        self.storage.add(e3)

        pending = self.storage.get_pending_for_build(build_id)
        assert len(pending) == 2
        uuids = {e.uuid for e in pending}
        assert e1.uuid in uuids
        assert e2.uuid in uuids
