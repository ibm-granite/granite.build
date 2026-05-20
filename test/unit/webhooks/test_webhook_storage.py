"""Unit tests for webhook subscription storage layer.

Tests the SQLWebhookStorage implementation which provides CRUD and
query operations for StoredWebhookSubscription records.
"""

import uuid

from gbserver.webhooks.models import StoredWebhookSubscription


class TestWebhookStorage:
    """Tests for webhook subscription storage operations."""

    @classmethod
    def setup_class(cls):
        """Create a single SQLWebhookStorage instance for all tests in this class."""
        from gbserver.webhooks.sql_storage import SQLWebhookStorage

        # Use a unique table name to avoid conflicts with other test runs
        cls.table_name = f"test_whsub_{uuid.uuid4().hex[:8]}"
        cls.storage = SQLWebhookStorage(table_name=cls.table_name)

    @classmethod
    def teardown_class(cls):
        """Drop the test table after all tests complete."""
        cls.storage.delete_table()

    def _make_subscription(self, **overrides) -> StoredWebhookSubscription:
        """Create a StoredWebhookSubscription with sensible defaults.

        Args:
            **overrides: Fields to override on the default subscription.

        Returns:
            A StoredWebhookSubscription instance.
        """
        defaults = {
            "space_name": "test-space",
            "build_id": "build-001",
            "webhook_url": "https://example.com/hook",
            "secret": "test-secret-key",
            "event_types": ["*"],
            "created_by": "testuser",
            "active": True,
        }
        defaults.update(overrides)
        return StoredWebhookSubscription(**defaults)

    def test_add_and_get(self):
        """Test adding a subscription and retrieving it by UUID."""
        sub = self._make_subscription()
        uuid_val = self.storage.add(sub)

        retrieved = self.storage.get_by_uuid(uuid_val)
        assert retrieved is not None
        assert retrieved.uuid == sub.uuid
        assert retrieved.space_name == "test-space"
        assert retrieved.build_id == "build-001"
        assert retrieved.webhook_url == "https://example.com/hook"
        assert retrieved.secret == "test-secret-key"
        assert retrieved.created_by == "testuser"
        assert retrieved.active is True

    def test_get_active_for_build(self):
        """Test filtering active subscriptions for a specific build.

        Setup: 2 subs for build-1 (1 active, 1 inactive), 1 for build-2.
        Expected: Only the active sub for build-1 is returned.
        """
        # Use unique build IDs to avoid interference from other tests
        build_id = f"build-active-{uuid.uuid4().hex[:8]}"
        other_build_id = f"build-other-{uuid.uuid4().hex[:8]}"

        active_sub = self._make_subscription(
            build_id=build_id, active=True, space_name="space-a"
        )
        inactive_sub = self._make_subscription(
            build_id=build_id, active=False, space_name="space-a"
        )
        other_build_sub = self._make_subscription(
            build_id=other_build_id, active=True, space_name="space-a"
        )

        self.storage.add(active_sub)
        self.storage.add(inactive_sub)
        self.storage.add(other_build_sub)

        results = self.storage.get_active_for_build(build_id)
        assert len(results) == 1
        assert results[0].uuid == active_sub.uuid
        assert results[0].build_id == build_id
        assert results[0].active is True

    def test_deactivate(self):
        """Test deactivating a single subscription by ID."""
        sub = self._make_subscription(
            active=True, build_id=f"build-deact-{uuid.uuid4().hex[:8]}"
        )
        self.storage.add(sub)

        # Verify it's active
        retrieved = self.storage.get_by_uuid(sub.uuid)
        assert retrieved.active is True

        # Deactivate
        self.storage.deactivate(sub.uuid)

        # Verify it's inactive
        retrieved = self.storage.get_by_uuid(sub.uuid)
        assert retrieved.active is False

    def test_deactivate_for_build(self):
        """Test deactivating all subscriptions for a build.

        Add 2 active subs for same build, deactivate_for_build returns 2.
        """
        build_id = f"build-deactall-{uuid.uuid4().hex[:8]}"
        other_build_id = f"build-keep-{uuid.uuid4().hex[:8]}"

        sub1 = self._make_subscription(build_id=build_id, active=True)
        sub2 = self._make_subscription(build_id=build_id, active=True)
        sub_other = self._make_subscription(build_id=other_build_id, active=True)

        self.storage.add(sub1)
        self.storage.add(sub2)
        self.storage.add(sub_other)

        count = self.storage.deactivate_for_build(build_id)
        assert count == 2

        # Verify both are now inactive
        retrieved1 = self.storage.get_by_uuid(sub1.uuid)
        retrieved2 = self.storage.get_by_uuid(sub2.uuid)
        assert retrieved1.active is False
        assert retrieved2.active is False

        # Verify the other build's sub is still active
        retrieved_other = self.storage.get_by_uuid(sub_other.uuid)
        assert retrieved_other.active is True

    def test_get_active_for_space(self):
        """Test filtering active space-wide subscriptions.

        Setup: 1 space-wide sub (build_id=None) in target-space, 1 per-build
        sub in target-space, 1 space-wide sub in another space.
        Expected: Only the active space-wide sub in target-space is returned.
        """
        target_space = f"space-target-{uuid.uuid4().hex[:8]}"
        other_space = f"space-other-{uuid.uuid4().hex[:8]}"

        # Space-wide subscription (build_id=None)
        space_sub = self._make_subscription(
            build_id=None, space_name=target_space, active=True
        )
        # Per-build subscription in same space (should NOT be returned)
        build_sub = self._make_subscription(
            build_id="build-123", space_name=target_space, active=True
        )
        # Space-wide subscription in different space (should NOT be returned)
        other_sub = self._make_subscription(
            build_id=None, space_name=other_space, active=True
        )
        # Inactive space-wide subscription (should NOT be returned)
        inactive_sub = self._make_subscription(
            build_id=None, space_name=target_space, active=False
        )

        self.storage.add(space_sub)
        self.storage.add(build_sub)
        self.storage.add(other_sub)
        self.storage.add(inactive_sub)

        results = self.storage.get_active_for_space(target_space)
        assert len(results) == 1
        assert results[0].uuid == space_sub.uuid
        assert results[0].space_name == target_space
        # build_id may be None or "" depending on storage layer serialization
        assert results[0].build_id is None or results[0].build_id == ""

    def test_get_by_space(self):
        """Test filtering subscriptions by space name.

        2 spaces, filter returns only matching space.
        """
        space_a = f"space-alpha-{uuid.uuid4().hex[:8]}"
        space_b = f"space-beta-{uuid.uuid4().hex[:8]}"

        sub_a1 = self._make_subscription(space_name=space_a, build_id="b1")
        sub_a2 = self._make_subscription(space_name=space_a, build_id="b2")
        sub_b = self._make_subscription(space_name=space_b, build_id="b3")

        self.storage.add(sub_a1)
        self.storage.add(sub_a2)
        self.storage.add(sub_b)

        results = self.storage.get_by_space(space_a)
        assert len(results) == 2
        result_uuids = {r.uuid for r in results}
        assert sub_a1.uuid in result_uuids
        assert sub_a2.uuid in result_uuids

        results_beta = self.storage.get_by_space(space_b)
        assert len(results_beta) == 1
        assert results_beta[0].uuid == sub_b.uuid
