from typing import Self

import pytest
from libgbtest.lineage.lineage import AbstractLineageTest
from libgbtest.lineage.mock_lineage_service import MockLineageService

from gbserver.lineage.wandb_jobstats import WandBLineageStore


class TestWandBLineage(AbstractLineageTest):

    def _get_tested_lineage_storage(self: Self):
        store = WandBLineageStore.__new__(WandBLineageStore)
        store._service = MockLineageService()
        # __new__ skips __init__ (which would build a real wandb client), so the
        # per-process caches __init__ sets up have to be seeded here too.
        store._recorded_until = {}
        return store


class TestRecordedCacheUnderFailure:
    """The store's TTL cache must never cache an unanswered dedup query.

    ``filter_unrecorded`` fails CLOSED, returning an EMPTY set on error. By value
    that is indistinguishable from "every candidate is already recorded", so a
    layer that decided what to cache from the return value alone would cache the
    whole candidate set on a sink outage and skip real targets for a TTL-long
    window. The failure flag is the only thing separating the two.
    """

    def _store(self, service):
        store = WandBLineageStore.__new__(WandBLineageStore)
        store._service = service
        store._recorded_until = {}
        return store

    def test_failed_query_caches_nothing(self):
        class _FailingService:
            def filter_unrecorded(
                self, target_ids, expected_counts=None, on_query_error=None
            ):
                if on_query_error is not None:
                    on_query_error(RuntimeError("timeout"))
                return set()

        store = self._store(_FailingService())

        result = store.filter_unrecorded({"t1", "t2"}, {"t1": 1, "t2": 1})

        assert result == set(), "a failed query must record nothing (fail closed)"
        assert store._recorded_until == {}, (
            "a failed query was cached as a 'recorded' verdict; real targets would "
            "be skipped until the TTL expired"
        )

    def test_failure_is_reported_to_the_caller(self):
        class _FailingService:
            def filter_unrecorded(
                self, target_ids, expected_counts=None, on_query_error=None
            ):
                if on_query_error is not None:
                    on_query_error(RuntimeError("timeout"))
                return set()

        store = self._store(_FailingService())
        seen = []

        store.filter_unrecorded({"t1"}, {"t1": 1}, on_query_error=seen.append)

        assert len(seen) == 1, (
            "the caller must hear about the failure; otherwise it reads the empty "
            "set as 'nothing to do' and advances its checkpoint"
        )

    def test_successful_query_still_caches_positive_verdicts(self):
        """The cache must keep working: it is what makes re-selection cheap."""

        class _Service:
            def filter_unrecorded(
                self, target_ids, expected_counts=None, on_query_error=None
            ):
                return {"t2"}

        store = self._store(_Service())

        result = store.filter_unrecorded({"t1", "t2"}, {"t1": 1, "t2": 1})

        assert result == {"t2"}
        assert ("t1", 1) in store._recorded_until
        assert ("t2", 1) not in store._recorded_until
