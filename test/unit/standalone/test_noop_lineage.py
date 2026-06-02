"""Tests that lineage is disabled (noop) in standalone mode.

Verifies that:
1. The standalone env defaults set GBSERVER_LINEAGE_PROVIDER=none
2. get_lineage_store() returns NoopLineageStore when provider is "none"
3. LineageServiceFactory.create("none") returns NoopLineageService
4. NoopLineageStore methods are safe no-ops with correct return types
"""

import importlib
import os

import pytest


class TestStandaloneLineageProviderDefault:
    """Verify standalone mode defaults GBSERVER_LINEAGE_PROVIDER to 'none'."""

    def test_standalone_sets_lineage_provider_none(self, monkeypatch):
        from gbserver.types import constants

        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")
        monkeypatch.delenv("GBSERVER_LINEAGE_PROVIDER", raising=False)
        importlib.reload(constants)

        try:
            assert os.environ.get("GBSERVER_LINEAGE_PROVIDER") == "none"
            assert constants.GBSERVER_LINEAGE_PROVIDER == "none"
        finally:
            importlib.reload(constants)

    def test_standalone_preserves_explicit_lineage_provider(self, monkeypatch):
        from gbserver.types import constants

        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")
        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", "wandb")
        importlib.reload(constants)

        try:
            assert os.environ.get("GBSERVER_LINEAGE_PROVIDER") == "wandb"
            assert constants.GBSERVER_LINEAGE_PROVIDER == "wandb"
        finally:
            importlib.reload(constants)


class TestGetLineageStoreStandalone:
    """Verify get_lineage_store() returns NoopLineageStore when provider is 'none'."""

    def test_returns_noop_store(self, monkeypatch):
        from gbserver.lineage import jobstats
        from gbserver.lineage.noop_jobstats import NoopLineageStore
        from gbserver.types import constants

        monkeypatch.setattr(constants, "GBSERVER_LINEAGE_PROVIDER", "none")

        jobstats.reset_lineage_store()
        try:
            store = jobstats.get_lineage_store()
            assert isinstance(store, NoopLineageStore)
        finally:
            jobstats.reset_lineage_store()

    def test_singleton_is_reused(self, monkeypatch):
        from gbserver.lineage import jobstats
        from gbserver.types import constants

        monkeypatch.setattr(constants, "GBSERVER_LINEAGE_PROVIDER", "none")

        jobstats.reset_lineage_store()
        try:
            store1 = jobstats.get_lineage_store()
            store2 = jobstats.get_lineage_store()
            assert store1 is store2
        finally:
            jobstats.reset_lineage_store()


class TestNoopLineageStoreReturnValues:
    """Verify NoopLineageStore methods return correct types without side effects."""

    @pytest.fixture()
    def store(self):
        from gbserver.lineage.noop_jobstats import NoopLineageStore

        return NoopLineageStore()

    def test_add_jobstats_for_build(self, store):
        # Should not raise
        result = store.add_jobstats_for_build(storage=None, build_id="build-123")
        assert result is None

    def test_add_jobstats_for_build_target(self, store):
        result = store.add_jobstats_for_build_target(
            storage=None, build_id="build-123", target_id="target-456"
        )
        assert result is None

    def test_add_jobstats_for_original_artifact(self, store):
        result = store.add_jobstats_for_original_artifact(artifact=None, sources=[])
        assert result is None

    def test_create_jobstats_for_target(self, store):
        events, events_dict = store.create_jobstats_for_target(
            storage=None, targetrun=None
        )
        assert events == []
        assert events_dict == {}

    def test_create_jobstats_for_original_artifact(self, store):
        result = store.create_jobstats_for_original_artifact(artifact=None, sources=[])
        assert result == {}

    def test_count_release_ids(self, store):
        count = store.count_release_ids("release-abc")
        assert count == 0

    def test_count_release_ids_with_target(self, store):
        count = store.count_release_ids("release-abc", target_id="target-1")
        assert count == 0

    def test_does_release_id_exist(self, store):
        exists = store.does_release_id_exist("release-abc", expected_count=1)
        assert exists is False


class TestNoopLineageServiceFactory:
    """Verify LineageServiceFactory returns NoopLineageService for 'none'."""

    def test_factory_creates_noop(self):
        from gbserver.lineage.openlineage_service import (
            LineageServiceFactory,
            NoopLineageService,
        )

        service = LineageServiceFactory.create("none")
        assert isinstance(service, NoopLineageService)

    def test_noop_service_emit_event(self):
        from gbserver.lineage.openlineage_service import LineageServiceFactory

        service = LineageServiceFactory.create("none")
        # Should not raise
        service.emit_event({"eventType": "COMPLETE", "run": {"runId": "x"}})

    def test_noop_service_search(self):
        from gbserver.lineage.openlineage_service import LineageServiceFactory

        service = LineageServiceFactory.create("none")
        total, results = service.search_lineage_by_tags(["tag=value"])
        assert total == 0
        assert results == []

    def test_noop_service_count_events(self):
        from gbserver.lineage.openlineage_service import LineageServiceFactory

        service = LineageServiceFactory.create("none")
        assert service.count_events_by_tags(["tag=value"]) == 0

    def test_noop_service_count_runs(self):
        from gbserver.lineage.openlineage_service import LineageServiceFactory

        service = LineageServiceFactory.create("none")
        assert service.count_runs_by_tags(["tag=value"]) == 0

    def test_noop_service_artifact_graph(self):
        from gbserver.lineage.openlineage_service import LineageServiceFactory

        service = LineageServiceFactory.create("none")
        result = service.get_artifact_graph(artifact_name="test")
        assert result is None
