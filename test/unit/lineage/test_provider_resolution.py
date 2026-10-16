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

"""Tests for resolving which lineage provider is configured.

The bug class these exist for: the write side used to build the W&B sink for *any*
value it did not recognize, while the read side raised on the same value. A typo
therefore recorded lineage to W&B while the UI reported lineage was unavailable --
two components disagreeing about the same setting, with no error anywhere. So what
is pinned here is that both sides accept exactly one set of values, and that
anything else fails on both.
"""

import pytest

from gbserver.lineage import jobstats
from gbserver.lineage.jobstats import (
    VALID_LINEAGE_PROVIDERS,
    UnknownLineageProvider,
    _resolve_lineage_provider,
)
from gbserver.lineage.openlineage_service import LineageServiceFactory


class TestDefaults:
    def test_standalone_defaults_to_none(self, monkeypatch):
        # Still "none", not "db". The index is fully wired on both sides, so this
        # is a deliberate hold on the rollout, not a missing piece: a deployment
        # opts in with GBSERVER_LINEAGE_PROVIDER=db. This assertion is what makes
        # flipping the default a conscious edit rather than an accident.
        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")
        monkeypatch.delenv("GBSERVER_LINEAGE_PROVIDER", raising=False)
        assert _resolve_lineage_provider() == "none"

    def test_non_standalone_defaults_to_wandb(self, monkeypatch):
        monkeypatch.setenv("GB_ENVIRONMENT", "STAGING")
        monkeypatch.delenv("GBSERVER_LINEAGE_PROVIDER", raising=False)
        assert _resolve_lineage_provider() == "wandb"

    def test_resolution_does_not_leak_into_the_environment(self, monkeypatch):
        import os

        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")
        monkeypatch.delenv("GBSERVER_LINEAGE_PROVIDER", raising=False)
        _resolve_lineage_provider()
        assert os.environ.get("GBSERVER_LINEAGE_PROVIDER") is None


class TestExplicitValues:
    @pytest.mark.parametrize("provider", VALID_LINEAGE_PROVIDERS)
    def test_every_valid_provider_resolves_to_itself(self, provider, monkeypatch):
        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", provider)
        assert _resolve_lineage_provider() == provider

    def test_db_is_a_valid_provider(self, monkeypatch):
        # The value that makes standalone serve a graph rather than a 404.
        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", "db")
        assert _resolve_lineage_provider() == "db"

    def test_surrounding_whitespace_is_tolerated(self, monkeypatch):
        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", "  db  ")
        assert _resolve_lineage_provider() == "db"

    def test_an_explicit_value_beats_the_standalone_default(self, monkeypatch):
        monkeypatch.setenv("GB_ENVIRONMENT", "STANDALONE")
        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", "wandb")
        assert _resolve_lineage_provider() == "wandb"


class TestUnknownProviderFailsLoudly:
    """A typo must not degrade -- in either direction."""

    def test_a_typo_raises(self, monkeypatch):
        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", "wandbb")
        with pytest.raises(UnknownLineageProvider):
            _resolve_lineage_provider()

    def test_the_error_names_the_valid_values(self, monkeypatch):
        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", "nope")
        with pytest.raises(UnknownLineageProvider) as excinfo:
            _resolve_lineage_provider()
        message = str(excinfo.value)
        assert "nope" in message
        for provider in VALID_LINEAGE_PROVIDERS:
            assert provider in message

    def test_an_empty_value_is_not_silently_a_default(self, monkeypatch):
        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", "")
        with pytest.raises(UnknownLineageProvider):
            _resolve_lineage_provider()

    @pytest.mark.live("lineage")
    def test_an_unknown_provider_no_longer_builds_the_wandb_sink(self, monkeypatch):
        # The regression this whole change is about: "db" (and any unrecognized
        # value) used to fall through to WandBLineageStore, sending lineage to a
        # backend the operator never configured.
        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", "definitely-not-a-provider")
        jobstats.reset_lineage_store()
        try:
            with pytest.raises(UnknownLineageProvider):
                jobstats.get_lineage_store()
        finally:
            jobstats.reset_lineage_store()


class TestBothSidesAgree:
    """The read and write sides must accept the same provider set.

    They resolve through one function but build from two registries, so this is
    what keeps the registries from drifting apart -- the split this change closed.
    """

    def test_every_valid_provider_builds_a_read_service(self):
        for provider in VALID_LINEAGE_PROVIDERS:
            assert LineageServiceFactory.create(provider) is not None

    def test_every_valid_provider_builds_a_write_store(self, monkeypatch):
        for provider in VALID_LINEAGE_PROVIDERS:
            monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", provider)
            jobstats.reset_lineage_store()
            try:
                assert jobstats.get_lineage_store() is not None
            finally:
                jobstats.reset_lineage_store()

    def test_the_read_side_rejects_what_the_write_side_rejects(self):
        with pytest.raises(ValueError):
            LineageServiceFactory.create("definitely-not-a-provider")


class TestDbProviderWiring:
    """ "db" wires the index both ways: the sink writes it, the service reads it."""

    def test_db_serves_the_graph_service(self):
        from gbserver.lineage.db_service import DBLineageService

        assert isinstance(LineageServiceFactory.create("db"), DBLineageService)

    @pytest.mark.live("lineage")
    def test_db_builds_the_index_sink(self, monkeypatch):
        from gbserver.lineage.db_jobstats import DBLineageStore

        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", "db")
        jobstats.reset_lineage_store()
        try:
            assert isinstance(jobstats.get_lineage_store(), DBLineageStore)
        finally:
            jobstats.reset_lineage_store()

    @pytest.mark.live("lineage")
    def test_db_records_centralized_lineage(self, monkeypatch):
        # What opens the gate: lineage-watch exits immediately for a store that
        # reports False (command_lineage_watch.py:91), so the index would never be
        # populated. This is the assertion that the watcher now stays up.
        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", "db")
        jobstats.reset_lineage_store()
        try:
            assert jobstats.get_lineage_store().records_centralized_lineage is True
        finally:
            jobstats.reset_lineage_store()

    @pytest.mark.live("lineage")
    def test_db_does_not_build_the_wandb_sink(self, monkeypatch):
        monkeypatch.setenv("GBSERVER_LINEAGE_PROVIDER", "db")
        jobstats.reset_lineage_store()
        try:
            assert type(jobstats.get_lineage_store()).__name__ != "WandBLineageStore"
        finally:
            jobstats.reset_lineage_store()
