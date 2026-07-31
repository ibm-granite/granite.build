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

"""Unit tests for WandBLineageService offline handling.

These build the service via ``__new__`` (skipping ``wandb.login`` / network)
and drive ``emit_event`` against a stubbed run, asserting that offline mode
skips artifact registration with a single WARNING while online mode registers
artifacts. Guards issue #181 Task 2: offline mode must not surface as an ERROR
or raise.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage.wandb_service import WandBLineageService


def _service() -> WandBLineageService:
    """Construct the service without running __init__ (no wandb.login)."""
    service = WandBLineageService.__new__(WandBLineageService)
    service._runs = {}
    return service


def _event() -> dict:
    return {
        "run": {"runId": "run-1", "facets": {}},
        "job": {"name": "job-1", "facets": {}, "namespace": "ns"},
        "eventType": "START",
        "inputs": [{"name": "in-ds"}],
        "outputs": [{"name": "out-ds"}],
    }


def _make_run(offline: bool) -> MagicMock:
    run = MagicMock()
    run.settings = SimpleNamespace(mode="offline" if offline else "online")
    run.tags = []
    run.config = MagicMock()
    run.summary = {}
    return run


@pytest.mark.parametrize("mode_offline", [True, False])
def test_is_offline_reads_settings_mode(mode_offline):
    service = _service()
    run = _make_run(offline=mode_offline)
    assert service._is_offline(run) is mode_offline


def test_is_offline_falls_back_to_offline_attribute():
    service = _service()
    # No settings.mode available; fall back to run.offline.
    run = SimpleNamespace(offline=True)
    assert service._is_offline(run) is True
    run = SimpleNamespace(offline=False)
    assert service._is_offline(run) is False


def test_is_offline_defaults_to_online():
    service = _service()
    assert service._is_offline(object()) is False


@pytest.mark.parametrize("mode", ["online", "run", "shared"])
def test_is_offline_online_modes_are_live(mode):
    """Modes with a live backend are not treated as offline."""
    service = _service()
    run = SimpleNamespace(settings=SimpleNamespace(mode=mode))
    assert service._is_offline(run) is False


@pytest.mark.parametrize("mode", ["offline", "disabled", "dryrun", "something-new"])
def test_is_offline_non_live_modes_skip_artifacts(mode):
    """Any mode outside the online allowlist fails safe as offline.

    Guards the allowlist (vs. an "offline"-only denylist): a new or renamed
    non-live mode must skip artifact registration, not raise against a dead
    backend.
    """
    service = _service()
    run = SimpleNamespace(settings=SimpleNamespace(mode=mode))
    assert service._is_offline(run) is True


def test_offline_skips_artifact_registration_with_single_warning(caplog):
    service = _service()
    run = _make_run(offline=True)

    with (
        patch.object(service, "_get_run", return_value=run),
        patch.object(service, "_register_artifacts") as register,
        caplog.at_level(logging.WARNING, logger="gbserver.lineage.wandb_service"),
    ):
        service.emit_event(_event())

    register.assert_not_called()
    run.use_artifact.assert_not_called()
    run.log_artifact.assert_not_called()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "offline" in warnings[0].getMessage().lower()
    # Offline must not surface as an ERROR.
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_online_registers_artifacts():
    service = _service()
    run = _make_run(offline=False)

    with (
        patch.object(service, "_get_run", return_value=run),
        patch.object(service, "_register_artifacts") as register,
    ):
        service.emit_event(_event())

    register.assert_called_once()
    # The run is passed through to artifact registration.
    assert register.call_args.args[0] is run
