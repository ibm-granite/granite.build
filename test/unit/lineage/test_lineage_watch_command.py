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

"""Anchor-flag guards in ``gbserver lineage-watch``.

The dispatch below the guards tests truthiness (``if force_build_id:``) while the
mutual-exclusivity guard tests ``is not None``. An empty string sits between the
two: it passes the guard and matches neither branch, so without an explicit
rejection the watcher starts unseeded while the operator believes an anchor was
applied. These tests pin that the empty value is rejected instead of silently
recording nothing, and that a real anchor still reaches ``seed_if_absent``.

All assert against ``result.output`` (combined stdout+stderr): CliRunner mixes the
streams by default on Click 8.1.x, where accessing ``result.stderr`` raises.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gbserver.commands import command_lineage_watch

MODULE = "gbserver.commands.command_lineage_watch"


class _StubbedCommand:
    """Stubs everything past the guards so only flag handling is exercised.

    ``stop_event.wait`` returns immediately, so a run that gets past the guards
    exits cleanly instead of blocking the test on the watcher's main-thread wait.
    """

    @pytest.fixture(autouse=True)
    def _stub_watcher(self):
        store = MagicMock()
        store.records_centralized_lineage = True
        with (
            patch(f"{MODULE}.get_lineage_store", return_value=store),
            patch(f"{MODULE}.get_admin_storage", return_value=MagicMock()),
            patch(f"{MODULE}.seed_if_absent") as seed,
            patch(f"{MODULE}.LineageWatcher") as watcher_cls,
        ):
            watcher_cls.return_value.stop_event.wait.return_value = None
            self._seed = seed
            yield

    def _invoke(self, *argv):
        return CliRunner().invoke(command_lineage_watch.cli, list(argv))


class TestEmptyAnchorRejected(_StubbedCommand):
    @pytest.mark.parametrize("flag", ["--base-build-id", "--force-build-id"])
    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_value_errors_and_does_not_seed(self, flag, value):
        result = self._invoke(flag, value)

        assert result.exit_code != 0, (
            f"{flag} with an empty value should exit non-zero, "
            f"got: {result.output!r}"
        )
        assert "empty value" in result.output
        # The failure mode this guards against is a silent no-op: exiting cleanly
        # without ever anchoring the checkpoint.
        self._seed.assert_not_called()


class TestValidAnchorStillDispatches(_StubbedCommand):
    def test_base_build_id_seeds_if_absent(self):
        result = self._invoke("--base-build-id", "from-latest")

        assert result.exit_code == 0, result.output
        self._seed.assert_called_once()
        assert self._seed.call_args.args[1] == "from-latest"
        assert not self._seed.call_args.kwargs.get("force", False)

    def test_force_build_id_seeds_with_force(self):
        result = self._invoke("--force-build-id", "some-build")

        assert result.exit_code == 0, result.output
        self._seed.assert_called_once()
        assert self._seed.call_args.args[1] == "some-build"
        assert self._seed.call_args.kwargs["force"] is True

    def test_both_flags_remain_mutually_exclusive(self):
        result = self._invoke("--base-build-id", "all", "--force-build-id", "b")

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output
        self._seed.assert_not_called()
