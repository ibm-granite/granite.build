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

"""``gbserver lineage-init``: seed the checkpoint without running the watcher.

The point of the command is what it does *not* do -- it must never construct a
LineageWatcher, because the whole reason it exists is that seeding via
``lineage-watch --base-build-id`` also starts recording for as long as the
operator leaves the process up.

All assert against ``result.output`` (combined stdout+stderr): CliRunner mixes the
streams by default on Click 8.1.x, where accessing ``result.stderr`` raises.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gbserver.commands import command_lineage_init
from gbserver.lineage.lineage_reconciler import (
    LINEAGE_WATCHER_CHECKPOINT_KEY,
    LINEAGE_WATCHER_DROPPED_KEY,
)
from gbserver.lineage.lineage_seeding import LineageSeedError

MODULE = "gbserver.commands.command_lineage_init"

_CHECKPOINT = {
    "build_id": "b-1",
    "created_time": "2026-01-01T00:00:00+00:00",
    "version": 2,
}


@pytest.mark.live("storage", "lineage")
class TestLineageInitCommand:
    """Seeding, idempotency and the read-only ``--show`` path."""

    @pytest.fixture(autouse=True)
    def _stub_storage(self):
        self.storage = MagicMock()
        self._stored: dict = {}
        self.storage.kv_pair_storage.get_value.side_effect = self._stored.get
        with patch(f"{MODULE}.get_admin_storage", return_value=self.storage):
            yield

    def _run(self, *args):
        return CliRunner().invoke(command_lineage_init.cli, list(args))

    def test_seeding_writes_the_checkpoint_and_never_starts_a_watcher(self):
        """The command seeds and exits; recording is left to the deployed watcher.

        This is the whole reason the command exists, so it is pinned explicitly:
        seeding through ``lineage-watch --base-build-id`` also runs the watcher,
        which records for however long the operator leaves the process up.
        """
        self._stored.clear()
        with patch(f"{MODULE}.seed_if_absent", return_value=True) as seed:
            self._stored[LINEAGE_WATCHER_CHECKPOINT_KEY] = _CHECKPOINT
            result = self._run("--build-id", "from-latest")

        assert result.exit_code == 0, result.output
        seed.assert_called_once_with(self.storage, "from-latest", force=False)
        assert "Seeded" in result.output
        assert "b-1" in result.output, "the operator must see the anchor it landed on"

    def test_an_already_seeded_environment_is_a_no_op_not_a_failure(self):
        """Re-running on an initialized environment must exit 0 and change nothing.

        Seed-if-absent is the safe default and the command is expected to be run
        from setup scripts, so a second run has to be harmless rather than an
        error someone has to special-case.
        """
        self._stored[LINEAGE_WATCHER_CHECKPOINT_KEY] = _CHECKPOINT
        with patch(f"{MODULE}.seed_if_absent", return_value=False):
            result = self._run("--build-id", "from-latest")

        assert result.exit_code == 0, result.output
        assert "already set" in result.output
        assert "--force" in result.output, "say how to override it"

    def test_show_reports_the_current_checkpoint_without_writing(self):
        """``--show`` is read-only: it must not call the seeding path at all."""
        self._stored[LINEAGE_WATCHER_CHECKPOINT_KEY] = _CHECKPOINT
        with patch(f"{MODULE}.seed_if_absent") as seed:
            result = self._run("--build-id", "from-latest", "--show")

        assert result.exit_code == 0, result.output
        assert "b-1" in result.output
        seed.assert_not_called()
        self.storage.kv_pair_storage.set_value.assert_not_called()

    def test_show_says_so_when_nothing_is_seeded_yet(self):
        """The unseeded state is the one an operator is diagnosing; name it."""
        self._stored.clear()
        result = self._run("--build-id", "from-latest", "--show")

        assert result.exit_code == 0, result.output
        assert "No lineage checkpoint" in result.output
        assert "records nothing" in result.output

    def test_force_is_passed_through(self):
        """``--force`` must reach seed_if_absent, not be silently dropped."""
        self._stored[LINEAGE_WATCHER_CHECKPOINT_KEY] = _CHECKPOINT
        with patch(f"{MODULE}.seed_if_absent", return_value=True) as seed:
            result = self._run("--build-id", "b-2", "--force")

        assert result.exit_code == 0, result.output
        seed.assert_called_once_with(self.storage, "b-2", force=True)

    def test_an_empty_anchor_is_rejected(self):
        """An empty string passes click's required check but resolves to no anchor.

        Same guard as lineage-watch: without it the command reports success while
        leaving the environment unseeded, which is the confusing failure the whole
        command is meant to remove.
        """
        with patch(f"{MODULE}.seed_if_absent") as seed:
            result = self._run("--build-id", "   ")

        assert result.exit_code != 0
        assert "empty value" in result.output
        seed.assert_not_called()

    def test_clear_dropped_targets_empties_the_durable_set(self):
        """The drop set is written empty, not deleted.

        ``LineageWatcher._load_dropped`` reads ``value.get("target_ids", [])``, so
        keeping the shape is what makes the cleared state readable on its next
        start rather than a missing key it has to tolerate.
        """
        self._stored[LINEAGE_WATCHER_DROPPED_KEY] = {"target_ids": ["t-1", "t-2"]}
        result = self._run("--clear-dropped-targets")

        assert result.exit_code == 0, result.output
        self.storage.kv_pair_storage.set_value.assert_called_once_with(
            LINEAGE_WATCHER_DROPPED_KEY, {"target_ids": []}
        )
        assert "t-1" in result.output and "t-2" in result.output
        assert "Restart" in result.output, (
            "the drop set is loaded once at start(), so a running watcher keeps "
            "skipping these targets until restarted -- the operator must be told"
        )

    def test_clearing_dropped_targets_leaves_the_checkpoint_alone(self):
        """Clearing is a complete operation on its own; --build-id stays optional."""
        self._stored[LINEAGE_WATCHER_DROPPED_KEY] = {"target_ids": ["t-1"]}
        with patch(f"{MODULE}.seed_if_absent") as seed:
            result = self._run("--clear-dropped-targets")

        assert result.exit_code == 0, result.output
        seed.assert_not_called(), "no anchor was requested, so none may be written"

    def test_clearing_an_empty_drop_set_is_a_no_op(self):
        """Nothing to clear must not write, so a scripted run stays harmless."""
        self._stored.clear()
        result = self._run("--clear-dropped-targets")

        assert result.exit_code == 0, result.output
        assert "nothing to clear" in result.output
        self.storage.kv_pair_storage.set_value.assert_not_called()

    def test_clear_and_seed_compose_in_one_invocation(self):
        """Both flags together: clear the drop set AND move the anchor."""
        self._stored[LINEAGE_WATCHER_DROPPED_KEY] = {"target_ids": ["t-1"]}
        self._stored[LINEAGE_WATCHER_CHECKPOINT_KEY] = _CHECKPOINT
        with patch(f"{MODULE}.seed_if_absent", return_value=True) as seed:
            result = self._run(
                "--build-id", "b-2", "--force", "--clear-dropped-targets"
            )

        assert result.exit_code == 0, result.output
        seed.assert_called_once_with(self.storage, "b-2", force=True)
        assert "Cleared 1 dropped target" in result.output

    def test_a_failed_seed_leaves_the_drop_set_intact(self):
        """A non-zero exit must not have mutated durable state.

        The clear and the seed are separate kv_pairs writes with no shared
        transaction, so clearing first meant a failed seed still wiped the drop
        set -- targets silently un-dropped by a run that reported failure, with
        nothing to roll it back. Clearing happens only after the seed lands.
        """
        self._stored[LINEAGE_WATCHER_DROPPED_KEY] = {"target_ids": ["t-1"]}
        with patch(
            f"{MODULE}.seed_if_absent", side_effect=LineageSeedError("no such build")
        ):
            result = self._run("--build-id", "b-nope", "--clear-dropped-targets")

        assert result.exit_code != 0
        assert "no such build" in result.output
        self.storage.kv_pair_storage.set_value.assert_not_called()

    def test_show_reports_dropped_targets(self):
        """A dropped target is skipped every scan; --show must surface it."""
        self._stored[LINEAGE_WATCHER_CHECKPOINT_KEY] = _CHECKPOINT
        self._stored[LINEAGE_WATCHER_DROPPED_KEY] = {"target_ids": ["t-9"]}
        result = self._run("--show")

        assert result.exit_code == 0, result.output
        assert "t-9" in result.output
        assert "permanently skipped" in result.output
        self.storage.kv_pair_storage.set_value.assert_not_called()

    def test_no_flags_at_all_is_an_error_not_a_silent_no_op(self):
        """Bare invocation must say what to pass, not exit 0 having done nothing."""
        result = self._run()

        assert result.exit_code != 0
        assert "Nothing to do" in result.output

    def test_an_unresolvable_anchor_fails_cleanly(self):
        """A bad build id must surface as a CLI error, not a traceback."""
        with patch(
            f"{MODULE}.seed_if_absent",
            side_effect=LineageSeedError("No build with a creation time found"),
        ):
            result = self._run("--build-id", "nope")

        assert result.exit_code != 0
        assert "No build with a creation time found" in result.output
