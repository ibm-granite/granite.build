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

"""Unit tests for ``gbserver lineage-seed``.

The watcher records nothing until its ``gb_status`` checkpoint exists (that is
the point of not auto-seeding it), so this command is the only supported way to
choose where centralized recording starts. These tests stub the admin storage and
the lineage store, so they run in CI without a cluster, PostgreSQL, or wandb
credentials.

They assert against ``result.output`` (combined stdout+stderr): CliRunner mixes
the streams by default on Click 8.1.x, where accessing ``result.stderr`` raises.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from gbserver.commands.command_lineage_seed import cli
from gbserver.lineage.lineage_reconciler import LINEAGE_WATCHER_CHECKPOINT_KEY
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status

_BASE = datetime(2026, 1, 1, 0, 0, 0)


def _target(build_id: str, uuid: str, finished_at: datetime) -> StoredTargetRun:
    return StoredTargetRun(
        uuid=uuid,
        build_id=build_id,
        environment_uri="env://test",
        status=Status.SUCCESS,
        finished_at=finished_at,
    )


class _StubStatusStorage:
    def __init__(self):
        self._values: dict = {}

    def get_value(self, key):
        return self._values.get(key)

    def set_value(self, key, value):
        self._values[key] = value


class _Harness:
    """Stub admin storage + lineage store, wired into the command's lookups."""

    def __init__(self, targets, records_centralized_lineage: bool = True):
        self.storage = MagicMock()
        self.storage.status_storage = _StubStatusStorage()

        def _get_by_where(where, query_control=None):
            matching = [t for t in targets if t.status.name == where["status"]]
            if "build_id" in where:
                matching = [t for t in matching if t.build_id == where["build_id"]]
            ordered = sorted(matching, key=lambda t: t.finished_at, reverse=True)
            if query_control is not None and query_control.pagination is not None:
                p = query_control.pagination
                start = p.index * p.size
                return ordered[start : start + p.size]
            return ordered

        self.storage.target_storage.get_by_where.side_effect = _get_by_where
        store = MagicMock()
        store.records_centralized_lineage = records_centralized_lineage
        self.store = store

    def __enter__(self):
        self._patches = [
            patch(
                "gbserver.commands.command_lineage_seed.get_admin_storage",
                return_value=self.storage,
            ),
            patch(
                "gbserver.commands.command_lineage_seed.get_lineage_store",
                return_value=self.store,
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False

    @property
    def checkpoint(self):
        return self.storage.status_storage.get_value(LINEAGE_WATCHER_CHECKPOINT_KEY)


class TestLineageSeedCommand:
    def test_from_latest_anchors_at_newest_successful_target(self):
        """--from-latest means "record from now on": the newest finished target."""
        newest_at = _BASE + timedelta(seconds=30)
        targets = [_target("b1", "t1", _BASE), _target("b2", "t2", newest_at)]

        with _Harness(targets) as h:
            result = CliRunner().invoke(cli, ["--from-latest"])

            assert result.exit_code == 0, result.output
            assert h.checkpoint == {
                "build_id": "b2",
                "finished_at": newest_at.isoformat(),
            }

    def test_build_id_anchors_at_that_builds_newest_target(self):
        """--build-id anchors at a chosen build, not at whatever is newest."""
        targets = [
            _target("b1", "t1", _BASE),
            _target("b1", "t2", _BASE + timedelta(seconds=5)),
            _target("b2", "t3", _BASE + timedelta(seconds=30)),
        ]

        with _Harness(targets) as h:
            result = CliRunner().invoke(cli, ["--build-id", "b1"])

            assert result.exit_code == 0, result.output
            # b1's newest target, not b2's newer one.
            assert h.checkpoint == {
                "build_id": "b1",
                "finished_at": (_BASE + timedelta(seconds=5)).isoformat(),
            }

    def test_all_anchors_before_all_history(self):
        """--all anchors older than any real target, so nothing is excluded."""
        with _Harness([_target("b1", "t1", _BASE)]) as h:
            result = CliRunner().invoke(cli, ["--all"])

            assert result.exit_code == 0, result.output
            assert datetime.fromisoformat(h.checkpoint["finished_at"]) < _BASE

    def test_exactly_one_mode_is_required(self):
        """Placing the watermark is a deliberate choice, so no mode is implied."""
        with _Harness([_target("b1", "t1", _BASE)]) as h:
            no_mode = CliRunner().invoke(cli, [])
            assert no_mode.exit_code != 0
            assert "exactly one of" in no_mode.output

            two_modes = CliRunner().invoke(cli, ["--from-latest", "--all"])
            assert two_modes.exit_code != 0
            assert "exactly one of" in two_modes.output

            assert h.checkpoint is None

    def test_existing_checkpoint_is_not_overwritten_without_force(self):
        """Re-seeding moves the watermark, so it must not happen by accident.

        Forward would skip lineage that was never recorded; backward would
        re-drive what is already recorded. Neither is a safe default.
        """
        existing = {"build_id": "b0", "finished_at": _BASE.isoformat()}
        newest_at = _BASE + timedelta(seconds=30)

        with _Harness([_target("b2", "t2", newest_at)]) as h:
            h.storage.status_storage.set_value(LINEAGE_WATCHER_CHECKPOINT_KEY, existing)

            result = CliRunner().invoke(cli, ["--from-latest"])

            assert result.exit_code != 0
            assert "already exists" in result.output
            assert h.checkpoint == existing

    def test_force_overwrites_an_existing_checkpoint(self):
        newest_at = _BASE + timedelta(seconds=30)

        with _Harness([_target("b2", "t2", newest_at)]) as h:
            h.storage.status_storage.set_value(
                LINEAGE_WATCHER_CHECKPOINT_KEY,
                {"build_id": "b0", "finished_at": _BASE.isoformat()},
            )

            result = CliRunner().invoke(cli, ["--from-latest", "--force"])

            assert result.exit_code == 0, result.output
            assert h.checkpoint == {
                "build_id": "b2",
                "finished_at": newest_at.isoformat(),
            }

    def test_no_successful_target_to_anchor_at_is_an_error(self):
        """Writing a checkpoint with nothing to anchor it to would be a guess."""
        with _Harness([]) as h:
            result = CliRunner().invoke(cli, ["--from-latest"])

            assert result.exit_code != 0
            assert "No successful target" in result.output
            assert h.checkpoint is None

    def test_unknown_build_id_is_an_error(self):
        with _Harness([_target("b1", "t1", _BASE)]) as h:
            result = CliRunner().invoke(cli, ["--build-id", "nope"])

            assert result.exit_code != 0
            assert "build nope" in result.output
            assert h.checkpoint is None

    def test_no_op_store_is_not_seeded(self):
        """With no centralized sink the watcher's recording leaf is a no-op, so a
        checkpoint would have nothing to drive."""
        with _Harness(
            [_target("b1", "t1", _BASE)], records_centralized_lineage=False
        ) as h:
            result = CliRunner().invoke(cli, ["--from-latest"])

            assert result.exit_code == 0, result.output
            assert h.checkpoint is None
