"""``gbserver lineage-index-init``: seed, keep, force and show the indexer checkpoint.

Drives the real indexer seeding (W&B source, whose API is a mock) against an
in-memory kv store, so every path asserts on what actually lands in gb_kv_pairs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from gbserver.commands import command_lineage_index_init
from gbserver.lineage import indexer as idx

MODULE = "gbserver.commands.command_lineage_index_init"
KEY = "lineage_index_checkpoint:wandb"
D1 = "2026-01-01T00:00:00Z"
D2 = "2026-02-01T00:00:00Z"


class _KV:
    def __init__(self):
        self.values = {}

    def get_value(self, key):
        return self.values.get(key)

    def set_value(self, key, value):
        self.values[key] = value

    def compare_and_set(self, key, expected, value):
        if self.values.get(key) != expected:
            return False
        self.set_value(key, value)
        return True


@pytest.fixture(name="env")
def env_fixture():
    storage = SimpleNamespace(kv_pair_storage=_KV(), lineage_row_storage=MagicMock())
    api = MagicMock()
    latest = SimpleNamespace(created_at="2026-03-01T00:00:00Z")
    indexer = idx.WandBLineageIndexer(sink=MagicMock(), rows=MagicMock(), api=api)
    with (
        patch(f"{MODULE}.get_admin_storage", return_value=storage),
        patch(f"{MODULE}.resolve_indexer_source", return_value="lineage_store"),
        patch(f"{MODULE}.create_indexer", return_value=indexer),
        patch.object(indexer, "_latest_job", return_value=latest),
    ):
        yield storage


def _run(*args):
    return CliRunner().invoke(command_lineage_index_init.cli, list(args))


@pytest.mark.parametrize(
    "spec, expected",
    [
        (D1, D1),
        ("from-latest", "2026-03-01T00:00:00Z"),
        ("all", "1970-01-01T00:00:00Z"),
    ],
)
def test_seeds_the_checkpoint(env, spec, expected):
    result = _run("--base-timestamp", spec)
    assert result.exit_code == 0, result.output
    assert env.kv_pair_storage.values[KEY]["timestamp"] == expected
    assert "Seeded" in result.output


def test_keeps_an_existing_checkpoint(env):
    env.kv_pair_storage.set_value(KEY, {"timestamp": D1})
    result = _run("--base-timestamp", D2)
    assert result.exit_code == 0, result.output
    assert env.kv_pair_storage.values[KEY]["timestamp"] == D1
    assert "kept it" in result.output


def test_force_replaces_an_existing_checkpoint(env):
    env.kv_pair_storage.set_value(KEY, {"timestamp": D1})
    result = _run("--base-timestamp", D2, "--force")
    assert result.exit_code == 0, result.output
    assert env.kv_pair_storage.values[KEY]["timestamp"] == D2


def test_a_failed_forced_seed_keeps_the_existing_checkpoint(env):
    env.kv_pair_storage.set_value(KEY, {"timestamp": D1})
    result = _run("--base-timestamp", "yesterday", "--force")
    assert result.exit_code != 0
    assert env.kv_pair_storage.values[KEY]["timestamp"] == D1


def test_rebuild_deletes_the_rows_and_reseeds_at_all(env):
    env.kv_pair_storage.set_value(KEY, {"timestamp": D1})
    result = _run("--rebuild")
    assert result.exit_code == 0, result.output
    env.lineage_row_storage.delete_all_rows.assert_called_once()
    env.lineage_row_storage.delete_table.assert_not_called()
    assert env.kv_pair_storage.values[KEY]["timestamp"] == "1970-01-01T00:00:00Z"


@pytest.mark.parametrize(
    "args", [("--rebuild", "--show"), ("--rebuild", "--base-timestamp", D1)]
)
def test_rebuild_rejects_a_conflicting_flag(env, args):
    result = _run(*args)
    assert result.exit_code != 0
    env.lineage_row_storage.delete_all_rows.assert_not_called()
    assert env.kv_pair_storage.values == {}


def test_show_prints_without_writing(env):
    assert "No lineage index checkpoint" in _run("--show").output
    env.kv_pair_storage.set_value(KEY, {"timestamp": D1})
    result = _run("--show")
    assert result.exit_code == 0
    assert D1 in result.output
    assert env.kv_pair_storage.values == {KEY: {"timestamp": D1}}


@pytest.mark.parametrize(
    "args", [("--show", "--base-timestamp", D1), ("--show", "--force")]
)
def test_show_must_be_alone(env, args):
    result = _run(*args)
    assert result.exit_code != 0
    assert "read-only" in result.output
    assert env.kv_pair_storage.values == {}


@pytest.mark.parametrize("args", [("--base-timestamp", "  "), ("--force",), ()])
def test_rejects_an_empty_or_missing_seed(env, args):
    result = _run(*args)
    assert result.exit_code != 0
    assert env.kv_pair_storage.values == {}


def test_a_source_with_no_indexer_says_so():
    with (
        patch(f"{MODULE}.resolve_indexer_source", return_value="bogus"),
        patch(f"{MODULE}.create_indexer", return_value=None),
    ):
        result = _run("--base-timestamp", "all")
    assert result.exit_code != 0
    assert "has no indexer" in result.output


def test_from_latest_on_an_empty_source_fails_without_the_flag(env):
    indexer = command_lineage_index_init.create_indexer("lineage_store")
    with patch.object(indexer, "_latest_job", return_value=None):
        result = _run("--base-timestamp", "from-latest")
    assert result.exit_code != 0
    assert KEY not in env.kv_pair_storage.values


def test_from_latest_on_an_empty_source_seeds_all_with_the_flag(env):
    """A fresh install (no job yet) indexes its first jobs instead of none."""
    indexer = command_lineage_index_init.create_indexer("lineage_store")
    with patch.object(indexer, "_latest_job", return_value=None):
        result = _run("--empty-from-beginning", "--base-timestamp", "from-latest")
    assert result.exit_code == 0, result.output
    assert env.kv_pair_storage.values[KEY]["timestamp"] == "1970-01-01T00:00:00Z"


def test_empty_from_beginning_does_not_change_a_non_empty_source(env):
    result = _run("--empty-from-beginning", "--base-timestamp", "from-latest")
    assert result.exit_code == 0, result.output
    assert env.kv_pair_storage.values[KEY]["timestamp"] == "2026-03-01T00:00:00Z"
