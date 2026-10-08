import json
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from merge_shard_results import merge  # noqa: E402

MODEL = "ibm-granite_granite-4.2-30b-fp8"


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_merge_concatenates_disjoint_ids_across_shards(tmp_path):
    shard0, shard1 = tmp_path / "shard-0", tmp_path / "shard-1"
    rel = f"{MODEL}/non_live/BFCL_v4_simple_python_result.json"
    _write_jsonl(shard0 / "result" / rel, [{"id": "simple_python_1", "result": "a"}])
    _write_jsonl(shard1 / "result" / rel, [{"id": "simple_python_2", "result": "b"}])

    dest = tmp_path / "merged"
    merge([shard0, shard1], dest)

    merged = _read_jsonl(dest / "result" / rel)
    assert {e["id"] for e in merged} == {"simple_python_1", "simple_python_2"}


def test_merge_only_includes_paths_that_exist_in_some_shard(tmp_path):
    shard0, shard1 = tmp_path / "shard-0", tmp_path / "shard-1"
    rel_a = f"{MODEL}/non_live/BFCL_v4_simple_python_result.json"
    rel_b = f"{MODEL}/live/BFCL_v4_live_simple_result.json"
    _write_jsonl(shard0 / "result" / rel_a, [{"id": "simple_python_1", "result": "a"}])
    _write_jsonl(shard1 / "result" / rel_b, [{"id": "live_simple_1", "result": "b"}])

    dest = tmp_path / "merged"
    merge([shard0, shard1], dest)

    assert (dest / "result" / rel_a).is_file()
    assert (dest / "result" / rel_b).is_file()


def test_merge_raises_on_duplicate_id_across_shards(tmp_path):
    shard0, shard1 = tmp_path / "shard-0", tmp_path / "shard-1"
    rel = f"{MODEL}/non_live/BFCL_v4_simple_python_result.json"
    _write_jsonl(shard0 / "result" / rel, [{"id": "simple_python_1", "result": "a"}])
    _write_jsonl(shard1 / "result" / rel, [{"id": "simple_python_1", "result": "b"}])

    with pytest.raises(AssertionError, match="simple_python_1"):
        merge([shard0, shard1], tmp_path / "merged")


def test_merge_copies_disjoint_memory_snapshot_subtrees(tmp_path):
    shard0, shard1 = tmp_path / "shard-0", tmp_path / "shard-1"
    snap_rel = f"{MODEL}/agentic/memory/kv/memory_snapshot"

    finance_dir = shard0 / "result" / snap_rel
    finance_dir.mkdir(parents=True)
    (finance_dir / "finance_final.json").write_text('{"k": "v0"}')
    (finance_dir / "prereq_checkpoints").mkdir()
    (
        finance_dir / "prereq_checkpoints" / "memory_kv_prereq_15-finance-0.json"
    ).write_text("{}")

    travel_dir = shard1 / "result" / snap_rel
    travel_dir.mkdir(parents=True)
    (travel_dir / "travel_final.json").write_text('{"k": "v1"}')

    dest = tmp_path / "merged"
    merge([shard0, shard1], dest)

    dest_snap = dest / "result" / snap_rel
    assert (dest_snap / "finance_final.json").read_text() == '{"k": "v0"}'
    assert (dest_snap / "travel_final.json").read_text() == '{"k": "v1"}'
    assert (
        dest_snap / "prereq_checkpoints" / "memory_kv_prereq_15-finance-0.json"
    ).is_file()


def test_merge_raises_on_colliding_memory_snapshot_file(tmp_path):
    shard0, shard1 = tmp_path / "shard-0", tmp_path / "shard-1"
    snap_rel = f"{MODEL}/agentic/memory/kv/memory_snapshot"

    for shard in (shard0, shard1):
        d = shard / "result" / snap_rel
        d.mkdir(parents=True)
        (d / "finance_final.json").write_text('{"k": "v"}')

    with pytest.raises(AssertionError, match="finance_final.json"):
        merge([shard0, shard1], tmp_path / "merged")


def test_merge_excludes_categories_absent_from_every_shard(tmp_path):
    # web_search is excluded at generate time (shard_test_ids.py), so no shard
    # ever produces a web_search result file -- merge should not invent one.
    shard0 = tmp_path / "shard-0"
    rel = f"{MODEL}/non_live/BFCL_v4_simple_python_result.json"
    _write_jsonl(shard0 / "result" / rel, [{"id": "simple_python_1", "result": "a"}])

    dest = tmp_path / "merged"
    merge([shard0], dest)

    assert not (
        dest / "result" / MODEL / "agentic" / "BFCL_v4_web_search_base_result.json"
    ).exists()
