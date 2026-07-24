# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
from services import file_service


@pytest.fixture
def tmp_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setitem(file_service.CONFIG, "UPLOAD_DIR", str(tmp_path))
    return str(tmp_path)


async def test_stream_split_jsonl_conserves_records(tmp_path):
    src = tmp_path / "src.jsonl"
    rows = [{"input": f"q{i}", "output": f"a{i}"} for i in range(100)]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    train = tmp_path / "t.jsonl"
    val = tmp_path / "v.jsonl"

    train_n, val_n = await file_service.stream_split_jsonl(
        src_path=str(src),
        train_path=str(train),
        val_path=str(val),
        train_set_percentage=80,
        seed="seed-123",
    )

    assert train_n + val_n == 100
    assert train_n > 0 and val_n > 0  # ratio produced both sides


async def test_stream_split_jsonl_is_reproducible(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text("\n".join(json.dumps({"input": str(i)}) for i in range(50)) + "\n")

    async def run(suffix):
        t = tmp_path / f"t_{suffix}.jsonl"
        v = tmp_path / f"v_{suffix}.jsonl"
        return await file_service.stream_split_jsonl(
            src_path=str(src),
            train_path=str(t),
            val_path=str(v),
            train_set_percentage=70,
            seed="fixed-seed",
        )

    assert await run("a") == await run("b")  # same seed -> same split counts


async def test_jsonl_mapping_keeps_only_target_columns(tmp_path):
    src = tmp_path / "src.jsonl"
    src.write_text(json.dumps({"q": "hi", "a": "yo", "junk": 1}) + "\n")
    dest = tmp_path / "out.jsonl"

    count = await file_service.remap_jsonl_file(
        str(src), str(dest), {"input": "q", "output": "a"}
    )

    assert count == 1
    rec = json.loads(dest.read_text().strip())
    assert rec == {"input": "hi", "output": "yo"}  # junk dropped, source->target


async def test_get_jsonl_data_limit_stops_early(tmp_upload_dir):
    path = os.path.join(tmp_upload_dir, "big.jsonl")
    with open(path, "w") as f:
        for i in range(1000):
            f.write(json.dumps({"i": i}) + "\n")

    rows = await file_service.get_jsonl_data("big.jsonl", limit=10)

    assert len(rows) == 10
    assert rows[0] == {"i": 0} and rows[9] == {"i": 9}
