"""Unit tests for build_sources.py: the quota maths, the rename, and the refusals."""

import json
from collections import Counter

import pytest
from build_sources import main, quotas


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _split(tmp_path, name, n, key="conversations"):
    return _write(
        tmp_path / f"{name}.jsonl",
        [{key: [{"role": "user", "content": f"{name}-{i}"}]} for i in range(n)],
    )


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_quotas_are_proportional_to_split_size():
    assert quotas([8486, 1398, 117], 1000) == [849, 140, 12]


def test_quotas_keep_at_least_one_row_of_a_tiny_split():
    """A 1% split rounding to zero would drop a domain from the subset silently."""
    assert quotas([10000, 1], 100) == [100, 1]


@pytest.mark.parametrize("target", [0, -1, 100, 10_000])
def test_quotas_take_everything_at_zero_or_beyond_the_corpus(target):
    assert quotas([60, 40], target) == [60, 40]


def test_rows_are_renamed_and_each_split_keeps_its_share(tmp_path, capsys):
    general = _split(tmp_path, "general", 90)
    tools = _split(tmp_path, "tools", 10)
    out = tmp_path / "out" / "train.jsonl"

    assert main(["x", str(out), "20", "7", str(general), str(tools)]) == 0

    rows = _read(out)
    assert len(rows) == 20
    assert all("messages" in r and "conversations" not in r for r in rows)
    domains = Counter(r["messages"][0]["content"].split("-")[0] for r in rows)
    assert domains == {"general": 18, "tools": 2}
    assert (
        f"GB_ARTIFACT_ID:corpus_source GB_ARTIFACT_PATH:{out}"
        in capsys.readouterr().out
    )


def test_the_selection_is_reproducible_from_the_seed(tmp_path):
    src = _split(tmp_path, "general", 200)
    a, b, c = (tmp_path / f"{n}.jsonl" for n in "abc")
    main(["x", str(a), "30", "1", str(src)])
    main(["x", str(b), "30", "1", str(src)])
    main(["x", str(c), "30", "2", str(src)])
    assert a.read_text() == b.read_text()
    assert a.read_text() != c.read_text()


def test_rows_already_spelled_messages_pass_through(tmp_path, capsys):
    src = _split(tmp_path, "general", 5, key="messages")
    main(["x", str(tmp_path / "o.jsonl"), "0", "1", str(src)])
    assert "renamed=0 already_messages=5 bad=0" in capsys.readouterr().out


def test_bad_rows_are_skipped_and_counted(tmp_path, capsys):
    src = tmp_path / "mixed.jsonl"
    src.write_text('{"conversations": []}\nnot json\n{"text": "no turns"}\n')
    main(["x", str(tmp_path / "o.jsonl"), "0", "1", str(src)])
    out = capsys.readouterr().out
    assert "unusable rows skipped: 2" in out
    assert "renamed=1 already_messages=0 bad=2" in out


def test_a_schema_change_is_refused(tmp_path):
    src = _write(tmp_path / "s.jsonl", [{"text": "a"}, {"text": "b"}])
    with pytest.raises(
        SystemExit, match="no row carried `conversations` or `messages`"
    ):
        main(["x", str(tmp_path / "o.jsonl"), "0", "1", str(src)])


def test_a_missing_source_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="source is not a file"):
        main(["x", str(tmp_path / "o.jsonl"), "0", "1", str(tmp_path / "nope.jsonl")])


def test_all_empty_sources_are_refused(tmp_path):
    src = _write(tmp_path / "e.jsonl", [])
    with pytest.raises(SystemExit, match="every source is empty"):
        main(["x", str(tmp_path / "o.jsonl"), "0", "1", str(src)])
