# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the fail-fast data/tokenizer check.

The point of ``data_check.py`` is that one cheap CPU pod reports *everything*
wrong with the dataset and tokenizer before ``train`` asks for a multi-node GPU
allocation. So the tests pin two things: that each failure mode the restored IBM
dataloader can hit is caught, and that several of them are reported together.

Fixtures are tiny real parquet/arrow files written in ``tmp_path`` (pyarrow is in
the step's venv); the tokenizer is a monkeypatched loader, since ``transformers``
is not a test-time dependency of the step.
"""

import copy
import json
import pathlib
import subprocess
import sys

import data_check as dc  # conftest.py puts src/ on sys.path
import pytest
import render_config as rc

pa = pytest.importorskip("pyarrow")

BASE = rc.DEFAULT_KNOBS

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "data_check.py"


# ── fixtures ────────────────────────────────────────────────────────────────────


class FakeTokenizer:
    """Enough of a ``transformers`` tokenizer for the three id asserts."""

    def __init__(self, bos=100257, eos=100257, pad=100256, vocab=100352):
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.pad_token_id = pad
        self._vocab = vocab

    def __len__(self):
        return self._vocab


def write_parquet(path, columns=("contents",), rows=2):
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({name: ["hello world"] * rows for name in columns})
    pq.write_table(table, path)
    return path


def write_arrow(path, columns=("tokens",), rows=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({name: [[1, 2, 3]] * rows for name in columns})
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return path


def parquet_tree(root, subs=("sub_a", "sub_b"), columns=("contents",)):
    """``dataset_path`` as the loader expects it: subfolders of shard files."""
    for sub in subs:
        write_parquet(root / sub / "shard-0000.parquet", columns=columns)
    return root


def knobs(root=None, **over):
    k = copy.deepcopy(BASE)
    if root is not None:
        k["dataset_path"] = str(root)
    k["tokenizer_name"] = over.pop("tokenizer_name", "/TOKENIZER")
    k.update(over)
    return k


def errs(root=None, tokenizer=None, **over):
    loader = (lambda name: tokenizer) if tokenizer is not None else _ok_loader
    return dc.check(knobs(root, **over), load_tokenizer=loader)


def _ok_loader(name):
    return FakeTokenizer()


# ── happy path ──────────────────────────────────────────────────────────────────


def test_hf_parquet_tree_ok(tmp_path):
    assert errs(parquet_tree(tmp_path)) == []


def test_arrow_tree_ok(tmp_path):
    root = tmp_path / "ds"
    write_arrow(root / "sub_a" / "shard-0000.arrow")
    assert errs(root, file_type="arrow", col_name="tokens") == []


def test_shards_may_be_nested_deeper(tmp_path):
    # StreamingDocDataset.setup() walks the sub-dataset recursively.
    root = tmp_path / "ds"
    write_parquet(root / "sub_a" / "part-0" / "shard-0000.parquet")
    assert errs(root) == []


def test_meta_folder_is_not_a_sub_dataset(tmp_path):
    # SamplingDataset's auto-discovery skips any entry whose name holds "meta".
    root = parquet_tree(tmp_path, subs=("sub_a",))
    (root / "meta").mkdir()
    (root / "meta" / "counts.csv").write_text("dataset/filename,tokens\n")
    assert errs(root) == []


def test_loose_files_at_the_top_are_not_sub_datasets(tmp_path):
    root = parquet_tree(tmp_path, subs=("sub_a",))
    write_parquet(root / "stray.parquet")
    assert errs(root) == []


# ── dataset_path ────────────────────────────────────────────────────────────────


def test_dataset_path_unset(tmp_path):
    # DEFAULT_KNOBS ships "" and render_config.validate does not reject it.
    found = errs(None)
    assert any("dataset_path" in e and "not set" in e for e in found)
    assert not any("Traceback" in e for e in found)


def test_dataset_path_missing(tmp_path):
    assert any("does not exist" in e for e in errs(tmp_path / "nope"))


def test_dataset_path_is_a_file(tmp_path):
    path = tmp_path / "file"
    path.write_text("x")
    assert any("dataset_path" in e for e in errs(path))


def test_dataset_path_empty_directory(tmp_path):
    assert any("no sub-dataset" in e for e in errs(tmp_path))


def test_dataset_path_only_meta(tmp_path):
    (tmp_path / "meta").mkdir()
    assert any("no sub-dataset" in e for e in errs(tmp_path))


# ── sub_datasets / weights ──────────────────────────────────────────────────────


def test_named_sub_datasets_ok(tmp_path):
    root = parquet_tree(tmp_path)
    assert errs(root, sub_datasets=["sub_a", "sub_b"]) == []


def test_typoed_sub_dataset_rejected(tmp_path):
    root = parquet_tree(tmp_path)
    found = errs(root, sub_datasets=["sub_a", "sub_c"])
    # One error, naming the typo: sub_a is fine and is not re-checked as missing.
    assert len(found) == 1
    assert "not found" in found[0] and "'sub_c'" in found[0]


def test_sub_datasets_accepts_csv_string(tmp_path):
    # parse_data_args() splits a comma-separated string.
    root = parquet_tree(tmp_path)
    assert errs(root, sub_datasets="sub_a, sub_b") == []
    assert any("sub_c" in e for e in errs(root, sub_datasets="sub_a,sub_c"))


def test_weights_length_must_match_named_sub_datasets(tmp_path):
    root = parquet_tree(tmp_path)
    found = errs(root, sub_datasets=["sub_a", "sub_b"], weights=[1])
    assert any("weights" in e for e in found)


def test_weights_length_must_match_discovered_sub_datasets(tmp_path):
    root = parquet_tree(tmp_path)  # two subfolders
    assert any("weights" in e for e in errs(root, weights=[1, 2, 3]))
    assert errs(root, weights=[1, 2]) == []


def test_weights_must_be_positive(tmp_path):
    root = parquet_tree(tmp_path)
    assert any("positive" in e for e in errs(root, weights=[1, 0]))
    assert any("positive" in e for e in errs(root, weights=[1, -1]))


def test_non_numeric_weight_rejected(tmp_path):
    root = parquet_tree(tmp_path)
    found = errs(root, weights=["1", "lots"])
    assert any("weights" in e for e in found)


# ── shard files ─────────────────────────────────────────────────────────────────


def test_sub_dataset_without_matching_files(tmp_path):
    root = parquet_tree(tmp_path, subs=("sub_a",))
    (root / "sub_b").mkdir()
    found = errs(root)
    assert any("sub_b" in e and "parquet" in e for e in found)


def test_every_sub_dataset_is_checked(tmp_path):
    root = parquet_tree(tmp_path, subs=("sub_a",))
    (root / "sub_b").mkdir()
    (root / "sub_c").mkdir()
    found = errs(root)
    assert any("sub_b" in e for e in found)
    assert any("sub_c" in e for e in found)


def test_arrow_files_do_not_satisfy_hf_parquet(tmp_path):
    root = tmp_path / "ds"
    write_arrow(root / "sub_a" / "shard-0000.arrow")
    assert any("parquet" in e for e in errs(root))


def test_parquet_files_do_not_satisfy_arrow(tmp_path):
    root = parquet_tree(tmp_path, subs=("sub_a",))
    found = errs(root, file_type="arrow", col_name="tokens")
    assert any("arrow" in e for e in found)


def test_bad_file_type(tmp_path):
    root = parquet_tree(tmp_path)
    assert any("file_type" in e for e in errs(root, file_type="jsonl"))


# ── column ──────────────────────────────────────────────────────────────────────


def test_missing_column_in_arrow(tmp_path):
    root = tmp_path / "ds"
    write_arrow(root / "sub_a" / "shard-0000.arrow", columns=("input_ids",))
    found = errs(root, file_type="arrow", col_name="tokens")
    assert any("tokens" in e and "input_ids" in e for e in found)


def test_column_among_several_is_ok(tmp_path):
    root = tmp_path / "ds"
    write_arrow(root / "sub_a" / "s.arrow", columns=("tokens", "doc_id"))
    assert errs(root, file_type="arrow", col_name="tokens") == []


def test_column_as_the_only_column_is_ok(tmp_path):
    root = tmp_path / "ds"
    write_arrow(root / "sub_a" / "s.arrow", columns=("tokens",))
    assert errs(root, file_type="arrow", col_name="tokens") == []


def test_missing_column_in_parquet(tmp_path):
    root = parquet_tree(tmp_path, subs=("sub_a",), columns=("text",))
    found = errs(root, col_name="contents")
    assert any("contents" in e for e in found)


def test_parquet_needs_exactly_one_legal_field(tmp_path):
    # ParquetHandler.open() asserts len({text,content,contents} & columns) == 1.
    root = parquet_tree(tmp_path, subs=("sub_a",), columns=("contents", "text"))
    found = errs(root, col_name="contents")
    assert any("contents" in e and "text" in e for e in found)


def test_parquet_col_name_outside_the_legal_fields(tmp_path):
    # ParquetHandler ignores col_name and reads the legal field it finds, so a
    # col_name of its own would silently train on a different column.
    root = parquet_tree(tmp_path, subs=("sub_a",), columns=("body", "text"))
    found = errs(root, col_name="body")
    assert any("body" in e for e in found)


def test_unreadable_shard_file(tmp_path):
    root = tmp_path / "ds"
    bad = root / "sub_a" / "shard-0000.parquet"
    bad.parent.mkdir(parents=True)
    bad.write_text("not parquet at all")
    found = errs(root)
    assert any("shard-0000.parquet" in e for e in found)
    assert not any("Traceback" in e for e in found)


# ── tokenizer ───────────────────────────────────────────────────────────────────


def test_tokenizer_unset(tmp_path):
    root = parquet_tree(tmp_path)
    found = dc.check({**knobs(root), "tokenizer_name": ""}, load_tokenizer=_ok_loader)
    assert any("tokenizer_name" in e and "not set" in e for e in found)


def test_tokenizer_does_not_load(tmp_path):
    root = parquet_tree(tmp_path)

    def boom(name):
        raise OSError(f"can't load tokenizer for {name!r}")

    found = dc.check(knobs(root), load_tokenizer=boom)
    assert any("tokenizer" in e and "/TOKENIZER" in e for e in found)
    assert not any("Traceback" in e for e in found)


def test_eos_mismatch(tmp_path):
    root = parquet_tree(tmp_path)
    found = errs(root, tokenizer=FakeTokenizer(eos=2))
    assert any("eos_token_id" in e for e in found)


def test_bos_mismatch(tmp_path):
    root = parquet_tree(tmp_path)
    assert any("bos_token_id" in e for e in errs(root, tokenizer=FakeTokenizer(bos=1)))


def test_pad_mismatch(tmp_path):
    root = parquet_tree(tmp_path)
    assert any("pad_token_id" in e for e in errs(root, tokenizer=FakeTokenizer(pad=0)))


def test_pad_none_on_the_tokenizer_is_not_a_failure(tmp_path):
    # model_wrapper/base.py guards each assert with `is not None` on the
    # tokenizer side, so a tokenizer without a pad token never trips it.
    root = parquet_tree(tmp_path)
    assert errs(root, tokenizer=FakeTokenizer(pad=None)) == []


def test_pad_none_on_both_sides_is_not_a_failure(tmp_path):
    root = parquet_tree(tmp_path)
    assert errs(root, tokenizer=FakeTokenizer(pad=None), pad_token_id=None) == []


def test_tokenizer_id_set_but_knob_null_is_a_failure(tmp_path):
    # The config would carry null and the assert compares against it.
    root = parquet_tree(tmp_path)
    found = errs(root, tokenizer=FakeTokenizer(pad=100256), pad_token_id=None)
    assert any("pad_token_id" in e for e in found)


def test_all_three_ids_reported_together(tmp_path):
    root = parquet_tree(tmp_path)
    found = errs(root, tokenizer=FakeTokenizer(bos=1, eos=2, pad=3))
    assert len([e for e in found if "token_id" in e]) == 3


# ── vocab ───────────────────────────────────────────────────────────────────────


def test_vocab_equal_to_vocab_size_is_allowed(tmp_path):
    root = parquet_tree(tmp_path)
    assert errs(root, tokenizer=FakeTokenizer(vocab=100352)) == []


def test_vocab_one_over_vocab_size_rejected(tmp_path):
    root = parquet_tree(tmp_path)
    found = errs(root, tokenizer=FakeTokenizer(vocab=100353))
    assert any("vocab" in e for e in found)


def test_vocab_under_vocab_size_is_allowed(tmp_path):
    root = parquet_tree(tmp_path)
    assert errs(root, tokenizer=FakeTokenizer(vocab=50000)) == []


def test_vocab_from_vocab_size_attribute(tmp_path):
    class NoLen:
        bos_token_id = 100257
        eos_token_id = 100257
        pad_token_id = 100256
        vocab_size = 100353

    root = parquet_tree(tmp_path)
    assert any("vocab" in e for e in errs(root, tokenizer=NoLen()))


# ── reporting everything at once ────────────────────────────────────────────────


def test_every_failure_is_reported(tmp_path):
    root = parquet_tree(tmp_path, subs=("sub_a",), columns=("text",))
    (root / "sub_b").mkdir()
    found = errs(
        root,
        sub_datasets=["sub_a", "sub_b", "sub_z"],
        weights=[1],
        col_name="contents",
        tokenizer=FakeTokenizer(eos=7, vocab=999999),
    )
    for needle in ("sub_z", "sub_b", "weights", "contents", "eos_token_id", "vocab"):
        assert any(needle in e for e in found), (needle, found)


# ── CLI ─────────────────────────────────────────────────────────────────────────


def _cli(*args, python_isolated=True):
    cmd = [sys.executable]
    if python_isolated:
        cmd.append("-I")
    return subprocess.run(
        [*cmd, str(_SRC), *args], capture_output=True, text=True, timeout=120
    )


def test_cli_help_needs_no_heavy_imports():
    # transformers and pyarrow are imported inside the functions that need them,
    # so the module is importable (and --help works) without either.
    result = _cli("--help")
    assert result.returncode == 0, result.stderr
    assert "--knobs" in result.stdout


def test_module_imports_without_transformers_or_pyarrow():
    code = (
        "import sys, importlib;"
        "sys.modules['pyarrow'] = None;"
        "sys.modules['transformers'] = None;"
        f"sys.path.insert(0, {str(_SRC.parent)!r});"
        "importlib.import_module('data_check')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_main_ok(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dc, "load_tokenizer", _ok_loader)
    root = parquet_tree(tmp_path)
    assert dc.main(["--knobs", json.dumps(knobs(root))]) == 0
    assert "data-check: ok" in capsys.readouterr().out


def test_main_reports_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dc, "load_tokenizer", _ok_loader)
    assert dc.main(["--knobs", json.dumps(knobs(tmp_path / "nope"))]) == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err
    assert "ok" not in captured.out


def test_cli_rejects_bad_json():
    result = _cli("--knobs", "{not json")
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
