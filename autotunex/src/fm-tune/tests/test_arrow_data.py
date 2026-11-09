"""Pin ``autotune/trainers/_arrow_data.py``: the Arrow backend both HF multi-GPU drivers share.

The file layout must not move (workers memory-map it, and the length-grouped
sampler reads the ``length`` column), and the per-row lengths, now one
vectorized sum per chunk, must equal the old per-row Python ``sum(mask)``.
"""

import pathlib

import pandas as pd
import pyarrow as pa
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from autotune.trainers._arrow_data import TokenizedArrowDataset, load_tokenize_and_save
from autotune.utils import tokenize_batch


@pytest.fixture(scope="module")
def word_tokenizer():
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({f"w{i}": i + 3 for i in range(100)})
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>", padding_side="left"
    )


def _frame(n=30):
    words = lambda k, o=0: " ".join(f"w{(i + o) % 100}" for i in range(k))  # noqa: E731
    return pd.DataFrame([{"input": words(i % 20 + 1, i), "output": words(i % 3 + 1)} for i in range(n)])


def test_file_matches_old_per_row_layout(tmp_path, word_tokenizer):
    df = _frame()
    src = tmp_path / "train.jsonl"
    df.to_json(src, orient="records", lines=True)
    out = tmp_path / "cache" / "train.arrow"

    # chunk_size=7 so lengths are concatenated across several chunks.
    n = load_tokenize_and_save(str(src), word_tokenizer, "input", "output", 16, str(out), chunk_size=7)

    expected = tokenize_batch(df, word_tokenizer, "input", "output", max_length=16)
    table = pa.ipc.open_file(pa.memory_map(str(out), "r")).read_all()
    assert n == len(table) == len(expected["input_ids"])
    assert table.column_names == ["input_ids", "attention_mask", "labels", "length"]
    assert table.schema.field("length").type == pa.int32()
    for key in ("input_ids", "attention_mask", "labels"):
        assert table.column(key).to_pylist() == expected[key]
    assert table.column("length").to_pylist() == [sum(m) for m in expected["attention_mask"]]


def test_dataset_reads_rows_and_lengths(tmp_path, word_tokenizer):
    df = _frame(10)
    src = tmp_path / "eval.jsonl"
    df.to_json(src, orient="records", lines=True)
    out = tmp_path / "eval.arrow"
    load_tokenize_and_save(str(src), word_tokenizer, "input", "output", 16, str(out))

    ds = TokenizedArrowDataset(str(out))
    assert len(ds) == len(ds.lengths)
    row = ds[0]
    assert set(row) == {"input_ids", "attention_mask", "labels"}
    assert int(row["attention_mask"].sum()) == int(ds.lengths[0])


@pytest.mark.parametrize("name", ["driver_multi_hf_fsdp.py", "driver_multi_hf_ds.py"])
def test_drivers_share_the_module(name):
    src = (pathlib.Path("autotune/trainers") / name).read_text()
    assert "from autotune.trainers._arrow_data import TokenizedArrowDataset, load_tokenize_and_save" in src
    assert "class _TokenizedArrowDataset" not in src
    assert "def _load_tokenize_and_save" not in src
