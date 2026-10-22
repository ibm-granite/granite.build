"""Pin ``tokenize_batch_arrays`` / ``tokenize_batch`` to the original per-row implementation.

The vectorized version exists for Ray Data throughput (see the ``ray_data``
backend in the HF multi-GPU drivers); it must produce the same rows, id for
id, including the sliding-window expansion, the empty-input drop and the
capacity error. ``_reference_tokenize_batch`` below is the pre-vectorization
implementation, kept verbatim as the oracle.
"""

import os

import numpy as np
import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from autotune.utils import tokenize_batch, tokenize_batch_arrays


def _reference_tokenize_batch(batch, tokenizer, input_col="input", output_col="output", max_length=1024, stride=512):
    if isinstance(batch, pd.DataFrame):
        examples = batch.to_dict(orient="records")
    else:
        examples = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

    input_ids_list, attention_masks_list, labels_list = [], [], []
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if stride > max_length:
        stride = max_length // 2

    for ex in examples:
        output_tokens = tokenizer(str(ex[output_col]), add_special_tokens=False)["input_ids"]
        capacity = max_length - len(output_tokens) - 1
        if capacity < 0:
            raise ValueError(f"Not enough capacity for output + EOS at given max_length = {max_length} tokens.")
        input_tokens = tokenizer(str(ex[input_col]), add_special_tokens=False)["input_ids"]
        start = 0
        while start < len(input_tokens):
            end = start + (max_length - len(output_tokens) - 1)
            chunk = input_tokens[start:end]
            combined = chunk + output_tokens + [tokenizer.eos_token_id]
            attention_mask = [1] * len(combined)
            labels = [-100] * len(chunk) + output_tokens + [tokenizer.eos_token_id]
            pad_len = max_length - len(combined)
            if pad_len > 0:
                combined = [tokenizer.pad_token_id] * pad_len + combined
                attention_mask = [0] * pad_len + attention_mask
                labels = [-100] * pad_len + labels
            input_ids_list.append(combined)
            attention_masks_list.append(attention_mask)
            labels_list.append(labels)
            start += stride

    return {"input_ids": input_ids_list, "attention_mask": attention_masks_list, "labels": labels_list}


@pytest.fixture(scope="module")
def word_tokenizer():
    """Offline one-token-per-word tokenizer, so token counts are exact by construction."""
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({f"w{i}": i + 3 for i in range(100)})
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")


def _words(n, offset=0):
    return " ".join(f"w{(i + offset) % 100}" for i in range(n))


# Input lengths straddling every window boundary for max_length=16, stride=4:
# empty (no rows), shorter than / equal to / one past the stride, exactly
# filling the capacity, and several windows long.
_CASES = [
    {"input": _words(n, n), "output": _words(m)}
    for n, m in [(0, 1), (1, 1), (3, 2), (4, 2), (5, 2), (13, 2), (14, 2), (40, 3), (25, 15)]
]


@pytest.mark.parametrize("as_dataframe", [True, False])
@pytest.mark.parametrize("max_length,stride", [(16, 4), (16, 16), (16, 40), (8, 3)])
def test_matches_reference(word_tokenizer, as_dataframe, max_length, stride):
    cases = [c for c in _CASES if len(c["output"].split()) + 1 <= max_length]
    df = pd.DataFrame(cases)
    batch = df if as_dataframe else {k: np.array(v, dtype=object) for k, v in df.to_dict(orient="list").items()}

    expected = _reference_tokenize_batch(df, word_tokenizer, max_length=max_length, stride=stride)
    assert tokenize_batch(batch, word_tokenizer, max_length=max_length, stride=stride) == expected

    arrays = tokenize_batch_arrays(batch, word_tokenizer, max_length=max_length, stride=stride)
    for key, rows in expected.items():
        assert arrays[key].dtype == np.int32
        assert arrays[key].shape == (len(rows), max_length)
        assert arrays[key].tolist() == rows


def test_empty_inputs_yield_zero_rows(word_tokenizer):
    arrays = tokenize_batch_arrays({"input": [""], "output": ["w1"]}, word_tokenizer, max_length=8)
    assert arrays["input_ids"].shape == (0, 8)


def test_output_too_long_raises(word_tokenizer):
    with pytest.raises(ValueError, match="Not enough capacity"):
        tokenize_batch_arrays({"input": ["w1"], "output": [_words(8)]}, word_tokenizer, max_length=8)


def test_nonpositive_stride_raises(word_tokenizer):
    with pytest.raises(ValueError, match="stride must be > 0"):
        tokenize_batch_arrays({"input": ["w1"], "output": ["w1"]}, word_tokenizer, stride=0)


GRANITE = os.environ.get("FMTUNE_PARITY_MODEL", "ibm-granite/granite-4.2-8b")


@pytest.mark.slow
def test_fast_tokenizer_matches_slow_on_granite():
    """``get_tokenizer`` switched to ``use_fast=True``; the rows it trains on must not move."""
    from transformers import AutoTokenizer

    try:
        slow = AutoTokenizer.from_pretrained(GRANITE, use_fast=False, padding_side="left", local_files_only=True)
        fast = AutoTokenizer.from_pretrained(GRANITE, use_fast=True, padding_side="left", local_files_only=True)
    except OSError:
        pytest.skip(f"{GRANITE} not in the local HF cache")
    # transformers 5 dropped the slow/fast split for most models: use_fast=False now
    # returns the same Rust backend. The tokenize_batch-vs-reference comparison
    # below still pins the rows, so only require the fast side to be fast.
    assert fast.is_fast
    for tok in (slow, fast):
        tok.pad_token = tok.eos_token

    msgs = [{"role": "user", "content": "Text A (premise): The café’s menu — naïve prices!\n\nText B: 価格は安い。"}]
    prompt = slow.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert prompt == fast.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    batch = {
        "input": [prompt, prompt * 40, "  leading and trailing whitespace \t\n", "<|im_start|>user\nhi<|im_end|>"],
        "output": ['{"label": "entailment"}', '{"label": "contradiction"}', "neutral", "ok"],
    }
    assert tokenize_batch(batch, fast, max_length=256, stride=128) == _reference_tokenize_batch(
        batch, slow, max_length=256, stride=128
    )


def test_compact_rows_round_trip(word_tokenizer):
    """``input_ids`` + ``pad_len``/``target_len`` must rebuild the full rows exactly."""
    import torch

    from autotune.trainers._ray_data import compact_rows, expand_rows

    cases = [c for c in _CASES if len(c["output"].split()) + 1 <= 16]
    arrays = tokenize_batch_arrays(pd.DataFrame(cases), word_tokenizer, max_length=16, stride=4)
    compact = compact_rows(arrays)
    assert set(compact) == {"input_ids", "pad_len", "target_len"}
    assert compact["pad_len"].dtype == compact["target_len"].dtype == np.int32

    rebuilt = expand_rows({k: torch.from_numpy(v) for k, v in compact.items()})
    for key, expected in arrays.items():
        assert rebuilt[key].dtype == torch.long
        assert rebuilt[key].tolist() == expected.tolist()


def test_batch_to_dataframe_restores_nested_lists():
    """Ray's numpy batches hold list columns as ndarrays; chat detection needs lists."""
    from autotune.trainers._ray_data import batch_to_dataframe

    msgs = np.empty(2, dtype=object)
    msgs[0] = np.array([{"role": "user", "content": "a"}], dtype=object)
    msgs[1] = np.array([{"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}], dtype=object)
    docs = np.empty(2, dtype=object)
    docs[0] = np.array([{"doc_id": "1", "text": "t", "tags": np.array(["x", "y"], dtype=object)}], dtype=object)
    docs[1] = np.array([], dtype=object)
    batch = {"input": msgs, "documents": docs, "output": np.array(["yes", "no"], dtype=object), "n": np.array([1, 2])}

    df = batch_to_dataframe(batch)
    assert df["input"].tolist() == [
        [{"role": "user", "content": "a"}],
        [{"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}],
    ]
    assert df["documents"].iloc[0] == [{"doc_id": "1", "text": "t", "tags": ["x", "y"]}]
    assert df["documents"].iloc[1] == []
    assert df["output"].tolist() == ["yes", "no"]
    assert df["n"].tolist() == [1, 2]


def _real_chat_template(df, tokenizer, input_col, thinking):
    from autotune.trainers.driver_multi_hf_fsdp import _apply_chat_template_to_df

    return _apply_chat_template_to_df(df, tokenizer, input_col, thinking)


def _expanded_rows(ds, n):
    """All rows of a materialized compact dataset, expanded, as sortable tuples."""
    import torch

    from autotune.trainers._ray_data import expand_rows

    got = ds.take_batch(n)
    assert got["input_ids"].dtype == np.int32
    full = expand_rows({k: torch.from_numpy(v) for k, v in got.items()})
    return sorted(zip(*(map(tuple, full[k].tolist()) for k in ("input_ids", "attention_mask", "labels"))))


def _reference_rows(tokenized):
    return sorted(zip(*(map(tuple, tokenized[k]) for k in ("input_ids", "attention_mask", "labels"))))


@pytest.mark.slow
@pytest.mark.parametrize("chat", [False, True], ids=["plain", "chat"])
def test_build_tokenized_datasets_matches_tokenize_batch(tmp_path, monkeypatch, chat):
    """End to end on a local Ray cluster: the same rows as the in-process path.

    ``chat`` runs a message-list jsonl through the driver's real
    ``_apply_chat_template_to_df``: Ray reads that column back as numpy arrays,
    and before ``batch_to_dataframe`` the template was skipped silently.
    Rows are compared as multisets because the train split is shuffled.
    """
    import json

    import ray

    from autotune.trainers._ray_data import build_tokenized_datasets
    from autotune.utils import get_tokenizer

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        tokenizer, _ = get_tokenizer(GRANITE)
    except OSError:
        pytest.skip(f"{GRANITE} not in the local HF cache")

    def record(i):
        text = " ".join(["word"] * (5 + 37 * i))
        return {"input": [{"role": "user", "content": text}] if chat else text, "output": f"label {i}"}

    rows = [record(i) for i in range(24)]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))

    half = pd.DataFrame(rows[:12])
    if chat:
        half = _real_chat_template(half, tokenizer, "input", False)
        assert half["input"].map(lambda s: isinstance(s, str) and "word" in s).all()
    expected = tokenize_batch(half, tokenizer, max_length=64)

    monkeypatch.setenv("RAY_DEDUP_LOGS", "0")
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True)
    try:
        train_ds, eval_ds, n = build_tokenized_datasets(
            str(path),
            str(path),
            percentage=0.5,
            seed=0,
            model_name_or_path=GRANITE,
            tokenizer_kwargs={},
            input_col="input",
            output_col="output",
            max_length=64,
            thinking=False,
            apply_chat_template=_real_chat_template,
        )
        assert n == len(expected["input_ids"]) == eval_ds.count()
        assert n > 12  # the sliding window expanded the long records
        assert _expanded_rows(train_ds, n) == _reference_rows(expected)
        assert _expanded_rows(eval_ds, n) == _reference_rows(expected)
    finally:
        ray.shutdown()
