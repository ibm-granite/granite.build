# SPDX-License-Identifier: Apache-2.0
"""student-sft's preprocessing cache (CustomSFTTrainer._prepare_dataset) under datasets 5.x.

The map chain starts with `with_transform(remove_none_values)`, as trl 0.26.2's
_prepare_dataset did, and datasets keeps that transform on the output of every map. datasets
5.2.0's save_to_disk refuses a format whose kwargs are not JSON serializable, so the
cache-MISS path died at `prepared.save_to_disk(...)` with "The format kwargs must be JSON
serializable, but key 'transform' isn't" (the #676 student-sft build test, dev image
dev-676-5da71432). These tests run both cache paths on a tiny offline tokenizer: the MISS
must save, the HIT must load the same rows, and both must hand the trainer a dataset with no
transform, as trl 1.18.2's own _prepare_dataset returns (it refuses a dataset that carries
one).

Needs torch, transformers, tokenizers, datasets and trl, so it runs in the step image (CPU
only) and skips elsewhere.
"""

import json
import os
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("tokenizers")

# Same directory; `sft` and `tokenizer` are pytest fixtures, used by name below.
from test_sft_labels import (  # noqa: E402,F401
    MESSAGES,
    RESPONSE_TEMPLATE,
    _args,
    sft,
    tokenizer,
)

# MESSAGES with one key that only some messages have: Arrow fills it with None in the others,
# which is what the remove_none_values transform at the top of the map chain strips.
ROWS = [
    {"messages": [dict(m) for m in MESSAGES[0]["messages"]]},
    {"messages": [dict(m) for m in MESSAGES[1]["messages"]]},
]
ROWS[1]["messages"][1]["name"] = "helper"


def _trainer(sft, cache_dir):
    trainer = types.SimpleNamespace(
        response_template=RESPONSE_TEMPLATE,
        precomputed_logits_dir="",
        _is_vlm=False,
        completion_only_loss=True,
        sft_cache_dir=str(cache_dir),
        sft_cache_key_blob={"test": "key"},
    )
    trainer._run_map_chain = types.MethodType(
        sft.CustomSFTTrainer._run_map_chain, trainer
    )
    return trainer


def _prepare(sft, tokenizer, trainer, rows):
    from datasets import Dataset

    return sft.CustomSFTTrainer._prepare_dataset(
        trainer, Dataset.from_list(rows), tokenizer, _args(), False, None, "train"
    )


def _rows(dataset):
    return [dataset[i] for i in range(len(dataset))]


def test_the_input_has_none_values_for_the_transform_to_strip():
    """Not vacuous: the rows really carry an Arrow-filled None."""
    from datasets import Dataset

    assert Dataset.from_list(ROWS)[0]["messages"][0]["name"] is None


def test_cache_miss_saves_and_returns_no_transform(sft, tokenizer, tmp_path, capsys):
    prepared = _prepare(sft, tokenizer, _trainer(sft, tmp_path), ROWS)
    assert "[sft-cache] MISS" in capsys.readouterr().out
    assert os.path.exists(tmp_path / "train" / "dataset_info.json")
    assert json.loads((tmp_path / "key.json").read_text()) == {"test": "key"}
    assert prepared.format["type"] is None
    assert "transform" not in prepared.format["format_kwargs"]
    assert all(-100 in r["labels"] for r in _rows(prepared))


def test_cache_hit_loads_the_rows_the_miss_saved(sft, tokenizer, tmp_path, capsys):
    miss = _rows(_prepare(sft, tokenizer, _trainer(sft, tmp_path), ROWS))
    capsys.readouterr()
    # A different input on the second call: a HIT must serve the cache, not re-run the chain.
    hit = _prepare(sft, tokenizer, _trainer(sft, tmp_path), ROWS[:1])
    assert "[sft-cache] HIT" in capsys.readouterr().out
    assert hit.format["type"] is None
    assert _rows(hit) == miss


def test_the_cached_rows_are_the_uncached_rows(sft, tokenizer, tmp_path):
    """The cache changes where the rows come from, not what they are: the uncached chain
    (no cache dir) gives the same rows, also with no transform."""
    from datasets import Dataset

    cached = _rows(_prepare(sft, tokenizer, _trainer(sft, tmp_path), ROWS))
    trainer = _trainer(sft, tmp_path)
    trainer.sft_cache_dir = None
    uncached = sft.CustomSFTTrainer._prepare_dataset(
        trainer, Dataset.from_list(ROWS), tokenizer, _args(), False, None, "train"
    )
    assert uncached.format["type"] is None
    assert _rows(uncached) == cached
