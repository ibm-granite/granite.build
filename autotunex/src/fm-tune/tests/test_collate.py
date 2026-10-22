"""Batch collation: SFT batches are sized to their own rows, not ``max_length``.

Every stored row is left-padded to ``max_length`` by ``tokenize_batch_arrays``,
and the drivers used to stack them as-is (``default_data_collator``), so every
micro-batch ran ``B x max_length`` tokens through the model regardless of how
short the rows were. These pin the two collators that replace it, the gating
that picks between them, and the length grouping that keeps a dynamic batch's
longest row close to the rest.
"""

import pathlib
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from autotune.trainers._collate import (
    PADDING_DYNAMIC,
    PADDING_FREE,
    DynamicPadCollator,
    FlatteningCollator,
    LengthGroupedTrainer,
    build_collator,
    group_by_length_enabled,
    resolve_padding_mode,
    row_lengths,
)
from autotune.trainers._ray_data import RayShardIterableDataset, compact_rows

SFT_DRIVERS = ["driver_single.py", "driver_multi_hf_fsdp.py", "driver_multi_hf_ds.py"]
PAD = 0


def _row(prompt, target, max_length):
    """One ``tokenize_batch_arrays``-shaped row: ``[pad | prompt | target]``."""
    pad = max_length - len(prompt) - len(target)
    return {
        "input_ids": [PAD] * pad + prompt + target,
        "attention_mask": [0] * pad + [1] * (len(prompt) + len(target)),
        "labels": [-100] * (pad + len(prompt)) + target,
    }


def _features(max_length=32, seed=0, n=4, as_tensors=False):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        p = rng.integers(1, 10)
        t = rng.integers(1, 5)
        rows.append(_row(list(rng.integers(3, 50, p)), list(rng.integers(3, 50, t)), max_length))
    if as_tensors:
        rows = [{k: torch.tensor(v) for k, v in r.items()} for r in rows]
    return rows


class TestDynamicPadCollator:
    @pytest.mark.parametrize("as_tensors", [False, True])
    def test_crops_to_longest_row_rounded_up(self, as_tensors):
        feats = _features(as_tensors=as_tensors)
        out = DynamicPadCollator(pad_to_multiple_of=8)(feats)
        longest = max(int(np.sum(np.asarray(f["attention_mask"]))) for f in feats)
        keep = -(-longest // 8) * 8
        assert out["input_ids"].shape == (len(feats), keep)
        for k in ("input_ids", "attention_mask", "labels"):
            full = torch.stack([torch.as_tensor(np.asarray(f[k])) for f in feats])
            assert torch.equal(out[k], full[:, -keep:])
        # Nothing real was cut: every row keeps all of its tokens.
        assert torch.equal(
            out["attention_mask"].sum(1), torch.as_tensor(row_lengths([f["attention_mask"] for f in feats]))
        )

    def test_never_wider_than_the_rows(self):
        feats = [_row([5] * 20, [6] * 10, 30)]
        out = DynamicPadCollator(pad_to_multiple_of=64)(feats)
        assert out["input_ids"].shape == (1, 30)

    def test_drops_extra_keys(self):
        feats = [dict(f, length=3) for f in _features()]
        assert set(DynamicPadCollator()(feats)) == {"input_ids", "attention_mask", "labels"}

    def test_loss_matches_full_padding(self):
        """Cropping changes nothing a model can see: pads are masked and -100 either way."""
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        cfg = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            pad_token_id=PAD,
            attn_implementation="eager",
        )
        model = LlamaForCausalLM(cfg).eval()
        feats = _features(max_length=48, seed=3)
        full = {k: torch.stack([torch.tensor(f[k]) for f in feats]) for k in feats[0]}
        cropped = DynamicPadCollator()(feats)
        assert cropped["input_ids"].shape[1] < full["input_ids"].shape[1]
        with torch.no_grad():
            loss_full = model(**full).loss
            loss_cropped = model(**cropped).loss
        assert torch.allclose(loss_full, loss_cropped, atol=1e-5)


class TestFlatteningCollator:
    def test_flattens_unpadded_rows(self):
        feats = [_row([7, 8, 9], [10, 11], 12), _row([3], [4, 5, 6], 12)]
        out = FlatteningCollator()(feats)
        assert out["input_ids"].tolist() == [[7, 8, 9, 10, 11, 3, 4, 5, 6]]
        assert out["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0, 1, 2, 3]]
        # First label of each sequence is the separator; it is a prompt token anyway.
        assert out["labels"].tolist() == [[-100, -100, -100, 10, 11, -100, 4, 5, 6]]
        assert out["cu_seq_lens_q"].tolist() == [0, 5, 9]
        assert out["max_length_q"] == 5
        assert "attention_mask" not in out

    def test_accepts_tensor_rows(self):
        lists = [_row([7, 8], [9], 6), _row([1, 2, 3], [4], 6)]
        tensors = [{k: torch.tensor(v) for k, v in r.items()} for r in lists]
        a, b = FlatteningCollator()(lists), FlatteningCollator()(tensors)
        assert torch.equal(a["input_ids"], b["input_ids"])
        assert torch.equal(a["labels"], b["labels"])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="flash attention needs CUDA")
    def test_loss_matches_padded_under_flash_attention(self):
        pytest.importorskip("flash_attn")
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        cfg = LlamaConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            pad_token_id=PAD,
        )
        feats = _features(max_length=48, seed=5)
        padded = {k: v.cuda() for k, v in DynamicPadCollator()(feats).items()}
        flat = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in FlatteningCollator()(feats).items()}
        losses = []
        for impl, batch in (("eager", padded), ("flash_attention_2", flat)):
            cfg._attn_implementation = impl
            torch.manual_seed(0)
            model = LlamaForCausalLM(cfg).to("cuda", torch.bfloat16).eval()
            with torch.no_grad():
                losses.append(model(**batch).loss.float())
        assert torch.allclose(losses[0], losses[1], atol=2e-2)


class TestResolvePaddingMode:
    DENSE = SimpleNamespace(model_type="granite", architectures=["GraniteForCausalLM"])
    HYBRID = SimpleNamespace(model_type="granitemoehybrid", architectures=["GraniteMoeHybridForCausalLM"])
    OK = dict(attn_implementation="flash_attention_2", device_kind="cuda", model_config=DENSE, peft_type="LORA")

    def test_auto_picks_padding_free_when_safe(self):
        assert resolve_padding_mode({}, **self.OK) == PADDING_FREE
        assert resolve_padding_mode({"padding_free": "auto"}, **self.OK) == PADDING_FREE

    @pytest.mark.parametrize(
        "override",
        [
            {"attn_implementation": "eager"},
            {"attn_implementation": "sdpa"},
            {"device_kind": "mps"},
            {"peft_type": "ALORA"},
            {"is_sr": True},
            {"model_config": HYBRID},
            {"model_config": SimpleNamespace(model_type="mamba2", architectures=None)},
        ],
    )
    def test_auto_falls_back_and_true_raises(self, override):
        kwargs = {**self.OK, **override}
        assert resolve_padding_mode({}, **kwargs) == PADDING_DYNAMIC
        with pytest.raises(ValueError, match="padding_free=true"):
            resolve_padding_mode({"padding_free": True}, **kwargs)

    @pytest.mark.parametrize("value", [False, "false", "off"])
    def test_false_always_dynamic(self, value):
        assert resolve_padding_mode({"padding_free": value}, **self.OK) == PADDING_DYNAMIC

    def test_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="auto\\|true\\|false"):
            resolve_padding_mode({"padding_free": "maybe"}, **self.OK)

    def test_build_collator(self):
        assert isinstance(build_collator(PADDING_FREE), FlatteningCollator)
        assert isinstance(build_collator(PADDING_DYNAMIC), DynamicPadCollator)

    def test_grouping_only_in_dynamic_mode(self):
        assert group_by_length_enabled({}, PADDING_DYNAMIC)
        assert not group_by_length_enabled({"group_by_length": False}, PADDING_DYNAMIC)
        assert not group_by_length_enabled({"group_by_length": True}, PADDING_FREE)


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"input_ids": [i]}


def _trainer(tmp_path, lengths, seed=42):
    from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments

    cfg = LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2)
    args = TrainingArguments(
        output_dir=str(tmp_path), per_device_train_batch_size=4, seed=seed, report_to="none", use_cpu=True
    )
    return LengthGroupedTrainer(
        model=LlamaForCausalLM(cfg), args=args, train_dataset=_ListDataset(len(lengths)), train_lengths=lengths
    )


class TestLengthGroupedTrainer:
    def test_same_seed_same_order_on_every_rank(self, tmp_path):
        """Ranks share one sampler order; BatchSamplerShard then deals out disjoint batches."""
        lengths = list(np.random.default_rng(0).integers(5, 500, 300))
        a = list(_trainer(tmp_path / "a", lengths)._get_train_sampler())
        b = list(_trainer(tmp_path / "b", lengths)._get_train_sampler())
        assert a == b
        assert sorted(a) == list(range(len(lengths)))

    def test_groups_similar_lengths(self, tmp_path):
        lengths = list(np.random.default_rng(1).integers(5, 1000, 400))
        order = list(_trainer(tmp_path, lengths)._get_train_sampler())
        bs = 4
        spread = np.mean([np.ptp([lengths[i] for i in order[j : j + bs]]) for j in range(0, len(order), bs)])
        random_spread = np.mean([np.ptp(lengths[j : j + bs]) for j in range(0, len(lengths), bs)])
        assert spread < random_spread / 4

    def test_without_lengths_uses_hf_default(self, tmp_path):
        trainer = _trainer(tmp_path, [1, 2, 3])
        trainer._train_lengths = None
        assert type(trainer._get_train_sampler()).__name__ == "RandomSampler"


class _FakeShard:
    """``iter_torch_batches`` over compact rows, like a Ray Data shard."""

    def __init__(self, rows):
        self.rows = compact_rows({k: np.asarray([r[k] for r in rows], dtype=np.int32) for k in rows[0]})

    def iter_torch_batches(self, batch_size, local_shuffle_buffer_size=None):
        n = len(self.rows["input_ids"])
        for i in range(0, n, batch_size):
            yield {k: torch.as_tensor(v[i : i + batch_size]) for k, v in self.rows.items()}


class TestRayShardGroupByLength:
    def _rows(self, n=200, max_length=64):
        rng = np.random.default_rng(2)
        return [_row([5] * int(rng.integers(1, 50)), [6, 7], max_length) for _ in range(n)]

    def test_yields_every_sample_once(self):
        rows = self._rows()
        ds = RayShardIterableDataset(_FakeShard(rows), batch_size=4, group_by_length=True)
        got = sorted(int(s["attention_mask"].sum()) for s in ds)
        assert got == sorted(sum(r["attention_mask"]) for r in rows)

    def test_respects_total_samples(self):
        ds = RayShardIterableDataset(_FakeShard(self._rows()), batch_size=4, total_samples=450, group_by_length=True)
        assert sum(1 for _ in ds) == 450

    def test_consecutive_runs_are_homogeneous(self):
        rows = self._rows()
        bs = 4
        grouped = [
            int(s["attention_mask"].sum()) for s in RayShardIterableDataset(_FakeShard(rows), bs, group_by_length=True)
        ]
        plain = [int(s["attention_mask"].sum()) for s in RayShardIterableDataset(_FakeShard(rows), bs)]
        spread = lambda xs: np.mean([np.ptp(xs[i : i + bs]) for i in range(0, len(xs), bs)])  # noqa: E731
        assert spread(grouped) < spread(plain) / 4


class TestDriversUseCollator:
    @pytest.mark.parametrize("name", SFT_DRIVERS)
    def test_no_max_length_stacking(self, name):
        src = (pathlib.Path("autotune/trainers") / name).read_text()
        assert "default_data_collator" not in src, f"{name} stacks rows at max_length again"
        assert "data_collator=build_collator(padding_mode)" in src
        assert "LengthGroupedTrainer(" in src

    @pytest.mark.parametrize("name", SFT_DRIVERS)
    def test_label_names_is_labels(self, name):
        """The flattened batch has no attention_mask; the old label_names dropped eval_loss."""
        src = (pathlib.Path("autotune/trainers") / name).read_text()
        assert re.search(r'label_names=\["labels"\]', src)
        assert '"attention_mask"]' not in src.split("label_names=")[1].split("\n")[0]
