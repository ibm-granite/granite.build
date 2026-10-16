"""Tests for autotune.utils — pure-logic functions (no model loading, no GPU)."""

import json
import os

import numpy as np
import pytest
import torch
from ray import tune

from autotune.constants import AutotunePrecision
from autotune.utils import (
    cleanup,
    filter_config,
    find_checkpoints,
    find_latest_checkpoint,
    flatten_dict,
    generate_unique_id,
    get_adapter_compatible_modules,
    get_autotune_precision,
    get_default_value,
    get_kwargs,
    get_param_space,
    get_strategy,
    get_torch_dtype,
    get_tuner_flag,
    init_random,
    is_jsonable,
    make_param_space,
    move_dir,
    parse_model_parameters,
    remove_dir,
    resolve_model_path,
    set_seed,
)


class TestFlattenDict:
    def test_empty(self):
        assert flatten_dict({}) == {}

    def test_single_level(self):
        assert flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested(self):
        out = flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
        assert out == {"a.b": 1, "a.c": 2, "d": 3}

    def test_deep_nesting(self):
        out = flatten_dict({"a": {"b": {"c": {"d": 1}}}})
        assert out == {"a.b.c.d": 1}

    def test_custom_separator(self):
        out = flatten_dict({"a": {"b": 1}}, sep="/")
        assert out == {"a/b": 1}

    def test_preserves_non_dict_values(self):
        out = flatten_dict({"a": [1, 2, 3], "b": "string"})
        assert out["a"] == [1, 2, 3]
        assert out["b"] == "string"


class TestGetTorchDtype:
    def test_fp32(self):
        assert get_torch_dtype("fp32") == torch.float32

    def test_bf16(self):
        assert get_torch_dtype("bf16") == torch.bfloat16

    def test_fp16_aliases_to_bf16(self):
        # Explicit behavior in current implementation
        assert get_torch_dtype("fp16") == torch.bfloat16

    def test_int8(self):
        assert get_torch_dtype("int8") == torch.int8

    def test_unknown_falls_back_to_fp32(self):
        assert get_torch_dtype("garbage") == torch.float32


class TestRandomSeeding:
    def test_init_random_returns_generator(self):
        rng = init_random(42)
        assert isinstance(rng, np.random.Generator)

    def test_init_random_is_deterministic(self):
        a = init_random(42).integers(0, 100, size=10)
        b = init_random(42).integers(0, 100, size=10)
        assert np.array_equal(a, b)

    def test_set_seed_makes_torch_deterministic(self):
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        assert torch.allclose(a, b)


class TestGetStrategy:
    def test_choice(self):
        spec = {"strategy": "choice", "values": ["a", "b", "c"]}
        s = get_strategy(spec)
        assert isinstance(s, tune.search.sample.Categorical)

    def test_uniform(self):
        spec = {"strategy": "uniform", "values": [0.0, 1.0]}
        s = get_strategy(spec)
        # Returns a Float domain
        assert s is not None

    def test_loguniform(self):
        spec = {"strategy": "loguniform", "values": [1e-5, 1e-1]}
        assert get_strategy(spec) is not None

    def test_randint(self):
        spec = {"strategy": "randint", "values": [1, 100]}
        assert get_strategy(spec) is not None

    def test_grid(self):
        spec = {"strategy": "grid", "values": [1, 2, 3]}
        assert get_strategy(spec) is not None

    def test_string(self):
        spec = {"strategy": "string", "values": ["x"], "default": "x"}
        s = get_strategy(spec)
        assert isinstance(s, tune.search.sample.Categorical)

    def test_missing_strategy_raises(self):
        with pytest.raises(AssertionError):
            get_strategy({"values": [1, 2]})

    def test_missing_values_raises(self):
        with pytest.raises(AssertionError):
            get_strategy({"strategy": "choice"})

    def test_uniform_wrong_arity_raises(self):
        with pytest.raises(AssertionError):
            get_strategy({"strategy": "uniform", "values": [0.0]})


class TestGetDefaultValueAndTunerFlag:
    def test_get_default(self):
        assert get_default_value({"default": 42}) == 42

    def test_get_default_missing(self):
        with pytest.raises(AssertionError):
            get_default_value({"foo": "bar"})

    def test_get_tuner_flag(self):
        assert get_tuner_flag({"for_tuner": True}) is True
        assert get_tuner_flag({"for_tuner": False}) is False

    def test_get_tuner_flag_missing(self):
        with pytest.raises(AssertionError):
            get_tuner_flag({})


class TestGetParamSpace:
    def test_empty_input_returns_empty_triple(self):
        space, defaults, flags = get_param_space({})
        assert space == {} and defaults == {} and flags == {}

    def test_missing_hyperparams_raises(self):
        with pytest.raises(ValueError):
            get_param_space({"description": "foo"})

    def test_typical_config(self):
        cfg = {
            "hyperparams": {
                "r": {
                    "strategy": "choice",
                    "values": [4, 8, 16],
                    "default": 8,
                    "for_tuner": True,
                },
                "lr": {
                    "strategy": "loguniform",
                    "values": [1e-5, 1e-3],
                    "default": 1e-4,
                    "for_tuner": False,
                },
            }
        }
        space, defaults, flags = get_param_space(cfg)
        assert set(space.keys()) == {"r", "lr"}
        assert defaults == {"r": 8, "lr": 1e-4}
        assert flags == {"r": True, "lr": False}


class TestMakeParamSpace:
    def test_skips_reserved_keys(self):
        cfg = {
            "tune_config": {"a": 1},
            "training_config": {"b": 2},
            "training_rl_config": {"c": 3},
            "lr": 0.001,
        }
        space, defaults = make_param_space(cfg)
        assert "tune_config" not in space
        assert "training_config" not in space
        assert "training_rl_config" not in space
        assert "lr" in space
        assert defaults["lr"] == 0.001


class TestFilterConfig:
    def test_filters_by_prefix(self):
        cfg = {"a.x": 1, "a.y": 2, "b.z": 3}
        out = filter_config(cfg, "a.")
        assert out == {"a.x": 1, "a.y": 2}

    def test_no_match_returns_empty(self):
        assert filter_config({"a": 1}, "z") == {}

    def test_empty_input(self):
        assert filter_config({}, "anything") == {}


class TestIsJsonable:
    def test_primitives(self):
        assert is_jsonable(42)
        assert is_jsonable(3.14)
        assert is_jsonable("hello")
        assert is_jsonable(None)
        assert is_jsonable([1, 2, 3])
        assert is_jsonable({"a": 1})

    def test_torch_tensor_not_jsonable(self):
        assert not is_jsonable(torch.tensor([1, 2, 3]))

    def test_circular_ref_not_jsonable(self):
        d = {}
        d["self"] = d
        assert not is_jsonable(d)

    def test_set_not_jsonable(self):
        assert not is_jsonable({1, 2, 3})


class TestGetAutotunePrecision:
    def test_fp32(self):
        assert get_autotune_precision("fp32") == AutotunePrecision.FP32

    def test_bf16(self):
        assert get_autotune_precision("bf16") == AutotunePrecision.BF16

    def test_fp16_maps_to_bf16(self):
        assert get_autotune_precision("fp16") == AutotunePrecision.BF16

    def test_int8(self):
        assert get_autotune_precision("int8") == AutotunePrecision.INT8

    def test_int4(self):
        assert get_autotune_precision("int4") == AutotunePrecision.INT4

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_autotune_precision("garbage")


class TestParseModelParameters:
    def test_billions_lowercase(self):
        assert parse_model_parameters("meta-llama/Llama-2-7b") == 7.0

    def test_billions_uppercase(self):
        assert parse_model_parameters("foo/model-13B") == 13.0

    def test_decimal_billions(self):
        assert parse_model_parameters("foo/model-1.5b") == 1.5

    def test_millions_converted_to_billions(self):
        assert parse_model_parameters("foo/model-500m") == 0.5

    def test_decimal_millions(self):
        assert parse_model_parameters("foo/model-1.3M") == pytest.approx(0.0013)

    def test_no_match_returns_none(self):
        assert parse_model_parameters("facebook/opt-125m") == 0.125
        assert parse_model_parameters("mistral-instruct") is None


class TestGenerateUniqueId:
    def test_default_length(self):
        assert len(generate_unique_id()) == 8

    def test_custom_length(self):
        assert len(generate_unique_id(length=16)) == 16

    def test_alphanumeric(self):
        uid = generate_unique_id(length=200)
        assert all(c.isalnum() for c in uid)

    def test_uniqueness_likely(self):
        ids = {generate_unique_id() for _ in range(100)}
        assert len(ids) >= 95  # accept rare collisions but mostly unique


class TestGetKwargs:
    def test_keyword_filter(self):
        cfg = {"trainer.lr": 0.1, "trainer.bs": 8, "model.dim": 256}
        out = get_kwargs(cfg, "trainer.", excluded=[])
        assert out == {"lr": 0.1, "bs": 8}

    def test_excluded(self):
        cfg = {"trainer.lr": 0.1, "trainer.bs": 8}
        out = get_kwargs(cfg, "trainer.", excluded=["trainer.lr"])
        assert out == {"bs": 8}

    def test_empty(self):
        assert get_kwargs({}, "any", []) == {}


class TestGetAdapterCompatibleModules:
    def test_non_alora(self):
        # Doesn't actually inspect the model — returns "all-linear"
        assert get_adapter_compatible_modules(model=None, alora=False) == "all-linear"

    def test_alora(self):
        # The alora-specific targeting is currently commented out; the function
        # returns "all-linear" for both alora and non-alora (no model inspection).
        out = get_adapter_compatible_modules(model=None, alora=True)
        assert out == "all-linear"


class TestFindCheckpoints:
    def test_empty_dir_returns_empty(self, tmp_path):
        assert find_checkpoints(str(tmp_path)) == []

    def test_finds_pl_checkpoint(self, tmp_path):
        ckpt_dir = tmp_path / "run1" / "checkpoint.ckpt"
        ckpt_dir.mkdir(parents=True)
        out = find_checkpoints(str(tmp_path))
        assert any("checkpoint.ckpt" in p for p in out)


class TestFindLatestCheckpoint:
    def test_no_dir(self, tmp_path):
        assert find_latest_checkpoint(str(tmp_path / "missing")) is None

    def test_no_checkpoints(self, tmp_path):
        assert find_latest_checkpoint(str(tmp_path)) is None

    def test_picks_latest_hf_checkpoint(self, tmp_path):
        c1 = tmp_path / "checkpoint-100"
        c2 = tmp_path / "checkpoint-200"
        c1.mkdir()
        c2.mkdir()
        # Make c2 newer
        os.utime(c2, (c2.stat().st_atime, c2.stat().st_mtime + 100))
        assert find_latest_checkpoint(str(tmp_path)) == str(c2)

    def test_unsupported_type_returns_none(self, tmp_path):
        c1 = tmp_path / "checkpoint-100"
        c1.mkdir()
        assert find_latest_checkpoint(str(tmp_path), ckpt_type="garbage") is None


class TestResolveModelPath:
    def test_hf_identifier_passthrough(self):
        assert resolve_model_path("facebook/opt-125m") == "facebook/opt-125m"

    def test_local_path_with_config(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        out = resolve_model_path(str(tmp_path))
        assert out == str(tmp_path)

    def test_local_path_with_subdir_config(self, tmp_path):
        sub = tmp_path / "model_v1"
        sub.mkdir()
        (sub / "config.json").write_text("{}")
        out = resolve_model_path(str(tmp_path))
        assert out == str(sub)

    def test_ambiguous_subdirs_raise(self, tmp_path):
        for name in ["a", "b"]:
            sub = tmp_path / name
            sub.mkdir()
            (sub / "config.json").write_text("{}")
        with pytest.raises(ValueError):
            resolve_model_path(str(tmp_path))

    def test_nonexistent_local_returns_resolved(self, tmp_path):
        # Not a HF id (starts with /) and doesn't exist — returns the abs path
        out = resolve_model_path(str(tmp_path / "missing"))
        assert "missing" in out

    def test_no_config_anywhere_returns_input_resolved(self, tmp_path):
        out = resolve_model_path(str(tmp_path))
        # Returns abs version of input
        assert os.path.abspath(str(tmp_path)) == out


class TestCleanupAndRemoveDir:
    def test_cleanup_removes_known_subdirs(self, tmp_path):
        for name in ["ray_results", "train_results", "lightning_logs", "logs"]:
            (tmp_path / name).mkdir()
            (tmp_path / name / "junk.log").write_text("x")
        cleanup(str(tmp_path))
        # cleanup recreates empty dirs after removing (rmtree + mkdir)
        for name in ["ray_results", "train_results", "lightning_logs", "logs"]:
            d = tmp_path / name
            assert d.exists()
            assert list(d.iterdir()) == []

    def test_cleanup_skips_missing(self, tmp_path):
        # Should not raise even if subdirs missing
        cleanup(str(tmp_path))

    def test_remove_dir_missing_noop(self, tmp_path):
        remove_dir(str(tmp_path / "does_not_exist"))  # no error

    def test_move_dir_missing_source_noop(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        # Source doesn't exist → silently no-op
        move_dir(str(tmp_path / "missing_src"), str(target))


class TestSerializableJson:
    """Indirect: ensure flatten_dict output is jsonable for typical configs."""

    def test_flatten_then_json(self):
        cfg = {"a": {"b": 1}, "c": "text", "d": [1, 2]}
        flat = flatten_dict(cfg)
        json.dumps(flat)  # would raise if not serializable
