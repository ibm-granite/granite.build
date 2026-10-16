import math
import os
import sys
import types
from pathlib import Path

import pytest
import tomllib


def _pyproject() -> dict:
    return tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())


def test_mlx_optional_extra_declared():
    data = _pyproject()
    extras = data["project"]["optional-dependencies"]
    assert "mlx" in extras, "expected an [mlx] optional extra"
    joined = " ".join(extras["mlx"])
    assert "mlx-lm" in joined
    assert "mlx" in joined
    # gated to Apple Silicon so Linux/CUDA installs never pull it
    assert "darwin" in joined
    assert "arm64" in joined


def test_supported_tuning_algos():
    from autotune.mlx_backend import supported_tuning_algos

    assert supported_tuning_algos() == {"sft", "lora", "qlora"}


def test_require_mlx_raises_helpful_error_when_missing(monkeypatch):
    # Simulate mlx not installed: block the import.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "mlx_lm" or name.startswith("mlx_lm.") or name == "mlx" or name.startswith("mlx."):
            raise ImportError("No module named mlx")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from autotune.mlx_backend import require_mlx

    with pytest.raises(ImportError, match=r"\[mlx\]"):
        require_mlx()


def test_cache_key_reflects_model_and_quant():
    from autotune.mlx_backend import cache_key

    assert cache_key("HuggingFaceTB/SmolLM2-135M-Instruct", True) == "SmolLM2-135M-Instruct__q4"
    assert cache_key("/tmp/local-model/", False) == "local-model__bf16"


def _install_fake_mlx_lm(monkeypatch, record):
    """Inject a fake mlx_lm exposing convert() that writes a config.json."""
    fake = types.ModuleType("mlx_lm")

    def convert(hf_path, mlx_path, quantize=False, q_bits=None, q_group_size=None, **kw):
        record.append({"hf_path": hf_path, "mlx_path": mlx_path, "quantize": quantize, "q_bits": q_bits})
        if os.path.exists(mlx_path):
            raise ValueError(f"Cannot save to the path {mlx_path} as it already exists.")
        os.makedirs(mlx_path)  # NOT exist_ok=True -- mirror real mlx_lm.convert's precondition
        with open(os.path.join(mlx_path, "config.json"), "w") as f:
            f.write("{}")

    fake.convert = convert
    fake.core = types.ModuleType("mlx.core")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", fake.core)
    return record


def test_ensure_mlx_model_converts_then_caches(tmp_path, monkeypatch):
    from autotune.mlx_backend import ensure_mlx_model

    calls = _install_fake_mlx_lm(monkeypatch, [])

    dest1 = ensure_mlx_model("org/Tiny-Model", quantize=True, cache_dir=str(tmp_path))
    assert dest1.endswith("Tiny-Model__q4")
    assert os.path.isfile(os.path.join(dest1, "config.json"))
    assert len(calls) == 1 and calls[0]["quantize"] is True and calls[0]["q_bits"] == 4

    # Second call hits the cache: convert() must NOT run again.
    dest2 = ensure_mlx_model("org/Tiny-Model", quantize=True, cache_dir=str(tmp_path))
    assert dest2 == dest1
    assert len(calls) == 1


def test_translate_config_lora_maps_and_computes_iters():
    from autotune.mlx_backend import translate_config

    tc = {"tuning_algorithm": "lora", "max_length": 512, "mlx_num_layers": 8}
    params = {
        "per_device_train_batch_size": 4,
        "learning_rate": 3e-5,
        "r": 16,
        "alpha_ratio": 2.0,
        "lora_dropout": 0.05,
        "gradient_accumulation_steps": 2,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
    }
    cfg = translate_config(tc, params, epochs=3, n_train_examples=100)

    assert cfg["fine_tune_type"] == "lora"
    assert cfg["quantize"] is False
    assert cfg["batch_size"] == 4
    # iters = ceil(100/4) * 3 = 75
    assert cfg["iters"] == math.ceil(100 / 4) * 3 == 75
    assert cfg["learning_rate"] == 3e-5
    assert cfg["max_seq_length"] == 512
    assert cfg["num_layers"] == 8
    assert cfg["grad_accumulation_steps"] == 2
    assert cfg["warmup"] == round(0.1 * 75)
    assert cfg["lr_scheduler_type"] == "cosine"
    assert cfg["lora_parameters"] == {"rank": 16, "dropout": 0.05, "scale": 2.0}


def test_translate_config_qlora_sets_quantize():
    from autotune.mlx_backend import translate_config

    tc = {"tuning_algorithm": "qlora", "max_length": 256}
    params = {"per_device_train_batch_size": 8, "learning_rate": 1e-5, "r": 8, "alpha_ratio": 1.0, "lora_dropout": 0.0}
    cfg = translate_config(tc, params, epochs=1, n_train_examples=64)
    assert cfg["quantize"] is True
    assert cfg["fine_tune_type"] == "lora"
    assert cfg["iters"] == math.ceil(64 / 8) * 1 == 8


def test_translate_config_sft_is_full_no_lora_params():
    from autotune.mlx_backend import translate_config

    tc = {"tuning_algorithm": "sft", "max_length": 128}
    params = {"per_device_train_batch_size": 2, "learning_rate": 5e-6}
    cfg = translate_config(tc, params, epochs=2, n_train_examples=10)
    assert cfg["fine_tune_type"] == "full"
    assert cfg["quantize"] is False
    assert cfg["lora_parameters"] is None
    assert cfg["iters"] == math.ceil(10 / 2) * 2 == 10


def test_translate_config_iters_never_zero():
    from autotune.mlx_backend import translate_config

    tc = {"tuning_algorithm": "lora", "max_length": 128}
    params = {"per_device_train_batch_size": 999, "learning_rate": 1e-5, "r": 8, "alpha_ratio": 2.0}
    cfg = translate_config(tc, params, epochs=1, n_train_examples=1)
    assert cfg["iters"] >= 1


def test_build_records_string_inputs():
    from autotune.mlx_backend import build_records

    recs, is_chat = build_records(["hello"], ["world"])
    assert is_chat is False
    assert recs == [{"prompt": "hello", "completion": "world"}]


def test_build_records_message_list_inputs():
    from autotune.mlx_backend import build_records

    msgs = [{"role": "user", "content": "hi"}]
    recs, is_chat = build_records([msgs], ["yo"])
    assert is_chat is True
    assert recs[0]["messages"][-1] == {"role": "assistant", "content": "yo"}
    assert recs[0]["messages"][0] == {"role": "user", "content": "hi"}


def _install_full_fake_mlx(monkeypatch, recorder):
    # mlx.core / mlx.nn / mlx.optimizers
    core = types.ModuleType("mlx.core")
    core.save_safetensors = lambda path, d: recorder.setdefault("saved", []).append(path)
    nn = types.ModuleType("mlx.nn")
    opts = types.ModuleType("mlx.optimizers")
    opts.Adam = lambda learning_rate=None: {"opt": "adam", "lr": learning_rate}
    opts.cosine_decay = lambda lr, steps: ("cos", lr, steps)
    opts.linear_schedule = lambda a, b, steps: ("lin", a, b, steps)
    opts.join_schedules = lambda scheds, bounds: ("join", scheds, bounds)
    utils_mod = types.ModuleType("mlx.utils")
    utils_mod.tree_flatten = lambda tree: [("a.b", 1)]
    mlx_mod = types.ModuleType("mlx")
    mlx_mod.core, mlx_mod.nn, mlx_mod.optimizers, mlx_mod.utils = core, nn, opts, utils_mod

    # mlx_lm.utils.load
    class _Model:
        layers = [object(), object(), object(), object()]

        def freeze(self):
            recorder["frozen"] = True

        def unfreeze(self):
            recorder["unfrozen"] = True

        def trainable_parameters(self):
            return {}

        def save_weights(self, path):
            recorder.setdefault("saved", []).append(path)

    class _Tok:
        def save_pretrained(self, dest):
            recorder["tok_saved"] = dest

    lm_utils = types.ModuleType("mlx_lm.utils")
    lm_utils.load = lambda path, **k: (_Model(), _Tok())
    lm_utils.save_config = lambda cfg, path: recorder.setdefault("saved", []).append(path)

    # mlx_lm.tuner.trainer
    trainer = types.ModuleType("mlx_lm.tuner.trainer")

    class TrainingArgs:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            recorder["targs"] = kw

    def train(model, args, optimizer, train_dataset, val_dataset, training_callback=None):
        recorder["train_called"] = True
        if training_callback:
            training_callback.on_train_loss_report({"train_loss": 1.5, "iteration": 1})
            training_callback.on_val_loss_report({"val_loss": 1.2, "iteration": 1})

    trainer.TrainingArgs = TrainingArgs
    trainer.train = train

    # mlx_lm.tuner.utils
    tuner_utils = types.ModuleType("mlx_lm.tuner.utils")
    tuner_utils.linear_to_lora_layers = lambda *a, **k: recorder.setdefault("lora_call", a)
    tuner_utils.print_trainable_parameters = lambda m: None

    # mlx_lm.tuner.datasets
    datasets = types.ModuleType("mlx_lm.tuner.datasets")
    datasets.CacheDataset = lambda ds: ds

    def _create_dataset(data, tok, cfg):
        recorder["dataset_cfg"] = cfg
        return {"data": data}

    datasets.create_dataset = _create_dataset

    lm = types.ModuleType("mlx_lm")
    lm.convert = lambda **k: None
    tuner_pkg = types.ModuleType("mlx_lm.tuner")

    for name, mod in {
        "mlx": mlx_mod,
        "mlx.core": core,
        "mlx.nn": nn,
        "mlx.optimizers": opts,
        "mlx.utils": utils_mod,
        "mlx_lm": lm,
        "mlx_lm.utils": lm_utils,
        "mlx_lm.tuner": tuner_pkg,
        "mlx_lm.tuner.trainer": trainer,
        "mlx_lm.tuner.utils": tuner_utils,
        "mlx_lm.tuner.datasets": datasets,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return recorder


def test_run_training_wires_mlx_and_returns_losses(tmp_path, monkeypatch):
    rec = _install_full_fake_mlx(monkeypatch, {})
    from autotune.mlx_backend import run_training

    cfg = {
        "fine_tune_type": "lora",
        "quantize": False,
        "batch_size": 2,
        "iters": 4,
        "learning_rate": 1e-5,
        "max_seq_length": 128,
        "num_layers": 2,
        "grad_accumulation_steps": 1,
        "warmup": 0,
        "lr_scheduler_type": "cosine",
        "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 2.0},
    }
    recs = [{"prompt": "a", "completion": "b"}]
    out = run_training(str(tmp_path), cfg, recs, False, recs, str(tmp_path / "adapters"))

    assert rec["train_called"] is True
    assert out["train_loss"] == 1.5 and out["eval_loss"] == 1.2
    assert out["loss"] == 1.2 and out["done"] is True
    assert rec["targs"]["batch_size"] == 1  # clamped from 2 to the 1-record dataset size
    assert rec["targs"]["iters"] == 4
    assert "model" in out and "tokenizer" in out
    assert rec["dataset_cfg"].mask_prompt is False


def test_run_training_does_not_clamp_when_dataset_large_enough(tmp_path, monkeypatch):
    rec = _install_full_fake_mlx(monkeypatch, {})
    from autotune.mlx_backend import run_training

    cfg = {
        "fine_tune_type": "lora",
        "quantize": False,
        "batch_size": 2,
        "iters": 4,
        "learning_rate": 1e-5,
        "max_seq_length": 128,
        "num_layers": 2,
        "grad_accumulation_steps": 1,
        "warmup": 0,
        "lr_scheduler_type": "cosine",
        "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 2.0},
    }
    recs = [{"prompt": "a", "completion": "b"}] * 4
    out = run_training(str(tmp_path), cfg, recs, False, recs, str(tmp_path / "adapters"))

    assert rec["targs"]["batch_size"] == 2  # min(2, 4, 4) == 2, unchanged
    assert out["done"] is True


def test_save_output_lora_writes_adapter(tmp_path, monkeypatch):
    rec = _install_full_fake_mlx(monkeypatch, {})
    from autotune.mlx_backend import run_training, save_output

    cfg = {
        "fine_tune_type": "lora",
        "quantize": False,
        "batch_size": 2,
        "iters": 2,
        "learning_rate": 1e-5,
        "max_seq_length": 64,
        "num_layers": 2,
        "grad_accumulation_steps": 1,
        "warmup": 0,
        "lr_scheduler_type": "cosine",
        "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 2.0},
    }
    recs = [{"prompt": "a", "completion": "b"}]
    out = run_training(str(tmp_path), cfg, recs, False, recs, str(tmp_path / "adapters"))
    dest = tmp_path / "model_out"
    save_output(out, cfg, str(dest))
    # save_config + save_safetensors both recorded a write under dest
    assert any(str(dest) in p for p in rec["saved"])


def test_autotune_mlx_preset_loads_and_has_mlx_defaults():
    import yaml

    p = Path(__file__).resolve().parents[1] / "autotune" / "configs" / "autotune_mlx.yaml"
    assert p.exists(), "autotune_mlx.yaml preset missing"
    data = yaml.safe_load(p.read_text())
    tc = data["training_config"]
    assert tc["num_gpus_per_trial"]["default"] == 1
    assert tc["use_flash_attention"]["default"] == "eager"
    assert tc["mlx_num_layers"]["default"] >= 1
    assert data["tune_config"]["max_concurrent_trials"]["default"] == 1
