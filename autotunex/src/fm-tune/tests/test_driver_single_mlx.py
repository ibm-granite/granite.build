import types


def _patch_backend(monkeypatch, calls):
    """Replace mlx_backend functions with recorders; stub ray.tune."""
    import autotune.trainers.driver_single_mlx as drv

    monkeypatch.setattr(
        drv.mlx_backend,
        "ensure_mlx_model",
        lambda model, quantize, **k: calls.setdefault("ensure", (model, quantize)) or "/mlx/model",
    )
    monkeypatch.setattr(
        drv.mlx_backend,
        "translate_config",
        lambda tc, params, epochs, n_train_examples: (
            calls.setdefault("translate", {"epochs": epochs, "n": n_train_examples, "algo": tc.get("tuning_algorithm")})
            or {"fine_tune_type": "lora", "quantize": tc.get("tuning_algorithm") == "qlora"}
        ),
    )
    monkeypatch.setattr(
        drv.mlx_backend,
        "build_records",
        lambda inputs, outputs: ([{"prompt": inputs[0], "completion": outputs[0]}], False),
    )
    monkeypatch.setattr(
        drv.mlx_backend,
        "run_training",
        lambda *a, **k: {
            "loss": 0.5,
            "train_loss": 0.6,
            "eval_loss": 0.5,
            "done": True,
            "model": object(),
            "tokenizer": object(),
        },
    )
    saved = {}
    monkeypatch.setattr(drv.mlx_backend, "save_output", lambda res, cfg, dest: saved.update({"dest": dest}))
    calls["saved"] = saved

    # stub tune.report + trial id
    reported = {}
    fake_ctx = types.SimpleNamespace(get_trial_id=lambda: "trialX")
    monkeypatch.setattr(drv.tune, "report", lambda d: reported.update(d))
    monkeypatch.setattr(drv.tune, "get_context", lambda: fake_ctx)
    calls["reported"] = reported

    # stub dataset loading to avoid disk I/O
    class _DS(list):
        def __getitem__(self, k):
            if isinstance(k, str):
                return ["hello"] if k == "input" else ["world"]
            return list.__getitem__(self, k)

        def select(self, rng):
            return self

    monkeypatch.setattr(drv, "_load_dataset_from_file", lambda p: _DS([0]))
    return calls


def _config(algo="lora", hpo=False):
    return {
        "training_config": {
            "tuning_algorithm": algo,
            "peft_type": "LORA",
            "model_name_or_path": "org/Tiny",
            "train_file": "t.jsonl",
            "validation_file": "v.jsonl",
            "output_dir": "/tmp/out",
            "output_model_name": "m",
            "max_length": 128,
            "num_train_epochs": 2,
            "hpo_num_epochs": 1,
            "hpo_search": hpo,
            "save_model": not hpo,
            "backend": "mlx",
        },
        "training_rl_config": {},
        "tuner_flags": {"learning_rate": False, "per_device_train_batch_size": False},
        "tune_config": {},
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 1,
    }


def test_driver_reports_metrics_and_calls_backend(monkeypatch):
    calls = _patch_backend(monkeypatch, {})
    from autotune.trainers.driver_single_mlx import train_driver_single_gpu

    result = train_driver_single_gpu(_config(algo="lora", hpo=False))
    assert result["done"] is True and result["loss"] == 0.5
    assert calls["ensure"] == ("org/Tiny", False)  # lora => not quantized
    assert calls["reported"]["loss"] == 0.5
    assert calls["saved"]["dest"].endswith("/models/m")  # final training saved
    # model/tokenizer stripped from the reported dict
    assert "model" not in calls["reported"] and "tokenizer" not in calls["reported"]


def test_driver_qlora_requests_quantized_convert(monkeypatch):
    calls = _patch_backend(monkeypatch, {})
    from autotune.trainers.driver_single_mlx import train_driver_single_gpu

    train_driver_single_gpu(_config(algo="qlora", hpo=True))
    assert calls["ensure"] == ("org/Tiny", True)  # qlora => quantized
    assert calls["translate"]["epochs"] == 1  # hpo => hpo_num_epochs
    assert "saved" in calls and "dest" not in calls["saved"]  # hpo: no save
