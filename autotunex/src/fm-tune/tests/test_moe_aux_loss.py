"""Pins autotune/trainers/_moe.py against the real transformers model classes (no weights)."""

import pytest

transformers = pytest.importorskip("transformers")

from autotune.trainers._moe import router_aux_loss_coef  # noqa: E402


def _meta_model(config):
    from accelerate import init_empty_weights

    with init_empty_weights():
        return transformers.AutoModelForCausalLM.from_config(config)


def _tiny(config_cls, **kw):
    base = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    base.update(kw)
    return config_cls(**base)


def test_granite_moe_cannot_record_router_logits():
    """transformers 5.9's GraniteMoe has no router_logits recorder: aux_loss is the int 0."""
    cfg = _tiny(transformers.GraniteMoeConfig, num_local_experts=4, num_experts_per_tok=2)
    assert router_aux_loss_coef(_meta_model(cfg)) == 0.0


def test_granite_hybrid_dense_mlp_is_still_treated_as_moe_by_trl():
    """Granite 4.0-H micro: zero experts, but the config carries output_router_logits."""
    cfg = _tiny(
        transformers.GraniteMoeHybridConfig,
        num_local_experts=0,
        layer_types=["mamba", "attention"],
        mamba_n_heads=4,
        mamba_d_head=32,
        mamba_d_state=16,
        mamba_n_groups=1,
        shared_intermediate_size=128,
    )
    assert router_aux_loss_coef(_meta_model(cfg)) == 0.0


def test_mixtral_records_router_logits_and_keeps_trls_default():
    cfg = _tiny(transformers.MixtralConfig, num_local_experts=4, num_experts_per_tok=2)
    assert router_aux_loss_coef(_meta_model(cfg)) is None


def test_dense_granite_is_not_moe():
    cfg = _tiny(transformers.GraniteConfig)
    assert router_aux_loss_coef(_meta_model(cfg)) is None


def test_peft_wrapped_model_is_unwrapped():
    peft = pytest.importorskip("peft")
    cfg = _tiny(transformers.GraniteMoeConfig, num_local_experts=4, num_experts_per_tok=2)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    wrapped = peft.get_peft_model(model, peft.LoraConfig(r=4, target_modules=["q_proj"]))
    assert router_aux_loss_coef(wrapped) == 0.0


@pytest.mark.parametrize(
    "path",
    [
        "autotune/trainers/driver_single_trl.py",
        "autotune/trainers/driver_multi_trl_ds.py",
        "autotune/trainers/driver_multi_trl_fsdp.py",
    ],
)
def test_trl_drivers_apply_it(path):
    from pathlib import Path

    assert "moe_router_aux_loss_coef(model)" in (Path(__file__).resolve().parents[1] / path).read_text()
