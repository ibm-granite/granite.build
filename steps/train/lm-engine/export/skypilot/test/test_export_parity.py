# SPDX-License-Identifier: Apache-2.0

"""End-to-end export of a tiny real model, inside the lm-engine image.

Everything in :mod:`test_export` monkeypatches the two upstream stages, which proves the
sequencing and says nothing about whether the conversion itself is faithful. This suite
is the other half: it builds a 2-layer ``GPTBaseConfig`` model, saves it the way
``lm_engine.training.unshard`` would, runs the *real*
``tools.model_conversion.export_to_huggingface``, copies the fixture tokenizer in, runs
the real checks, and then runs :func:`export.check_parity` over both models.

**It SKIPS outside the image.** ``lm_engine`` and ``tools`` only exist on the family
image built by ``steps/train/lm-engine/Makefile`` (``tools/`` is not an installed
package -- it lives in the source tree at ``/app/src`` on ``PYTHONPATH``), so on a
developer host every test here is skipped. It is written to run unattended in the image:
no host paths, no network, CPU only, seconds of compute.

It also *prints* the observed max |delta logits| for each model_type. That number is the
evidence for the step's ``parity_tolerance`` default, so run it with ``-s`` (or ``-rA``)
when you want to see it::

    pytest test/test_export_parity.py -s

For reference, upstream's own ``tests/model_conversion_test.py`` holds the same
comparison to ``atol=3e-7`` in fp32 (``2.5e-5`` for ``granitemoehybrid``) on random
weights, so anything in that neighbourhood means the conversion is exact and the step's
0.05 default has four orders of magnitude of headroom.

``granitemoehybrid`` is not covered here: its exporter requires Mamba2 sequence mixers,
whose kernels are a separate dependency, and upstream already tests that path. What this
step adds on top -- that the four muP multipliers are verified for it too -- is covered
on the host, in ``test_export_checks.py``.
"""

import json
import pathlib
import shutil

import pytest

pytest.importorskip("lm_engine", reason="lm_engine only exists inside the family image")
pytest.importorskip(
    "tools.model_conversion",
    reason="tools/ is not an installed package; it lives at /app/src in the image",
)

import export  # noqa: E402

_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"

#: Big enough to hold the fixture tokenizer's special ids (100256, 100257) so the
#: exported config and the copied tokenizer agree and ``check_tokenizer`` is exercised
#: for real rather than against a stand-in.
VOCAB_SIZE = 100352
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_HEADS = 16
NUM_KV_HEADS = 4
INTERMEDIATE_SIZE = 256
SEQUENCE_LENGTH = 1024

BOS = 100257
EOS = 100257
PAD = 100256

#: Set explicitly rather than left to ``attention_multiplier_method``, so the exported
#: Granite config states a real number and the muP check has something to compare.
ATTENTION_MULTIPLIER = 0.125

#: muP multipliers for the ``granite`` case, all four distinct and none equal to ``1``:
#: upstream writes ``1`` for a multiplier it lost, so a value of ``1`` here would make a
#: lost multiplier indistinguishable from a kept one.
M_EMB = 12.0
M_WIDTH = 4.0
M_RESIDUAL = 0.263


def _config(mup: bool):
    """A tiny ``GPTBaseConfig``, shaped after upstream's ``get_dense_test_config``.

    rope + rmsnorm + swiglu + ``mlp_type: MLP`` is the one shape both
    ``_export_granite_config`` and ``_export_llama_config`` accept (each asserts on all
    four). ``mup=False`` additionally leaves every multiplier ``None``, which
    ``_export_llama_config`` asserts for ``attention_multiplier``.
    """
    from lm_engine.training.models import GPTBaseConfig

    return GPTBaseConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=SEQUENCE_LENGTH,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        position_embedding_type="rope",
        normalization_function="rmsnorm",
        layer_norm_epsilon=1e-5,
        embedding_dropout=0,
        initializer_range=0.02,
        rope_theta=10000000,
        rope_scaling=None,
        rope_dim=None,
        init_method="normal",
        embedding_init_method="normal",
        use_depth_scaled_init=False,
        router_aux_loss_coef=0.001,
        tie_word_embeddings=True,
        bos_token_id=BOS,
        eos_token_id=EOS,
        pad_token_id=PAD,
        m_emb=M_EMB if mup else None,
        m_width=M_WIDTH if mup else None,
        m_residual=M_RESIDUAL if mup else None,
        sequence_mixer_blocks=[
            {
                "sequence_mixer_type": "softmax_attention",
                "add_bias": False,
                "num_attention_heads": NUM_HEADS,
                "num_key_value_heads": NUM_KV_HEADS,
                "head_dim": None,
                "softmax_dropout": 0,
                "dropout": 0,
                "attention_multiplier": ATTENTION_MULTIPLIER if mup else None,
                "attention_multiplier_method": None if mup else "1 / sqrt(head_dim)",
                "attention_gate": False,
                "exclusive_self_attention": False,
                "sliding_window": None,
            }
            for _ in range(NUM_LAYERS)
        ],
        mlp_blocks=[
            {
                "mlp_type": "MLP",
                "activation_function": "swiglu",
                "add_bias": False,
                "intermediate_size": INTERMEDIATE_SIZE,
                "dropout": 0,
            }
            for _ in range(NUM_LAYERS)
        ],
    )


def _save_unsharded(config, path: pathlib.Path) -> None:
    """Write an unsharded lm-engine checkpoint, exactly as stage 1 leaves one.

    ``LLMAdapter_HF.save_pretrained`` is what ``lm_engine.training.unshard`` reaches
    through ``ModelWrapper.save_pretrained``: ``config.json`` plus the safetensors, and
    deliberately no tokenizer -- which is why ``export_to_huggingface`` finds none to
    re-save and the step's own copy is the only tokenizer the export ever gets.
    """
    import lm_engine.training  # noqa: F401  # registers the lm_engine model classes
    import torch
    from lm_engine.training.hf_adapter import LLMAdapter_HF, get_causal_lm_class

    torch.manual_seed(1234)
    model = get_causal_lm_class(config.model_type)(config).to(torch.float32)
    model.eval()
    LLMAdapter_HF(model).save_pretrained(str(path), safe_serialization=True)


def _expected(mup: bool) -> dict:
    expected = {
        "rope_theta": 10000000,
        "vocab_size": VOCAB_SIZE,
        "tie_word_embeddings": True,
        "bos_token_id": BOS,
        "eos_token_id": EOS,
        "pad_token_id": PAD,
        "sequence_length": SEQUENCE_LENGTH,
    }
    if mup:
        expected.update(
            m_emb=M_EMB,
            m_width=M_WIDTH,
            m_residual=M_RESIDUAL,
            attention_multiplier=ATTENTION_MULTIPLIER,
        )
    return expected


@pytest.mark.parametrize(
    "model_type, mup", [("llama", False), ("granite", True)], ids=["llama", "granite"]
)
def test_a_tiny_model_round_trips_through_the_real_export(
    tmp_path, model_type, mup, capsys
):
    unsharded = tmp_path / "unsharded"
    hf_dir = tmp_path / "hf"
    tokenizer_dir = tmp_path / "tok"
    shutil.copytree(_FIXTURES / "tokenizer", tokenizer_dir)

    _save_unsharded(_config(mup), unsharded)

    export.convert(unsharded, model_type, hf_dir)
    copied = export.copy_tokenizer(tokenizer_dir, hf_dir)
    assert sorted(copied) == [
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    assert export.run_checks(hf_dir, tokenizer_dir, _expected(mup)) == []

    delta = export.check_parity(unsharded, hf_dir)
    with capsys.disabled():
        print(
            f"\nPARITY {model_type}: max |delta logits| = {delta:.3e} "
            f"(step default parity_tolerance = 0.05)"
        )
    assert delta < 0.05


def test_the_exported_config_states_the_trained_rope_theta(tmp_path):
    """The one incident the whole step exists for, end to end on a real export."""
    unsharded = tmp_path / "unsharded"
    hf_dir = tmp_path / "hf"
    _save_unsharded(_config(mup=False), unsharded)
    export.convert(unsharded, "llama", hf_dir)

    cfg = json.loads((hf_dir / "config.json").read_text())
    sites = dict(
        [("rope_theta", cfg.get("rope_theta"))]
        + [
            (f"{parent}.rope_theta", (cfg.get(parent) or {}).get("rope_theta"))
            for parent in ("rope_parameters", "rope_scaling")
            if isinstance(cfg.get(parent), dict)
        ]
    )
    assert 10000000 in sites.values(), sites


def test_the_mup_multipliers_survive_a_real_granite_export(tmp_path):
    unsharded = tmp_path / "unsharded"
    hf_dir = tmp_path / "hf"
    _save_unsharded(_config(mup=True), unsharded)
    export.convert(unsharded, "granite", hf_dir)

    cfg = json.loads((hf_dir / "config.json").read_text())
    assert cfg["embedding_multiplier"] == pytest.approx(M_EMB)
    assert cfg["logits_scaling"] == pytest.approx(M_WIDTH)
    assert cfg["residual_multiplier"] == pytest.approx(M_RESIDUAL)
    assert cfg["attention_multiplier"] == pytest.approx(ATTENTION_MULTIPLIER)
