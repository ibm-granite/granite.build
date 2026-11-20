# SPDX-License-Identifier: Apache-2.0

"""Tests for the post-export verification in ``src/export_checks.py``.

Each case stands for a way a Hugging Face export has quietly shipped a wrong model
in this project: a dropped ``rope_theta``, a re-saved tokenizer, a collapsed muP
multiplier. The fixtures under ``fixtures/`` are the two config shapes transformers
writes (top-level and ``rope_parameters``-nested) plus the damaged variants.
"""

import json
import pathlib
import shutil
import subprocess
import sys

import pytest
from export_checks import check_config, check_tokenizer

_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def _cfg(name: str) -> dict:
    return json.loads((_FIXTURES / f"config_{name}.json").read_text())


@pytest.fixture
def expected() -> dict:
    return json.loads((_FIXTURES / "expected_g4_350m_mup.json").read_text())


@pytest.fixture
def expected_sp(expected) -> dict:
    """The same run without muP: the recipe exports it as ``llama``."""
    return {
        k: v
        for k, v in expected.items()
        if k not in ("m_emb", "m_residual", "m_width", "attention_multiplier")
    }


# ---------------------------------------------------------------- check_config


def test_a_faithful_granite_export_reports_nothing(expected):
    assert check_config(_cfg("granite_faithful"), expected) == []


def test_an_empty_expected_dict_checks_nothing(expected):
    assert check_config(_cfg("granite_rope_missing"), {}) == []


def test_rope_theta_absent_everywhere_is_an_error(expected):
    errors = check_config(_cfg("granite_rope_missing"), expected)
    assert len(errors) == 1
    assert "rope_theta" in errors[0]


def test_rope_theta_nested_under_rope_parameters_is_accepted(expected):
    assert check_config(_cfg("granite_rope_nested"), expected) == []


def test_rope_theta_under_rope_scaling_is_accepted(expected):
    cfg = _cfg("granite_rope_nested")
    cfg["rope_scaling"] = cfg.pop("rope_parameters")
    assert check_config(cfg, expected) == []


def test_a_nested_rope_theta_that_disagrees_with_training_is_an_error(expected):
    cfg = _cfg("granite_rope_nested")
    cfg["rope_parameters"]["rope_theta"] = 10000
    errors = check_config(cfg, expected)
    assert len(errors) == 1
    assert "10000" in errors[0] and "rope_parameters.rope_theta" in errors[0]


def test_rope_theta_in_both_places_with_the_same_value_is_fine(expected):
    cfg = _cfg("granite_faithful")
    cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": 10000000}
    assert check_config(cfg, expected) == []


def test_rope_theta_in_both_places_with_different_values_is_an_error(expected):
    """Which copy the server honours depends on the transformers version."""
    errors = check_config(_cfg("granite_rope_disagree"), expected)
    assert len(errors) == 1
    assert "rope_theta" in errors[0]
    assert "rope_parameters.rope_theta" in errors[0]


def test_a_null_top_level_rope_theta_does_not_shadow_the_nested_one(expected):
    cfg = _cfg("granite_rope_nested")
    cfg["rope_theta"] = None
    assert check_config(cfg, expected) == []


def test_a_rope_theta_that_is_null_everywhere_is_an_error(expected):
    cfg = _cfg("granite_rope_nested")
    cfg["rope_theta"] = None
    cfg["rope_parameters"]["rope_theta"] = None
    errors = check_config(cfg, expected)
    assert len(errors) == 1
    assert "rope_theta" in errors[0]


def test_a_collapsed_mup_multiplier_is_an_error(expected):
    errors = check_config(_cfg("granite_mup_collapsed"), expected)
    assert len(errors) == 2
    assert any("residual_multiplier" in e and "m_residual" in e for e in errors)
    assert any("logits_scaling" in e and "m_width" in e for e in errors)


def test_a_multiplier_off_in_the_last_bit_is_an_error(expected):
    """Config scalars are a JSON round-trip of one float, so they compare exactly."""
    cfg = _cfg("granite_faithful")
    cfg["residual_multiplier"] = 0.263 + 2**-53
    errors = check_config(cfg, expected)
    assert len(errors) == 1
    assert "residual_multiplier" in errors[0]


def test_an_absent_multiplier_field_is_an_error(expected):
    cfg = _cfg("granite_faithful")
    del cfg["attention_multiplier"]
    errors = check_config(cfg, expected)
    assert len(errors) == 1
    assert "attention_multiplier" in errors[0]


def test_the_multipliers_are_not_checked_for_a_llama_export(expected_sp):
    assert check_config(_cfg("llama_faithful"), expected_sp) == []


def test_a_collapsed_mup_multiplier_is_an_error_for_a_moe_hybrid_export(expected):
    """``_export_granitemoehybrid_config`` writes the same four multiplier fields.

    Upstream's MoE/hybrid exporter (``tools/model_conversion/granitemoehybrid.py``,
    pinned at a9c6e2b8) sets ``embedding_multiplier`` / ``residual_multiplier`` /
    ``logits_scaling`` / ``attention_multiplier`` with exactly the same
    ``1 if ... is None`` fallback as the dense ``granite`` exporter, so a MoE muP run
    can lose a multiplier the same way -- and the recipe reaches that path whenever
    ``EXPORT_MODEL_TYPE: auto`` resolves to ``granitemoehybrid``.
    """
    errors = check_config(_cfg("granitemoehybrid_mup_collapsed"), expected)
    assert len(errors) == 2
    assert any("residual_multiplier" in e and "m_residual" in e for e in errors)
    assert any("logits_scaling" in e and "m_width" in e for e in errors)


def test_a_faithful_moe_hybrid_export_reports_nothing(expected):
    cfg = _cfg("granitemoehybrid_mup_collapsed")
    cfg["residual_multiplier"] = expected["m_residual"]
    cfg["logits_scaling"] = expected["m_width"]
    assert check_config(cfg, expected) == []


def test_an_absent_multiplier_field_is_an_error_for_a_moe_hybrid_export(expected):
    """A lost multiplier is normally written as ``1``, but absence is no better."""
    cfg = _cfg("granitemoehybrid_mup_collapsed")
    cfg["residual_multiplier"] = expected["m_residual"]
    cfg["logits_scaling"] = expected["m_width"]
    del cfg["embedding_multiplier"]
    errors = check_config(cfg, expected)
    assert len(errors) == 1
    assert "embedding_multiplier" in errors[0]


def test_max_position_embeddings_is_compared_against_sequence_length(expected):
    cfg = _cfg("granite_faithful")
    cfg["max_position_embeddings"] = 4096
    errors = check_config(cfg, expected)
    assert len(errors) == 1
    assert "max_position_embeddings" in errors[0] and "sequence_length" in errors[0]


def test_vocab_size_is_compared(expected):
    cfg = _cfg("granite_faithful")
    cfg["vocab_size"] = 100352 + 64
    assert len(check_config(cfg, expected)) == 1


def test_tie_word_embeddings_missing_is_not_the_same_as_false(expected):
    """HF defaults for it differ by architecture, so the file must state it."""
    cfg = _cfg("granite_faithful")
    del cfg["tie_word_embeddings"]
    errors = check_config(cfg, dict(expected, tie_word_embeddings=False))
    assert len(errors) == 1
    assert "tie_word_embeddings" in errors[0]


def test_tie_word_embeddings_false_matches_a_false_expectation(expected):
    cfg = _cfg("granite_faithful")
    cfg["tie_word_embeddings"] = False
    assert check_config(cfg, dict(expected, tie_word_embeddings=False)) == []


def test_tie_word_embeddings_does_not_accept_an_integer_stand_in(expected):
    cfg = _cfg("granite_faithful")
    cfg["tie_word_embeddings"] = 1
    errors = check_config(cfg, expected)
    assert len(errors) == 1
    assert "tie_word_embeddings" in errors[0]


def test_a_null_pad_token_id_on_both_sides_is_fine(expected):
    cfg = _cfg("granite_faithful")
    cfg["pad_token_id"] = None
    assert check_config(cfg, dict(expected, pad_token_id=None)) == []


def test_a_pad_token_id_null_only_in_the_export_is_an_error(expected):
    cfg = _cfg("granite_faithful")
    cfg["pad_token_id"] = None
    errors = check_config(cfg, expected)
    assert len(errors) == 1
    assert "pad_token_id" in errors[0]


def test_a_pad_token_id_null_only_in_training_is_an_error(expected):
    errors = check_config(_cfg("granite_faithful"), dict(expected, pad_token_id=None))
    assert len(errors) == 1
    assert "pad_token_id" in errors[0]


def test_a_model_type_the_recipe_did_not_ask_for_is_an_error(expected):
    errors = check_config(_cfg("llama_faithful"), dict(expected, model_type="granite"))
    assert any("model_type" in e for e in errors)


def test_every_config_failure_is_reported_not_just_the_first(expected):
    cfg = _cfg("granite_mup_collapsed")
    del cfg["rope_theta"]
    cfg["vocab_size"] = 1
    cfg["max_position_embeddings"] = 1
    cfg["pad_token_id"] = None
    errors = check_config(cfg, expected)
    assert len(errors) == 6
    for field in (
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "pad_token_id",
        "residual_multiplier",
        "logits_scaling",
    ):
        assert any(field in e for e in errors), field


# ------------------------------------------------------------- check_tokenizer


@pytest.fixture
def tok(tmp_path) -> tuple[pathlib.Path, pathlib.Path]:
    """A byte-identical src/dst pair of tokenizer directories."""
    src = tmp_path / "unsharded"
    dst = tmp_path / "hf"
    shutil.copytree(_FIXTURES / "tokenizer", src)
    shutil.copytree(_FIXTURES / "tokenizer", dst)
    return src, dst


@pytest.fixture
def tok_cfg() -> dict:
    return json.loads((_FIXTURES / "config_granite_faithful.json").read_text())


def test_an_identical_tokenizer_tree_reports_nothing(tok, tok_cfg):
    src, dst = tok
    assert check_tokenizer(src, dst, tok_cfg) == []


def test_a_resaved_tokenizer_json_is_a_byte_difference(tok, tok_cfg):
    """The real incident: save_pretrained leaked the trainer's truncation setting."""
    src, dst = tok
    data = json.loads((dst / "tokenizer.json").read_text())
    data["truncation"] = {"max_length": 466, "strategy": "LongestFirst"}
    (dst / "tokenizer.json").write_text(json.dumps(data, indent=2))
    errors = check_tokenizer(src, dst, tok_cfg)
    assert len(errors) == 1
    assert "tokenizer.json" in errors[0]


def test_a_whitespace_only_difference_is_still_a_byte_difference(tok, tok_cfg):
    """Byte-identity, not JSON equality: a re-serialisation is a re-save."""
    src, dst = tok
    data = json.loads((dst / "tokenizer.json").read_text())
    (dst / "tokenizer.json").write_text(json.dumps(data))
    errors = check_tokenizer(src, dst, tok_cfg)
    assert len(errors) == 1
    assert "tokenizer.json" in errors[0]


def test_a_same_sized_difference_is_caught(tok, tok_cfg):
    """A size or mtime check would miss this one."""
    src, dst = tok
    raw = (dst / "tokenizer.json").read_bytes()
    (dst / "tokenizer.json").write_bytes(raw.replace(b'"a": 0', b'"a": 9'))
    assert len(raw) == len((dst / "tokenizer.json").read_bytes())
    assert len(check_tokenizer(src, dst, tok_cfg)) == 1


def test_a_tokenizer_file_missing_from_the_export_is_an_error(tok, tok_cfg):
    """export_to_huggingface writes no tokenizer at all when it cannot load one."""
    src, dst = tok
    (dst / "tokenizer.json").unlink()
    errors = check_tokenizer(src, dst, tok_cfg)
    assert len(errors) == 1
    assert "tokenizer.json" in errors[0]


def test_an_empty_export_directory_reports_every_missing_file(tok, tok_cfg):
    src, dst = tok
    for path in dst.iterdir():
        path.unlink()
    errors = check_tokenizer(src, dst, tok_cfg)
    assert len(errors) == 3


def test_a_tokenizer_file_only_in_the_export_is_an_error(tok, tok_cfg):
    """It was synthesised by the exporter, so nothing trained it."""
    src, dst = tok
    (dst / "added_tokens.json").write_text('{"<|pad|>": 100256}\n')
    errors = check_tokenizer(src, dst, tok_cfg)
    assert len(errors) == 1
    assert "added_tokens.json" in errors[0]


def test_non_tokenizer_files_in_the_export_are_ignored(tok, tok_cfg):
    src, dst = tok
    (dst / "config.json").write_text(json.dumps(tok_cfg))
    (dst / "model.safetensors").write_bytes(b"\0")
    (dst / "generation_config.json").write_text("{}")
    assert check_tokenizer(src, dst, tok_cfg) == []


def test_a_source_directory_with_no_tokenizer_at_all_is_an_error(tmp_path, tok_cfg):
    src = tmp_path / "unsharded"
    dst = tmp_path / "hf"
    src.mkdir()
    dst.mkdir()
    errors = check_tokenizer(src, dst, tok_cfg)
    assert len(errors) == 1
    assert "no tokenizer" in errors[0].lower()


def test_a_missing_directory_is_an_error(tmp_path, tok_cfg):
    errors = check_tokenizer(tmp_path / "nope", tmp_path / "also-nope", tok_cfg)
    assert len(errors) == 2


def test_a_declared_token_resolving_to_another_id_is_an_error(tok, tok_cfg):
    src, dst = tok
    errors = check_tokenizer(src, dst, dict(tok_cfg, pad_token_id=100255))
    assert len(errors) == 1
    assert "pad_token" in errors[0] and "100255" in errors[0]


def test_a_declared_token_outside_the_vocabulary_is_an_error(tok, tok_cfg):
    src, dst = tok
    for path in (src, dst):
        data = json.loads((path / "tokenizer.json").read_text())
        data["added_tokens"] = [data["added_tokens"][1]]
        (path / "tokenizer.json").write_text(json.dumps(data, indent=2))
    errors = check_tokenizer(src, dst, tok_cfg)
    assert len(errors) == 1
    assert "<|pad|>" in errors[0]


def test_a_token_the_tokenizer_does_not_declare_is_not_compared(tok, tok_cfg):
    """A tokenizer with no pad_token is a style choice, not a mis-export."""
    src, dst = tok
    for path in (src, dst):
        for name in ("tokenizer_config.json", "special_tokens_map.json"):
            data = json.loads((path / name).read_text())
            del data["pad_token"]
            (path / name).write_text(json.dumps(data, indent=2))
    assert check_tokenizer(src, dst, tok_cfg) == []


def test_a_declared_token_with_a_null_id_in_the_config_is_an_error(tok, tok_cfg):
    src, dst = tok
    errors = check_tokenizer(src, dst, dict(tok_cfg, pad_token_id=None))
    assert len(errors) == 1
    assert "pad_token" in errors[0]


def test_the_ids_come_from_added_tokens_json_too(tok, tok_cfg):
    src, dst = tok
    for path in (src, dst):
        data = json.loads((path / "tokenizer.json").read_text())
        data["added_tokens"] = []
        (path / "tokenizer.json").write_text(json.dumps(data, indent=2))
        (path / "added_tokens.json").write_text(
            '{"<|pad|>": 100256, "<|end_of_text|>": 100257}\n'
        )
    assert check_tokenizer(src, dst, tok_cfg) == []


def test_every_tokenizer_failure_is_reported_not_just_the_first(tok, tok_cfg):
    src, dst = tok
    (dst / "tokenizer_config.json").unlink()
    raw = (dst / "tokenizer.json").read_bytes()
    (dst / "tokenizer.json").write_bytes(raw.replace(b'"a": 0', b'"a": 9'))
    (dst / "vocab.json").write_text("{}")
    errors = check_tokenizer(src, dst, tok_cfg)
    assert len(errors) == 3


def test_unreadable_json_is_reported_not_raised(tok, tok_cfg):
    src, dst = tok
    for path in (src, dst):
        (path / "tokenizer_config.json").write_text("{ not json")
    errors = check_tokenizer(src, dst, tok_cfg)
    assert len(errors) == 1
    assert "tokenizer_config.json" in errors[0]


# ------------------------------------------------------------------ purity


def test_the_module_imports_under_python_isolated_mode():
    """It runs inside the export pod after the heavy work: stdlib only, no torch."""
    code = (
        f"import sys; sys.path.insert(0, {str(_SRC)!r}); import export_checks;"
        "assert not {m for m in sys.modules if m.split('.')[0] in"
        " {'torch', 'transformers', 'yaml', 'numpy', 'safetensors'}};"
        "print(sorted(export_checks.__all__))"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "check_config" in out.stdout and "check_tokenizer" in out.stdout
