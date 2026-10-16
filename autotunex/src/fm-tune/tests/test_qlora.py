"""Tests for QLoRA (4-bit quantized LoRA) support.

Covers:
- constants registration (qlora is a known tuning algo mapping to PeftType.LORA)
- the shipped autotune.yaml qlora tuner section (well-formed, mirrors lora)
- the get_qlora_quantization_config / prepare_qlora_model helpers

These are pure-logic tests: no model loading, no GPU, no bitsandbytes runtime
required (the BitsAndBytesConfig construction is mocked so the test runs without
a bitsandbytes install).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import yaml
from peft import PeftType

from autotune.config import AutotuneConfig
from autotune.constants import AUTOTUNE_TUNING_ALGO, AUTOTUNE_TUNING_TO_PEFT_TYPE

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTOTUNE_YAML = REPO_ROOT / "autotune" / "configs" / "autotune.yaml"


class TestConstants:
    def test_qlora_is_a_known_algo(self):
        assert "qlora" in AUTOTUNE_TUNING_ALGO

    def test_qlora_maps_to_lora_peft_type(self):
        # QLoRA == LoRA on a quantized base; PEFT has no dedicated QLoRA type.
        assert AUTOTUNE_TUNING_TO_PEFT_TYPE["qlora"] == PeftType.LORA
        assert AUTOTUNE_TUNING_TO_PEFT_TYPE["qlora"] == AUTOTUNE_TUNING_TO_PEFT_TYPE["lora"]


class TestShippedYamlSection:
    def _tuners(self):
        raw = yaml.safe_load(AUTOTUNE_YAML.read_text())
        return raw["tuners_config"]

    def test_qlora_section_exists(self):
        tuners = self._tuners()
        assert "qlora" in tuners
        assert tuners["qlora"]["title"] == "QLoRA"

    def test_qlora_hyperparams_match_lora(self):
        # 4-bit NF4 is fixed in the driver, so the QLoRA tunable surface is
        # identical to LoRA's — same hyperparams, same for_tuner flags.
        tuners = self._tuners()
        lora_hp = tuners["lora"]["hyperparams"]
        qlora_hp = tuners["qlora"]["hyperparams"]
        assert set(qlora_hp.keys()) == set(lora_hp.keys())
        for key in lora_hp:
            assert qlora_hp[key]["for_tuner"] == lora_hp[key]["for_tuner"], key
            assert qlora_hp[key]["default"] == lora_hp[key]["default"], key

    def test_config_lookup_finds_qlora(self):
        cfg = AutotuneConfig()
        cfg.load(str(AUTOTUNE_YAML))
        section = cfg.get_tuner_config_dict("qlora")
        assert section
        assert section["title"] == "QLoRA"


class TestQuantizationHelper:
    def test_get_qlora_quantization_config_4bit_nf4(self):
        # Mock BitsAndBytesConfig so the helper's argument wiring is verified
        # without requiring a bitsandbytes install (its post_init checks the
        # installed bnb version).
        with patch("autotune.utils.BitsAndBytesConfig") as bnb:
            from autotune.utils import get_qlora_quantization_config

            get_qlora_quantization_config()
            bnb.assert_called_once_with(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

    def test_get_qlora_quantization_config_custom_compute_dtype(self):
        with patch("autotune.utils.BitsAndBytesConfig") as bnb:
            from autotune.utils import get_qlora_quantization_config

            get_qlora_quantization_config(compute_dtype=torch.float16)
            _, kwargs = bnb.call_args
            assert kwargs["bnb_4bit_compute_dtype"] == torch.float16

    def test_prepare_qlora_model_delegates_to_peft(self):
        sentinel_model = MagicMock(name="quantized_model")
        prepared = MagicMock(name="prepared_model")
        with patch("autotune.utils.prepare_model_for_kbit_training", return_value=prepared) as prep:
            from autotune.utils import prepare_qlora_model

            out = prepare_qlora_model(sentinel_model, use_gradient_checkpointing=True)
            prep.assert_called_once_with(sentinel_model, use_gradient_checkpointing=True)
            assert out is prepared
