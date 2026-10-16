"""Tests for tokenizer customization (get_tokenizer, resize_model_embeddings, extract_tokenizer_kwargs)."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from autotune.utils import (
    extract_tokenizer_kwargs,
    get_tokenizer,
    resize_model_embeddings,
)

pytestmark = pytest.mark.slow

MODEL_NAME = "facebook/opt-125m"

LLMON_TOKENS = [
    "<|open|>",
    "<|close|>",
    "<|open_end|>",
    "<|self_close|>",
    "<|.|>",
    "<|:|>",
]


class TestGetTokenizer:
    def test_default_returns_tuple(self):
        tokenizer, num_new = get_tokenizer(MODEL_NAME)
        assert num_new == 0
        assert hasattr(tokenizer, "encode")

    def test_pad_token_fallback(self):
        tokenizer, _ = get_tokenizer(MODEL_NAME)
        assert tokenizer.pad_token is not None

    def test_additional_special_tokens(self):
        tokenizer, num_new = get_tokenizer(
            MODEL_NAME,
            additional_special_tokens=LLMON_TOKENS,
        )
        assert num_new == len(LLMON_TOKENS)
        for tok in LLMON_TOKENS:
            ids = tokenizer.encode(tok, add_special_tokens=False)
            assert len(ids) == 1, f"Token {tok} should be a single ID, got {ids}"

    def test_additional_tokens(self):
        tokenizer, num_new = get_tokenizer(
            MODEL_NAME,
            additional_tokens=["newtok_alpha", "newtok_beta"],
        )
        assert num_new == 2

    def test_pad_token_override(self):
        tokenizer, _ = get_tokenizer(MODEL_NAME, pad_token="<pad>")
        assert tokenizer.pad_token == "<pad>"

    def test_eos_token_override(self):
        original_tok, _ = get_tokenizer(MODEL_NAME)
        original_eos = original_tok.eos_token
        tokenizer, _ = get_tokenizer(MODEL_NAME, eos_token="<custom_eos>")
        assert tokenizer.eos_token == "<custom_eos>"
        assert tokenizer.eos_token != original_eos

    def test_bos_token_override(self):
        tokenizer, _ = get_tokenizer(MODEL_NAME, bos_token="<custom_bos>")
        assert tokenizer.bos_token == "<custom_bos>"

    def test_no_duplicate_special_tokens(self):
        tokenizer, num_new = get_tokenizer(
            MODEL_NAME,
            additional_special_tokens=["<|open|>"],
        )
        first_vocab_size = len(tokenizer)
        tokenizer2, num_new2 = get_tokenizer(
            MODEL_NAME,
            additional_special_tokens=["<|open|>", "<|close|>"],
        )
        assert num_new2 == 2
        assert len(tokenizer2) == first_vocab_size + 1


class TestResizeModelEmbeddings:
    @pytest.fixture
    def model_and_tokenizer(self):
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
        tokenizer, num_new = get_tokenizer(
            MODEL_NAME,
            additional_special_tokens=LLMON_TOKENS,
        )
        return model, tokenizer, num_new

    def test_resize_increases_embedding_size(self, model_and_tokenizer):
        model, tokenizer, num_new = model_and_tokenizer
        original_size = model.get_input_embeddings().weight.shape[0]
        resize_model_embeddings(model, tokenizer, num_new)
        new_size = model.get_input_embeddings().weight.shape[0]
        assert new_size > original_size
        assert new_size >= len(tokenizer)

    def test_pad_to_multiple_of_64(self, model_and_tokenizer):
        model, tokenizer, num_new = model_and_tokenizer
        resize_model_embeddings(model, tokenizer, num_new, pad_to_multiple_of=64)
        new_size = model.get_input_embeddings().weight.shape[0]
        assert new_size % 64 == 0

    def test_no_resize_when_not_needed(self):
        """Backward compat: no tokens added => no resize, even if vocab isn't multiple of 64."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
        tokenizer, _ = get_tokenizer(MODEL_NAME)
        original_size = model.get_input_embeddings().weight.shape[0]
        resize_model_embeddings(model, tokenizer, num_new_tokens=0)
        assert model.get_input_embeddings().weight.shape[0] == original_size

    def test_backward_compat_no_resize_with_default_pad(self):
        """No resize happens when num_new_tokens=0 regardless of pad_to_multiple_of."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
        tokenizer, _ = get_tokenizer(MODEL_NAME)
        original_size = model.get_input_embeddings().weight.shape[0]
        resize_model_embeddings(model, tokenizer, num_new_tokens=0, pad_to_multiple_of=64)
        assert model.get_input_embeddings().weight.shape[0] == original_size

    # def test_new_embeddings_initialized_to_mean(self, model_and_tokenizer):
    #     model, tokenizer, num_new = model_and_tokenizer
    #     old_embeddings = model.get_input_embeddings().weight.data.clone()
    #     expected_mean = old_embeddings.mean(dim=0)
    #     resize_model_embeddings(model, tokenizer, num_new, pad_to_multiple_of=1)
    #     new_embeddings = model.get_input_embeddings().weight.data
    #     for i in range(num_new):
    #         torch.testing.assert_close(
    #             new_embeddings[-(num_new - i)],
    #             expected_mean,
    #             atol=1e-5,
    #             rtol=1e-5,
    #         )


class TestExtractTokenizerKwargs:
    def test_extracts_present_keys(self):
        config = {
            "model_name_or_path": "some/model",
            "additional_special_tokens": ["<tok>"],
            "pad_token": "<pad>",
            "num_train_epochs": 3,
        }
        result = extract_tokenizer_kwargs(config)
        assert result == {
            "additional_special_tokens": ["<tok>"],
            "pad_token": "<pad>",
        }

    def test_ignores_none_values(self):
        config = {
            "additional_special_tokens": None,
            "pad_token": "<pad>",
        }
        result = extract_tokenizer_kwargs(config)
        assert result == {"pad_token": "<pad>"}

    def test_empty_config(self):
        assert extract_tokenizer_kwargs({}) == {}

    def test_all_tokenizer_keys(self):
        config = {
            "tokenizer_name_or_path": "some/tokenizer",
            "additional_special_tokens": ["<a>"],
            "additional_tokens": ["b"],
            "pad_token": "<pad>",
            "eos_token": "<eos>",
            "bos_token": "<bos>",
        }
        result = extract_tokenizer_kwargs(config)
        assert len(result) == 6
