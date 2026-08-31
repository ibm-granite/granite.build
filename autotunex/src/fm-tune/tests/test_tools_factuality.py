"""Tests for autotune.tools.build_factuality_dataset — pure helpers."""

import json
from unittest.mock import MagicMock

from autotune.tools.build_factuality_dataset import (
    _build_messages,
    _extract_contexts,
    _guardian_block,
    _record_correction,
    _record_detection,
    _unique_strings,
)


class TestUniqueStrings:
    def test_preserves_order(self):
        assert _unique_strings(["b", "a", "c", "a", "b"]) == ["b", "a", "c"]

    def test_empty(self):
        assert _unique_strings([]) == []

    def test_no_duplicates(self):
        assert _unique_strings(["a", "b", "c"]) == ["a", "b", "c"]


class TestGuardianBlock:
    def test_detection_for_prompt(self):
        block = _guardian_block(is_detection=True, for_prompt=True)
        assert "yes" in block.lower()
        assert "json" in block.lower()
        assert "score" in block

    def test_detection_no_prompt(self):
        block = _guardian_block(is_detection=True, for_prompt=False)
        assert "yes" in block.lower()
        # No JSON dict format hint
        assert "json dict" not in block.lower()

    def test_correction_for_prompt(self):
        block = _guardian_block(is_detection=False, for_prompt=True)
        assert "corrected" in block.lower()
        assert "correction" in block.lower()

    def test_correction_no_prompt(self):
        block = _guardian_block(is_detection=False, for_prompt=False)
        assert "corrected" in block.lower()
        assert "json dict" not in block.lower()


class TestExtractContexts:
    def test_basic(self):
        rec = {
            "c_a1": {"text": "context 1"},
            "c_a2": {"text": "context 2"},
            "other": "ignored",
        }
        out = _extract_contexts(rec)
        assert sorted(out) == ["context 1", "context 2"]

    def test_dedup(self):
        rec = {
            "c_a1": {"text": "same"},
            "c_a2": {"text": "same"},
            "c_a3": {"text": "different"},
        }
        out = _extract_contexts(rec)
        assert len(out) == 2

    def test_skips_none_text(self):
        rec = {"c_a1": {"text": None}, "c_a2": {"text": "real"}}
        assert _extract_contexts(rec) == ["real"]

    def test_skips_non_c_a_keys(self):
        rec = {"foo": {"text": "x"}, "bar": "no dict"}
        assert _extract_contexts(rec) == []


class TestBuildMessages:
    def test_three_messages(self):
        msgs = _build_messages("query", "response", is_detection=True, for_prompt=False)
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "query"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "response"
        assert msgs[2]["role"] == "user"
        assert "guardian" in msgs[2]["content"]


class TestRecordDetection:
    def test_chat_format(self):
        dp = {
            "query": "Q",
            "response": {"text": "R", "label": "Yes"},
            "c_a1": {"text": "context"},
        }
        out = _record_detection(dp, tokenizer=None, fmt="chat", for_prompt=False)
        # In chat format, input is a list of messages
        assert isinstance(out["input"], list)
        assert len(out["input"]) == 3
        # Output is JSON dict with score
        parsed = json.loads(out["output"])
        assert parsed == {"score": "yes"}
        assert "documents" in out

    def test_formatted_uses_tokenizer(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = "TEMPLATED"
        dp = {
            "query": "Q",
            "response": {"text": "R", "label": "No"},
            "c_a1": {"text": "context"},
        }
        out = _record_detection(dp, tokenizer=tok, fmt="formatted", for_prompt=False)
        assert out["input"] == "TEMPLATED"
        assert json.loads(out["output"]) == {"score": "no"}


class TestRecordCorrection:
    def test_chat_format_short_response(self):
        tok = MagicMock()
        # Tokenizer returns 5 tokens — under max_length
        tok.return_value = {"input_ids": [1, 2, 3, 4, 5]}
        dp = {
            "query": "Q",
            "response": {"text": "R", "label": "Yes"},
            "correction": {"text": "C"},
            "c_a1": {"text": "context"},
        }
        out = _record_correction(dp, tokenizer=tok, fmt="chat", for_prompt=False, max_length=100, include_meta=False)
        assert out is not None
        parsed = json.loads(out["output"])
        assert parsed == {"correction": "C"}

    def test_length_gate_filters_long_correction(self):
        tok = MagicMock()
        # Returns 1000 tokens — exceeds max_length
        tok.return_value = {"input_ids": list(range(1000))}
        dp = {
            "query": "Q",
            "response": {"text": "R", "label": "Yes"},
            "correction": {"text": "C" * 5000},
            "c_a1": {"text": "context"},
        }
        out = _record_correction(dp, tokenizer=tok, fmt="chat", for_prompt=False, max_length=100, include_meta=False)
        assert out is None

    def test_missing_correction_uses_none(self):
        tok = MagicMock()
        tok.return_value = {"input_ids": [1, 2, 3]}
        dp = {
            "query": "Q",
            "response": {"text": "R", "label": "Yes"},
            "c_a1": {"text": "context"},
        }
        out = _record_correction(dp, tokenizer=tok, fmt="chat", for_prompt=False, max_length=100, include_meta=False)
        parsed = json.loads(out["output"])
        assert parsed == {"correction": "none"}

    def test_include_meta_adds_fields(self):
        tok = MagicMock()
        tok.return_value = {"input_ids": [1, 2, 3]}
        dp = {
            "query": "QUERY",
            "response": {"text": "RESPONSE", "label": "Yes"},
            "correction": {"text": "C"},
            "c_a1": {"text": "ctx"},
        }
        out = _record_correction(dp, tokenizer=tok, fmt="chat", for_prompt=False, max_length=100, include_meta=True)
        assert out["query"] == "QUERY"
        assert out["response"] == "RESPONSE"
