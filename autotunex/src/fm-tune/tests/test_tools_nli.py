"""Tests for autotune.tools.build_nli_dataset — pure helpers + CLI."""

import json
from unittest.mock import MagicMock

from autotune.tools.build_nli_dataset import _format_label, _make_record, main

GRANITE4_TEMPLATE = "{%- if documents %}{%- set x = x + (document | tojson) %}{%- endif %}"
GRANITE42_TEMPLATE = "{%- if enable_thinking %}<|im_start|>assistant<think>{%- endif %}"


def _tok(template=GRANITE4_TEMPLATE):
    tok = MagicMock()
    tok.chat_template = template
    tok.apply_chat_template.return_value = "RENDERED"
    tok.return_value = {"input_ids": [1, 2, 3]}
    return tok


def _dp(**over):
    dp = {"premise": "A man is eating.", "hypothesis": "A person eats.", "label": "entailment"}
    dp.update(over)
    return dp


class TestFormatLabel:
    def test_canonical_labels(self):
        for label in ("entailment", "contradiction", "neutral"):
            assert json.loads(_format_label(label)) == {"label": label}

    def test_case_and_whitespace_normalized(self):
        assert json.loads(_format_label("  Entailment ")) == {"label": "entailment"}
        assert json.loads(_format_label("NEUTRAL")) == {"label": "neutral"}

    def test_rejects_non_canonical(self):
        assert _format_label("maybe") is None
        assert _format_label("") is None

    def test_rejects_non_strings(self):
        assert _format_label(None) is None
        assert _format_label(1) is None


class TestMakeRecord:
    def test_chat_format_shape(self):
        out = _make_record(_dp(), tokenizer=None, fmt="chat")
        assert [m["role"] for m in out["input"]] == ["user"]
        assert "A man is eating." in out["input"][0]["content"]
        assert "A person eats." in out["input"][0]["content"]
        assert json.loads(out["output"]) == {"label": "entailment"}

    def test_formatted_uses_template(self):
        out = _make_record(_dp(), tokenizer=_tok(), fmt="formatted")
        assert out["input"] == "RENDERED"

    def test_thinking_disabled_by_default(self):
        tok = _tok(GRANITE42_TEMPLATE)
        _make_record(_dp(), tokenizer=tok, fmt="formatted")
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is False

    def test_never_passes_documents(self):
        tok = _tok()
        _make_record(_dp(), tokenizer=tok, fmt="formatted")
        assert "documents" not in tok.apply_chat_template.call_args.kwargs

    def test_missing_premise(self):
        assert _make_record(_dp(premise=" "), tokenizer=None, fmt="chat") is None

    def test_missing_hypothesis(self):
        assert _make_record({"premise": "P", "label": "neutral"}, tokenizer=None, fmt="chat") is None

    def test_bad_label_dropped(self):
        assert _make_record(_dp(label="unknown"), tokenizer=None, fmt="chat") is None


class TestMainBothFormats:
    def _write_raw(self, tmp_path, records):
        p = tmp_path / "raw_nli.json"
        p.write_text(json.dumps(records), encoding="utf-8")
        return p

    def test_writes_two_files(self, tmp_path, monkeypatch):
        raw = [_dp(), _dp(label="neutral"), _dp(label="bogus")]
        in_path = self._write_raw(tmp_path, raw)
        out_dir = tmp_path / "out"
        monkeypatch.setattr("autotune.tools.build_nli_dataset.load_tokenizer", lambda *a, **k: _tok())
        assert main(["--input", str(in_path), "--output-dir", str(out_dir), "--model", "dummy"]) == 0

        # Names come from the input stem + format.
        formatted = out_dir / "raw_nli_formatted.jsonl"
        chat = out_dir / "raw_nli_chat.jsonl"
        assert formatted.exists() and chat.exists()

        f_lines = formatted.read_text(encoding="utf-8").strip().split("\n")
        c_lines = chat.read_text(encoding="utf-8").strip().split("\n")
        # The bogus-label record is dropped from both.
        assert len(f_lines) == len(c_lines) == 2
        for line in f_lines:
            assert isinstance(json.loads(line)["input"], str)
        for line in c_lines:
            rec = json.loads(line)
            assert isinstance(rec["input"], list)
            assert json.loads(rec["output"])["label"] in ("entailment", "neutral")

    def test_missing_input_returns_1(self, tmp_path, capsys):
        rc = main(["--input", str(tmp_path / "nope.json"), "--output-dir", str(tmp_path / "out")])
        assert rc == 1
        assert "not found" in capsys.readouterr().err


class TestImportability:
    def test_module_imports_without_eval_intrinsics(self):
        # The prompt block used to come from a sibling repo (eval_intrinsics),
        # which made this module unimportable here; it now lives in autotune.tools.
        import importlib

        mod = importlib.import_module("autotune.tools.build_nli_dataset")
        assert mod.get_nli_messages("P", "H")[0]["role"] == "user"


class TestSingleFormat:
    def test_single_format_writes_one_file(self, tmp_path, monkeypatch):
        in_path = tmp_path / "raw_nli.json"
        in_path.write_text(json.dumps([_dp()]), encoding="utf-8")
        out_dir = tmp_path / "out"
        monkeypatch.setattr("autotune.tools.build_nli_dataset.load_tokenizer", lambda *a, **k: _tok())
        rc = main(["--input", str(in_path), "--output-dir", str(out_dir), "--format", "chat", "--model", "dummy"])
        assert rc == 0
        assert (out_dir / "raw_nli_chat.jsonl").exists()
        assert not (out_dir / "raw_nli_formatted.jsonl").exists()
